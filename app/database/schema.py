"""Idempotent PostgreSQL schema setup and migration for Stage 1.

The migration preserves the old workspace-shaped `sessions` table by renaming it
before creating the real login/work-session table. Existing workspaces and documents
are retained and grouped under an ended "Previous session history" session.
"""

from psycopg import sql

from app.database.connection import get_pool


def _prepare_legacy_sessions(connection) -> list[str]:
    """Rename the old workspace-shaped sessions table, if this is an upgrade."""
    exists = connection.execute(
        "SELECT to_regclass('public.sessions') AS table_name"
    ).fetchone()["table_name"]
    if not exists:
        return []

    columns = connection.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'sessions'
        """
    ).fetchall()
    column_names = {row["column_name"] for row in columns}
    if "started_at" in column_names:
        return []  # Already the new login/work-session table.

    # The earlier application used sessions(id, user_id, name, bytes_used, ...)
    # as its workspace table. Keep it as a migration source instead of dropping it.
    suffix = 1
    while True:
        target = "legacy_workspace_sessions" if suffix == 1 else f"legacy_workspace_sessions_{suffix}"
        target_exists = connection.execute(
            "SELECT to_regclass(%s) AS table_name", (f"public.{target}",)
        ).fetchone()["table_name"]
        if not target_exists:
            connection.execute(
                sql.SQL("ALTER TABLE public.sessions RENAME TO {}").format(sql.Identifier(target))
            )
            return [target]
        suffix += 1


def _migrate_legacy_workspace_rows(connection, legacy_table_names: list[str]) -> None:
    for table_name in legacy_table_names:
        columns = connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
            """,
            (table_name,),
        ).fetchall()
        names = {row["column_name"] for row in columns}
        if not {"id", "user_id", "name"}.issubset(names):
            continue
        bytes_expr = "COALESCE(bytes_used, 0)" if "bytes_used" in names else "0"
        created_expr = "COALESCE(created_at, NOW())" if "created_at" in names else "NOW()"
        deleted_expr = "deleted_at" if "deleted_at" in names else "NULL"
        query = sql.SQL(
            """
            INSERT INTO workspaces (id, user_id, name, bytes_used, created_at, deleted_at)
            SELECT id, user_id,
                   CASE WHEN name ~* '^session[[:space:]]+[0-9]+$'
                        THEN regexp_replace(name, '^session', 'Workspace', 'i')
                        ELSE name END,
                   {bytes_expr}, {created_expr}, {deleted_expr}
            FROM {table}
            ON CONFLICT (id) DO NOTHING
            """
        ).format(
            bytes_expr=sql.SQL(bytes_expr),
            created_expr=sql.SQL(created_expr),
            deleted_expr=sql.SQL(deleted_expr),
            table=sql.Identifier("public", table_name),
        )
        connection.execute(query)


def create_tables() -> None:
    """Create tables and apply safe additive migrations; call after opening the pool."""
    with get_pool().connection() as connection:
        with connection.transaction():
            # The native PostgreSQL installation has pgvector available; extension state is per database.
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            legacy_names = _prepare_legacy_sessions(connection)

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # Earlier schemas did not keep a username; populate one without losing accounts.
            connection.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT")
            connection.execute(
                """
                UPDATE users
                SET username = 'user_' || substr(md5(id), 1, 12)
                WHERE username IS NULL OR btrim(username) = ''
                """
            )
            connection.execute("ALTER TABLE users ALTER COLUMN username SET NOT NULL")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_unique ON users(username)")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at TIMESTAMPTZ NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT sessions_status_allowed_check CHECK (status IN ('ACTIVE', 'ENDED'))
                )
                """
            )
            # Add missing fields if this database was created by an intermediate build.
            connection.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS name TEXT")
            connection.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ACTIVE'")
            connection.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
            connection.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
            connection.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ NULL")
            connection.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
            connection.execute("UPDATE sessions SET name = 'Work session' WHERE name IS NULL")
            # Normalize labels used by the session-history UI in earlier builds.
            connection.execute(
                "UPDATE sessions SET name = 'Session ' || substring(name from '[0-9]+$') "
                "WHERE name ~* '^work session [0-9]+$'"
            )
            connection.execute("ALTER TABLE sessions ALTER COLUMN name SET NOT NULL")
            connection.execute(
                """
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='sessions_status_allowed_check'
                          AND conrelid='public.sessions'::regclass
                    ) THEN
                        ALTER TABLE sessions ADD CONSTRAINT sessions_status_allowed_check
                            CHECK (status IN ('ACTIVE', 'ENDED'));
                    END IF;
                END $$
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_started ON sessions(user_id, started_at DESC)")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_id TEXT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TIMESTAMPTZ NOT NULL,
                    revoked BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute("ALTER TABLE refresh_tokens ADD COLUMN IF NOT EXISTS session_id TEXT NULL")
            connection.execute("ALTER TABLE refresh_tokens ALTER COLUMN session_id DROP NOT NULL")
            connection.execute(
                """
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='refresh_tokens_session_id_fkey'
                          AND conrelid='public.refresh_tokens'::regclass
                          AND confrelid <> 'public.sessions'::regclass
                    ) THEN
                        ALTER TABLE refresh_tokens DROP CONSTRAINT refresh_tokens_session_id_fkey;
                    END IF;
                END $$
                """
            )
            connection.execute(
                """
                UPDATE refresh_tokens r SET session_id=NULL
                WHERE session_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM sessions s WHERE s.id=r.session_id AND s.user_id=r.user_id
                )
                """
            )
            connection.execute(
                """
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='refresh_tokens_session_id_fkey'
                          AND conrelid='public.refresh_tokens'::regclass
                          AND confrelid <> 'public.sessions'::regclass
                    ) THEN
                        ALTER TABLE refresh_tokens DROP CONSTRAINT refresh_tokens_session_id_fkey;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='refresh_tokens_session_id_fkey'
                          AND conrelid='public.refresh_tokens'::regclass
                          AND confrelid='public.sessions'::regclass
                    ) THEN
                        ALTER TABLE refresh_tokens ADD CONSTRAINT refresh_tokens_session_id_fkey
                            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
                    END IF;
                END $$
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_session_id ON refresh_tokens(session_id)")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_id TEXT NULL,
                    name TEXT NOT NULL,
                    bytes_used BIGINT NOT NULL DEFAULT 0 CHECK (bytes_used >= 0),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deleted_at TIMESTAMPTZ NULL
                )
                """
            )
            connection.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS session_id TEXT NULL")
            connection.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS bytes_used BIGINT NOT NULL DEFAULT 0")
            connection.execute("ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL")
            # Some intermediate schemas declared this FK column NOT NULL even though the
            # migration needs a temporary NULL before assigning historical sessions.
            connection.execute("ALTER TABLE workspaces ALTER COLUMN session_id DROP NOT NULL")
            connection.execute(
                """
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='workspaces_session_id_fkey'
                          AND conrelid='public.workspaces'::regclass
                          AND confrelid <> 'public.sessions'::regclass
                    ) THEN
                        ALTER TABLE workspaces DROP CONSTRAINT workspaces_session_id_fkey;
                    END IF;
                END $$
                """
            )
            connection.execute(
                """
                UPDATE workspaces w SET session_id=NULL
                WHERE session_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM sessions s WHERE s.id=w.session_id AND s.user_id=w.user_id
                )
                """
            )
            _migrate_legacy_workspace_rows(connection, legacy_names)
            # Keep the UI's workspace terminology consistent for existing rows as well.
            connection.execute(
                """
                UPDATE workspaces
                SET name = regexp_replace(name, '^session', 'Workspace', 'i')
                WHERE name ~* '^session[[:space:]]+[0-9]+$'
                """
            )

            # Chat is durable workspace data, not browser-only session state. Keep it
            # independently of login/logout so a user can return to the same notebook.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    telemetry JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace_created "
                "ON chat_messages(user_id, workspace_id, created_at, id)"
            )

            # Ensure workspace rows referenced by existing documents exist, even if an earlier
            # partial migration did not copy the old sessions table first.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    original_name TEXT NOT NULL,
                    s3_key TEXT NULL,
                    storage_uri TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size_bytes BIGINT NOT NULL,
                    content_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deleted_at TIMESTAMPTZ NULL
                )
                """
            )
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS workspace_id TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS s3_key TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash TEXT")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_uri TEXT")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_type TEXT NOT NULL DEFAULT 'application/octet-stream'")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'PENDING'")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
            # Stage 3 indexing state lives on each immutable document-version row, not on the
            # logical source. Re-indexing one version therefore cannot overwrite another version's state.
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_status TEXT NOT NULL DEFAULT 'NOT_INDEXED'")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_target_artifact_id TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedded_artifact_id TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedded_chunk_count INTEGER NOT NULL DEFAULT 0")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_error_code TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_error_message TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_requested_at TIMESTAMPTZ NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_started_at TIMESTAMPTZ NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_completed_at TIMESTAMPTZ NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")

            # Legacy documents used session_id for a workspace. Map it before dropping that column.
            connection.execute(
                """
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema='public' AND table_name='documents' AND column_name='session_id'
                    ) THEN
                        EXECUTE 'UPDATE documents SET workspace_id = session_id WHERE workspace_id IS NULL';
                    END IF;
                END $$
                """
            )
            connection.execute(
                """
                UPDATE documents d
                SET s3_key = regexp_replace(d.storage_uri, '^s3://[^/]+/', '')
                WHERE d.s3_key IS NULL AND d.storage_uri IS NOT NULL
                """
            )
            connection.execute(
                "UPDATE documents SET s3_key = 'legacy/' || user_id || '/' || id WHERE s3_key IS NULL"
            )
            connection.execute(
                "UPDATE documents SET storage_uri = 's3://legacy/' || s3_key WHERE storage_uri IS NULL"
            )
            connection.execute(
                """
                INSERT INTO workspaces (id, user_id, name, bytes_used, created_at)
                SELECT d.workspace_id, d.user_id, 'Migrated workspace', 0, MIN(d.created_at)
                FROM documents d
                LEFT JOIN workspaces w ON w.id = d.workspace_id
                WHERE d.workspace_id IS NOT NULL AND w.id IS NULL
                GROUP BY d.workspace_id, d.user_id
                ON CONFLICT (id) DO NOTHING
                """
            )

            # A real source record represents one logical document over time. Every pre-existing
            # document becomes version 1 of its own source to avoid guessing source relationships.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_sources (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    source_name TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deleted_at TIMESTAMPTZ NULL
                )
                """
            )
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id TEXT NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS version_number INTEGER NULL")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_current BOOLEAN NOT NULL DEFAULT FALSE")
            connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS supersedes_document_id TEXT NULL")
            connection.execute(
                """
                DO $$
                DECLARE d RECORD; src_id TEXT; src_name TEXT; src_slug TEXT;
                BEGIN
                    FOR d IN SELECT id, user_id, workspace_id, original_name, created_at
                             FROM documents WHERE source_id IS NULL AND workspace_id IS NOT NULL
                    LOOP
                        src_id := 'src_' || substr(md5(d.id || random()::text || clock_timestamp()::text), 1, 16);
                        src_name := regexp_replace(d.original_name, '\\.[^.]*$', '');
                        src_slug := lower(regexp_replace(src_name, '[^a-zA-Z0-9]+', '-', 'g'));
                        src_slug := trim(BOTH '-' FROM src_slug);
                        IF src_slug = '' THEN src_slug := 'document'; END IF;
                        INSERT INTO document_sources (id, user_id, workspace_id, source_name, source_key, created_at, updated_at)
                        VALUES (src_id, d.user_id, d.workspace_id, left(src_name, 200), left(src_slug, 48) || '-' || substr(md5(src_id), 1, 6), d.created_at, d.created_at);
                        UPDATE documents
                        SET source_id = src_id,
                            version_number = 1,
                            is_current = (status = 'UPLOADED' AND deleted_at IS NULL)
                        WHERE id = d.id;
                    END LOOP;
                END $$
                """
            )
            connection.execute("UPDATE documents SET version_number = 1 WHERE version_number IS NULL")
            connection.execute("UPDATE documents SET is_current = FALSE WHERE deleted_at IS NOT NULL")

            # Make sure all existing workspace rows belong to a preserved historical session.
            connection.execute(
                """
                INSERT INTO sessions (id, user_id, name, status, started_at, last_seen_at, ended_at, created_at)
                SELECT 'sess_import_' || substr(md5(w.user_id), 1, 16),
                       w.user_id, 'Previous session history', 'ENDED',
                       MIN(w.created_at), MAX(w.created_at), MAX(w.created_at), MIN(w.created_at)
                FROM workspaces w
                WHERE w.session_id IS NULL
                GROUP BY w.user_id
                ON CONFLICT (id) DO NOTHING
                """
            )
            connection.execute(
                """
                UPDATE workspaces w
                SET session_id = s.id
                FROM sessions s
                WHERE w.session_id IS NULL AND s.user_id = w.user_id AND s.name = 'Previous session history'
                """
            )
            connection.execute("ALTER TABLE workspaces ALTER COLUMN session_id SET NOT NULL")
            connection.execute(
                """
                DO $$ BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='workspaces_session_id_fkey'
                          AND conrelid='public.workspaces'::regclass
                          AND confrelid <> 'public.sessions'::regclass
                    ) THEN
                        ALTER TABLE workspaces DROP CONSTRAINT workspaces_session_id_fkey;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='workspaces_session_id_fkey'
                          AND conrelid='public.workspaces'::regclass
                          AND confrelid='public.sessions'::regclass
                    ) THEN
                        ALTER TABLE workspaces ADD CONSTRAINT workspaces_session_id_fkey
                            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE;
                    END IF;
                END $$
                """
            )

            # Add document FKs only after old rows have been assigned a source and workspace.
            connection.execute(
                """
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='documents_workspace_id_fkey'
                          AND conrelid='public.documents'::regclass
                    ) THEN
                        ALTER TABLE documents ADD CONSTRAINT documents_workspace_id_fkey
                            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='documents_source_id_fkey'
                          AND conrelid='public.documents'::regclass
                    ) THEN
                        ALTER TABLE documents ADD CONSTRAINT documents_source_id_fkey
                            FOREIGN KEY (source_id) REFERENCES document_sources(id) ON DELETE CASCADE;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='documents_supersedes_document_id_fkey'
                          AND conrelid='public.documents'::regclass
                    ) THEN
                        ALTER TABLE documents ADD CONSTRAINT documents_supersedes_document_id_fkey
                            FOREIGN KEY (supersedes_document_id) REFERENCES documents(id) ON DELETE SET NULL;
                    END IF;
                END $$
                """
            )
            connection.execute("ALTER TABLE documents ALTER COLUMN workspace_id SET NOT NULL")
            connection.execute("ALTER TABLE documents ALTER COLUMN source_id SET NOT NULL")
            connection.execute("ALTER TABLE documents ALTER COLUMN version_number SET DEFAULT 1")
            connection.execute("ALTER TABLE documents ALTER COLUMN version_number SET NOT NULL")
            connection.execute("ALTER TABLE documents ALTER COLUMN s3_key SET NOT NULL")

            # Replace legacy status checks with a version-aware lifecycle.
            connection.execute(
                """
                DO $$ DECLARE c RECORD; BEGIN
                    FOR c IN
                        SELECT conname FROM pg_constraint
                        WHERE conrelid='public.documents'::regclass
                          AND contype='c'
                          AND pg_get_constraintdef(oid) ILIKE '%status%'
                          AND conname <> 'documents_status_allowed_check'
                    LOOP
                        EXECUTE format('ALTER TABLE public.documents DROP CONSTRAINT %I', c.conname);
                    END LOOP;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='documents_status_allowed_check'
                          AND conrelid='public.documents'::regclass
                    ) THEN
                        ALTER TABLE public.documents ADD CONSTRAINT documents_status_allowed_check
                            CHECK (status IN ('PENDING', 'UPLOADED', 'FAILED', 'SUPERSEDED', 'DELETED'));
                    END IF;
                END $$
                """
            )
            # Remove the earlier NOT VALID size check if it exists. PostgreSQL still applies a
            # NOT VALID CHECK to updated legacy rows, which could prevent users from retiring a
            # previously uploaded file larger than today's 20 MiB limit. Enforce the limit on new
            # document-version inserts instead; immutable identity fields prevent later size edits.
            connection.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_file_size_allowed_check")
            connection.execute(
                """
                CREATE OR REPLACE FUNCTION enforce_document_version_size_on_insert()
                RETURNS trigger AS $$
                BEGIN
                    IF NEW.file_size_bytes <= 0 OR NEW.file_size_bytes > 20971520 THEN
                        RAISE EXCEPTION 'new document versions must be between 1 byte and 20 MiB';
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """
            )
            connection.execute("DROP TRIGGER IF EXISTS trg_document_version_size_limit ON documents")
            connection.execute(
                """
                CREATE TRIGGER trg_document_version_size_limit
                BEFORE INSERT ON documents
                FOR EACH ROW EXECUTE FUNCTION enforce_document_version_size_on_insert()
                """
            )
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_s3_key ON documents(s3_key)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_version ON documents(source_id, version_number)")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_one_current_per_source
                ON documents(source_id)
                WHERE is_current = TRUE AND deleted_at IS NULL AND status = 'UPLOADED'
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_workspace_id ON documents(workspace_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_workspace_file_hash "
                "ON documents(workspace_id, file_hash) WHERE deleted_at IS NULL"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sources_workspace_id ON document_sources(workspace_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sources_user_id ON document_sources(user_id)")

            # An upload intent binds one pending immutable document-version record to one
            # unique staging key. The browser receives permissions for the staging key only;
            # the API promotes verified bytes to documents.s3_key after completion.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_intents (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    staging_key TEXT NOT NULL UNIQUE,
                    expected_sha256 CHAR(64) NOT NULL CHECK (expected_sha256 ~ '^[a-f0-9]{64}$'),
                    expected_size_bytes BIGINT NOT NULL CHECK (expected_size_bytes > 0 AND expected_size_bytes <= 20971520),
                    expected_content_type TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ NULL,
                    cancelled_at TIMESTAMPTZ NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_upload_intents_user_id ON upload_intents(user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_upload_intents_workspace_id ON upload_intents(workspace_id)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_upload_intents_open_expiry "
                "ON upload_intents(expires_at) WHERE completed_at IS NULL AND cancelled_at IS NULL"
            )

            # Stage 2 durable processing state. These tables intentionally store only PII finding
            # metadata (entity type/confidence/location), never the original matched sensitive value.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'QUEUED',
                    progress_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    artifact_id TEXT NULL,
                    pii_finding_count INTEGER NOT NULL DEFAULT 0 CHECK (pii_finding_count >= 0),
                    error_code TEXT NULL,
                    error_message TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ NULL,
                    completed_at TIMESTAMPTZ NULL,
                    finished_at TIMESTAMPTZ NULL,
                    CONSTRAINT ingestion_runs_status_allowed_check CHECK (status IN (
                        'QUEUED','EXTRACTING','PII_SCANNING','NLP_ENRICHMENT','CHUNKING',
                        'UPLOADING_ARTIFACT','SANITIZED','QUARANTINED','FAILED'
                    ))
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS extracted_artifacts (
                    id TEXT PRIMARY KEY,
                    ingestion_run_id TEXT NOT NULL UNIQUE REFERENCES ingestion_runs(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    document_version_number INTEGER NOT NULL,
                    artifact_type TEXT NOT NULL DEFAULT 'STRUCTURED_EXTRACTION',
                    status TEXT NOT NULL CHECK (status IN ('SANITIZED','QUARANTINED')),
                    s3_bucket TEXT NOT NULL,
                    s3_key TEXT NOT NULL UNIQUE,
                    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
                    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
                    schema_version TEXT NOT NULL DEFAULT '1.0',
                    element_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
                    pii_entity_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                """
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='ingestion_runs_artifact_id_fkey'
                          AND conrelid='public.ingestion_runs'::regclass
                    ) THEN
                        ALTER TABLE ingestion_runs ADD CONSTRAINT ingestion_runs_artifact_id_fkey
                            FOREIGN KEY (artifact_id) REFERENCES extracted_artifacts(id) ON DELETE SET NULL;
                    END IF;
                END $$
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_findings (
                    id TEXT PRIMARY KEY,
                    ingestion_run_id TEXT NOT NULL REFERENCES ingestion_runs(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    source_element_id TEXT NULL,
                    field_path TEXT NULL,
                    entity_type TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
                    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
                    action TEXT NOT NULL DEFAULT 'REDACTED_AND_QUARANTINED',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_runs_document_created ON ingestion_runs(document_id, created_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_runs_workspace_status ON ingestion_runs(workspace_id, status, created_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_workspace_created ON extracted_artifacts(workspace_id, created_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_document_version ON extracted_artifacts(document_id, document_version_number DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_privacy_findings_workspace_created ON privacy_findings(workspace_id, created_at DESC)")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_one_active_run_per_document
                ON ingestion_runs(document_id)
                WHERE status IN ('QUEUED','EXTRACTING','PII_SCANNING','NLP_ENRICHMENT','CHUNKING','UPLOADING_ARTIFACT')
                """
            )
            # A worker may have been interrupted by a local server restart. Keep that run visible but
            # make a retry possible after a day; this migration does not delete any document data.
            connection.execute(
                """
                UPDATE ingestion_runs SET status='FAILED', error_code='WORKER_INTERRUPTED',
                    error_message='Processing was interrupted; retry the ingestion run.',
                    finished_at=NOW(), updated_at=NOW()
                WHERE status IN ('QUEUED','EXTRACTING','PII_SCANNING','NLP_ENRICHMENT','CHUNKING','UPLOADING_ARTIFACT')
                  AND updated_at < NOW() - INTERVAL '24 hours'
                """
            )

            # Stage 3: store text/provenance and vector values in separate tables. The vector
            # width is fixed at 1024 for BGE-M3; changing the embedding model/dimension requires
            # an explicit re-index/migration rather than silently mixing incomparable vectors.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    chunk_key TEXT NOT NULL,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    source_id TEXT NULL REFERENCES document_sources(id) ON DELETE SET NULL,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    document_version_number INTEGER NOT NULL CHECK (document_version_number > 0),
                    artifact_id TEXT NOT NULL REFERENCES extracted_artifacts(id) ON DELETE CASCADE,
                    ingestion_run_id TEXT NOT NULL REFERENCES ingestion_runs(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    chunk_type TEXT NOT NULL CHECK (chunk_type IN ('text','table')),
                    content TEXT NOT NULL,
                    content_sha256 CHAR(64) NOT NULL CHECK (content_sha256 ~ '^[a-f0-9]{64}$'),
                    token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
                    source_element_id TEXT NULL,
                    source_element_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    source_location JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source_locations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    heading_context JSONB NOT NULL DEFAULT '[]'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT document_chunks_document_chunk_key_unique UNIQUE (document_id, chunk_key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
                    embedding vector(1024) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL DEFAULT 1024 CHECK (embedding_dimensions = 1024),
                    content_sha256 CHAR(64) NOT NULL CHECK (content_sha256 ~ '^[a-f0-9]{64}$'),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_workspace_user ON document_chunks(workspace_id, user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id, chunk_index)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_artifact ON document_chunks(artifact_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(embedding_model)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw_cosine "
                "ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
            )
            connection.execute(
                """
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='documents_embedding_target_artifact_id_fkey'
                          AND conrelid='public.documents'::regclass
                    ) THEN
                        ALTER TABLE documents ADD CONSTRAINT documents_embedding_target_artifact_id_fkey
                            FOREIGN KEY (embedding_target_artifact_id) REFERENCES extracted_artifacts(id) ON DELETE SET NULL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='documents_embedded_artifact_id_fkey'
                          AND conrelid='public.documents'::regclass
                    ) THEN
                        ALTER TABLE documents ADD CONSTRAINT documents_embedded_artifact_id_fkey
                            FOREIGN KEY (embedded_artifact_id) REFERENCES extracted_artifacts(id) ON DELETE SET NULL;
                    END IF;
                END $$
                """
            )

            # A document row is one immutable version. Mutable lifecycle columns (status,
            # is_current, deleted_at and supersedes_document_id) may change; identity and object
            # coordinates must not be rewritten after the version is created.
            connection.execute(
                """
                CREATE OR REPLACE FUNCTION protect_document_version_identity()
                RETURNS trigger AS $$
                BEGIN
                    IF NEW.id IS DISTINCT FROM OLD.id
                       OR NEW.source_id IS DISTINCT FROM OLD.source_id
                       OR NEW.workspace_id IS DISTINCT FROM OLD.workspace_id
                       OR NEW.user_id IS DISTINCT FROM OLD.user_id
                       OR NEW.original_name IS DISTINCT FROM OLD.original_name
                       OR NEW.s3_key IS DISTINCT FROM OLD.s3_key
                       OR NEW.storage_uri IS DISTINCT FROM OLD.storage_uri
                       OR NEW.file_hash IS DISTINCT FROM OLD.file_hash
                       OR NEW.file_size_bytes IS DISTINCT FROM OLD.file_size_bytes
                       OR NEW.content_type IS DISTINCT FROM OLD.content_type
                       OR NEW.version_number IS DISTINCT FROM OLD.version_number THEN
                        RAISE EXCEPTION 'document version identity fields are immutable';
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """
            )
            connection.execute("DROP TRIGGER IF EXISTS trg_document_version_identity_immutable ON documents")
            connection.execute(
                """
                CREATE TRIGGER trg_document_version_identity_immutable
                BEFORE UPDATE ON documents
                FOR EACH ROW EXECUTE FUNCTION protect_document_version_identity()
                """
            )

            # bytes_used represents files promoted into final storage. Pending uploads do not
            # count until confirmation succeeds.
            connection.execute(
                """
                UPDATE workspaces w
                SET bytes_used = COALESCE((
                    SELECT SUM(d.file_size_bytes)
                    FROM documents d
                    WHERE d.workspace_id = w.id AND d.deleted_at IS NULL
                      AND d.status IN ('UPLOADED', 'SUPERSEDED')
                ), 0)
                """
            )
            # Remove the old overloaded column only after document references were migrated.
            connection.execute("ALTER TABLE documents DROP COLUMN IF EXISTS session_id CASCADE")
