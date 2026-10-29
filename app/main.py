"""Sourcebook API: authentication, versioned S3 documents, Stage 2 ingestion and Stage 3 vector search."""

import base64
import logging
import os
from datetime import datetime, timezone
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Cookie,
    Depends,
    FastAPI,
    HTTPException,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from botocore.exceptions import ClientError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator
from psycopg import Connection
from psycopg.types.json import Jsonb

from app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    CORS_ORIGINS,
    COOKIE_SECURE,
    MAX_FILE_SIZE_BYTES,
    JWT_SECRET,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    STORAGE_PROVIDER,
    PRESIGNED_UPLOAD_EXPIRE_SECONDS,
    REFRESH_TOKEN_EXPIRE_DAYS,
    S3_BUCKET_NAME,
)
from app.database.connection import close_db_pool, get_db, get_pool, open_db_pool
from app.database.schema import create_tables
from app.storage.s3_compatible import ensure_bucket
from app.s3_client import (
    copy_object,
    create_download_url,
    create_presigned_upload_post,
    delete_object,
    head_object,
)
from app.security import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    hash_token,
    new_id,
    verify_password,
)
from app.ingestion.vector_index import (
    Stage3Error,
    delete_document_vectors,
    delete_source_vectors,
    embedding_health,
    process_document_embedding,
    queue_document_index,
    semantic_search,
)
from app.ingestion.rag_answer import (
    answer_workspace_query,
    rag_health,
    stage4_rerank_workspace_query,
    stage5_generate_llm_answer,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("stage1_backend")

app = FastAPI(
    title="Sourcebook — Stage 1 to Stage 5 End-to-End RAG API",
    version="0.5.0",
    description=(
        "Sourcebook accounts, workspaces, immutable document versions, direct-to-S3 uploads, "
        "Stage 2 extraction/privacy/chunk preparation, and Stage 3 local Ollama embeddings, "
        "PostgreSQL/pgvector storage and workspace-scoped semantic search."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

bearer_scheme = HTTPBearer(auto_error=False)

MAX_PASSWORD_BYTES = 72  # bcrypt input limit
ALLOWED_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def password_must_fit_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError("Password must be at most 72 bytes when UTF-8 encoded")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)

    @field_validator("password")
    @classmethod
    def password_must_fit_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError("Password must be at most 72 bytes when UTF-8 encoded")
        return value


class WorkspaceCreateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)


class ChatMessageCreateRequest(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=100_000)
    citations: list[dict] = Field(default_factory=list)
    telemetry: dict = Field(default_factory=dict)


class SemanticSearchRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    include_superseded: bool = False


class Stage4RerankRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=8000)
    candidate_k: int = Field(default=20, ge=3, le=50)
    top_k: int = Field(default=3, ge=1, le=10)
    include_superseded: bool = False


class Stage5GenerateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    reranked_chunks: list[dict] = Field(default_factory=list)
    chat_model: str | None = Field(default=None, max_length=100)
    history: list[dict[str, str]] = Field(default_factory=list)


class RagAskRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=8000)
    candidate_k: int = Field(default=20, ge=3, le=50)
    top_k: int = Field(default=3, ge=1, le=10)
    include_superseded: bool = False
    history: list[dict[str, str]] = Field(default_factory=list)
    chat_model: str | None = Field(default=None, max_length=100)


@app.on_event("startup")
def startup() -> None:
    # Secrets are supplied through .env; never silently start with a known signing key.
    if len(JWT_SECRET) < 32:
        raise RuntimeError("Set JWT_SECRET in .env to a random value of at least 32 characters.")
    if STORAGE_PROVIDER == "minio" and (not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY):
        raise RuntimeError("Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY in .env before starting MinIO mode.")

    # MinIO is the local-first default. AWS S3 buckets remain externally managed.
    ensure_bucket()
    open_db_pool()
    create_tables()
    logger.info("Storage and PostgreSQL initialized; Sourcebook schema is ready")


@app.on_event("shutdown")
def shutdown() -> None:
    close_db_pool()


def set_refresh_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key="refresh_token",
        value=raw_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/api/v1/auth",
    )


def _session_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "started_at": row["started_at"],
        "ended_at": row.get("ended_at"),
    }


def _next_number(names: list[str], pattern: str) -> int:
    numbers = []
    for name in names:
        match = re.fullmatch(pattern, str(name).strip(), flags=re.IGNORECASE)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def create_working_session(connection: Connection, user_id: str) -> dict:
    """Create a new login/work period without evicting prior history."""
    # Serializes session creation and numbering for this account.
    connection.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", (user_id,))
    connection.execute(
        """
        UPDATE sessions s
        SET status='ENDED', ended_at=COALESCE(s.ended_at, NOW())
        WHERE s.user_id=%s AND s.status='ACTIVE'
          AND NOT EXISTS (
              SELECT 1 FROM refresh_tokens r
              WHERE r.session_id=s.id AND r.revoked=FALSE AND r.expires_at>NOW()
          )
        """,
        (user_id,),
    )
    names = connection.execute(
        "SELECT name FROM sessions WHERE user_id=%s",
        (user_id,),
    ).fetchall()
    number = _next_number([row["name"] for row in names], r"(?:Work\s+)?Session\s+(\d+)")
    row = connection.execute(
        """
        INSERT INTO sessions (id, user_id, name, status, started_at, last_seen_at)
        VALUES (%s, %s, %s, 'ACTIVE', NOW(), NOW())
        RETURNING id, name, status, started_at, ended_at
        """,
        (new_id("sess"), user_id, f"Session {number}"),
    ).fetchone()
    return row


def create_workspace_for_session(connection: Connection, user_id: str, session_id: str, name: str | None = None) -> dict:
    """Create a workspace with numbering scoped to one session; keep prior workspaces."""
    parent = connection.execute(
        "SELECT id FROM sessions WHERE id=%s AND user_id=%s AND status='ACTIVE' FOR UPDATE",
        (session_id, user_id),
    ).fetchone()
    if not parent:
        raise HTTPException(status_code=403, detail="Cannot create a workspace outside an active work session")

    existing_names = connection.execute(
        "SELECT name FROM workspaces WHERE user_id=%s AND session_id=%s",
        (user_id, session_id),
    ).fetchall()
    number = _next_number([row["name"] for row in existing_names], r"Workspace\s+(\d+)")
    workspace_name = name.strip() if name and name.strip() else f"Workspace {number}"
    workspace = connection.execute(
        """
        INSERT INTO workspaces (id, user_id, session_id, name)
        VALUES (%s, %s, %s, %s)
        RETURNING id, session_id, name, bytes_used, created_at
        """,
        (new_id("wsp"), user_id, session_id, workspace_name),
    ).fetchone()
    return workspace


def persist_refresh_token(
    connection: Connection, user_id: str, session_id: str, raw_token: str, expires_at
) -> None:
    connection.execute(
        """
        INSERT INTO refresh_tokens (id, user_id, session_id, token_hash, expires_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (new_id("rt"), user_id, session_id, hash_token(raw_token), expires_at),
    )


def issue_tokens(connection: Connection, user: dict, session: dict, response: Response) -> dict:
    access_token, _ = create_access_token(user["id"], user["email"], session["id"])
    raw_refresh, refresh_expiry = create_refresh_token()
    persist_refresh_token(connection, user["id"], session["id"], raw_refresh, refresh_expiry)
    connection.execute("UPDATE sessions SET last_seen_at=NOW() WHERE id=%s", (session["id"],))
    set_refresh_cookie(response, raw_refresh)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {"id": user["id"], "username": user["username"], "email": user["email"]},
        "session": _session_dict(session),
    }


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer access token required")
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        session_id = payload.get("sid")
        if not user_id or not session_id:
            raise HTTPException(status_code=401, detail="Access token is missing its working session")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Access token expired or invalid") from None

    user = connection.execute(
        """
        SELECT u.id, u.username, u.email, s.id AS session_id, s.name AS session_name
        FROM users u
        JOIN sessions s ON s.id=%s AND s.user_id=u.id AND s.status='ACTIVE'
        WHERE u.id=%s AND u.is_active=TRUE
        """,
        (session_id, user_id),
    ).fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User or working session is inactive")
    return user


def _end_session_from_refresh_token(connection: Connection, raw_token: str | None) -> None:
    if not raw_token:
        return
    row = connection.execute(
        "SELECT session_id FROM refresh_tokens WHERE token_hash=%s FOR UPDATE",
        (hash_token(raw_token),),
    ).fetchone()
    connection.execute(
        "UPDATE refresh_tokens SET revoked=TRUE WHERE token_hash=%s",
        (hash_token(raw_token),),
    )
    if row and row["session_id"]:
        connection.execute(
            """
            UPDATE sessions SET status='ENDED', ended_at=COALESCE(ended_at, NOW())
            WHERE id=%s AND status='ACTIVE'
            """,
            (row["session_id"],),
        )
        connection.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE session_id=%s", (row["session_id"],))


def _user_payload(user: dict) -> dict:
    return {"id": user["id"], "username": user["username"], "email": user["email"]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/auth/signup", status_code=status.HTTP_201_CREATED)
def signup(
    request: SignupRequest,
    response: Response,
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    user_id = new_id("usr")
    username = request.username.strip()
    email = str(request.email).lower().strip()
    user = connection.execute(
        """
        INSERT INTO users (id, username, email, password_hash)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING id, username, email
        """,
        (user_id, username, email, hash_password(request.password)),
    ).fetchone()
    if user is None:
        raise HTTPException(status_code=409, detail="Username or email is already registered")

    session = create_working_session(connection, user["id"])
    workspace = create_workspace_for_session(connection, user["id"], session["id"], "Workspace 1")
    result = issue_tokens(connection, user, session, response)
    result["workspace"] = workspace
    return result


@app.post("/api/v1/auth/login")
def login(
    request: LoginRequest,
    response: Response,
    connection: Annotated[Connection, Depends(get_db)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> dict:
    user = connection.execute(
        "SELECT id, username, email, password_hash FROM users WHERE email=%s AND is_active=TRUE",
        (str(request.email).lower().strip(),),
    ).fetchone()
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # If the browser logs in again without first calling logout, retire that cookie's session.
    _end_session_from_refresh_token(connection, refresh_token)
    session = create_working_session(connection, user["id"])
    workspace = create_workspace_for_session(connection, user["id"], session["id"], "Workspace 1")
    result = issue_tokens(connection, user, session, response)
    result["workspace"] = workspace
    return result


@app.post("/api/v1/auth/refresh")
def refresh_access_token(
    response: Response,
    connection: Annotated[Connection, Depends(get_db)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> dict:
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token cookie is missing")

    token_hash_value = hash_token(refresh_token)
    token_row = connection.execute(
        """
        SELECT u.id, u.username, u.email, r.session_id, s.name AS session_name,
               s.status AS session_status, s.started_at AS session_started_at, s.ended_at AS session_ended_at
        FROM refresh_tokens r
        JOIN users u ON u.id=r.user_id
        LEFT JOIN sessions s ON s.id=r.session_id
        WHERE r.token_hash=%s AND r.revoked=FALSE AND r.expires_at>NOW() AND u.is_active=TRUE
          AND (r.session_id IS NULL OR s.status='ACTIVE')
        FOR UPDATE OF r
        """,
        (token_hash_value,),
    ).fetchone()
    if not token_row:
        raise HTTPException(status_code=401, detail="Refresh token expired, revoked, or its session ended")

    if token_row["session_id"]:
        session = {
            "id": token_row["session_id"],
            "name": token_row["session_name"],
            "status": token_row["session_status"],
            "started_at": token_row["session_started_at"],
            "ended_at": token_row["session_ended_at"],
        }
    else:
        # One-time compatibility path for a refresh cookie created before work sessions existed.
        session = create_working_session(connection, token_row["id"])
        create_workspace_for_session(connection, token_row["id"], session["id"], "Workspace 1")

    connection.execute("UPDATE refresh_tokens SET revoked=TRUE WHERE token_hash=%s", (token_hash_value,))
    user = {"id": token_row["id"], "username": token_row["username"], "email": token_row["email"]}
    result = issue_tokens(connection, user, session, response)
    result["user"] = _user_payload(user)
    return result


@app.post("/api/v1/auth/logout")
def logout(
    response: Response,
    connection: Annotated[Connection, Depends(get_db)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> dict:
    _end_session_from_refresh_token(connection, refresh_token)
    response.delete_cookie("refresh_token", path="/api/v1/auth")
    return {"message": "Logged out"}


@app.get("/api/v1/sessions")
def list_sessions(
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    session_rows = connection.execute(
        """
        SELECT id, name, status, started_at, ended_at
        FROM sessions WHERE user_id=%s
        ORDER BY started_at DESC, created_at DESC
        """,
        (user["id"],),
    ).fetchall()
    workspace_rows = connection.execute(
        """
        SELECT w.id, w.session_id, w.name, w.bytes_used, w.created_at,
               COUNT(d.id) FILTER (WHERE d.status='UPLOADED' AND d.is_current=TRUE AND d.deleted_at IS NULL) AS document_count
        FROM workspaces w
        LEFT JOIN documents d ON d.workspace_id=w.id
        WHERE w.user_id=%s AND w.deleted_at IS NULL
        GROUP BY w.id
        ORDER BY w.created_at ASC
        """,
        (user["id"],),
    ).fetchall()
    workspace_by_session: dict[str, list[dict]] = {}
    for row in workspace_rows:
        workspace_by_session.setdefault(row["session_id"], []).append(row)

    result = []
    for session in session_rows:
        record = _session_dict(session)
        record["workspaces"] = workspace_by_session.get(session["id"], [])
        result.append(record)
    return {"sessions": result, "current_session_id": user["session_id"]}


@app.get("/api/v1/workspaces")
def list_workspaces(
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    rows = connection.execute(
        """
        SELECT w.id, w.session_id, w.name, w.bytes_used, w.created_at,
               s.status AS session_status
        FROM workspaces w JOIN sessions s ON s.id=w.session_id
        WHERE w.user_id=%s AND w.deleted_at IS NULL
        ORDER BY s.started_at DESC, w.created_at ASC
        """,
        (user["id"],),
    ).fetchall()
    return {"workspaces": rows, "current_session_id": user["session_id"]}


@app.post("/api/v1/workspaces", status_code=status.HTTP_201_CREATED)
def create_workspace(
    request: WorkspaceCreateRequest,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    workspace = create_workspace_for_session(connection, user["id"], user["session_id"], request.name)
    return {"workspace": workspace}


def _require_owned_workspace(connection: Connection, user_id: str, workspace_id: str) -> dict:
    workspace = connection.execute(
        """
        SELECT id, user_id, session_id, name
        FROM workspaces
        WHERE id=%s AND user_id=%s AND deleted_at IS NULL
        """,
        (workspace_id, user_id),
    ).fetchone()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@app.get("/api/v1/workspaces/{workspace_id}/chat-history")
def get_workspace_chat_history(
    workspace_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Return durable chat turns for this user's workspace, in chronological order."""
    _require_owned_workspace(connection, user["id"], workspace_id)
    rows = connection.execute(
        """
        SELECT id, role, content, citations, telemetry, created_at
        FROM chat_messages
        WHERE user_id=%s AND workspace_id=%s
        ORDER BY created_at ASC, id ASC
        """,
        (user["id"], workspace_id),
    ).fetchall()
    return {"workspace_id": workspace_id, "messages": rows}


@app.post("/api/v1/workspaces/{workspace_id}/chat-history", status_code=status.HTTP_201_CREATED)
def append_workspace_chat_message(
    workspace_id: str,
    request: ChatMessageCreateRequest,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Persist one user or assistant turn; login sessions do not own chat retention."""
    _require_owned_workspace(connection, user["id"], workspace_id)
    row = connection.execute(
        """
        INSERT INTO chat_messages (id, user_id, workspace_id, role, content, citations, telemetry)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, role, content, citations, telemetry, created_at
        """,
        (
            new_id("msg"), user["id"], workspace_id, request.role, request.content,
            Jsonb(request.citations), Jsonb(request.telemetry),
        ),
    ).fetchone()
    return {"workspace_id": workspace_id, "message": row}


@app.delete("/api/v1/workspaces/{workspace_id}/chat-history")
def clear_workspace_chat_history(
    workspace_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Clear only the selected workspace's chat, never documents or other workspaces."""
    _require_owned_workspace(connection, user["id"], workspace_id)
    connection.execute(
        "DELETE FROM chat_messages WHERE user_id=%s AND workspace_id=%s",
        (user["id"], workspace_id),
    )
    return {"workspace_id": workspace_id, "deleted": True}


def _safe_filename(filename: str | None) -> str:
    cleaned = (filename or "document.txt").replace("\\", "/")
    name = PurePosixPath(cleaned).name.strip()
    return (name or "document.txt")[:255]


def _slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug[:48] or "document"


def _content_type_for(filename: str) -> str:
    suffix = PurePosixPath(filename).suffix.lower()
    content_type = ALLOWED_TYPES.get(suffix)
    if not content_type:
        allowed = ", ".join(ALLOWED_TYPES.keys())
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed extensions: {allowed}")
    return content_type


ACTIVE_INGESTION_STATUSES = (
    "QUEUED", "EXTRACTING", "PII_SCANNING", "NLP_ENRICHMENT", "CHUNKING", "UPLOADING_ARTIFACT"
)
FINAL_INGESTION_STATUSES = {"SANITIZED", "QUARANTINED"}


def _queue_ingestion_run(
    connection: Connection, document_id: str, user_id: str, force: bool = False
) -> tuple[dict, bool]:
    """Create one queued run per immutable document version; return (run, newly_queued)."""
    document = connection.execute(
        """
        SELECT id, user_id, workspace_id, status, deleted_at
        FROM documents WHERE id=%s AND user_id=%s AND deleted_at IS NULL
          AND status IN ('UPLOADED','SUPERSEDED')
        FOR UPDATE
        """,
        (document_id, user_id),
    ).fetchone()
    if not document:
        raise HTTPException(status_code=404, detail="Uploaded document version not found")

    active = connection.execute(
        """
        SELECT id, status, created_at FROM ingestion_runs
        WHERE document_id=%s AND status IN (
            'QUEUED','EXTRACTING','PII_SCANNING','NLP_ENRICHMENT','CHUNKING','UPLOADING_ARTIFACT'
        ) ORDER BY created_at DESC LIMIT 1
        """,
        (document_id,),
    ).fetchone()
    if active:
        connection.commit()
        return dict(active), False

    latest = connection.execute(
        "SELECT id, status, created_at FROM ingestion_runs WHERE document_id=%s ORDER BY created_at DESC LIMIT 1",
        (document_id,),
    ).fetchone()
    if latest and latest["status"] in FINAL_INGESTION_STATUSES and not force:
        connection.commit()
        return dict(latest), False

    run_id = new_id("ing")
    connection.execute(
        """
        INSERT INTO ingestion_runs (id, document_id, user_id, workspace_id, status, progress_metadata)
        VALUES (%s,%s,%s,%s,'QUEUED',%s)
        """,
        (run_id, document_id, user_id, document["workspace_id"], Jsonb({"stage": "queued"})),
    )
    connection.commit()
    return {"id": run_id, "status": "QUEUED", "created_at": None}, True


def _schedule_document_ingestion(background_tasks: BackgroundTasks, document_id: str, user_id: str, run_id: str) -> None:
    from app.ingestion.pipeline import process_document_run
    background_tasks.add_task(process_document_run, run_id, document_id, user_id)


def _recalculate_workspace_bytes(connection: Connection, workspace_id: str) -> None:
    connection.execute(
        """
        UPDATE workspaces w SET bytes_used=COALESCE((
            SELECT SUM(d.file_size_bytes) FROM documents d
            WHERE d.workspace_id=w.id AND d.deleted_at IS NULL
              AND d.status IN ('UPLOADED', 'SUPERSEDED')
        ), 0) WHERE w.id=%s
        """,
        (workspace_id,),
    )


class UploadIntentRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=255)
    file_size_bytes: int = Field(gt=0, le=MAX_FILE_SIZE_BYTES)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_id: str | None = Field(default=None, max_length=100)

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


def _upload_response_document(row: dict) -> dict:
    return {
        "id": row["id"],
        "source_id": row["source_id"],
        "source_name": row.get("source_name"),
        "source_key": row.get("source_key"),
        "workspace_id": row["workspace_id"],
        "filename": row["original_name"],
        "file_size_bytes": row["file_size_bytes"],
        "content_type": row["content_type"],
        "file_hash": row["file_hash"],
        "version_number": row["version_number"],
        "previous_version_number": row.get("previous_version_number"),
        "change_detected": row.get("change_detected", False),
        "status": row["status"],
        "is_current": row["is_current"],
        "storage_uri": row["storage_uri"],
        "s3_key": row["s3_key"],
        "embedding_status": row.get("embedding_status", "NOT_INDEXED"),
        "embedding_model": row.get("embedding_model"),
        "embedding_dimensions": row.get("embedding_dimensions"),
        "embedded_chunk_count": row.get("embedded_chunk_count", 0),
    }


def _return_duplicate(connection: Connection, document_id: str, workspace_id: str, source_id: str | None) -> dict:
    row = connection.execute(
        """
        SELECT d.id, d.source_id, s.source_name, s.source_key, d.workspace_id,
               d.original_name, d.file_size_bytes, d.content_type, d.file_hash,
               d.version_number, d.status, d.is_current, d.storage_uri, d.s3_key
        FROM documents d JOIN document_sources s ON s.id=d.source_id
        WHERE d.id=%s AND d.workspace_id=%s AND d.deleted_at IS NULL
          AND s.deleted_at IS NULL
        """,
        (document_id, workspace_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=409, detail="An identical document is already being uploaded")
    result = _upload_response_document(row)
    result["previous_version_number"] = None
    result["change_detected"] = False
    return {
        "duplicate": True,
        "message": "Identical SHA-256 content already exists in this workspace. No new S3 upload or document version was created.",
        "document": result,
    }


def _verify_uploaded_object(head: dict, intent: dict, expected_checksum_b64: str) -> None:
    metadata = {str(key).lower(): str(value) for key, value in (head.get("Metadata") or {}).items()}
    if int(head.get("ContentLength", -1)) != int(intent["expected_size_bytes"]):
        raise HTTPException(status_code=422, detail="Uploaded object size does not match the upload intent")
    if int(head.get("ContentLength", -1)) <= 0 or int(head.get("ContentLength", -1)) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded object exceeds the 20 MiB limit")
    if head.get("ContentType") != intent["expected_content_type"]:
        raise HTTPException(status_code=422, detail="Uploaded object content type does not match the upload intent")
    if metadata.get("document-id") != intent["document_id"]:
        raise HTTPException(status_code=422, detail="Uploaded object does not match the expected document version")
    if metadata.get("sha256") != intent["expected_sha256"]:
        raise HTTPException(status_code=422, detail="Uploaded object SHA-256 metadata does not match the upload intent")
    checksum = head.get("ChecksumSHA256")
    # AWS S3 validates the signed x-amz-checksum-sha256 form field during POST. When S3
    # returns the stored checksum, compare it again; metadata alone is not used as the checksum.
    if checksum and checksum != expected_checksum_b64:
        raise HTTPException(status_code=422, detail="S3 checksum does not match the requested SHA-256")


@app.post("/api/v1/documents/upload-intents", status_code=status.HTTP_201_CREATED)
def create_document_upload_intent(
    request: UploadIntentRequest,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Reserve an immutable document-version record and issue a short-lived S3 POST policy."""
    filename = _safe_filename(request.filename)
    content_type = _content_type_for(filename)
    digest = request.sha256
    actual_workspace_id = request.workspace_id

    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME is not configured")

    try:
        workspace = connection.execute(
            """
            SELECT id, session_id FROM workspaces
            WHERE id=%s AND user_id=%s AND deleted_at IS NULL
            """,
            (actual_workspace_id, user["id"]),
        ).fetchone()
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        if workspace["session_id"] != user["session_id"]:
            raise HTTPException(status_code=403, detail="Previous work sessions are read-only. Upload into a workspace in your current session.")

        # Deduplicate against the current version in this workspace before allocating a new ID.
        if not request.source_id:
            identical = connection.execute(
                """
                SELECT d.id FROM documents d
                JOIN document_sources s ON s.id=d.source_id
                WHERE d.workspace_id=%s AND d.user_id=%s AND d.file_hash=%s
                  AND d.is_current=TRUE AND d.status='UPLOADED'
                  AND d.deleted_at IS NULL AND s.deleted_at IS NULL
                ORDER BY d.created_at DESC LIMIT 1
                """,
                (actual_workspace_id, user["id"], digest),
            ).fetchone()
            if identical:
                return _return_duplicate(connection, identical["id"], actual_workspace_id, None)
            pending_same = connection.execute(
                """
                SELECT id FROM documents
                WHERE workspace_id=%s AND user_id=%s AND file_hash=%s
                  AND status='PENDING' AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                (actual_workspace_id, user["id"], digest),
            ).fetchone()
            if pending_same:
                raise HTTPException(status_code=409, detail="An identical document upload is already in progress in this workspace")

        previous = None
        if request.source_id:
            source = connection.execute(
                """
                SELECT id, workspace_id, user_id, source_name, source_key
                FROM document_sources
                WHERE id=%s AND workspace_id=%s AND user_id=%s AND deleted_at IS NULL
                FOR UPDATE
                """,
                (request.source_id, actual_workspace_id, user["id"]),
            ).fetchone()
            if not source:
                raise HTTPException(status_code=404, detail="Document source not found in this workspace")
            previous = connection.execute(
                """
                SELECT id, version_number, file_hash, original_name
                FROM documents
                WHERE source_id=%s AND is_current=TRUE AND status='UPLOADED' AND deleted_at IS NULL
                ORDER BY version_number DESC LIMIT 1
                """,
                (request.source_id,),
            ).fetchone()
            if previous and previous["file_hash"] == digest:
                return _return_duplicate(connection, previous["id"], actual_workspace_id, request.source_id)
            version_row = connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version FROM documents WHERE source_id=%s",
                (request.source_id,),
            ).fetchone()
            actual_source_id = source["id"]
            source_name = source["source_name"]
            source_key = source["source_key"]
            version_number = int(version_row["next_version"])
        else:
            source_name = PurePosixPath(filename).stem[:200] or "Document"
            actual_source_id = new_id("src")
            source_key = f"{_slugify(source_name)}-{actual_source_id[-12:]}"
            connection.execute(
                """
                INSERT INTO document_sources (id, user_id, workspace_id, source_name, source_key)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (actual_source_id, user["id"], actual_workspace_id, source_name, source_key),
            )
            version_number = 1

        document_id = new_id("doc")  # This ID never changes; it identifies one immutable version.
        upload_intent_id = new_id("upl")
        final_key = (
            f"rag/v1/users/{user['id']}/sessions/{user['session_id']}"
            f"/workspaces/{actual_workspace_id}/sources/{source_key}"
            f"/versions/v{version_number:04d}/{document_id}/{filename}"
        )
        staging_key = (
            f"rag-staging/v1/users/{user['id']}/sessions/{user['session_id']}"
            f"/workspaces/{actual_workspace_id}/intents/{upload_intent_id}/{filename}"
        )
        storage_uri = f"s3://{S3_BUCKET_NAME}/{final_key}"

        connection.execute(
            """
            INSERT INTO documents (
                id, source_id, workspace_id, user_id, original_name, s3_key, storage_uri,
                file_hash, file_size_bytes, content_type, status, version_number,
                is_current, supersedes_document_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', %s, FALSE, %s)
            """,
            (
                document_id, actual_source_id, actual_workspace_id, user["id"], filename,
                final_key, storage_uri, digest, request.file_size_bytes, content_type,
                version_number, previous["id"] if previous else None,
            ),
        )
        connection.execute(
            """
            INSERT INTO upload_intents (
                id, document_id, user_id, workspace_id, staging_key,
                expected_sha256, expected_size_bytes, expected_content_type, expires_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                      NOW() + (%s * INTERVAL '1 second'))
            """,
            (
                upload_intent_id, document_id, user["id"], actual_workspace_id, staging_key,
                digest, request.file_size_bytes, content_type, PRESIGNED_UPLOAD_EXPIRE_SECONDS,
            ),
        )

        try:
            presigned = create_presigned_upload_post(
                staging_key=staging_key,
                content_type=content_type,
                document_id=document_id,
                sha256_hex=digest,
                max_file_size_bytes=MAX_FILE_SIZE_BYTES,
                expires_in=PRESIGNED_UPLOAD_EXPIRE_SECONDS,
            )
        except Exception as exc:
            connection.rollback()
            logger.exception("Could not create S3 upload policy for document_id=%s", document_id)
            raise HTTPException(status_code=502, detail="Could not authorize upload to S3. Check bucket configuration and AWS permissions.") from exc

        connection.commit()
        return {
            "duplicate": False,
            "upload_intent_id": upload_intent_id,
            "expires_in_seconds": PRESIGNED_UPLOAD_EXPIRE_SECONDS,
            "message": "Upload authorized. Send the file directly to the supplied S3 form, then confirm the upload.",
            "document": {
                "id": document_id,
                "source_id": actual_source_id,
                "source_name": source_name,
                "source_key": source_key,
                "workspace_id": actual_workspace_id,
                "filename": filename,
                "file_size_bytes": request.file_size_bytes,
                "content_type": content_type,
                "file_hash": digest,
                "version_number": version_number,
                "previous_version_number": previous["version_number"] if previous else None,
                "change_detected": bool(previous and previous["file_hash"] != digest),
                "status": "PENDING",
                "is_current": False,
                "storage_uri": storage_uri,
                "s3_key": final_key,
            },
            "s3_upload": {"url": presigned["url"], "fields": presigned["fields"], "method": "POST"},
        }
    except HTTPException:
        connection.rollback()
        raise
    except Exception as exc:
        connection.rollback()
        logger.exception("Could not create upload intent")
        raise HTTPException(status_code=500, detail="Could not create an upload intent") from exc


@app.post("/api/v1/documents/upload-intents/{intent_id}/complete")
def complete_document_upload(
    intent_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Verify the S3 staging object, promote it to an immutable key and finalize PostgreSQL metadata."""
    intent = connection.execute(
        """
        SELECT ui.id AS upload_intent_id, ui.document_id, ui.user_id, ui.workspace_id,
               ui.staging_key, ui.expected_sha256, ui.expected_size_bytes,
               ui.expected_content_type, ui.expires_at, ui.completed_at, ui.cancelled_at,
               d.source_id, d.original_name, d.s3_key, d.storage_uri, d.file_hash,
               d.file_size_bytes, d.content_type, d.version_number, d.status AS document_status,
               d.is_current, s.source_name, s.source_key
        FROM upload_intents ui
        JOIN documents d ON d.id=ui.document_id
        JOIN document_sources s ON s.id=d.source_id
        WHERE ui.id=%s AND ui.user_id=%s AND d.user_id=%s
        FOR UPDATE OF ui, d
        """,
        (intent_id, user["id"], user["id"]),
    ).fetchone()
    if not intent:
        raise HTTPException(status_code=404, detail="Upload intent not found")
    if intent["completed_at"]:
        connection.commit()
        row = {
            **intent,
            "id": intent["document_id"],
            "source_id": intent["source_id"],
            "source_name": intent["source_name"],
            "source_key": intent["source_key"],
            "workspace_id": intent["workspace_id"],
            "filename": intent["original_name"],
            "status": intent["document_status"],
        }
        return {
            "duplicate": False,
            "already_completed": True,
            "message": "This upload was already finalized.",
            "document": {
                "id": intent["document_id"], "source_id": intent["source_id"],
                "source_name": intent["source_name"], "source_key": intent["source_key"],
                "workspace_id": intent["workspace_id"], "filename": intent["original_name"],
                "file_size_bytes": intent["file_size_bytes"], "content_type": intent["content_type"],
                "file_hash": intent["file_hash"], "version_number": intent["version_number"],
                "status": intent["document_status"], "is_current": intent["is_current"],
                "storage_uri": intent["storage_uri"], "s3_key": intent["s3_key"],
            },
        }
    if intent["cancelled_at"] or intent["document_status"] != "PENDING":
        connection.rollback()
        raise HTTPException(status_code=409, detail="This upload intent is no longer pending")

    # A valid signature does not make an upload intent permanent. Expired intents can no
    # longer be finalized; mark them failed and remove any staged bytes where possible.
    if intent["expires_at"] <= datetime.now(timezone.utc):
        connection.execute(
            "UPDATE upload_intents SET cancelled_at=COALESCE(cancelled_at,NOW()) WHERE id=%s",
            (intent_id,),
        )
        connection.execute(
            "UPDATE documents SET status='FAILED', is_current=FALSE WHERE id=%s AND status='PENDING'",
            (intent["document_id"],),
        )
        connection.commit()
        try:
            delete_object(intent["staging_key"])
        except Exception:
            logger.warning("Expired staging object cleanup failed for intent_id=%s", intent_id, exc_info=True)
        raise HTTPException(status_code=410, detail="Upload permission expired. Please start the upload again.")

    checksum_b64 = base64.b64encode(bytes.fromhex(intent["expected_sha256"])).decode("ascii")
    try:
        staging_head = head_object(intent["staging_key"])
    except ClientError as exc:
        connection.rollback()
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            raise HTTPException(status_code=409, detail="The S3 upload was not found. Upload the file before confirming it.") from exc
        logger.exception("Could not inspect staging S3 object for intent_id=%s", intent_id)
        raise HTTPException(status_code=502, detail="Could not verify the uploaded S3 object") from exc
    except Exception as exc:
        connection.rollback()
        logger.exception("Could not inspect staging S3 object for intent_id=%s", intent_id)
        raise HTTPException(status_code=502, detail="Could not verify the uploaded S3 object") from exc

    try:
        _verify_uploaded_object(staging_head, intent, checksum_b64)
    except HTTPException:
        connection.execute(
            "UPDATE documents SET status='FAILED', is_current=FALSE WHERE id=%s AND status='PENDING'",
            (intent["document_id"],),
        )
        connection.execute("UPDATE upload_intents SET cancelled_at=NOW() WHERE id=%s", (intent_id,))
        connection.commit()
        try:
            delete_object(intent["staging_key"])
        except Exception:
            logger.warning("Could not remove invalid staging object for intent_id=%s", intent_id, exc_info=True)
        raise

    # Promote inside S3. The final key includes the immutable document-version ID and is never
    # reused by a future version. The browser cannot overwrite this key with the presigned form.
    try:
        copy_object(intent["staging_key"], intent["s3_key"])
        final_head = head_object(intent["s3_key"])
        _verify_uploaded_object(final_head, intent, checksum_b64)
    except HTTPException:
        connection.rollback()
        raise
    except Exception as exc:
        connection.rollback()
        logger.exception("Could not promote staging object for document_id=%s", intent["document_id"])
        raise HTTPException(status_code=502, detail="S3 accepted the upload but could not promote it to its final immutable key") from exc

    try:
        connection.execute("SELECT id FROM document_sources WHERE id=%s FOR UPDATE", (intent["source_id"],))
        current_after_upload = connection.execute(
            """
            SELECT id, version_number FROM documents
            WHERE source_id=%s AND is_current=TRUE AND status='UPLOADED' AND deleted_at IS NULL
            ORDER BY version_number DESC LIMIT 1 FOR UPDATE
            """,
            (intent["source_id"],),
        ).fetchone()
        preceding_version = connection.execute(
            """
            SELECT id, version_number, file_hash FROM documents
            WHERE source_id=%s AND version_number < %s AND status IN ('UPLOADED','SUPERSEDED')
              AND deleted_at IS NULL
            ORDER BY version_number DESC LIMIT 1
            """,
            (intent["source_id"], intent["version_number"]),
        ).fetchone()

        if current_after_upload and int(current_after_upload["version_number"]) > int(intent["version_number"]):
            connection.execute(
                "UPDATE documents SET status='SUPERSEDED', is_current=FALSE, supersedes_document_id=%s WHERE id=%s",
                (preceding_version["id"] if preceding_version else None, intent["document_id"]),
            )
        else:
            connection.execute(
                """
                UPDATE documents SET status='SUPERSEDED', is_current=FALSE
                WHERE source_id=%s AND is_current=TRUE AND status='UPLOADED' AND deleted_at IS NULL
                """,
                (intent["source_id"],),
            )
            connection.execute(
                """
                UPDATE documents SET status='UPLOADED', is_current=TRUE, supersedes_document_id=%s
                WHERE id=%s AND user_id=%s
                """,
                (preceding_version["id"] if preceding_version else None, intent["document_id"], user["id"]),
            )
        connection.execute("UPDATE document_sources SET updated_at=NOW(), deleted_at=NULL WHERE id=%s", (intent["source_id"],))
        connection.execute(
            "UPDATE upload_intents SET completed_at=NOW() WHERE id=%s AND completed_at IS NULL",
            (intent_id,),
        )
        connection.execute(
            "UPDATE workspaces SET bytes_used=bytes_used + %s WHERE id=%s AND user_id=%s",
            (intent["expected_size_bytes"], intent["workspace_id"], user["id"]),
        )
        connection.commit()
    except Exception as exc:
        connection.rollback()
        logger.exception("PostgreSQL finalization failed for document_id=%s", intent["document_id"])
        # Leave staging/final objects in place so completion can safely be retried. The intent
        # and immutable document record remain PENDING until the database transaction succeeds.
        raise HTTPException(status_code=500, detail="S3 upload is complete but PostgreSQL finalization failed. Retry confirmation.") from exc

    try:
        delete_object(intent["staging_key"])
    except Exception:
        # A bucket lifecycle rule should expire any leftovers under rag-staging/.
        logger.warning("Upload finalized but staging cleanup failed for intent_id=%s", intent_id, exc_info=True)

    final = connection.execute(
        """
        SELECT d.id, d.source_id, s.source_name, s.source_key, d.workspace_id,
               d.original_name, d.file_size_bytes, d.content_type, d.file_hash,
               d.version_number, d.status, d.is_current, d.storage_uri, d.s3_key
        FROM documents d JOIN document_sources s ON s.id=d.source_id
        WHERE d.id=%s AND d.user_id=%s
        """,
        (intent["document_id"], user["id"]),
    ).fetchone()
    connection.commit()
    result_doc = _upload_response_document(final)
    result_doc["previous_version_number"] = preceding_version["version_number"] if preceding_version else None
    result_doc["change_detected"] = bool(preceding_version and preceding_version["file_hash"] != intent["expected_sha256"])
    # Stage 2 starts only after the immutable source object and PostgreSQL metadata are committed.
    run, queued = _queue_ingestion_run(connection, intent["document_id"], user["id"], force=False)
    result_doc["ingestion_run_id"] = run["id"]
    if queued:
        _schedule_document_ingestion(background_tasks, intent["document_id"], user["id"], run["id"])
    return {
        "duplicate": False,
        "message": "S3 object verified and PostgreSQL metadata committed. Stage 2 document processing has been queued.",
        "document": result_doc,
        "ingestion": {"run_id": run["id"], "status": run["status"]},
    }


@app.post("/api/v1/documents/upload-intents/{intent_id}/cancel")
def cancel_document_upload(
    intent_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Cancel an unfinished upload and remove its temporary S3 object if present."""
    intent = connection.execute(
        """
        SELECT ui.id, ui.document_id, ui.staging_key, ui.completed_at, ui.cancelled_at
        FROM upload_intents ui WHERE ui.id=%s AND ui.user_id=%s FOR UPDATE
        """,
        (intent_id, user["id"]),
    ).fetchone()
    if not intent:
        raise HTTPException(status_code=404, detail="Upload intent not found")
    if intent["completed_at"]:
        connection.rollback()
        raise HTTPException(status_code=409, detail="A completed upload cannot be cancelled")
    connection.execute("UPDATE upload_intents SET cancelled_at=COALESCE(cancelled_at,NOW()) WHERE id=%s", (intent_id,))
    connection.execute(
        "UPDATE documents SET status='FAILED', is_current=FALSE WHERE id=%s AND user_id=%s AND status='PENDING'",
        (intent["document_id"], user["id"]),
    )
    connection.commit()
    try:
        delete_object(intent["staging_key"])
    except Exception:
        logger.warning("Upload cancelled but staging cleanup failed for intent_id=%s", intent_id, exc_info=True)
    return {"message": "Upload cancelled. Any staged object will be removed or expired by the S3 lifecycle policy."}


@app.get("/api/v1/documents")
def list_documents(
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
    workspace_id: str | None = None,
) -> dict:
    if workspace_id:
        owned = connection.execute(
            "SELECT id FROM workspaces WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
            (workspace_id, user["id"]),
        ).fetchone()
        if not owned:
            raise HTTPException(status_code=404, detail="Workspace not found")
        rows = connection.execute(
            """
            SELECT d.id, d.source_id, s.source_name, s.source_key, d.workspace_id,
                   d.original_name, d.s3_key, d.file_hash, d.file_size_bytes, d.content_type,
                   d.version_number, d.is_current, d.supersedes_document_id,
                   d.status, d.storage_uri, d.created_at,
                   d.embedding_status, d.embedding_model, d.embedding_dimensions, d.embedded_chunk_count,
                   d.embedded_artifact_id, d.embedding_error_code,
                   (SELECT ir.status FROM ingestion_runs ir WHERE ir.document_id=d.id ORDER BY ir.created_at DESC LIMIT 1) AS ingestion_status,
                   (SELECT ir.pii_finding_count FROM ingestion_runs ir WHERE ir.document_id=d.id ORDER BY ir.created_at DESC LIMIT 1) AS pii_finding_count
            FROM documents d JOIN document_sources s ON s.id=d.source_id
            WHERE d.user_id=%s AND d.workspace_id=%s AND d.deleted_at IS NULL
              AND s.deleted_at IS NULL
            ORDER BY s.created_at ASC, d.version_number DESC
            """,
            (user["id"], workspace_id),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT d.id, d.source_id, s.source_name, s.source_key, d.workspace_id,
                   d.original_name, d.s3_key, d.file_hash, d.file_size_bytes, d.content_type,
                   d.version_number, d.is_current, d.supersedes_document_id,
                   d.status, d.storage_uri, d.created_at,
                   d.embedding_status, d.embedding_model, d.embedding_dimensions, d.embedded_chunk_count,
                   d.embedded_artifact_id, d.embedding_error_code,
                   (SELECT ir.status FROM ingestion_runs ir WHERE ir.document_id=d.id ORDER BY ir.created_at DESC LIMIT 1) AS ingestion_status,
                   (SELECT ir.pii_finding_count FROM ingestion_runs ir WHERE ir.document_id=d.id ORDER BY ir.created_at DESC LIMIT 1) AS pii_finding_count
            FROM documents d JOIN document_sources s ON s.id=d.source_id
            WHERE d.user_id=%s AND d.deleted_at IS NULL AND s.deleted_at IS NULL
            ORDER BY d.created_at DESC
            """,
            (user["id"],),
        ).fetchall()
    return {"documents": rows}


@app.post("/api/v1/ingestion/documents/{document_id}/process", status_code=status.HTTP_202_ACCEPTED)
def start_document_ingestion(
    document_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
    force: bool = False,
) -> dict:
    """Queue/retry Stage 2 for one uploaded document version (all ownership checks are server-side)."""
    run, queued = _queue_ingestion_run(connection, document_id, user["id"], force=force)
    if queued:
        _schedule_document_ingestion(background_tasks, document_id, user["id"], run["id"])
    elif run.get("status") == "SANITIZED":
        try:
            idx_res = queue_document_index(connection, document_id, user["id"], force=force)
            if idx_res.get("queued"):
                background_tasks.add_task(process_document_embedding, document_id, user["id"], idx_res["artifact_id"])
        except Exception:
            pass
    return {"document_id": document_id, "run_id": run["id"], "status": run["status"], "queued": queued}


@app.post("/api/v1/ingestion/workspaces/{workspace_id}/process-pending", status_code=status.HTTP_202_ACCEPTED)
def process_pending_workspace_documents(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    workspace = connection.execute(
        "SELECT id FROM workspaces WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
        (workspace_id, user["id"]),
    ).fetchone()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    documents = connection.execute(
        """
        SELECT id FROM documents
        WHERE user_id=%s AND workspace_id=%s AND status IN ('UPLOADED','SUPERSEDED') AND deleted_at IS NULL
        ORDER BY created_at ASC
        """,
        (user["id"], workspace_id),
    ).fetchall()
    queued_ids = []
    skipped = 0
    for row in documents:
        run, queued = _queue_ingestion_run(connection, row["id"], user["id"], force=False)
        if queued:
            _schedule_document_ingestion(background_tasks, row["id"], user["id"], run["id"])
            queued_ids.append({"document_id": row["id"], "run_id": run["id"]})
        else:
            skipped += 1
    return {"workspace_id": workspace_id, "queued_count": len(queued_ids), "skipped_count": skipped, "queued": queued_ids}


@app.post("/api/v1/ingestion/process-pending", status_code=status.HTTP_202_ACCEPTED)
def process_pending_user_documents(
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Queue unprocessed/failed document versions from every workspace owned by the user."""
    documents = connection.execute(
        """
        SELECT d.id FROM documents d
        WHERE d.user_id=%s AND d.status IN ('UPLOADED','SUPERSEDED') AND d.deleted_at IS NULL
        ORDER BY d.created_at ASC
        """,
        (user["id"],),
    ).fetchall()
    queued_ids = []
    skipped = 0
    for row in documents:
        run, queued = _queue_ingestion_run(connection, row["id"], user["id"], force=False)
        if queued:
            _schedule_document_ingestion(background_tasks, row["id"], user["id"], run["id"])
            queued_ids.append({"document_id": row["id"], "run_id": run["id"]})
        else:
            skipped += 1
    return {"queued_count": len(queued_ids), "skipped_count": skipped, "queued": queued_ids}


@app.get("/api/v1/embeddings/health")
def check_embedding_provider(
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Make a real authenticated embedding request to the configured local Ollama model."""
    try:
        return embedding_health()
    except Stage3Error as exc:
        raise HTTPException(status_code=503, detail={"code": exc.code, "message": exc.public_message}) from exc


@app.post("/api/v1/ingestion/documents/{document_id}/index", status_code=status.HTTP_202_ACCEPTED)
def start_document_embedding_indexing(
    document_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
    force: bool = False,
) -> dict:
    """Index the latest approved Stage 2 artifact for exactly one immutable document version."""
    try:
        result = queue_document_index(connection, document_id, user["id"], force=force)
    except Stage3Error as exc:
        http_status = 404 if exc.code == "DOCUMENT_NOT_AVAILABLE" else 409
        raise HTTPException(status_code=http_status, detail={"code": exc.code, "message": exc.public_message}) from exc
    if result.get("queued"):
        background_tasks.add_task(process_document_embedding, document_id, user["id"], result["artifact_id"])
    return result


@app.post("/api/v1/ingestion/workspaces/{workspace_id}/index-pending", status_code=status.HTTP_202_ACCEPTED)
def index_pending_workspace_documents(
    workspace_id: str,
    background_tasks: BackgroundTasks,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Queue Stage 3 for eligible, processed document versions in one owned workspace."""
    workspace = connection.execute(
        "SELECT id FROM workspaces WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
        (workspace_id, user["id"]),
    ).fetchone()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    documents = connection.execute(
        """
        SELECT id FROM documents
        WHERE user_id=%s AND workspace_id=%s AND deleted_at IS NULL
          AND status IN ('UPLOADED','SUPERSEDED')
        ORDER BY created_at ASC, id ASC
        """,
        (user["id"], workspace_id),
    ).fetchall()
    queued_items = []
    skipped_items = []
    for row in documents:
        try:
            result = queue_document_index(connection, row["id"], user["id"], force=False)
        except Stage3Error as exc:
            skipped_items.append({"document_id": row["id"], "code": exc.code, "message": exc.public_message})
            continue
        if result.get("queued"):
            background_tasks.add_task(process_document_embedding, row["id"], user["id"], result["artifact_id"])
            queued_items.append({"document_id": row["id"], "artifact_id": result["artifact_id"], "status": "QUEUED"})
        else:
            skipped_items.append({"document_id": row["id"], "status": result.get("status"), "message": result.get("message", "Already indexed or already in progress.")})
    return {
        "workspace_id": workspace_id,
        "queued_count": len(queued_items),
        "skipped_count": len(skipped_items),
        "queued": queued_items,
        "skipped": skipped_items,
    }


@app.get("/api/v1/ingestion/documents/{document_id}/embedding-status")
def get_document_embedding_status(
    document_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Return embedding status and row counts for one document version only."""
    document = connection.execute(
        """
        SELECT d.id, d.workspace_id, d.source_id, d.original_name, d.version_number,
               d.is_current, d.embedding_status, d.embedding_target_artifact_id,
               d.embedded_artifact_id, d.embedding_model, d.embedding_dimensions,
               d.embedded_chunk_count, d.embedding_error_code, d.embedding_error_message,
               d.embedding_requested_at, d.embedding_started_at, d.embedding_completed_at,
               d.embedding_updated_at
        FROM documents d JOIN workspaces w ON w.id=d.workspace_id
        WHERE d.id=%s AND d.user_id=%s AND d.deleted_at IS NULL AND w.deleted_at IS NULL
        """,
        (document_id, user["id"]),
    ).fetchone()
    if not document:
        raise HTTPException(status_code=404, detail="Document version not found")
    counts = connection.execute(
        """
        SELECT COUNT(DISTINCT dc.id) AS chunk_count,
               COUNT(DISTINCT ce.chunk_id) AS vector_count
        FROM document_chunks dc
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id=dc.id
        WHERE dc.document_id=%s
        """,
        (document_id,),
    ).fetchone()
    return {"document": document, "stored_chunk_count": counts["chunk_count"], "stored_vector_count": counts["vector_count"]}


@app.delete("/api/v1/ingestion/documents/{document_id}/index")
def delete_document_embedding_index(
    document_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Remove the vectors for a single document version without deleting its source file/artifact."""
    owned = connection.execute(
        "SELECT id FROM documents WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
        (document_id, user["id"]),
    ).fetchone()
    if not owned:
        raise HTTPException(status_code=404, detail="Document version not found")
    deleted_chunks = delete_document_vectors(connection, document_id)
    connection.commit()
    return {
        "document_id": document_id,
        "deleted_chunk_count": deleted_chunks,
        "message": "Only this document version's chunks and vectors were removed; the source file and other versions were preserved.",
    }


@app.post("/api/v1/search")
def search_workspace_documents(
    request: SemanticSearchRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Return nearest chunks from one authorized workspace with source metadata for citations."""
    try:
        return semantic_search(
            user_id=user["id"],
            workspace_id=request.workspace_id,
            query=request.query,
            top_k=request.top_k,
            include_superseded=request.include_superseded,
        )
    except Stage3Error as exc:
        if exc.code == "WORKSPACE_NOT_FOUND":
            http_status = 404
        elif exc.code == "OLLAMA_UNAVAILABLE":
            http_status = 503
        else:
            http_status = 422
        raise HTTPException(status_code=http_status, detail={"code": exc.code, "message": exc.public_message}) from exc


@app.get("/api/v1/rag/health")
@app.get("/api/v1/ollama/models")
def check_rag_pipeline_health(
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Inspect Stage 3 embeddings, Stage 4 reranker, and Stage 5 local Ollama LLM readiness."""
    try:
        return rag_health()
    except Stage3Error as exc:
        raise HTTPException(status_code=503, detail={"code": exc.code, "message": exc.public_message}) from exc


@app.post("/api/v1/stage4/rerank")
def run_stage4_reranker(
    request: Stage4RerankRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Stage 4: Retrieve 20 hybrid candidate chunks and rerank down to Top K (3 or 5)."""
    try:
        return stage4_rerank_workspace_query(
            user_id=user["id"],
            workspace_id=request.workspace_id,
            query=request.query,
            candidate_k=request.candidate_k,
            top_k=request.top_k,
            include_superseded=request.include_superseded,
        )
    except Stage3Error as exc:
        http_status = 404 if exc.code == "WORKSPACE_NOT_FOUND" else (503 if exc.code == "OLLAMA_UNAVAILABLE" else 422)
        raise HTTPException(status_code=http_status, detail={"code": exc.code, "message": exc.public_message}) from exc


@app.post("/api/v1/stage5/generate")
def run_stage5_llm_generator(
    request: Stage5GenerateRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Stage 5: Pass Stage 4 reranked chunks + user query to any local Ollama LLM for the final answer."""
    return stage5_generate_llm_answer(
        query=request.query,
        reranked_chunks=request.reranked_chunks,
        chat_model=request.chat_model,
        history=request.history,
    )


@app.post("/api/v1/ask")
@app.post("/api/v1/chat")
def ask_workspace_question(
    request: RagAskRequest,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Stage 3 -> Stage 4 (Rerank 20->Top K) -> Stage 5 (Ollama LLM Final Answer) End-to-End RAG."""
    try:
        return answer_workspace_query(
            user_id=user["id"],
            workspace_id=request.workspace_id,
            query=request.query,
            candidate_k=request.candidate_k,
            top_k=request.top_k,
            include_superseded=request.include_superseded,
            history=request.history,
            chat_model=request.chat_model,
        )
    except Stage3Error as exc:
        if exc.code == "WORKSPACE_NOT_FOUND":
            http_status = 404
        elif exc.code == "OLLAMA_UNAVAILABLE":
            http_status = 503
        else:
            http_status = 422
        raise HTTPException(status_code=http_status, detail={"code": exc.code, "message": exc.public_message}) from exc


@app.get("/api/v1/ingestion/documents/{document_id}/status")
def get_document_ingestion_status(
    document_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    document = connection.execute(
        """
        SELECT id, workspace_id, version_number, embedding_status, embedded_chunk_count,
               embedding_model, embedding_dimensions, embedding_error_code, embedding_error_message
        FROM documents WHERE id=%s AND user_id=%s AND deleted_at IS NULL
        """,
        (document_id, user["id"]),
    ).fetchone()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    run = connection.execute(
        """
        SELECT id, status, progress_metadata, pii_finding_count, error_code, error_message,
               artifact_id, created_at, started_at, finished_at, updated_at
        FROM ingestion_runs WHERE document_id=%s ORDER BY created_at DESC LIMIT 1
        """,
        (document_id,),
    ).fetchone()
    if not run:
        return {
            "document_id": document_id,
            "status": "NOT_STARTED",
            "embedding_status": document["embedding_status"],
            "embedded_chunk_count": document["embedded_chunk_count"],
            "message": "Stage 2 has not been started for this document.",
        }
    artifact = None
    if run["artifact_id"]:
        artifact = connection.execute(
            """
            SELECT id, status, artifact_type, s3_bucket, s3_key, sha256, size_bytes, schema_version,
                   element_counts, pii_entity_counts, metadata, created_at
            FROM extracted_artifacts WHERE id=%s AND user_id=%s AND workspace_id=%s
            """,
            (run["artifact_id"], user["id"], document["workspace_id"]),
        ).fetchone()
    return {
        "document_id": document_id,
        "workspace_id": document["workspace_id"],
        "version_number": document["version_number"],
        "embedding_status": document["embedding_status"],
        "embedded_chunk_count": document["embedded_chunk_count"],
        "embedding_model": document["embedding_model"],
        "embedding_dimensions": document["embedding_dimensions"],
        "embedding_error_code": document["embedding_error_code"],
        "embedding_error_message": document["embedding_error_message"],
        "run": run,
        "artifact": artifact,
    }


@app.get("/api/v1/ingestion/documents/{document_id}/artifact")
def get_document_ingestion_artifact(
    document_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    """Download and verify the latest Stage 2 JSON artifact (including main_content and chunks) for a document version."""
    import hashlib
    import json
    from app.s3_client import download_object_bytes

    document = connection.execute(
        "SELECT id, workspace_id, version_number FROM documents WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
        (document_id, user["id"]),
    ).fetchone()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    artifact_row = connection.execute(
        """
        SELECT id, ingestion_run_id, status, s3_bucket, s3_key, sha256, size_bytes, schema_version, created_at
        FROM extracted_artifacts
        WHERE document_id=%s AND user_id=%s AND workspace_id=%s
        ORDER BY created_at DESC, id DESC LIMIT 1
        """,
        (document_id, user["id"], document["workspace_id"]),
    ).fetchone()
    if not artifact_row:
        raise HTTPException(status_code=404, detail="Stage 2 extraction artifact not found for this document")
    try:
        raw_bytes = download_object_bytes(artifact_row["s3_key"], bucket=artifact_row["s3_bucket"])
    except Exception as exc:
        logger.exception("Could not download Stage 2 artifact id=%s", artifact_row["id"])
        raise HTTPException(status_code=502, detail="Could not download Stage 2 artifact from S3") from exc
    if hashlib.sha256(raw_bytes).hexdigest().lower() != str(artifact_row["sha256"]).lower():
        raise HTTPException(status_code=502, detail="Stage 2 artifact checksum mismatch")
    payload = json.loads(raw_bytes.decode("utf-8"))
    return {
        "document_id": document_id,
        "workspace_id": document["workspace_id"],
        "version_number": document["version_number"],
        "artifact_metadata": artifact_row,
        "main_content": payload.get("main_content", ""),
        "artifact": payload,
    }


@app.get("/api/v1/ingestion/workspaces/{workspace_id}/artifacts")
def list_workspace_artifacts(
    workspace_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    workspace = connection.execute(
        "SELECT id FROM workspaces WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
        (workspace_id, user["id"]),
    ).fetchone()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    rows = connection.execute(
        """
        SELECT a.id, a.document_id, d.original_name, a.document_version_number, a.status,
               a.s3_bucket, a.s3_key, a.sha256, a.size_bytes, a.schema_version,
               a.element_counts, a.pii_entity_counts, a.metadata, a.created_at,
               r.id AS ingestion_run_id, r.pii_finding_count
        FROM extracted_artifacts a
        JOIN documents d ON d.id=a.document_id
        JOIN ingestion_runs r ON r.id=a.ingestion_run_id
        WHERE a.user_id=%s AND a.workspace_id=%s
        ORDER BY a.created_at DESC
        """,
        (user["id"], workspace_id),
    ).fetchall()
    return {"workspace_id": workspace_id, "artifacts": rows}


@app.get("/api/v1/documents/{document_id}/download-url")
def get_document_download_url(
    document_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    doc = connection.execute(
        """
        SELECT id, original_name, s3_key, status FROM documents
        WHERE id=%s AND user_id=%s AND deleted_at IS NULL AND status IN ('UPLOADED','SUPERSEDED')
        """,
        (document_id, user["id"]),
    ).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document version not found")
    try:
        url = create_download_url(doc["s3_key"], doc["original_name"])
    except Exception as exc:
        logger.exception("Could not create download URL for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail="Could not create a temporary download link") from exc
    return {"download_url": url, "expires_in_seconds": 300, "filename": doc["original_name"]}


@app.delete("/api/v1/documents/{document_id}")
def delete_document_version(
    document_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    doc = connection.execute(
        """
        SELECT id, source_id, workspace_id, s3_key, status, is_current
        FROM documents WHERE id=%s AND user_id=%s AND deleted_at IS NULL
        """,
        (document_id, user["id"]),
    ).fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="Document version not found")

    intent_rows = connection.execute(
        "SELECT staging_key FROM upload_intents WHERE document_id=%s",
        (document_id,),
    ).fetchall()
    object_keys = list(dict.fromkeys([doc["s3_key"], *(row["staging_key"] for row in intent_rows)]))
    try:
        for object_key in object_keys:
            if object_key:
                delete_object(object_key)
    except Exception as exc:
        logger.exception("Could not delete S3 object for document_id=%s", document_id)
        raise HTTPException(status_code=502, detail="Could not delete the file from S3; database metadata was not changed") from exc

    # Retire only this immutable version's indexed chunks; chunk_embeddings cascades via FK.
    delete_document_vectors(connection, document_id)
    connection.execute(
        "UPDATE documents SET status='DELETED', is_current=FALSE, deleted_at=NOW() WHERE id=%s",
        (document_id,),
    )
    connection.execute(
        "UPDATE upload_intents SET cancelled_at=COALESCE(cancelled_at,NOW()) "
        "WHERE document_id=%s AND completed_at IS NULL",
        (document_id,),
    )
    restored = None
    if doc["is_current"]:
        restored = connection.execute(
            """
            SELECT id FROM documents
            WHERE source_id=%s AND deleted_at IS NULL AND status='SUPERSEDED'
            ORDER BY version_number DESC LIMIT 1
            """,
            (doc["source_id"],),
        ).fetchone()
        if restored:
            connection.execute(
                "UPDATE documents SET status='UPLOADED', is_current=TRUE WHERE id=%s",
                (restored["id"],),
            )
        else:
            connection.execute(
                "UPDATE document_sources SET deleted_at=NOW(), updated_at=NOW() WHERE id=%s",
                (doc["source_id"],),
            )
    remaining = connection.execute(
        "SELECT id FROM documents WHERE source_id=%s AND deleted_at IS NULL LIMIT 1",
        (doc["source_id"],),
    ).fetchone()
    if not remaining:
        connection.execute(
            "UPDATE document_sources SET deleted_at=NOW(), updated_at=NOW() WHERE id=%s",
            (doc["source_id"],),
        )
    _recalculate_workspace_bytes(connection, doc["workspace_id"])
    connection.commit()
    return {
        "message": "Document version deleted from S3 and marked as deleted in PostgreSQL.",
        "restored_previous_version": restored["id"] if restored else None,
    }


@app.delete("/api/v1/document-sources/{source_id}")
def delete_document_source(
    source_id: str,
    user: Annotated[dict, Depends(get_current_user)],
    connection: Annotated[Connection, Depends(get_db)],
) -> dict:
    source = connection.execute(
        """
        SELECT id, workspace_id FROM document_sources
        WHERE id=%s AND user_id=%s AND deleted_at IS NULL
        """,
        (source_id, user["id"]),
    ).fetchone()
    if not source:
        raise HTTPException(status_code=404, detail="Document source not found")
    rows = connection.execute(
        "SELECT id, s3_key FROM documents WHERE source_id=%s AND deleted_at IS NULL",
        (source_id,),
    ).fetchall()
    intent_rows = connection.execute(
        """
        SELECT ui.staging_key FROM upload_intents ui
        JOIN documents d ON d.id=ui.document_id
        WHERE d.source_id=%s
        """,
        (source_id,),
    ).fetchall()
    object_keys = list(dict.fromkeys([*(row["s3_key"] for row in rows), *(row["staging_key"] for row in intent_rows)]))
    try:
        for object_key in object_keys:
            if object_key:
                delete_object(object_key)
    except Exception as exc:
        logger.exception("Could not delete all S3 versions for source_id=%s", source_id)
        raise HTTPException(status_code=502, detail="Could not delete all versions from S3; check bucket permissions") from exc

    # Source deletion deliberately removes vectors for this logical source's versions only.
    delete_source_vectors(connection, source_id)
    connection.execute(
        """
        UPDATE documents SET status='DELETED', is_current=FALSE, deleted_at=NOW()
        WHERE source_id=%s AND deleted_at IS NULL
        """,
        (source_id,),
    )
    connection.execute(
        "UPDATE upload_intents SET cancelled_at=COALESCE(cancelled_at,NOW()) "
        "WHERE document_id IN (SELECT id FROM documents WHERE source_id=%s) AND completed_at IS NULL",
        (source_id,),
    )
    connection.execute(
        "UPDATE document_sources SET deleted_at=NOW(), updated_at=NOW() WHERE id=%s",
        (source_id,),
    )
    _recalculate_workspace_bytes(connection, source["workspace_id"])
    connection.commit()
    return {"message": "Document source and all of its versions were deleted from S3 and retired in PostgreSQL.", "versions_deleted": len(rows)}
