"""Environment-based configuration module."""

import os
from dotenv import load_dotenv

try:
    from psycopg.conninfo import make_conninfo
except ModuleNotFoundError:  # Allows lightweight config inspection before deps are installed.
    def make_conninfo(**parameters: object) -> str:
        def quote(value: object) -> str:
            text_value = str(value).replace("\\", "\\\\").replace("'", "\\'")
            return f"'{text_value}'"
        return " ".join(f"{key}={quote(value)}" for key, value in parameters.items())

load_dotenv()

# DATABASE_URL is optional. If it is not set, build a connection string from
# individual PostgreSQL environment variables.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    DATABASE_URL = make_conninfo(
        host=os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", os.getenv("PGHOST", "localhost"))),
        port=os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", os.getenv("PGPORT", "5432"))),
        dbname=os.getenv("POSTGRES_DB", os.getenv("DB_NAME", os.getenv("PGDATABASE", "rag_db"))),
        user=os.getenv("POSTGRES_USER", os.getenv("DB_USER", os.getenv("PGUSER", "postgres"))),
        password=os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", os.getenv("PGPASSWORD", ""))),
    )

# Storage is S3-compatible. MinIO is the default for local-first development; set
# STORAGE_PROVIDER=s3 explicitly to use AWS S3 or another configured S3 endpoint.
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "minio").strip().lower()
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"
MINIO_ENDPOINT = (
    os.getenv("MINIO_ENDPOINT") or os.getenv("S3_ENDPOINT_URL") or "http://localhost:9000"
).strip().rstrip("/")
# Presigned links are generated against the browser-reachable endpoint. In Docker,
# the storage SDK uses http://minio:9000 while browsers normally use localhost:9000.
MINIO_PUBLIC_ENDPOINT = (
    os.getenv("MINIO_PUBLIC_ENDPOINT") or MINIO_ENDPOINT
).strip().rstrip("/")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sourcebook").strip() or "sourcebook"
S3_BUCKET_NAME = (
    MINIO_BUCKET if STORAGE_PROVIDER == "minio" else os.getenv("S3_BUCKET_NAME", "").strip()
)
S3_ENDPOINT_URL = MINIO_ENDPOINT if STORAGE_PROVIDER == "minio" else (os.getenv("S3_ENDPOINT_URL") or None)

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MiB per file
PRESIGNED_UPLOAD_EXPIRE_SECONDS = int(os.getenv("PRESIGNED_UPLOAD_EXPIRE_SECONDS", "900"))
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
# Stage 3 local embedding service (Ollama + BGE-M3 by default).
# Keep default batch size conservative (4) so multi-hundred-chunk documents do not
# overflow Ollama's llama.cpp physical batch buffer (n_ubatch / n_batch).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3").strip()
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "4")))
OLLAMA_NUM_CTX = max(512, int(os.getenv("OLLAMA_NUM_CTX", "4096")))
OLLAMA_SAFE_CHUNK_CHARS = max(300, int(os.getenv("OLLAMA_SAFE_CHUNK_CHARS", "1200")))
OLLAMA_MAX_RETRIES = max(1, int(os.getenv("OLLAMA_MAX_RETRIES", "4")))
OLLAMA_REQUEST_TIMEOUT_SECONDS = max(1, float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "90")))

# Automatically run Stage 3 embedding indexing as soon as Stage 2 sanitization completes.
AUTO_EMBED_ON_INGESTION = os.getenv("AUTO_EMBED_ON_INGESTION", "true").strip().lower() != "false"

# Stage 4 Reranking & Grounded Answer Synthesis via local Ollama
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2").strip()
OLLAMA_RERANK_MODEL = os.getenv("OLLAMA_RERANK_MODEL", "bge-m3").strip()
RAG_CANDIDATE_POOL_SIZE = max(5, min(50, int(os.getenv("RAG_CANDIDATE_POOL_SIZE", "20"))))
RAG_RERANK_TOP_K = max(1, min(10, int(os.getenv("RAG_RERANK_TOP_K", "3"))))
OLLAMA_CHAT_TEMPERATURE = float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.1"))

# Public configuration constants available to application modules.
__all__ = [name for name in globals() if name.isupper()]
