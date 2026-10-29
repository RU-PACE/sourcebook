"""Stage 3: embed approved Stage 2 chunks and persist them in PostgreSQL/pgvector.

The implementation intentionally uses LangChain for the Ollama embedding interface only.
Psycopg 3 remains the database layer so all workspace authorization and version cleanup are
explicit and easy to audit.

Includes:
- Process-level serialization, control-character/repetition sanitization, adaptive batch
  bisection, runner-restart retry with backoff, and sub-segment mean-pooling so that large
  multi-hundred-chunk documents never crash Ollama's local llama.cpp runner.
- Hybrid lexical/acronym reranking + pinpoint sentence/bullet extraction (`focused_answer` /
  `matched_excerpt`) in `semantic_search` so simple questions (e.g. "What is NER?") return the
  exact matching pointer/definition instead of burying the answer in a multi-topic wall of text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Keep pure chunk-validation helpers importable in light-weight test environments.
    from psycopg import Connection

try:
    from app.config import (
        EMBEDDING_BATCH_SIZE,
        EMBEDDING_DIMENSIONS,
        EMBEDDING_PROVIDER,
        OLLAMA_BASE_URL,
        OLLAMA_EMBEDDING_MODEL,
        OLLAMA_MAX_RETRIES,
        OLLAMA_NUM_CTX,
        OLLAMA_SAFE_CHUNK_CHARS,
        OLLAMA_REQUEST_TIMEOUT_SECONDS,
        S3_BUCKET_NAME,
    )
except ModuleNotFoundError as exc:
    # Production installs psycopg through requirements.txt. This fallback exists only so that
    # pure Stage 3 data-contract tests can run without application dependencies installed.
    if exc.name not in {"psycopg", "psycopg.conninfo"}:
        raise
    EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "4")))
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3").strip()
    OLLAMA_NUM_CTX = max(512, int(os.getenv("OLLAMA_NUM_CTX", "4096")))
    OLLAMA_SAFE_CHUNK_CHARS = max(300, int(os.getenv("OLLAMA_SAFE_CHUNK_CHARS", "1200")))
    OLLAMA_MAX_RETRIES = max(1, int(os.getenv("OLLAMA_MAX_RETRIES", "4")))
    OLLAMA_REQUEST_TIMEOUT_SECONDS = max(1.0, float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "90")))
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()

logger = logging.getLogger("sourcebook_stage3")
DATABASE_VECTOR_DIMENSIONS = 1024

# Serialize embedding calls against the single local Ollama runner so concurrent FastAPI
# background tasks do not overwhelm llama.cpp or crash each other during model reloads.
_OLLAMA_LOCK = threading.RLock()

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "define",
        "definition", "describe", "tell", "me", "about", "do", "does", "explain",
        "for", "from", "give", "how", "i", "in", "is", "it", "its", "mean", "means",
        "of", "on", "or", "please", "short", "simple", "that", "the", "their", "there",
        "these", "this", "to", "us", "was", "were", "what", "when", "where", "which",
        "who", "why", "with", "would", "you", "your",
    }
)


def _get_pool():
    from app.db_client import get_pool

    return get_pool()


def _download_object_bytes(s3_key: str, bucket: str) -> bytes:
    from app.s3_client import download_object_bytes

    return download_object_bytes(s3_key, bucket=bucket)


def _execute_many(connection: Any, query: str, parameter_rows: Any) -> None:
    """Execute a parameterized statement for many rows using the Psycopg 3 cursor API."""
    with connection.cursor() as cursor:
        cursor.executemany(query, parameter_rows)


class Stage3Error(Exception):
    """Safe-to-return Stage 3 error with a stable API error code."""

    def __init__(self, code: str, public_message: str):
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


def _get_embeddings_client():
    """Build the configured local embedding client lazily."""
    if EMBEDDING_PROVIDER != "ollama":
        raise Stage3Error(
            "UNSUPPORTED_EMBEDDING_PROVIDER",
            "Only the Ollama embedding provider is currently implemented.",
        )
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise Stage3Error(
            "LANGCHAIN_OLLAMA_NOT_INSTALLED",
            "Install requirements-stage3.txt to enable Stage 3 embeddings.",
        ) from exc

    timeout = OLLAMA_REQUEST_TIMEOUT_SECONDS
    try:
        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_ctx=OLLAMA_NUM_CTX,
            client_kwargs={"timeout": timeout},
        )
    except TypeError:
        # Compatibility path for earlier langchain-ollama versions that do not
        # forward client_kwargs. Keep the existing model/context behavior intact.
        try:
            return OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
                num_ctx=OLLAMA_NUM_CTX,
            )
        except TypeError:
            return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def _assert_dimension_configuration() -> None:
    if EMBEDDING_DIMENSIONS != DATABASE_VECTOR_DIMENSIONS:
        raise Stage3Error(
            "EMBEDDING_SCHEMA_DIMENSION_MISMATCH",
            "This database schema is configured for 1,024-dimensional BGE-M3 vectors. Keep EMBEDDING_DIMENSIONS=1024 or perform an explicit schema migration before changing it.",
        )


def _validate_vector(vector: Any, *, context: str) -> list[float]:
    _assert_dimension_configuration()
    if not isinstance(vector, (list, tuple)):
        raise Stage3Error("INVALID_EMBEDDING_RESPONSE", f"Ollama returned an invalid vector for {context}.")
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise Stage3Error(
            "EMBEDDING_DIMENSION_MISMATCH",
            f"The configured model returned {len(vector)} dimensions; Sourcebook expects {EMBEDDING_DIMENSIONS}.",
        )
    values: list[float] = []
    for value in vector:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise Stage3Error("INVALID_EMBEDDING_RESPONSE", f"Ollama returned a non-numeric vector for {context}.") from exc
        if not math.isfinite(numeric):
            raise Stage3Error("INVALID_EMBEDDING_RESPONSE", f"Ollama returned a non-finite vector value for {context}.")
        values.append(numeric)
    return values


def _vector_literal(values: list[float]) -> str:
    """Serialize validated numeric values into pgvector's text input format."""
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def _sanitize_embedding_text(text: str) -> str:
    """Clean control characters and pathological symbol runs that can crash llama.cpp tokenizers."""
    if not text:
        return ""
    cleaned = _CONTROL_CHARS_RE.sub(" ", text)
    cleaned = re.sub(r"\.{5,}", "...", cleaned)
    cleaned = re.sub(r"-{5,}", "---", cleaned)
    cleaned = re.sub(r"_{5,}", "___", cleaned)
    cleaned = re.sub(r"={5,}", "===", cleaned)
    cleaned = re.sub(r"[ \t]{3,}", "  ", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip() or " "


def _split_text_for_ollama(text: str, max_chars: int) -> list[str]:
    """Split text exceeding max_chars into overlapping segments so no single input exceeds
    Ollama's physical micro-batch limit (n_ubatch=512 tokens in llama.cpp)."""
    cleaned = _sanitize_embedding_text(text)
    if len(cleaned) <= max_chars:
        return [cleaned]

    overlap = min(150, max(20, max_chars // 6))
    segments: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            search_from = max(start + 1, end - (max_chars // 4))
            window = cleaned[search_from:end]
            break_pos = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "), window.rfind(" "))
            if break_pos > 0:
                end = search_from + break_pos + 1
        piece = cleaned[start:end].strip()
        if piece:
            segments.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return segments or [cleaned[:max_chars]]


def _mean_pool_vectors(vectors: list[list[float]], *, context: str) -> list[float]:
    """Combine sub-segment embeddings into a single L2-normalized vector of EMBEDDING_DIMENSIONS."""
    if len(vectors) == 1:
        return _validate_vector(vectors[0], context=context)
    validated = [_validate_vector(vec, context=context) for vec in vectors]
    dim = len(validated[0])
    pooled = [0.0] * dim
    for vec in validated:
        for idx, val in enumerate(vec):
            pooled[idx] += val
    count = float(len(validated))
    pooled = [val / count for val in pooled]
    norm = math.sqrt(sum(val * val for val in pooled))
    if norm > 1e-12 and math.isfinite(norm):
        pooled = [val / norm for val in pooled]
    return _validate_vector(pooled, context=context)


def _call_ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    """Invoke Ollama embed_documents with retry + backoff if the local runner restarts."""
    last_exc: Exception | None = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        client = _get_embeddings_client()
        try:
            with _OLLAMA_LOCK:
                if hasattr(client, "embed_documents"):
                    raw_vectors = client.embed_documents(texts)
                else:
                    raw_vectors = [client.embed_query(item) for item in texts]
            if not isinstance(raw_vectors, (list, tuple)) or len(raw_vectors) != len(texts):
                raise Stage3Error("INVALID_EMBEDDING_RESPONSE", "Ollama returned an unexpected number of vectors.")
            return [list(vec) for vec in raw_vectors]
        except Stage3Error:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < OLLAMA_MAX_RETRIES:
                backoff = min(3.0, 0.5 * (2 ** (attempt - 1)))
                logger.warning(
                    "Ollama embed attempt %s/%s failed (%s: %s); retrying in %.2fs",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def _embed_single_text_resilient(text: str, *, context: str, max_chars: int | None = None) -> list[float]:
    """Embed a single text string, progressively splitting into smaller windows if Ollama's
    physical batch buffer rejects the input size."""
    effective_max_chars = max_chars or OLLAMA_SAFE_CHUNK_CHARS
    segments = _split_text_for_ollama(text, effective_max_chars)

    try:
        if len(segments) == 1:
            client = _get_embeddings_client()
            with _OLLAMA_LOCK:
                if hasattr(client, "embed_documents"):
                    res = _call_ollama_embed_batch(segments)
                    return _validate_vector(res[0], context=context)
                vec = client.embed_query(segments[0])
                return _validate_vector(vec, context=context)
        else:
            seg_vectors: list[list[float]] = []
            for seg in segments:
                seg_res = _call_ollama_embed_batch([seg])
                seg_vectors.append(seg_res[0])
            return _mean_pool_vectors(seg_vectors, context=context)
    except Stage3Error:
        raise
    except Exception as first_exc:
        for fallback_chars in (600, 350):
            if fallback_chars >= effective_max_chars:
                continue
            try:
                smaller_segments = _split_text_for_ollama(text, fallback_chars)
                seg_vectors = []
                for seg in smaller_segments:
                    seg_res = _call_ollama_embed_batch([seg])
                    seg_vectors.append(seg_res[0])
                logger.info(
                    "Recovered %s by splitting into %s sub-segments of <=%s chars",
                    context,
                    len(smaller_segments),
                    fallback_chars,
                )
                return _mean_pool_vectors(seg_vectors, context=context)
            except Stage3Error:
                raise
            except Exception:
                continue
        raise first_exc


def _embed_batch_resilient(texts: list[str], *, start_offset: int) -> list[list[float]]:
    """Embed a batch of chunk texts; bisect on failure so one dense chunk never crashes the batch."""
    if not texts:
        return []

    sanitized = [_sanitize_embedding_text(t) for t in texts]
    if all(len(t) <= OLLAMA_SAFE_CHUNK_CHARS for t in sanitized):
        try:
            raw_vectors = _call_ollama_embed_batch(sanitized)
            return [
                _validate_vector(vec, context=f"chunk batch offset {start_offset + idx}")
                for idx, vec in enumerate(raw_vectors)
            ]
        except Stage3Error:
            raise
        except Exception as exc:
            logger.warning(
                "Batch embedding at offset %s (size=%s) failed (%s); bisecting batch for resilience.",
                start_offset,
                len(texts),
                exc,
            )

    if len(texts) > 1:
        mid = len(texts) // 2
        left = _embed_batch_resilient(texts[:mid], start_offset=start_offset)
        right = _embed_batch_resilient(texts[mid:], start_offset=start_offset + mid)
        return left + right

    # Single chunk path: if it already failed at OLLAMA_SAFE_CHUNK_CHARS, step down immediately to 600 chars
    initial_max = (
        OLLAMA_SAFE_CHUNK_CHARS
        if len(sanitized[0]) > OLLAMA_SAFE_CHUNK_CHARS
        else min(600, OLLAMA_SAFE_CHUNK_CHARS)
    )
    single_vec = _embed_single_text_resilient(
        texts[0],
        context=f"chunk offset {start_offset}",
        max_chars=initial_max,
    )
    return [single_vec]


def _extract_query_keywords(query: str) -> tuple[list[str], str]:
    """Extract meaningful query terms and core target phrase (e.g. 'NER', 'machine learning')."""
    raw_tokens = re.findall(r"[A-Za-z0-9_+-]{2,}", query)
    keywords = [
        tok
        for tok in raw_tokens
        if tok.lower() not in _QUERY_STOPWORDS or (tok.isupper() and len(tok) >= 2)
    ]
    if not keywords:
        keywords = raw_tokens
    core_phrase = " ".join(keywords).strip()
    return keywords, core_phrase


def _split_chunk_into_atomic_units(chunk_text: str) -> list[str]:
    """Split a chunk into individual pointers, table rows, or sentence groups for pinpoint extraction."""
    # Strip leading Section/Subsection context lines when extracting the focused answer
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    body_lines = [
        line
        for line in lines
        if not line.startswith(("Section:", "Subsection:"))
    ]
    if not body_lines:
        body_lines = lines

    # Check if the chunk has multiple bullet/numbered lines or table rows
    bullet_re = re.compile(r"^(?:[-*+•▪‣]|\d{1,3}[.)]|Row\s+\d+:)\s+")
    units: list[str] = []
    current_bullet: list[str] = []
    for line in body_lines:
        if bullet_re.match(line):
            if current_bullet:
                units.append(" ".join(current_bullet).strip())
            current_bullet = [line]
        elif current_bullet:
            current_bullet.append(line)
        else:
            # Split prose paragraph into sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            units.extend(sentences if len(sentences) > 1 else [line])
    if current_bullet:
        units.append(" ".join(current_bullet).strip())

    return [u for u in units if u]


def _score_text_against_query(text: str, keywords: list[str], core_phrase: str) -> float:
    """Compute a lexical/definition relevance score for a candidate chunk or sentence unit."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    score = 0.0

    # Exact phrase match bonus (e.g. "machine learning" or "named entity recognition")
    if core_phrase and len(core_phrase) >= 2:
        phrase_pattern = re.compile(rf"\b{re.escape(core_phrase.lower())}\b")
        if phrase_pattern.search(text_lower):
            score += 0.45
        # Definition lead-in bonus (e.g., "NER: ...", "NER is ...", "Machine Learning (ML) is ...")
        def_pattern = re.compile(
            rf"(?:^|[\n•*-]|\d+[.)]\s*)\s*{re.escape(core_phrase.lower())}(?:\s*\([^)]+\))?\s*(?::|—|–|-|\bis\b|\brefers\s+to\b)"
        )
        if def_pattern.search(text_lower):
            score += 0.55

    matched_keywords = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if re.search(rf"\b{re.escape(kw_lower)}\b", text_lower):
            matched_keywords += 1
            # Extra weight for exact case-sensitive acronym match (e.g., "NER", "RAG", "LLM")
            if kw.isupper() and len(kw) >= 2 and re.search(rf"\b{re.escape(kw)}\b", text):
                score += 0.25

    score += 0.35 * (matched_keywords / max(1, len(keywords)))
    return score


def _extract_focused_excerpt(chunk_text: str, query: str) -> str:
    """Extract the most relevant 1-2 sentences or single bullet pointer from a larger chunk."""
    cleaned = chunk_text.strip()
    if not cleaned:
        return ""
    keywords, core_phrase = _extract_query_keywords(query)
    if not keywords:
        return cleaned

    units = _split_chunk_into_atomic_units(cleaned)
    if len(units) <= 1:
        return units[0] if units else cleaned

    scored = [
        (_score_text_against_query(unit, keywords, core_phrase), idx, unit)
        for idx, unit in enumerate(units)
    ]
    best_score, best_idx, best_unit = max(scored, key=lambda item: (item[0], -item[1]))
    if best_score <= 0:
        # Return first substantive unit without Section/Subsection headers
        return "\n".join(units[:2]).strip()

    # If the best unit is a short sentence (< 160 chars) and not a self-contained bullet/row,
    # include the immediately following sentence for completeness if it is part of the same thought.
    bullet_re = re.compile(r"^(?:[-*+•▪‣]|\d{1,3}[.)]|Row\s+\d+:)\s+")
    if (
        len(best_unit) < 160
        and best_idx + 1 < len(units)
        and not bullet_re.match(units[best_idx + 1])
    ):
        return f"{best_unit} {units[best_idx + 1]}".strip()
    return best_unit.strip()


def embedding_health() -> dict[str, Any]:
    """Make a real local Ollama embedding call and validate its output dimensions."""
    _assert_dimension_configuration()
    client = _get_embeddings_client()
    try:
        with _OLLAMA_LOCK:
            vector = client.embed_query("Sourcebook local embedding health check")
    except Exception as exc:
        logger.exception("Ollama embedding health check failed")
        raise Stage3Error(
            "OLLAMA_UNAVAILABLE",
            "Could not generate an embedding. Check that Ollama is running and the configured model is pulled.",
        ) from exc
    validated = _validate_vector(vector, context="health check")
    return {
        "ok": True,
        "provider": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_EMBEDDING_MODEL,
        "dimensions": len(validated),
    }


def _document_chunk_id(document_id: str, chunk_key: str) -> str:
    """Stable within a document version; different document versions never share chunk ids."""
    digest = hashlib.sha256(f"{document_id}\0{chunk_key}".encode("utf-8")).hexdigest()[:32]
    return f"chk_{digest}"


def prepare_chunk_records(
    document: dict[str, Any], artifact: dict[str, Any], artifact_row: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate Stage 2 artifact identity/policy and normalize chunks for database persistence."""
    if not isinstance(artifact, dict):
        raise Stage3Error("INVALID_ARTIFACT_JSON", "The Stage 2 artifact is not a JSON object.")

    privacy = artifact.get("privacy") or {}
    artifact_doc = artifact.get("document") or {}
    if str(artifact_row.get("status")) != "SANITIZED" or str(privacy.get("artifact_status")) != "SANITIZED":
        raise Stage3Error("ARTIFACT_NOT_SANITIZED", "Only SANITIZED Stage 2 artifacts can be embedded.")
    if (
        privacy.get("embedding_eligible") is not True
        or (artifact_row.get("metadata") or {}).get("embedding_eligible") is not True
    ):
        raise Stage3Error(
            "ARTIFACT_NOT_EMBEDDING_ELIGIBLE",
            "The Stage 2 policy did not approve this artifact for embedding.",
        )

    expected_identity = {
        "document_id": str(document["id"]),
        "workspace_id": str(document["workspace_id"]),
        "document_version_number": int(document["version_number"]),
    }
    for key, expected in expected_identity.items():
        value = artifact_doc.get(key)
        if value is None or str(value) != str(expected):
            raise Stage3Error(
                "ARTIFACT_IDENTITY_MISMATCH",
                f"The Stage 2 artifact {key} does not match its database record.",
            )
    artifact_source_id = artifact_doc.get("document_source_id", artifact_doc.get("source_id"))
    expected_source_id = document.get("source_id")
    if artifact_source_id is not None and expected_source_id is not None:
        if str(artifact_source_id) != str(expected_source_id):
            raise Stage3Error(
                "ARTIFACT_IDENTITY_MISMATCH",
                "The Stage 2 artifact source ID does not match this document version.",
            )
    artifact_source_hash = str(artifact_doc.get("sha256") or "").lower()
    document_source_hash = str(document.get("file_hash") or "").lower()
    if not artifact_source_hash or artifact_source_hash != document_source_hash:
        raise Stage3Error(
            "ARTIFACT_SOURCE_CHECKSUM_MISMATCH",
            "The artifact source checksum does not match this document version.",
        )

    chunk_items = artifact.get("chunks")
    if not isinstance(chunk_items, list):
        raise Stage3Error(
            "INVALID_ARTIFACT_CHUNKS", "The Stage 2 artifact does not contain a valid chunks array."
        )

    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for position, chunk in enumerate(chunk_items):
        if not isinstance(chunk, dict):
            raise Stage3Error("INVALID_ARTIFACT_CHUNKS", "A chunk in the Stage 2 artifact is malformed.")
        text = str(chunk.get("text") or chunk.get("main_content") or "").strip()
        if not text:
            continue
        main_content = str(chunk.get("main_content") or text).strip()
        chunk_type = str(chunk.get("chunk_type") or "text").lower()
        if chunk_type not in {"text", "table"}:
            raise Stage3Error("UNSUPPORTED_CHUNK_TYPE", f"Unsupported Stage 2 chunk type: {chunk_type}.")

        chunk_key = str(chunk.get("chunk_id") or f"index-{position}")
        if chunk_key in seen_keys:
            raise Stage3Error(
                "DUPLICATE_CHUNK_KEY", "The Stage 2 artifact contains duplicate chunk identifiers."
            )
        seen_keys.add(chunk_key)

        source_ids = chunk.get("source_element_ids")
        if not isinstance(source_ids, list):
            source_id = chunk.get("source_element_id")
            source_ids = [str(source_id)] if source_id is not None else []
        source_locations = chunk.get("source_locations")
        if not isinstance(source_locations, list):
            source_locations = []
        source_location = chunk.get("source_location")
        if not isinstance(source_location, dict):
            source_location = {}
        heading_context = chunk.get("heading_context")
        if not isinstance(heading_context, list):
            heading_context = []
        chunk_metadata = chunk.get("metadata")
        if not isinstance(chunk_metadata, dict):
            chunk_metadata = {}
        chunking_config = chunk.get("chunking_config")
        if not isinstance(chunking_config, dict):
            chunking_config = {}

        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records.append(
            {
                "id": _document_chunk_id(str(document["id"]), chunk_key),
                "chunk_key": chunk_key,
                "document_id": str(document["id"]),
                "source_id": str(document["source_id"]) if document.get("source_id") else None,
                "user_id": str(document["user_id"]),
                "workspace_id": str(document["workspace_id"]),
                "document_version_number": int(document["version_number"]),
                "artifact_id": str(artifact_row["id"]),
                "ingestion_run_id": str(artifact_row["ingestion_run_id"]),
                "chunk_index": int(chunk.get("chunk_index", position)),
                "chunk_type": chunk_type,
                "content": text,
                "main_content": main_content,
                "content_sha256": content_sha256,
                "token_count": max(0, int(chunk.get("token_count") or 0)),
                "source_element_id": str(chunk["source_element_id"])
                if chunk.get("source_element_id") is not None
                else None,
                "source_element_ids": source_ids,
                "source_location": source_location,
                "source_locations": source_locations,
                "heading_context": heading_context,
                "metadata": {
                    "main_content": main_content,
                    "chunk_metadata": chunk_metadata,
                    "chunking_config": chunking_config,
                    "source_element_type": chunk.get("source_element_type"),
                },
            }
        )

    if not records:
        raise Stage3Error(
            "NO_EMBEDDABLE_CHUNKS",
            "The approved Stage 2 artifact contains no non-empty text/table chunks.",
        )
    return records


def _get_document_and_artifact(
    document_id: str, user_id: str, artifact_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _get_pool().connection() as connection:
        document = connection.execute(
            """
            SELECT d.id, d.source_id, d.user_id, d.workspace_id, d.original_name, d.status,
                   d.deleted_at, d.version_number, d.file_hash, d.is_current,
                   w.deleted_at AS workspace_deleted_at
            FROM documents d
            JOIN workspaces w ON w.id=d.workspace_id
            WHERE d.id=%s AND d.user_id=%s
            """,
            (document_id, user_id),
        ).fetchone()
        if not document or document["deleted_at"] is not None or document["workspace_deleted_at"] is not None:
            raise Stage3Error(
                "DOCUMENT_NOT_AVAILABLE", "The document or its workspace is no longer available."
            )

        artifact_row = connection.execute(
            """
            SELECT a.id, a.document_id, a.user_id, a.workspace_id, a.document_version_number,
                   a.status, a.s3_bucket, a.s3_key, a.sha256, a.metadata, a.created_at,
                   a.ingestion_run_id
            FROM extracted_artifacts a
            JOIN ingestion_runs r ON r.id=a.ingestion_run_id
            WHERE a.id=%s AND a.document_id=%s AND a.user_id=%s AND a.workspace_id=%s
              AND a.document_version_number=%s AND r.id=a.ingestion_run_id
            """,
            (artifact_id, document_id, user_id, document["workspace_id"], document["version_number"]),
        ).fetchone()
        if not artifact_row:
            raise Stage3Error(
                "ARTIFACT_NOT_FOUND",
                "The selected Stage 2 artifact was not found for this document version.",
            )

        latest = connection.execute(
            """
            SELECT id, status FROM extracted_artifacts
            WHERE document_id=%s AND user_id=%s AND workspace_id=%s
            ORDER BY created_at DESC, id DESC LIMIT 1
            """,
            (document_id, user_id, document["workspace_id"]),
        ).fetchone()
        if not latest or str(latest["id"]) != str(artifact_id):
            raise Stage3Error(
                "ARTIFACT_SUPERSEDED",
                "A newer Stage 2 artifact exists. Queue indexing for the latest artifact instead.",
            )
        if artifact_row["status"] != "SANITIZED":
            raise Stage3Error(
                "ARTIFACT_NOT_SANITIZED", "Only SANITIZED Stage 2 artifacts can be embedded."
            )
        return dict(document), dict(artifact_row)


def queue_document_index(
    connection: Connection, document_id: str, user_id: str, force: bool = False
) -> dict[str, Any]:
    """Queue indexing for the latest approved Stage 2 artifact of exactly one version."""
    document = connection.execute(
        """
        SELECT d.id, d.workspace_id, d.status, d.deleted_at, d.embedding_status,
               d.embedding_target_artifact_id, d.embedded_artifact_id, d.embedding_model,
               d.embedded_chunk_count, w.deleted_at AS workspace_deleted_at
        FROM documents d JOIN workspaces w ON w.id=d.workspace_id
        WHERE d.id=%s AND d.user_id=%s
        """,
        (document_id, user_id),
    ).fetchone()
    if not document or document["deleted_at"] is not None or document["workspace_deleted_at"] is not None:
        raise Stage3Error("DOCUMENT_NOT_AVAILABLE", "The document or its workspace was not found.")
    if document["status"] not in {"UPLOADED", "SUPERSEDED"}:
        raise Stage3Error(
            "DOCUMENT_NOT_READY", "Only completed uploaded document versions can be indexed."
        )

    artifact = connection.execute(
        """
        SELECT a.id, a.status, a.metadata, a.document_version_number, a.created_at
        FROM extracted_artifacts a
        WHERE a.document_id=%s AND a.user_id=%s AND a.workspace_id=%s
        ORDER BY a.created_at DESC, a.id DESC LIMIT 1
        """,
        (document_id, user_id, document["workspace_id"]),
    ).fetchone()
    if not artifact:
        raise Stage3Error(
            "STAGE2_NOT_COMPLETE", "Run Stage 2 extraction for this document before indexing it."
        )
    if artifact["status"] != "SANITIZED":
        raise Stage3Error(
            "ARTIFACT_NOT_SANITIZED",
            "The latest Stage 2 artifact is quarantined; it cannot be embedded.",
        )
    metadata = artifact.get("metadata") or {}
    if metadata.get("embedding_eligible") is not True:
        raise Stage3Error(
            "ARTIFACT_NOT_EMBEDDING_ELIGIBLE",
            "Stage 2 policy has not approved this artifact for embedding.",
        )

    if document["embedding_status"] in {"QUEUED", "EMBEDDING"}:
        return {
            "document_id": document_id,
            "artifact_id": document["embedding_target_artifact_id"],
            "status": document["embedding_status"],
            "queued": False,
            "message": "Embedding is already queued or in progress for this document version.",
        }

    already_indexed = (
        document["embedding_status"] == "INDEXED"
        and str(document["embedded_artifact_id"] or "") == str(artifact["id"])
        and str(document["embedding_model"] or "") == OLLAMA_EMBEDDING_MODEL
    )
    if already_indexed and not force:
        return {
            "document_id": document_id,
            "artifact_id": artifact["id"],
            "status": "INDEXED",
            "queued": False,
            "already_indexed": True,
            "chunk_count": int(document["embedded_chunk_count"] or 0),
        }

    connection.execute(
        """
        UPDATE documents
        SET embedding_status='QUEUED', embedding_target_artifact_id=%s,
            embedding_error_code=NULL, embedding_error_message=NULL,
            embedding_requested_at=NOW(), embedding_started_at=NULL,
            embedding_completed_at=NULL, embedding_updated_at=NOW()
        WHERE id=%s AND user_id=%s AND deleted_at IS NULL
        """,
        (artifact["id"], document_id, user_id),
    )
    connection.commit()
    return {
        "document_id": document_id,
        "artifact_id": artifact["id"],
        "status": "QUEUED",
        "queued": True,
        "already_indexed": False,
    }


def _load_artifact_json(artifact_row: dict[str, Any]) -> dict[str, Any]:
    bucket = str(artifact_row.get("s3_bucket") or S3_BUCKET_NAME)
    try:
        data = _download_object_bytes(str(artifact_row["s3_key"]), bucket=bucket)
    except Exception as exc:
        logger.exception("Could not download Stage 2 artifact id=%s", artifact_row.get("id"))
        raise Stage3Error(
            "ARTIFACT_DOWNLOAD_FAILED", "Could not download the private Stage 2 artifact from S3."
        ) from exc
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash.lower() != str(artifact_row["sha256"]).lower():
        raise Stage3Error(
            "ARTIFACT_CHECKSUM_MISMATCH",
            "The Stage 2 artifact checksum does not match its database metadata.",
        )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage3Error(
            "INVALID_ARTIFACT_JSON", "The Stage 2 artifact is not valid UTF-8 JSON."
        ) from exc
    return payload


def _generate_document_vectors(texts: list[str]) -> list[list[float]]:
    """Generate 1,024-D vectors for all chunks with adaptive batch sizing, bisection, and sub-chunk pooling."""
    _assert_dimension_configuration()
    _get_embeddings_client()

    vectors: list[list[float]] = []
    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            batch_vectors = _embed_batch_resilient(batch, start_offset=start)
        except Stage3Error:
            raise
        except Exception as exc:
            logger.exception("Ollama embed_documents failed for batch beginning at chunk %s", start)
            raise Stage3Error(
                "OLLAMA_EMBEDDING_FAILED",
                "Ollama failed while embedding document chunks. Check the Ollama service, model and available memory.",
            ) from exc
        if not isinstance(batch_vectors, (list, tuple)) or len(batch_vectors) != len(batch):
            raise Stage3Error(
                "INVALID_EMBEDDING_RESPONSE", "Ollama returned an unexpected number of vectors."
            )
        vectors.extend(batch_vectors)
    return vectors


def _mark_failed(document_id: str, user_id: str, artifact_id: str, error: Exception) -> None:
    if isinstance(error, Stage3Error):
        code, message = error.code, error.public_message
    else:
        code, message = (
            "EMBEDDING_INDEX_FAILED",
            "Stage 3 indexing failed. Review server logs for the document version.",
        )
    try:
        with _get_pool().connection() as connection:
            connection.execute(
                """
                UPDATE documents
                SET embedding_status='FAILED', embedding_error_code=%s,
                    embedding_error_message=%s, embedding_updated_at=NOW()
                WHERE id=%s AND user_id=%s AND deleted_at IS NULL
                  AND embedding_target_artifact_id=%s AND embedding_status IN ('QUEUED','EMBEDDING')
                """,
                (code, message, document_id, user_id, artifact_id),
            )
    except Exception:
        logger.exception("Could not persist Stage 3 failure status for document_id=%s", document_id)


def process_document_embedding(document_id: str, user_id: str, artifact_id: str) -> None:
    """Generate and atomically replace embeddings for exactly one immutable document version."""
    try:
        with _OLLAMA_LOCK:
            with _get_pool().connection() as connection:
                with connection.transaction():
                    state = connection.execute(
                        """
                        SELECT embedding_status, embedding_target_artifact_id, deleted_at
                        FROM documents WHERE id=%s AND user_id=%s FOR UPDATE
                        """,
                        (document_id, user_id),
                    ).fetchone()
                    if not state or state["deleted_at"] is not None:
                        return
                    if (
                        state["embedding_status"] != "QUEUED"
                        or str(state["embedding_target_artifact_id"] or "") != str(artifact_id)
                    ):
                        return
                    connection.execute(
                        """
                        UPDATE documents SET embedding_status='EMBEDDING', embedding_started_at=NOW(),
                            embedding_updated_at=NOW(), embedding_error_code=NULL, embedding_error_message=NULL
                        WHERE id=%s AND user_id=%s
                        """,
                        (document_id, user_id),
                    )

            document, artifact_row = _get_document_and_artifact(document_id, user_id, artifact_id)
            artifact = _load_artifact_json(artifact_row)
            records = prepare_chunk_records(document, artifact, artifact_row)
            vectors = _generate_document_vectors([record["content"] for record in records])
            if len(vectors) != len(records):
                raise Stage3Error(
                    "EMBEDDING_COUNT_MISMATCH",
                    "The number of generated vectors does not match the chunk count.",
                )

        try:
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise Stage3Error(
                "PSYCOPG_NOT_INSTALLED", "Install the application's requirements to persist vectors."
            ) from exc
        chunk_rows = []
        embedding_rows = []
        for record, vector in zip(records, vectors, strict=True):
            chunk_rows.append(
                (
                    record["id"],
                    record["chunk_key"],
                    record["document_id"],
                    record["source_id"],
                    record["user_id"],
                    record["workspace_id"],
                    record["document_version_number"],
                    record["artifact_id"],
                    record["ingestion_run_id"],
                    record["chunk_index"],
                    record["chunk_type"],
                    record["content"],
                    record["content_sha256"],
                    record["token_count"],
                    record["source_element_id"],
                    Jsonb(record["source_element_ids"]),
                    Jsonb(record["source_location"]),
                    Jsonb(record["source_locations"]),
                    Jsonb(record["heading_context"]),
                    Jsonb(record["metadata"]),
                )
            )
            embedding_rows.append(
                (
                    record["id"],
                    _vector_literal(vector),
                    OLLAMA_EMBEDDING_MODEL,
                    EMBEDDING_DIMENSIONS,
                    record["content_sha256"],
                )
            )

        with _get_pool().connection() as connection:
            with connection.transaction():
                current = connection.execute(
                    """
                    SELECT d.id, d.source_id, d.workspace_id, d.version_number, d.deleted_at,
                           d.embedding_status, d.embedding_target_artifact_id,
                           w.deleted_at AS workspace_deleted_at
                    FROM documents d JOIN workspaces w ON w.id=d.workspace_id
                    WHERE d.id=%s AND d.user_id=%s FOR UPDATE
                    """,
                    (document_id, user_id),
                ).fetchone()
                if (
                    not current
                    or current["deleted_at"] is not None
                    or current["workspace_deleted_at"] is not None
                    or current["embedding_status"] != "EMBEDDING"
                    or str(current["embedding_target_artifact_id"] or "") != str(artifact_id)
                ):
                    raise Stage3Error(
                        "DOCUMENT_CHANGED_DURING_INDEXING",
                        "The document changed or was deleted while embeddings were being generated.",
                    )
                latest = connection.execute(
                    """
                    SELECT id, status FROM extracted_artifacts
                    WHERE document_id=%s AND user_id=%s AND workspace_id=%s
                    ORDER BY created_at DESC, id DESC LIMIT 1
                    """,
                    (document_id, user_id, current["workspace_id"]),
                ).fetchone()
                if not latest or str(latest["id"]) != str(artifact_id) or latest["status"] != "SANITIZED":
                    raise Stage3Error(
                        "ARTIFACT_SUPERSEDED",
                        "A newer or non-approved artifact exists; this indexing attempt was not committed.",
                    )

                connection.execute("DELETE FROM document_chunks WHERE document_id=%s", (document_id,))
                _execute_many(
                    connection,
                    """
                    INSERT INTO document_chunks (
                        id, chunk_key, document_id, source_id, user_id, workspace_id,
                        document_version_number, artifact_id, ingestion_run_id, chunk_index,
                        chunk_type, content, content_sha256, token_count, source_element_id,
                        source_element_ids, source_location, source_locations, heading_context, metadata
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                    """,
                    chunk_rows,
                )
                _execute_many(
                    connection,
                    """
                    INSERT INTO chunk_embeddings (
                        chunk_id, embedding, embedding_model, embedding_dimensions, content_sha256
                    ) VALUES (%s, %s::vector, %s, %s, %s)
                    """,
                    embedding_rows,
                )
                connection.execute(
                    """
                    UPDATE documents
                    SET embedding_status='INDEXED', embedding_target_artifact_id=%s,
                        embedded_artifact_id=%s, embedding_model=%s, embedding_dimensions=%s,
                        embedded_chunk_count=%s, embedding_error_code=NULL, embedding_error_message=NULL,
                        embedding_completed_at=NOW(), embedding_updated_at=NOW()
                    WHERE id=%s AND user_id=%s
                    """,
                    (
                        artifact_id,
                        artifact_id,
                        OLLAMA_EMBEDDING_MODEL,
                        EMBEDDING_DIMENSIONS,
                        len(records),
                        document_id,
                        user_id,
                    ),
                )
        logger.info(
            "Stage 3 indexed document_id=%s artifact_id=%s chunks=%s model=%s dimensions=%s",
            document_id,
            artifact_id,
            len(records),
            OLLAMA_EMBEDDING_MODEL,
            EMBEDDING_DIMENSIONS,
        )
    except Exception as exc:
        logger.exception("Stage 3 indexing failed document_id=%s artifact_id=%s", document_id, artifact_id)
        _mark_failed(document_id, user_id, artifact_id, exc)


def delete_document_vectors(connection: Connection, document_id: str) -> int:
    """Delete chunks and cascaded vectors for exactly one immutable document version."""
    count = connection.execute(
        "SELECT COUNT(*) AS count FROM document_chunks WHERE document_id=%s",
        (document_id,),
    ).fetchone()["count"]
    connection.execute("DELETE FROM document_chunks WHERE document_id=%s", (document_id,))
    connection.execute(
        """
        UPDATE documents SET embedding_status='NOT_INDEXED', embedding_target_artifact_id=NULL,
            embedded_artifact_id=NULL, embedding_model=NULL, embedding_dimensions=NULL,
            embedded_chunk_count=0, embedding_error_code=NULL, embedding_error_message=NULL,
            embedding_requested_at=NULL, embedding_started_at=NULL, embedding_completed_at=NULL,
            embedding_updated_at=NOW()
        WHERE id=%s
        """,
        (document_id,),
    )
    return int(count or 0)


def delete_source_vectors(connection: Connection, source_id: str) -> int:
    """Delete vectors for all versions of one logical source, never adjacent sources."""
    count = connection.execute(
        "SELECT COUNT(*) AS count FROM document_chunks WHERE source_id=%s",
        (source_id,),
    ).fetchone()["count"]
    connection.execute("DELETE FROM document_chunks WHERE source_id=%s", (source_id,))
    connection.execute(
        """
        UPDATE documents SET embedding_status='NOT_INDEXED', embedding_target_artifact_id=NULL,
            embedded_artifact_id=NULL, embedding_model=NULL, embedding_dimensions=NULL,
            embedded_chunk_count=0, embedding_error_code=NULL, embedding_error_message=NULL,
            embedding_requested_at=NULL, embedding_started_at=NULL, embedding_completed_at=NULL,
            embedding_updated_at=NOW()
        WHERE source_id=%s
        """,
        (source_id,),
    )
    return int(count or 0)


def semantic_search(
    *,
    user_id: str,
    workspace_id: str,
    query: str,
    top_k: int = 5,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Workspace-scoped hybrid (cosine + lexical/definition boost) search with pinpoint answer excerpts."""
    clean_query = query.strip()
    if not clean_query:
        raise Stage3Error("EMPTY_QUERY", "Search query cannot be empty.")
    if len(clean_query) > 8000:
        raise Stage3Error("QUERY_TOO_LONG", "Search query must be 8,000 characters or fewer.")
    if top_k < 1 or top_k > 50:
        raise Stage3Error("INVALID_TOP_K", "top_k must be between 1 and 50.")

    search_started_at = perf_counter()
    with _get_pool().connection() as authorization_connection:
        workspace = authorization_connection.execute(
            "SELECT id FROM workspaces WHERE id=%s AND user_id=%s AND deleted_at IS NULL",
            (workspace_id, user_id),
        ).fetchone()
        if not workspace:
            raise Stage3Error("WORKSPACE_NOT_FOUND", "Workspace was not found.")

    embedding_started_at = perf_counter()
    try:
        client = _get_embeddings_client()
        sanitized_query = _sanitize_embedding_text(clean_query)
        if len(sanitized_query) <= OLLAMA_SAFE_CHUNK_CHARS:
            with _OLLAMA_LOCK:
                raw_query_vec = client.embed_query(sanitized_query)
            query_vector = _validate_vector(raw_query_vec, context="search query")
        else:
            query_vector = _embed_single_text_resilient(
                sanitized_query, context="search query", max_chars=OLLAMA_SAFE_CHUNK_CHARS
            )
    except Stage3Error:
        raise
    except Exception as exc:
        logger.exception("Ollama query embedding failed")
        raise Stage3Error(
            "OLLAMA_UNAVAILABLE",
            "Could not embed the query. Check the Ollama service and configured model.",
        ) from exc

    query_embedding_ms = round((perf_counter() - embedding_started_at) * 1000, 2)
    vector_text = _vector_literal(query_vector)
    candidate_limit = min(max(int(top_k) * 4, 20), 100)
    retrieval_started_at = perf_counter()
    with _get_pool().connection() as connection:
        connection.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        rows = connection.execute(
            """
            SELECT dc.id AS chunk_id, dc.chunk_key AS source_chunk_id,
                   dc.document_id, d.original_name AS filename,
                   d.version_number, d.is_current, dc.chunk_index, dc.chunk_type,
                   dc.content, dc.source_element_id, dc.source_element_ids,
                   dc.source_location, dc.source_locations, dc.heading_context,
                   dc.metadata, dc.artifact_id, ce.embedding_model,
                   (1 - (ce.embedding <=> %s::vector))::double precision AS similarity
            FROM chunk_embeddings ce
            JOIN document_chunks dc ON dc.id=ce.chunk_id
            JOIN documents d ON d.id=dc.document_id
            JOIN workspaces w ON w.id=dc.workspace_id AND w.user_id=dc.user_id
            WHERE dc.user_id=%s AND dc.workspace_id=%s
              AND w.deleted_at IS NULL AND d.deleted_at IS NULL
              AND dc.artifact_id=d.embedded_artifact_id
              AND dc.artifact_id=(
                  SELECT a_latest.id FROM extracted_artifacts a_latest
                  WHERE a_latest.document_id=d.id AND a_latest.user_id=d.user_id
                    AND a_latest.workspace_id=d.workspace_id
                  ORDER BY a_latest.created_at DESC, a_latest.id DESC LIMIT 1
              )
              AND EXISTS (
                  SELECT 1 FROM extracted_artifacts a_approved
                  WHERE a_approved.id=dc.artifact_id AND a_approved.status='SANITIZED'
                    AND a_approved.metadata->>'embedding_eligible'='true'
              )
              AND ce.embedding_model=%s
              AND (%s::boolean OR d.is_current=TRUE)
            ORDER BY ce.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                vector_text,
                user_id,
                workspace_id,
                OLLAMA_EMBEDDING_MODEL,
                bool(include_superseded),
                vector_text,
                candidate_limit,
            ),
        ).fetchall()
    pgvector_search_ms = round((perf_counter() - retrieval_started_at) * 1000, 2)

    keywords, core_phrase = _extract_query_keywords(clean_query)
    scored_candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        cosine_sim = float(item["similarity"])
        item["similarity"] = cosine_sim
        content_text = str(item.get("content") or "")
        focused = _extract_focused_excerpt(content_text, clean_query)
        item["focused_answer"] = focused
        item["matched_excerpt"] = focused

        # Hybrid score combines semantic similarity with exact lexical/acronym/definition match
        lex_boost = _score_text_against_query(focused or content_text, keywords, core_phrase)
        hybrid_score = round(cosine_sim + (0.18 * lex_boost), 6)
        item["hybrid_score"] = hybrid_score
        item["citation"] = {
            "document_id": item["document_id"],
            "filename": item["filename"],
            "version_number": item["version_number"],
            "source_chunk_id": item["source_chunk_id"],
            "source_element_id": item["source_element_id"],
            "source_element_ids": item["source_element_ids"],
            "source_location": item["source_location"],
            "source_locations": item["source_locations"],
            "heading_context": item["heading_context"],
        }
        scored_candidates.append((hybrid_score, -idx, item))

    scored_candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    results = [entry[2] for entry in scored_candidates[: int(top_k)]]

    return {
        "workspace_id": workspace_id,
        "query": clean_query,
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "include_superseded": bool(include_superseded),
        "top_k": int(top_k),
        "result_count": len(results),
        "timings_ms": {
            "query_embedding_ms": query_embedding_ms,
            "pgvector_search_ms": pgvector_search_ms,
            "total_ms": round((perf_counter() - search_started_at) * 1000, 2),
        },
        "results": results,
    }
