"""Stage 2 ingestion execution: source retrieval, LangGraph, artifact persistence, and automatic Stage 3 vector indexing."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
import uuid

try:
    from psycopg.types.json import Jsonb
except ModuleNotFoundError:
    Jsonb = lambda value: value  # type: ignore[misc,assignment]

try:
    from app.config import AUTO_EMBED_ON_INGESTION, S3_BUCKET_NAME
except ModuleNotFoundError:
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
    AUTO_EMBED_ON_INGESTION = os.getenv("AUTO_EMBED_ON_INGESTION", "true").strip().lower() != "false"

from app.ingestion.chunking import count_tokens
from app.ingestion.extractors import build_main_content
from app.ingestion.graph import build_ingestion_graph

logger = logging.getLogger("sourcebook_ingestion")


def _new_id(prefix: str) -> str:
    try:
        from app.security import new_id

        return new_id(prefix)
    except ModuleNotFoundError:
        return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _get_pool():
    from app.db_client import get_pool

    return get_pool()


def _download_object_to_path(s3_key: str, destination_path: str) -> None:
    from app.s3_client import download_object_to_path

    download_object_to_path(s3_key, destination_path)


def upload_bytes(data: bytes, s3_key: str, content_type: str, metadata: dict[str, str] | None = None) -> str:
    from app.s3_client import upload_bytes as _s3_upload_bytes

    return _s3_upload_bytes(data, s3_key, content_type, metadata)


def _update_run(
    run_id: str,
    status: str,
    details: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    finished: bool = False,
) -> None:
    """Update durable ingestion status without ever logging document content or PII values."""
    payload = Jsonb(details or {})
    with _get_pool().connection() as connection:
        connection.execute(
            """
            UPDATE ingestion_runs
            SET status=%s, progress_metadata=%s,
                error_code=%s, error_message=%s,
                started_at=COALESCE(started_at, NOW()),
                finished_at=CASE WHEN %s THEN NOW() ELSE finished_at END,
                updated_at=NOW()
            WHERE id=%s
            """,
            (status, payload, error_code, error_message, finished, run_id),
        )
        connection.commit()


def _artifact_writer(state: dict[str, Any]) -> dict[str, Any]:
    document = state["document"]
    run_id = state["run_id"]
    policy = state["policy"]
    pii_findings = state.get("pii_findings", [])
    entities = state.get("entities", [])
    relationships = state.get("relationships", [])
    chunks = state.get("chunks", [])
    elements = state.get("sanitized_elements", [])
    main_content = state.get("main_content")
    if main_content is None:
        main_content = build_main_content(elements)
    assets = state.get("assets", [])
    asset_descriptors = {item["asset_id"]: item for item in state.get("sanitized_asset_descriptors", [])}

    asset_metadata: list[dict[str, Any]] = []
    asset_keys: list[str] = []
    for raw_asset in assets:
        safe_desc = asset_descriptors.get(raw_asset.asset_id, {})
        extension = PurePosixPath(raw_asset.filename).suffix.lower() or ".bin"
        asset_key = (
            f"rag-quarantine-assets/v1/users/{document['user_id']}/workspaces/{document['workspace_id']}"
            f"/documents/{document['id']}/runs/{run_id}/{raw_asset.asset_id}{extension}"
        )
        asset_sha = hashlib.sha256(raw_asset.data).hexdigest()
        upload_bytes(
            raw_asset.data,
            asset_key,
            raw_asset.content_type or mimetypes.guess_type(raw_asset.filename)[0] or "application/octet-stream",
            {
                "document-id": document["id"],
                "ingestion-run-id": run_id,
                "sha256": asset_sha,
                "privacy-state": "quarantined-unreviewed",
            },
        )
        asset_keys.append(asset_key)
        asset_metadata.append(
            {
                "asset_id": raw_asset.asset_id,
                "s3_key": asset_key,
                "filename": safe_desc.get("filename", raw_asset.filename),
                "content_type": raw_asset.content_type,
                "size_bytes": len(raw_asset.data),
                "sha256": asset_sha,
                "source_location": safe_desc.get("source_location", {}),
                "description": safe_desc.get("description", ""),
                "policy_status": "QUARANTINED_UNSCANNED_VISUAL",
                "access_note": "No presigned download URL is issued by the Stage 2 API for unreviewed visual assets.",
            }
        )

    document_status = policy["status"]
    artifact_prefix = "rag-quarantine" if document_status == "QUARANTINED" else "rag-artifacts"
    artifact_key = (
        f"{artifact_prefix}/v1/users/{document['user_id']}/sessions/{document['session_id']}"
        f"/workspaces/{document['workspace_id']}/documents/{document['id']}/runs/{run_id}/sanitized-extraction.json"
    )
    type_counts = Counter(str(item.get("type", "unknown")) for item in elements)
    entity_counts = Counter(str(item.get("label", "unknown")) for item in entities)
    pii_counts = Counter(str(item.get("entity_type", "unknown")) for item in pii_findings)
    main_content_tokens = count_tokens(main_content) if main_content else 0

    artifact = {
        "artifact_schema_version": "1.1",
        "artifact_id": _new_id("art"),
        "ingestion_run_id": run_id,
        "document": {
            "document_id": document["id"],
            "document_source_id": document["source_id"],
            "document_version_number": document["version_number"],
            "workspace_id": document["workspace_id"],
            "session_id": document["session_id"],
            "sha256": document["file_hash"],
            "filename": state.get("sanitized_filename", "document"),
            "file_type": document["extension"],
            "content_type": document["content_type"],
        },
        "processing": {
            "parser_name": state.get("parser_name"),
            "parser_version": state.get("parser_version"),
            "guardrail": "presidio-analyzer-anonymizer",
            "guardrail_policy_version": "sourcebook-pii-policy-v1",
            "nlp_model": "spaCy configured English model",
            "chunking_strategy": "structure-aware-token-chunks-v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warnings": state.get("extraction_warnings", []),
        },
        "privacy": {
            "decision": policy["decision"],
            "artifact_status": document_status,
            "pii_detected": bool(pii_findings),
            "pii_finding_count": len(pii_findings),
            "pii_entity_counts": dict(pii_counts),
            "raw_pii_values_in_derived_artifact": False,
            "raw_pii_values_in_findings_table": False,
            "original_source_retained_in_private_s3": True,
            "embedding_eligible": bool(policy.get("embedding_eligible")),
            "visual_assets_quarantined": bool(asset_metadata),
            "visual_assets_policy": "QUARANTINE_UNREVIEWED_IMAGES",
            "reason_codes": policy.get("reason_codes", []),
        },
        "main_content": main_content,
        "content_format": "markdown",
        "elements": elements,
        "tables": [item for item in elements if str(item.get("type", "")).lower() == "table"],
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
        "assets": asset_metadata,
        "summary": {
            "main_content_char_count": len(main_content),
            "main_content_token_count": main_content_tokens,
            "element_count": len(elements),
            "element_type_counts": dict(type_counts),
            "table_count": sum(1 for item in elements if str(item.get("type", "")).lower() == "table"),
            "asset_count": len(asset_metadata),
            "entity_count": len(entities),
            "entity_type_counts": dict(entity_counts),
            "relationship_candidate_count": len(relationships),
            "chunk_count": len(chunks),
            "text_chunk_count": sum(1 for item in chunks if item.get("chunk_type") == "text"),
            "table_chunk_count": sum(1 for item in chunks if item.get("chunk_type") == "table"),
            "chunk_token_count": sum(int(item.get("token_count", 0)) for item in chunks),
            "pii_finding_count": len(pii_findings),
        },
    }
    artifact_bytes = json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    upload_bytes(
        artifact_bytes,
        artifact_key,
        "application/json",
        {
            "document-id": document["id"],
            "ingestion-run-id": run_id,
            "sha256": artifact_sha256,
            "privacy-state": document_status.lower(),
            "schema-version": "1.1",
        },
    )
    return {
        "artifact_id": artifact["artifact_id"],
        "s3_key": artifact_key,
        "bucket": S3_BUCKET_NAME,
        "sha256": artifact_sha256,
        "size_bytes": len(artifact_bytes),
        "status": document_status,
        "pii_findings": pii_findings,
        "pii_entity_counts": dict(pii_counts),
        "element_type_counts": dict(type_counts),
        "element_count": len(elements),
        "table_count": artifact["summary"]["table_count"],
        "asset_count": len(asset_metadata),
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "chunk_count": len(chunks),
        "main_content_char_count": len(main_content),
        "main_content_token_count": main_content_tokens,
        "embedding_eligible": bool(policy.get("embedding_eligible")),
        "artifact_json": artifact,
        "asset_keys": asset_keys,
    }


def process_document_run(run_id: str, document_id: str, user_id: str) -> None:
    """Process one immutable uploaded document version and automatically chain into Stage 3 embedding."""
    with _get_pool().connection() as connection:
        document = connection.execute(
            """
            SELECT d.id, d.source_id, d.workspace_id, d.user_id, d.original_name, d.s3_key,
                   d.storage_uri, d.file_hash, d.file_size_bytes, d.content_type,
                   d.version_number, d.status AS document_status, d.deleted_at,
                   w.session_id
            FROM documents d
            JOIN workspaces w ON w.id=d.workspace_id
            WHERE d.id=%s AND d.user_id=%s AND d.deleted_at IS NULL
              AND d.status IN ('UPLOADED','SUPERSEDED')
            """,
            (document_id, user_id),
        ).fetchone()
        if not document:
            _update_run(
                run_id,
                "FAILED",
                {"stage": "document_lookup"},
                "DOCUMENT_NOT_AVAILABLE",
                "Document is not available for processing.",
                True,
            )
            return
        latest = connection.execute(
            "SELECT status FROM ingestion_runs WHERE id=%s AND document_id=%s",
            (run_id, document_id),
        ).fetchone()
        if not latest or latest["status"] not in {"QUEUED", "FAILED"}:
            return
        document = dict(document)

    document["extension"] = PurePosixPath(document["original_name"]).suffix.lower()

    def status_callback(status: str, details: dict[str, Any]) -> None:
        _update_run(run_id, status, details)

    try:
        with tempfile.TemporaryDirectory(prefix="sourcebook-ingestion-") as temp_dir:
            local_path = PurePosixPath(temp_dir) / ("original" + document["extension"])
            _download_object_to_path(document["s3_key"], str(local_path))
            local_digest = hashlib.sha256()
            with open(local_path, "rb") as file_handle:
                for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                    local_digest.update(block)
            if local_digest.hexdigest().lower() != str(document["file_hash"]).lower():
                raise ValueError("SOURCE_CHECKSUM_MISMATCH")

            graph = build_ingestion_graph()
            final_state = graph.invoke(
                {
                    "document": document,
                    "local_path": str(local_path),
                    "extension": document["extension"],
                    "run_id": run_id,
                    "user_id": user_id,
                    "workspace_id": document["workspace_id"],
                    "status_callback": status_callback,
                    "artifact_writer": _artifact_writer,
                }
            )

        artifact_result = final_state["artifact"]
        artifact_result.pop("artifact_json", None)
        pii_findings = artifact_result.pop("pii_findings")
        asset_keys = artifact_result.pop("asset_keys")
        final_status = artifact_result["status"]
        should_auto_embed = (
            AUTO_EMBED_ON_INGESTION
            and final_status == "SANITIZED"
            and bool(artifact_result.get("embedding_eligible"))
            and int(artifact_result.get("chunk_count") or 0) > 0
        )

        try:
            with _get_pool().connection() as connection:
                with connection.transaction():
                    connection.execute(
                        """
                        INSERT INTO extracted_artifacts (
                            id, ingestion_run_id, document_id, user_id, workspace_id, document_version_number,
                            artifact_type, status, s3_bucket, s3_key, sha256, size_bytes, schema_version,
                            element_counts, pii_entity_counts, metadata, created_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,'STRUCTURED_EXTRACTION',%s,%s,%s,%s,%s,'1.1',%s,%s,%s,NOW())
                        """,
                        (
                            artifact_result["artifact_id"],
                            run_id,
                            document_id,
                            user_id,
                            document["workspace_id"],
                            document["version_number"],
                            final_status,
                            artifact_result["bucket"],
                            artifact_result["s3_key"],
                            artifact_result["sha256"],
                            artifact_result["size_bytes"],
                            Jsonb(
                                {
                                    "main_content_char_count": artifact_result["main_content_char_count"],
                                    "main_content_token_count": artifact_result["main_content_token_count"],
                                    "element_count": artifact_result["element_count"],
                                    "element_type_counts": artifact_result["element_type_counts"],
                                    "table_count": artifact_result["table_count"],
                                    "asset_count": artifact_result["asset_count"],
                                    "entity_count": artifact_result["entity_count"],
                                    "relationship_count": artifact_result["relationship_count"],
                                    "chunk_count": artifact_result["chunk_count"],
                                }
                            ),
                            Jsonb(artifact_result["pii_entity_counts"]),
                            Jsonb(
                                {
                                    "embedding_eligible": artifact_result["embedding_eligible"],
                                    "visual_asset_keys": asset_keys,
                                    "visual_assets_quarantined": bool(asset_keys),
                                    "raw_pii_values_in_derived_artifact": False,
                                    "raw_pii_values_in_findings_table": False,
                                    "original_source_retained_in_private_s3": True,
                                    "has_main_content": True,
                                }
                            ),
                        ),
                    )
                    connection.execute("DELETE FROM document_chunks WHERE document_id=%s", (document_id,))
                    connection.execute(
                        """
                        UPDATE documents
                        SET embedding_status=%s, embedding_target_artifact_id=%s,
                            embedded_artifact_id=NULL, embedding_model=NULL, embedding_dimensions=NULL,
                            embedded_chunk_count=0, embedding_error_code=NULL, embedding_error_message=NULL,
                            embedding_requested_at=CASE WHEN %s THEN NOW() ELSE NULL END,
                            embedding_started_at=NULL,
                            embedding_completed_at=NULL, embedding_updated_at=NOW()
                        WHERE id=%s AND user_id=%s
                        """,
                        (
                            "QUEUED" if should_auto_embed else "NOT_INDEXED",
                            artifact_result["artifact_id"] if should_auto_embed else None,
                            should_auto_embed,
                            document_id,
                            user_id,
                        ),
                    )
                    for finding in pii_findings:
                        connection.execute(
                            """
                            INSERT INTO privacy_findings (
                                id, ingestion_run_id, document_id, user_id, workspace_id,
                                source_element_id, field_path, entity_type, confidence,
                                start_offset, end_offset, action
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            """,
                            (
                                _new_id("pii"),
                                run_id,
                                document_id,
                                user_id,
                                document["workspace_id"],
                                finding.get("source_element_id"),
                                finding.get("field_path"),
                                finding["entity_type"],
                                finding["confidence"],
                                finding["start_offset"],
                                finding["end_offset"],
                                finding["action"],
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE ingestion_runs
                        SET status=%s, progress_metadata=%s, artifact_id=%s,
                            pii_finding_count=%s, completed_at=NOW(), finished_at=NOW(), updated_at=NOW(),
                            error_code=NULL, error_message=NULL
                        WHERE id=%s
                        """,
                        (
                            final_status,
                            Jsonb(
                                {
                                    "stage": "embedding_queued" if should_auto_embed else "completed",
                                    "auto_embedding": should_auto_embed,
                                    "main_content_char_count": artifact_result["main_content_char_count"],
                                    "main_content_token_count": artifact_result["main_content_token_count"],
                                    "element_count": artifact_result["element_count"],
                                    "table_count": artifact_result["table_count"],
                                    "asset_count": artifact_result["asset_count"],
                                    "entity_count": artifact_result["entity_count"],
                                    "relationship_count": artifact_result["relationship_count"],
                                    "chunk_count": artifact_result["chunk_count"],
                                    "pii_entity_counts": artifact_result["pii_entity_counts"],
                                    "artifact_s3_key": artifact_result["s3_key"],
                                }
                            ),
                            artifact_result["artifact_id"],
                            len(pii_findings),
                            run_id,
                        ),
                    )
        except Exception:
            for generated_key in [artifact_result.get("s3_key"), *asset_keys]:
                if generated_key:
                    try:
                        from app.s3_client import delete_object

                        delete_object(generated_key)
                    except Exception:
                        logger.exception("Could not clean up unreferenced Stage 2 artifact key=%s", generated_key)
            raise

        logger.info(
            "Stage 2 completed run_id=%s document_id=%s status=%s elements=%s chunks=%s auto_embed=%s",
            run_id,
            document_id,
            final_status,
            artifact_result["element_count"],
            artifact_result["chunk_count"],
            should_auto_embed,
        )

        # Automatically execute Stage 3 vector indexing as part of the unified upload pipeline!
        if should_auto_embed:
            try:
                from app.ingestion.vector_index import process_document_embedding

                process_document_embedding(document_id, user_id, artifact_result["artifact_id"])
            except Exception:
                logger.exception(
                    "Auto Stage 3 embedding failed after Stage 2 completion document_id=%s artifact_id=%s",
                    document_id,
                    artifact_result["artifact_id"],
                )
    except Exception as exc:
        message = str(exc)
        code = "INGESTION_FAILED"
        if "PII guardrail" in message or "presidio" in message.lower() or "spaCy model" in message:
            code = "PII_GUARDRAIL_UNAVAILABLE"
            safe_message = "The PII guardrail could not be initialized. The document was not approved for indexing."
        elif "Docling" in message or "extract" in message.lower() or "worksheet" in message.lower() or "DOCX" in message:
            code = "DOCUMENT_EXTRACTION_FAILED"
            safe_message = "Document extraction failed. Check the parser dependency and file format."
        elif "SOURCE_CHECKSUM_MISMATCH" in message:
            code = "SOURCE_CHECKSUM_MISMATCH"
            safe_message = "The stored S3 object did not match the document SHA-256 metadata."
        elif "LangGraph" in message:
            code = "WORKFLOW_DEPENDENCY_MISSING"
            safe_message = "The Stage 2 workflow dependency is unavailable. Install requirements-stage2.txt."
        else:
            safe_message = "Stage 2 processing failed. Review the server logs using the ingestion run ID."
        logger.exception("Stage 2 ingestion failed run_id=%s document_id=%s code=%s", run_id, document_id, code)
        try:
            _update_run(run_id, "FAILED", {"stage": "failed"}, code, safe_message, True)
        except Exception:
            logger.exception("Could not persist ingestion failure status run_id=%s", run_id)
