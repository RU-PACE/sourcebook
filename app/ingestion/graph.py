"""LangGraph orchestration for Stage 2 extraction, guardrails and artifact storage.

The graph is intentionally in-process for this first implementation. Use a durable queue/worker
and persistent checkpointing before deploying high-volume production workloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypedDict

from app.ingestion.chunking import build_chunks
from app.ingestion.extractors import build_main_content, extract_document
from app.ingestion.nlp_enrichment import enrich_elements
from app.ingestion.pii_guardrail import redact_structure
from app.ingestion.policy import decide_artifact_policy


class IngestionState(TypedDict, total=False):
    document: dict[str, Any]
    local_path: str
    extension: str
    run_id: str
    user_id: str
    workspace_id: str
    status_callback: Callable[[str, dict[str, Any]], None]
    artifact_writer: Callable[[dict[str, Any]], dict[str, Any]]
    parser_name: str
    parser_version: str
    elements: list[dict[str, Any]]
    assets: list[Any]
    extraction_warnings: list[str]
    sanitized_elements: list[dict[str, Any]]
    sanitized_asset_descriptors: list[dict[str, Any]]
    sanitized_filename: str
    main_content: str
    pii_findings: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    policy: dict[str, Any]
    artifact: dict[str, Any]


def _set_status(state: IngestionState, status: str, details: dict[str, Any] | None = None) -> None:
    callback = state.get("status_callback")
    if callback:
        callback(status, details or {})


def _extract_node(state: IngestionState) -> dict[str, Any]:
    _set_status(state, "EXTRACTING", {"stage": "extraction_started"})
    result = extract_document(Path(state["local_path"]), state["extension"])
    return {
        "parser_name": result.parser_name,
        "parser_version": result.parser_version,
        "elements": result.elements,
        "assets": result.assets,
        "extraction_warnings": result.warnings,
    }


def _pii_guardrail_node(state: IngestionState) -> dict[str, Any]:
    _set_status(state, "PII_SCANNING", {"stage": "pii_scan_started"})
    document = state["document"]
    descriptors = []
    for asset in state.get("assets", []):
        descriptors.append(
            {
                "element_id": asset.asset_id,
                "asset_id": asset.asset_id,
                "filename": asset.filename,
                "content_type": asset.content_type,
                "size_bytes": len(asset.data),
                "sha256": __import__("hashlib").sha256(asset.data).hexdigest(),
                "source_location": asset.source_location,
                "description": asset.description or "",
                "policy_status": "QUARANTINED_UNSCANNED_VISUAL",
            }
        )
    candidate = {
        "elements": state.get("elements", []),
        "assets": descriptors,
        "document_metadata": {
            "element_id": "document_metadata",
            "filename": document.get("original_name", "document"),
        },
    }
    sanitized, findings = redact_structure(candidate)
    sanitized_elements = sanitized["elements"]
    sanitized_main_content = build_main_content(sanitized_elements)
    policy = decide_artifact_policy(findings, state.get("extraction_warnings", []))
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["entity_type"]] = counts.get(finding["entity_type"], 0) + 1
    _set_status(
        state,
        "PII_SCANNING",
        {
            "pii_detection_count": len(findings),
            "pii_entity_counts": counts,
            "policy_decision": policy["decision"],
        },
    )
    return {
        "sanitized_elements": sanitized_elements,
        "sanitized_asset_descriptors": sanitized["assets"],
        "sanitized_filename": sanitized["document_metadata"]["filename"],
        "main_content": sanitized_main_content,
        "pii_findings": findings,
        "policy": {**policy, "pii_entity_counts": counts},
    }


def _nlp_node(state: IngestionState) -> dict[str, Any]:
    _set_status(state, "NLP_ENRICHMENT", {"stage": "entity_and_relation_extraction_started"})
    entities, relationships = enrich_elements(state.get("sanitized_elements", []))
    return {"entities": entities, "relationships": relationships}


def _chunk_node(state: IngestionState) -> dict[str, Any]:
    _set_status(state, "CHUNKING", {"stage": "structure_aware_chunking_started"})
    chunks = build_chunks(state.get("sanitized_elements", []))
    return {"chunks": chunks}


def _artifact_node(state: IngestionState) -> dict[str, Any]:
    _set_status(state, "UPLOADING_ARTIFACT", {"stage": "artifact_upload_started"})
    writer = state.get("artifact_writer")
    if not writer:
        raise RuntimeError("Stage 2 artifact writer is not configured")
    artifact = writer(state)
    return {"artifact": artifact}


def build_ingestion_graph():
    """Build a new graph for each run; no long-lived graph state is shared across users."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Stage 2 workflow requires LangGraph. Install requirements-stage2.txt.") from exc

    graph = StateGraph(IngestionState)
    graph.add_node("extract", _extract_node)
    graph.add_node("pii_guardrail", _pii_guardrail_node)
    graph.add_node("nlp_enrichment", _nlp_node)
    graph.add_node("chunk", _chunk_node)
    graph.add_node("persist_artifact", _artifact_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "pii_guardrail")
    graph.add_edge("pii_guardrail", "nlp_enrichment")
    graph.add_edge("nlp_enrichment", "chunk")
    graph.add_edge("chunk", "persist_artifact")
    graph.add_edge("persist_artifact", END)
    return graph.compile()
