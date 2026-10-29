"""Sensitive-data detection and redaction for Sourcebook ingestion.

Only high-sensitivity identifiers are handled in this initial policy. Ordinary names,
organizations, dates, years, locations, timestamps and general NER results are deliberately
excluded so that technical documents and prose are not quarantined for benign entities.

Raw matched values are never returned in findings or logged. The sanitizer redacts the value
in-place and returns provenance-only findings for the audit table.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, NamedTuple


# Deliberately narrow allowlist. Do not add PERSON, ORGANIZATION, DATE_TIME or LOCATION here
# unless a future explicit privacy policy requires them.
SENSITIVE_PRESIDIO_ENTITIES = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_BANK_NUMBER",
        "US_SSN",
        "US_ITIN",
        # High-sensitivity identifiers supported by relevant Presidio recognizers.
        "IN_AADHAAR",
        "IN_PAN",
        "IN_PASSPORT",
        "IN_VOTER",
    }
)

DEFAULT_SCORE_THRESHOLD = 0.35

_CONTEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PHONE_NUMBER",
        re.compile(
            r"(?i)\b(?:mobile|phone|telephone|contact|whatsapp|cell)(?:\s+number)?"
            r"\s*(?:is|:|#|=|-)?\s*(?P<value>(?:\+?91[\s.-]?)?[6-9](?:[\s.-]?\d){9})\b"
        ),
    ),
    (
        "BANK_ACCOUNT",
        re.compile(
            r"(?i)\b(?:bank\s+account(?:\s+(?:number|no\.?))?|"
            r"account\s+(?:number|no\.?|#)|a/c\s*(?:number|no\.?))"
            r"\s*(?:is\s*|[:=#-]\s*)?(?P<value>(?:\d[\s-]?){8,18}\d)\b"
        ),
    ),
    (
        "UPI_ID",
        re.compile(
            r"(?i)\b(?:upi\s*(?:id|address)?|vpa|payment\s+address)"
            r"\s*(?:is\s*|[:=]\s*)?(?P<value>[A-Z0-9._-]{2,256}@[A-Z][A-Z0-9.-]{1,63})\b"
        ),
    ),
    (
        "BANK_ROUTING_CODE",
        re.compile(
            r"(?i)\bIFSC(?:\s+code)?\s*(?:is\s*|[:=]\s*)?"
            r"(?P<value>[A-Z]{4}0[A-Z0-9]{6})\b"
        ),
    ),
    (
        "TRANSACTION_ID",
        re.compile(
            r"(?i)\b(?:UTR|RRN|transaction\s*(?:id|reference|ref|number|no\.?)|"
            r"txn\s*(?:id|ref|no\.?))\s*(?:is\s*|[:=#-]\s*)?"
            r"(?P<value>[A-Z0-9][A-Z0-9-]{7,34})\b"
        ),
    ),
)


# These fields are structural identifiers or processing metadata rather than free text. They
# must not be rewritten, because other stages rely on them to link chunks and artifacts.
SKIP_SCAN_KEYS = {
    "element_id",
    "asset_id",
    "source_ref",
    "document_id",
    "document_version_id",
    "workspace_id",
    "user_id",
    "session_id",
    "ingestion_run_id",
    "artifact_id",
    "sha256",
    "file_hash",
    "content_type",
    "s3_key",
    "artifact_key",
    "status",
    "version_number",
    "page",
    "row",
    "column",
    "bbox",
    "score",
    "confidence",
    "count",
    "row_count",
    "start_offset",
    "end_offset",
    "start",
    "end",
    "type",
    "table_id",
    "format",
    "delimiter",
    "policy_status",
}


class _Span(NamedTuple):
    start: int
    end: int
    entity_type: str
    confidence: float
    source: str = "presidio"


_ENTITY_PRIORITY = {
    "CREDIT_CARD": 120,
    "IN_AADHAAR": 118,
    "IN_PAN": 118,
    "IN_PASSPORT": 118,
    "IN_VOTER": 118,
    "US_SSN": 118,
    "US_ITIN": 118,
    "IBAN_CODE": 116,
    "US_BANK_NUMBER": 116,
    "BANK_ACCOUNT": 116,
    "TRANSACTION_ID": 114,
    "UPI_ID": 112,
    "BANK_ROUTING_CODE": 110,
    "EMAIL_ADDRESS": 100,
    "PHONE_NUMBER": 98,
}


@lru_cache(maxsize=1)
def _engine():
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise RuntimeError(
            "PII guardrail dependencies are missing. Install requirements-stage2.txt."
        ) from exc

    model_name = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model_name}],
    }
    try:
        nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize PII detection with spaCy model '{model_name}'. "
            "Install the model using: python -m spacy download en_core_web_sm"
        ) from exc


def _presidio_spans(text: str, analyzer) -> list[_Span]:
    threshold = float(os.getenv("PII_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD)))
    try:
        supported = set(analyzer.get_supported_entities(language="en"))
    except Exception:
        supported = set(SENSITIVE_PRESIDIO_ENTITIES)
    allowed = sorted(SENSITIVE_PRESIDIO_ENTITIES.intersection(supported))
    if not allowed:
        return []

    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=allowed,
            score_threshold=threshold,
        )
    except TypeError:
        results = analyzer.analyze(text=text, language="en", entities=allowed)

    spans: list[_Span] = []
    for result in results:
        entity_type = str(getattr(result, "entity_type", ""))
        score = float(getattr(result, "score", 0.0))
        start = int(getattr(result, "start", -1))
        end = int(getattr(result, "end", -1))
        if entity_type not in SENSITIVE_PRESIDIO_ENTITIES:
            continue
        if score < threshold or start < 0 or end <= start or end > len(text):
            continue
        spans.append(_Span(start, end, entity_type, round(score, 4), "presidio"))
    return spans


def _context_spans(text: str) -> list[_Span]:
    """Find context-labelled financial/payment identifiers; don't scan arbitrary numbers."""
    spans: list[_Span] = []
    for entity_type, pattern in _CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span("value")
            raw_value = match.group("value")
            if entity_type == "PHONE_NUMBER":
                digits = re.sub(r"\D", "", raw_value)
                if digits.startswith("91") and len(digits) == 12:
                    digits = digits[2:]
                if len(digits) != 10:
                    continue
            elif entity_type == "BANK_ACCOUNT":
                digits = re.sub(r"\D", "", raw_value)
                if not 9 <= len(digits) <= 18:
                    continue
            elif entity_type == "TRANSACTION_ID":
                if len(re.sub(r"[^A-Z0-9]", "", raw_value.upper())) < 8:
                    continue
            spans.append(_Span(start, end, entity_type, 0.99, "context_pattern"))
    return spans


def _merge_overlapping_spans(spans: list[_Span]) -> list[_Span]:
    """Resolve overlapping recognizer hits into one deterministic redaction per region."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    groups: list[list[_Span]] = []
    current: list[_Span] = [ordered[0]]
    group_end = ordered[0].end
    for span in ordered[1:]:
        if span.start < group_end:
            current.append(span)
            group_end = max(group_end, span.end)
        else:
            groups.append(current)
            current = [span]
            group_end = span.end
    groups.append(current)

    merged: list[_Span] = []
    for group in groups:
        winner = max(
            group,
            key=lambda item: (
                _ENTITY_PRIORITY.get(item.entity_type, 0),
                item.confidence,
                item.end - item.start,
            ),
        )
        merged.append(
            _Span(
                start=min(item.start for item in group),
                end=max(item.end for item in group),
                entity_type=winner.entity_type,
                confidence=max(item.confidence for item in group),
                source=winner.source,
            )
        )
    return merged


def _redact_text(text: str, spans: list[_Span]) -> tuple[str, list[_Span]]:
    spans = _merge_overlapping_spans(spans)
    sanitized = text
    for span in reversed(spans):
        sanitized = sanitized[: span.start] + f"[{span.entity_type}]" + sanitized[span.end :]
    return sanitized, spans


def _scan_text(text: str, analyzer, element_id: str, field_path: str):
    if not text or not text.strip():
        return text, []

    spans = _presidio_spans(text, analyzer)
    spans.extend(_context_spans(text))
    sanitized, redacted_spans = _redact_text(text, spans)
    findings = [
        {
            "entity_type": span.entity_type,
            "confidence": round(float(span.confidence), 4),
            "source_element_id": element_id,
            "field_path": field_path,
            "start_offset": int(span.start),
            "end_offset": int(span.end),
            "action": "REDACTED",
        }
        for span in redacted_spans
    ]
    return sanitized, findings


def redact_structure(elements: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Redact high-sensitivity values while preserving document structure and main_content."""
    analyzer = _engine()
    findings: list[dict[str, Any]] = []

    def walk(value: Any, element_id: str, path: str, key: str | None = None) -> Any:
        if isinstance(value, dict):
            result = {}
            current_element = str(value.get("element_id") or element_id)
            has_same_main = (
                "text" in value
                and "main_content" in value
                and isinstance(value.get("text"), str)
                and value.get("main_content") == value.get("text")
            )
            has_same_table_data = (
                "columns" in value
                and "rows" in value
                and isinstance(value.get("table_data"), dict)
            )
            for child_key, child_value in value.items():
                child_path = f"{path}.{child_key}" if path else child_key
                if child_key in SKIP_SCAN_KEYS:
                    result[child_key] = child_value
                elif child_key == "main_content" and has_same_main:
                    # Will be synchronized from redacted 'text' below to avoid duplicate PII findings
                    continue
                elif child_key == "table_data" and has_same_table_data:
                    # Will be synchronized from redacted 'columns' and 'rows' below
                    continue
                else:
                    result[child_key] = walk(child_value, current_element, child_path, child_key)
            if has_same_main:
                result["main_content"] = result.get("text", "")
            if has_same_table_data:
                result["table_data"] = {
                    "columns": result.get("columns", []),
                    "rows": result.get("rows", []),
                }
            return result
        if isinstance(value, list):
            return [walk(item, element_id, f"{path}[{index}]", key) for index, item in enumerate(value)]
        if isinstance(value, str):
            sanitized, field_findings = _scan_text(value, analyzer, element_id, path)
            findings.extend(field_findings)
            return sanitized
        return value

    sanitized = walk(elements, "unknown", "elements")
    return sanitized, findings
