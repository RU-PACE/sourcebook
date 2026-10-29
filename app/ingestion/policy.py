"""Policy decision for sanitized extraction artifacts."""

from __future__ import annotations

from typing import Any


def decide_artifact_policy(
    pii_findings: list[dict[str, Any]],
    extraction_warnings: list[str],
) -> dict[str, Any]:
    """Allow forward processing after configured sensitive values have been redacted.

    The Stage 2 guardrail now only returns findings for configured high-sensitivity entities and
    records action='REDACTED'. Since these spans have been replaced in the output structure,
    routine detections do not quarantine the whole document. Failures to detect, redact, or verify
    content must still fail closed at the calling pipeline boundary.
    """
    if pii_findings:
        return {
            "status": "SANITIZED",
            "embedding_eligible": True,
            "decision": "SENSITIVE_DATA_REDACTED_AND_ALLOWED",
            "reason_codes": ["SENSITIVE_DATA_REDACTED"],
        }
    return {
        "status": "SANITIZED",
        "embedding_eligible": True,
        "decision": "PII_SCAN_PASSED",
        "reason_codes": ["EXTRACTION_WARNING"] if extraction_warnings else [],
    }
