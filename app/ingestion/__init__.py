"""Stage 2-4 document ingestion: extraction, privacy guardrails, vector indexing, and RAG QA."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_RAG_ANSWER_PATH = Path(__file__).resolve().with_name("rag_answer.py")
if "app.ingestion.rag_answer" not in sys.modules and _RAG_ANSWER_PATH.is_file():
    _spec = importlib.util.spec_from_file_location("app.ingestion.rag_answer", _RAG_ANSWER_PATH)
    if _spec is not None and _spec.loader is not None:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["app.ingestion.rag_answer"] = _mod
        _spec.loader.exec_module(_mod)
        rag_answer = _mod
