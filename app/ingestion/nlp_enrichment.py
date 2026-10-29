"""Lightweight, provenance-aware spaCy entity and dependency-relation extraction."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_nlp():
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("NLP enrichment requires spaCy. Install requirements-stage2.txt.") from exc
    model_name = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
    try:
        return spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is unavailable. Install it using: python -m spacy download en_core_web_sm"
        ) from exc


def enrich_elements(elements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract entities and conservative subject-verb-object candidates from sanitized text."""
    nlp = _load_nlp()
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    seen_entities: set[tuple[str, str, str]] = set()

    for element in elements:
        text = str(element.get("text") or "")
        if not text.strip() and str(element.get("type", "")).lower() == "table":
            columns = element.get("columns") or (element.get("table_data") or {}).get("columns") or []
            rows = element.get("rows") or (element.get("table_data") or {}).get("rows") or []
            table_name = element.get("table_name") or element.get("sheet_name") or "Table"
            header = " | ".join(str(value) for value in columns)
            row_text = []
            for row_index, row in enumerate(rows, start=1):
                cells = []
                for column_index, value in enumerate(row):
                    if value is None or not str(value).strip():
                        continue
                    column_name = str(columns[column_index]) if column_index < len(columns) and str(columns[column_index]).strip() else f"Column {column_index + 1}"
                    cells.append(f"{column_name}: {value}")
                if cells:
                    row_text.append(f"Row {row_index}: " + "; ".join(cells))
            text = "\n".join(part for part in [f"Table: {table_name}", f"Columns: {header}" if header else "", *row_text] if part)
        if len(text.strip()) < 2:
            continue
        doc = nlp(text[:200_000])
        element_id = str(element.get("element_id", "unknown"))
        source_location = element.get("source_location", {})
        for ent in doc.ents:
            surface = ent.text.strip()
            if not surface or (surface.startswith("[") and surface.endswith("]")):
                continue
            key = (surface.casefold(), ent.label_, element_id)
            if key in seen_entities:
                continue
            seen_entities.add(key)
            entities.append({
                "entity_id": f"ent_{len(entities) + 1:07d}",
                "text": surface,
                "label": ent.label_,
                "source_element_id": element_id,
                "source_location": source_location,
            })

        # This is deliberately a heuristic candidate extractor, not a claim of complete relation
        # extraction. Only emit a relation when a verb has an explicit subject and object.
        for token in doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass", "csubj"}]
            objects = [child for child in token.children if child.dep_ in {"dobj", "obj", "attr", "oprd"}]
            for subject in subjects[:2]:
                for obj in objects[:2]:
                    subject_text = " ".join(t.text for t in subject.subtree).strip()
                    object_text = " ".join(t.text for t in obj.subtree).strip()
                    if not subject_text or not object_text:
                        continue
                    if (subject_text.startswith("[") and subject_text.endswith("]")) or (object_text.startswith("[") and object_text.endswith("]")):
                        continue
                    relationships.append({
                        "relationship_id": f"rel_{len(relationships) + 1:07d}",
                        "subject": subject_text,
                        "predicate": token.lemma_.upper(),
                        "object": object_text,
                        "source_element_id": element_id,
                        "source_location": source_location,
                        "extraction_method": "spacy_dependency_heuristic",
                    })
    return entities, relationships
