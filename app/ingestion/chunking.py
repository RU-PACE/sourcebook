"""Structure-aware, concept-focused token chunking with document provenance and main_content preservation.

Combines structure-aware section tracking with recursive semantic boundary splitting (inspired by
RecursiveCharacterTextSplitter + Parent-Child context headers):
- Preserves section/subsection hierarchy as contextual prefixes on every chunk.
- Separates distinct bullet/numbered pointers and concept definitions so queries like "What is NER?"
  retrieve the specific concept rather than a 6-topic wall of text.
- Merges only genuinely tiny continuation sentences within the same section up to SOFT_MERGE_TOKENS.
- Renders tables row-wise with repeated column headers and exact row-range citations.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "300"))
SOFT_MERGE_TOKENS = int(os.getenv("SOFT_MERGE_TOKENS", "90"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))
TOKENIZER_NAME = "cl100k_base"

_TEXT_TYPES = {
    "text",
    "paragraph",
    "list_item",
    "bullet_item",
    "numbered_item",
    "title",
    "section_header",
    "section_heading",
    "heading",
    "subtitle",
    "narrative_text",
    "caption",
    "footnote",
    "formula",
    "code",
    "quote",
    "reference",
    "picture",
    "figure",
    "chart",
    "key_value_region",
    "key_value",
    "form",
    "handwritten_text",
    "checkbox_selected",
    "checkbox_unselected",
    "page_header",
    "page_footer",
    "abstract",
    "summary",
}
_HEADING_TYPES = {"heading", "title", "section_header", "section_heading", "subtitle"}
_TABLE_TYPES = {"table", "document_index", "table_markdown"}
_LIST_ITEM_TYPES = {"list_item", "bullet_item", "numbered_item"}

# Detects a bullet/numbered pointer line or a concept definition lead-in (e.g. "1. NER: ...", "• Machine Learning - ...")
_BULLET_START_RE = re.compile(r"^(?:[-*+•▪‣]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")
_DEFINITION_START_RE = re.compile(
    r"^(?:[-*+•▪‣]\s+|\d{1,3}[.)]\s+)?[A-Z][A-Za-z0-9 /()_-]{1,50}\s*(?::|—|–|-\s|\bis\s+(?:a|an|the)\b)"
)


@lru_cache(maxsize=1)
def _get_encoder():
    """Return tiktoken's embedding-model tokenizer, or a reversible whitespace fallback."""
    try:
        import tiktoken

        return tiktoken.get_encoding(TOKENIZER_NAME)
    except ImportError:
        return None


def _encode(text: str) -> list[Any]:
    encoder = _get_encoder()
    if encoder is not None:
        return encoder.encode(text, disallowed_special=())
    return re.findall(r"\s+|[^\s]+", text, flags=re.UNICODE)


def _decode(tokens: list[Any]) -> str:
    encoder = _get_encoder()
    if encoder is not None:
        return encoder.decode(tokens)
    return "".join(str(token) for token in tokens)


def count_tokens(text: str) -> int:
    """Count model-aligned tokens where tiktoken is installed."""
    return len(_encode(text))


def _table_parts(element: dict[str, Any]) -> tuple[list[str], list[list[Any]], str]:
    table_data = element.get("table_data") or {}
    columns = element.get("columns") or table_data.get("columns") or []
    rows = element.get("rows") or table_data.get("rows") or []
    title = str(
        element.get("table_name") or element.get("sheet_name") or element.get("name") or "Table"
    ).strip()
    if not columns and not rows and (element.get("text") or element.get("main_content")):
        parsed = _markdown_table(str(element.get("text") or element.get("main_content") or ""))
        if parsed:
            columns, rows = parsed
    return [str(value).strip() for value in columns], list(rows), title


def _markdown_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip() and "|" in line]
    parsed: list[list[str]] = []
    for line in lines:
        if line.startswith("#"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        if any(cells):
            parsed.append(cells)
    if not parsed:
        return None
    return parsed[0], parsed[1:]


def _context_text(headings: list[dict[str, Any]]) -> str:
    if not headings:
        return ""
    labels = [
        f"Section: {item['text']}" if index == 0 else f"Subsection: {item['text']}"
        for index, item in enumerate(headings)
    ]
    return "\n".join(labels)


def _source_location(element: dict[str, Any]) -> dict[str, Any]:
    value = element.get("source_location") or {}
    if isinstance(value, dict) and value:
        return dict(value)
    keys = (
        "page",
        "line",
        "line_start",
        "line_end",
        "worksheet",
        "sheet_name",
        "row",
        "column",
        "bbox",
        "start_row",
        "source_ref",
        "paragraph_index",
        "table_index",
    )
    return {key: element[key] for key in keys if element.get(key) is not None}


def _unique_sources(
    elements: list[dict[str, Any]], headings: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]]]:
    ids: list[str] = []
    locations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for element in [*headings, *elements]:
        source_id = element.get("element_id")
        if source_id is None:
            continue
        source_id = str(source_id)
        if source_id in seen:
            continue
        seen.add(source_id)
        ids.append(source_id)
        locations.append({"source_element_id": source_id, "location": _source_location(element)})
    return ids, locations


def _make_chunk(
    chunks: list[dict[str, Any]],
    *,
    text: str,
    main_content: str | None = None,
    chunk_type: str,
    elements: list[dict[str, Any]],
    headings: list[dict[str, Any]],
    token_count: int,
    metadata: dict[str, Any] | None = None,
    source_location_override: dict[str, Any] | None = None,
) -> None:
    source_ids, source_locations = _unique_sources(elements, headings)
    fallback_location = source_locations[0]["location"] if source_locations else {}
    if source_location_override is not None:
        fallback_location = source_location_override
        overridden_id = str(elements[0].get("element_id", "")) if elements else ""
        for source in source_locations:
            if source.get("source_element_id") == overridden_id:
                source["location"] = {**(source.get("location") or {}), **source_location_override}
    heading_context = [
        {
            "element_id": str(item.get("element_id", "")),
            "text": str(item.get("text") or ""),
            "level": int(item.get("heading_level") or index + 1),
            "source_location": _source_location(item),
        }
        for index, item in enumerate(headings)
    ]
    source_types = list(dict.fromkeys(str(item.get("type") or "text").lower() for item in elements))
    clean_text = text.strip()
    clean_main = (main_content if main_content is not None else clean_text).strip()
    chunks.append(
        {
            "chunk_id": f"chk_{len(chunks) + 1:07d}",
            "chunk_index": len(chunks),
            "chunk_type": chunk_type,
            "text": clean_text,
            "main_content": clean_main,
            "token_count": token_count,
            "source_element_id": str(elements[0].get("element_id"))
            if elements and elements[0].get("element_id") is not None
            else (source_ids[0] if source_ids else None),
            "source_element_ids": source_ids,
            "source_element_type": source_types[0]
            if len(source_types) == 1
            else ("mixed_text" if source_types else chunk_type),
            "source_location": fallback_location,
            "source_locations": source_locations,
            "heading_context": heading_context,
            "metadata": metadata or {},
            "chunking_config": {
                "strategy": "structure-aware-token-chunks-v2",
                "tokenizer": TOKENIZER_NAME,
                "max_tokens": MAX_CHUNK_TOKENS,
                "soft_merge_tokens": SOFT_MERGE_TOKENS,
                "overlap_tokens": CHUNK_OVERLAP_TOKENS if chunk_type == "text" else 0,
            },
        }
    )


def _find_clean_split_end(tokens: list[Any], start: int, max_end: int) -> int:
    """Prefer splitting at a newline or sentence boundary near max_end rather than mid-word."""
    if max_end >= len(tokens):
        return len(tokens)
    window_len = max_end - start
    if window_len < 32:
        return max_end

    lookback = max(8, window_len // 4)
    search_start = max(start + 1, max_end - lookback)
    tail_text = _decode(tokens[search_start:max_end])

    best_match = None
    for match in re.finditer(r"(?:\n+|(?<=[.!?])\s+|(?<=[;:])\s+)", tail_text):
        best_match = match
    if best_match is not None and best_match.end() > 0:
        prefix_tail = tail_text[: best_match.end()]
        prefix_tokens = _encode(prefix_tail)
        candidate_end = search_start + len(prefix_tokens)
        if start < candidate_end <= max_end:
            return candidate_end
    return max_end


def _split_multi_pointer_element(element: dict[str, Any]) -> list[dict[str, Any]]:
    """If a single extracted block contains multiple bullet/numbered pointers (e.g. a PDF list block),
    split it into individual pointer elements so each concept can be retrieved with high precision."""
    text = str(element.get("main_content") or element.get("text") or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [element]

    bullet_indices = [idx for idx, line in enumerate(lines) if _BULLET_START_RE.match(line)]
    if len(bullet_indices) < 2:
        return [element]

    parts: list[dict[str, Any]] = []
    # Keep any lead-in line before the first bullet attached to the first bullet if short, or separate
    first_bullet = bullet_indices[0]
    lead_in = "\n".join(lines[:first_bullet]).strip() if first_bullet > 0 else ""

    for pos, start_idx in enumerate(bullet_indices):
        end_idx = bullet_indices[pos + 1] if pos + 1 < len(bullet_indices) else len(lines)
        pointer_text = "\n".join(lines[start_idx:end_idx]).strip()
        if pos == 0 and lead_in and count_tokens(lead_in) <= 25:
            pointer_text = f"{lead_in}\n{pointer_text}"
        elif pos == 0 and lead_in:
            parts.append({**element, "text": lead_in, "main_content": lead_in, "type": "paragraph"})
        parts.append({**element, "text": pointer_text, "main_content": pointer_text, "type": "list_item"})

    return parts if parts else [element]


def _should_start_new_semantic_chunk(
    current_group: list[dict[str, Any]],
    current_tokens: int,
    next_element: dict[str, Any],
    next_tokens: int,
    budget: int,
) -> bool:
    """Decide whether next_element should start a new chunk instead of merging into current_group."""
    if not current_group:
        return False
    if current_tokens + next_tokens + 2 > budget:
        return True

    next_type = str(next_element.get("type") or "text").lower()
    next_text = str(next_element.get("main_content") or next_element.get("text") or "").strip()
    prev_type = str(current_group[-1].get("type") or "text").lower()
    prev_text = str(current_group[-1].get("main_content") or current_group[-1].get("text") or "").strip()

    # Keep chunks page-scoped when page numbers are known so citations point to one exact page
    prev_page = _source_location(current_group[-1]).get("page")
    next_page = _source_location(next_element).get("page")
    if prev_page is not None and next_page is not None and prev_page != next_page:
        return True

    # Separate distinct bullet/numbered pointers or concept definitions once either side has substance
    next_is_pointer = next_type in _LIST_ITEM_TYPES or bool(_BULLET_START_RE.match(next_text)) or bool(_DEFINITION_START_RE.match(next_text))
    prev_is_pointer = prev_type in _LIST_ITEM_TYPES or bool(_BULLET_START_RE.match(prev_text)) or bool(_DEFINITION_START_RE.match(prev_text))
    if next_is_pointer and prev_is_pointer and (current_tokens >= 18 or next_tokens >= 18):
        return True

    # Once the current chunk reaches SOFT_MERGE_TOKENS (~90 tokens), prefer starting a fresh chunk
    # at the next paragraph boundary rather than packing unrelated paragraphs together.
    if current_tokens >= SOFT_MERGE_TOKENS:
        return True

    return False


def _emit_element_group(
    chunks: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    headings: list[dict[str, Any]],
    prefix: str,
    budget: int,
) -> None:
    if not elements:
        return
    body_tokens: list[Any] = []
    spans: list[tuple[int, int, dict[str, Any]]] = []
    separator_tokens = _encode("\n\n")
    for element in elements:
        el_text = str(element.get("main_content") or element.get("text") or "").strip()
        if not el_text:
            continue
        if body_tokens:
            body_tokens.extend(separator_tokens)
        text_tokens = _encode(el_text)
        start = len(body_tokens)
        body_tokens.extend(text_tokens)
        spans.append((start, len(body_tokens), element))

    if not body_tokens:
        return

    start = 0
    overlap = min(CHUNK_OVERLAP_TOKENS, max(0, budget // 4))
    while start < len(body_tokens):
        raw_end = min(start + budget, len(body_tokens))
        end = _find_clean_split_end(body_tokens, start, raw_end)
        selected_tokens = body_tokens[start:end]
        selected_elements = [
            element
            for span_start, span_end, element in spans
            if span_start < end and span_end > start
        ]
        if not selected_elements:
            start = end
            continue
        piece = _decode(selected_tokens).strip()
        full_text = f"{prefix}{piece}".strip()
        _make_chunk(
            chunks,
            text=full_text,
            main_content=piece,
            chunk_type="text",
            elements=selected_elements,
            headings=headings,
            token_count=count_tokens(full_text),
            metadata={
                "section_title": headings[-1].get("text") if headings else None,
                "source_element_count": len(selected_elements),
            },
        )
        if end >= len(body_tokens):
            break
        start = max(end - overlap, start + 1)


def _text_flow_chunks(
    chunks: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    headings: list[dict[str, Any]],
) -> None:
    if not elements:
        return
    context = _context_text(headings)
    prefix = f"{context}\n\n" if context else ""
    prefix_tokens = _encode(prefix)
    if len(prefix_tokens) >= MAX_CHUNK_TOKENS - 8:
        while headings and len(_encode(f"{_context_text(headings)}\n\n")) >= MAX_CHUNK_TOKENS - 8:
            headings = headings[1:]
        context = _context_text(headings)
        prefix = f"{context}\n\n" if context else ""
        prefix_tokens = _encode(prefix)
    budget = max(1, MAX_CHUNK_TOKENS - len(prefix_tokens) - 8)

    # Expand multi-bullet blocks into fine-grained semantic units first
    expanded_elements: list[dict[str, Any]] = []
    for element in elements:
        expanded_elements.extend(_split_multi_pointer_element(element))

    current_group: list[dict[str, Any]] = []
    current_tokens = 0
    for element in expanded_elements:
        el_text = str(element.get("main_content") or element.get("text") or "").strip()
        if not el_text:
            continue
        el_tokens = count_tokens(el_text)
        if _should_start_new_semantic_chunk(current_group, current_tokens, element, el_tokens, budget):
            _emit_element_group(chunks, current_group, headings, prefix, budget)
            current_group = []
            current_tokens = 0
        current_group.append(element)
        current_tokens += el_tokens + (2 if len(current_group) > 1 else 0)

    if current_group:
        _emit_element_group(chunks, current_group, headings, prefix, budget)


def _table_chunks(
    chunks: list[dict[str, Any]],
    element: dict[str, Any],
    headings: list[dict[str, Any]],
) -> None:
    columns, rows, title = _table_parts(element)
    if not columns and not rows:
        fallback = str(element.get("main_content") or element.get("text") or "").strip()
        if fallback:
            context = _context_text(headings)
            prefix_lines = ([context] if context else []) + [f"Table: {title}"]
            prefix = "\n".join(prefix_lines) + "\n"
            prefix_tokens = _encode(prefix)
            budget = max(1, MAX_CHUNK_TOKENS - len(prefix_tokens) - 8)
            body_tokens = _encode(fallback)
            start = 0
            overlap = min(CHUNK_OVERLAP_TOKENS, max(0, budget // 4))
            while start < len(body_tokens):
                end = min(start + budget, len(body_tokens))
                piece = _decode(body_tokens[start:end]).strip()
                text = f"{prefix}{piece}".strip()
                _make_chunk(
                    chunks,
                    text=text,
                    main_content=piece,
                    chunk_type="table",
                    elements=[element],
                    headings=headings,
                    token_count=count_tokens(text),
                    metadata={
                        "table_id": element.get("table_id"),
                        "table_name": title,
                        "structured_rows": False,
                    },
                )
                if end >= len(body_tokens):
                    break
                start = max(end - overlap, start + 1)
        return

    source_base = _source_location(element)
    context = _context_text(headings)
    header_lines = ([context] if context else []) + [f"Table: {title}"]
    if columns:
        header_lines.append("Columns: " + " | ".join(columns))
    header = "\n".join(header_lines)
    header_tokens = _encode(header + "\n")
    row_budget = max(1, MAX_CHUNK_TOKENS - len(header_tokens) - 8)
    start_row = int(source_base.get("start_row") or 1)
    data_row_start = start_row + 1

    row_segments: list[tuple[int, str, list[Any]]] = []
    for index, row in enumerate(rows):
        pairs: list[str] = []
        for column_index, value in enumerate(row):
            if value is None or not str(value).strip():
                continue
            label = (
                columns[column_index]
                if column_index < len(columns) and columns[column_index]
                else f"Column {column_index + 1}"
            )
            pairs.append(f"{label}: {value}")
        if pairs:
            row_text = f"Row {data_row_start + index}: " + "; ".join(pairs)
            row_segments.append((data_row_start + index, row_text, _encode(row_text)))

    if not row_segments:
        _make_chunk(
            chunks,
            text=header,
            main_content=header,
            chunk_type="table",
            elements=[element],
            headings=headings,
            token_count=count_tokens(header),
            metadata={
                "table_id": element.get("table_id"),
                "table_name": title,
                "column_headers": columns,
                "row_start": None,
                "row_end": None,
            },
            source_location_override={**source_base, "row_start": None, "row_end": None},
        )
        return

    batch: list[tuple[int, str, list[Any]]] = []
    batch_tokens = 0

    def flush_batch() -> None:
        nonlocal batch, batch_tokens
        if not batch:
            return
        body = "\n".join(item[1] for item in batch)
        text = f"{header}\n{body}".strip()
        row_start, row_end = batch[0][0], batch[-1][0]
        _make_chunk(
            chunks,
            text=text,
            main_content=body,
            chunk_type="table",
            elements=[element],
            headings=headings,
            token_count=count_tokens(text),
            metadata={
                "table_id": element.get("table_id"),
                "table_name": title,
                "column_headers": columns,
                "row_start": row_start,
                "row_end": row_end,
            },
            source_location_override={**source_base, "row_start": row_start, "row_end": row_end},
        )
        batch = []
        batch_tokens = 0

    for row_number, row_text, tokens in row_segments:
        if len(tokens) > row_budget:
            flush_batch()
            part_start = 0
            row_overlap = min(CHUNK_OVERLAP_TOKENS, max(0, row_budget // 4))
            while part_start < len(tokens):
                part_end = min(part_start + row_budget, len(tokens))
                piece = _decode(tokens[part_start:part_end]).strip()
                text = f"{header}\n{piece}".strip()
                _make_chunk(
                    chunks,
                    text=text,
                    main_content=piece,
                    chunk_type="table",
                    elements=[element],
                    headings=headings,
                    token_count=count_tokens(text),
                    metadata={
                        "table_id": element.get("table_id"),
                        "table_name": title,
                        "column_headers": columns,
                        "row_start": row_number,
                        "row_end": row_number,
                        "row_part": part_start // max(1, row_budget - row_overlap) + 1,
                    },
                    source_location_override={
                        **source_base,
                        "row_start": row_number,
                        "row_end": row_number,
                    },
                )
                if part_end >= len(tokens):
                    break
                part_start = max(part_end - row_overlap, part_start + 1)
            continue
        if batch and batch_tokens + len(tokens) + 1 > row_budget:
            flush_batch()
        batch.append((row_number, row_text, tokens))
        batch_tokens += len(tokens) + (1 if len(batch) > 1 else 0)
    flush_batch()


def _infer_heading_level(element: dict[str, Any], element_type: str) -> int:
    raw_level = element.get("heading_level")
    if isinstance(raw_level, int) and raw_level >= 1:
        return min(6, raw_level)
    if str(raw_level or "").isdigit():
        return max(1, min(6, int(raw_level)))
    if element_type == "title":
        return 1
    return 2


def build_chunks(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build searchable chunks while preserving sections, tabular meaning, and source citations."""
    chunks: list[dict[str, Any]] = []
    headings: list[dict[str, Any]] = []
    text_flow: list[dict[str, Any]] = []
    seen_running_headers: set[str] = set()
    heading_materialized = True

    def flush_text_flow() -> None:
        nonlocal text_flow, heading_materialized
        if text_flow:
            _text_flow_chunks(chunks, text_flow, headings.copy())
            text_flow = []
            heading_materialized = True

    def flush_orphan_heading_if_needed(next_level: int | None = None) -> None:
        """Ensure a standalone heading with no body text is not lost when replaced by a sibling/parent heading."""
        nonlocal heading_materialized
        if heading_materialized or not headings:
            return
        if next_level is not None and next_level > int(headings[-1].get("heading_level") or 1):
            return
        leaf_heading = headings[-1]
        parent_headings = headings[:-1]
        _text_flow_chunks(chunks, [leaf_heading], parent_headings.copy())
        heading_materialized = True

    for element in elements:
        element_type = str(element.get("type") or "text").lower()
        text = str(element.get("main_content") or element.get("text") or "").strip()

        if element_type in {"page_header", "page_footer"}:
            norm = text.casefold()
            if not norm or norm in seen_running_headers:
                continue
            seen_running_headers.add(norm)

        if element_type in _HEADING_TYPES:
            if not text:
                continue
            flush_text_flow()
            level = _infer_heading_level(element, element_type)
            flush_orphan_heading_if_needed(next_level=level)
            headings = [
                heading for heading in headings if int(heading.get("heading_level") or 1) < level
            ]
            headings.append({**element, "heading_level": level, "text": text, "main_content": text})
            heading_materialized = False
            continue

        if element_type in _TABLE_TYPES:
            flush_text_flow()
            _table_chunks(chunks, element, headings.copy())
            heading_materialized = True
            continue

        if not text:
            flush_text_flow()
            continue

        if element_type in {"picture", "figure", "chart", "image"} and not text:
            flush_text_flow()
            continue

        text_flow.append({**element, "text": text, "main_content": text, "type": element_type})

    flush_text_flow()
    flush_orphan_heading_if_needed(next_level=None)
    return chunks
