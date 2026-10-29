"""Format-specific document extraction into a normalized, provenance-aware representation.

Uses Docling as the primary layout- and structure-aware extractor across PDF, Word (DOCX),
and Excel (XLSX) when available, combined with high-recall format-specific extractors
(openpyxl, xlrd, python-docx/OpenXML, pypdf, csv, markdown/text) so that tables, headings,
key-value regions, lists, and prose are accurately extracted and assembled into a unified
`main_content` representation.

The normalized representation deliberately does not include raw document bytes. Extracted
binary pictures are returned separately so the pipeline can place them in restricted storage.
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
}
MAX_TABULAR_ROWS = 100_000
MAX_SPREADSHEET_CELLS = 250_000

_HEADING_LABELS = {"heading", "title", "section_header", "section_heading", "subtitle"}
_LIST_LABELS = {"list_item", "bullet_item", "numbered_item"}
_TABLE_LABELS = {"table", "document_index", "table_markdown"}
_KV_LABELS = {"key_value_region", "form", "key_value"}
_PICTURE_LABELS = {"picture", "figure", "chart", "image"}


@dataclass
class ExtractedAsset:
    asset_id: str
    filename: str
    content_type: str
    data: bytes
    source_location: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass
class ExtractionResult:
    parser_name: str
    parser_version: str
    elements: list[dict[str, Any]]
    assets: list[ExtractedAsset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    main_content: str = ""


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float):
        if value != value:  # NaN check
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "<na>", "none"} and not isinstance(value, str):
        return ""
    return text


def _normalize_whitespace(text: str) -> str:
    """Clean excessive horizontal whitespace while preserving intentional line breaks."""
    if not text:
        return ""
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(cleaned_lines).strip()


def _format_table_text(
    table_name: str,
    columns: list[str],
    rows: list[list[str]],
    *,
    sheet_name: str | None = None,
    start_row: int = 1,
) -> str:
    """Render a structured table into a human- and LLM-readable Markdown + row representation."""
    clean_cols = [_text_value(col) or f"Column {idx + 1}" for idx, col in enumerate(columns)]
    if not clean_cols and rows:
        max_width = max((len(r) for r in rows), default=0)
        clean_cols = [f"Column {idx + 1}" for idx in range(max_width)]

    lines: list[str] = []
    title_label = table_name or sheet_name or "Table"
    if sheet_name and sheet_name != title_label:
        lines.append(f"### Table: {title_label} (Sheet: {sheet_name})")
    else:
        lines.append(f"### Table: {title_label}")

    if clean_cols:
        header_cells = [col.replace("|", "\\|").replace("\n", " ") for col in clean_cols]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join("---" for _ in header_cells) + " |")

    for row in rows:
        if not any(_text_value(cell) for cell in row):
            continue
        padded = [
            _text_value(row[idx]).replace("|", "\\|").replace("\n", " ") if idx < len(row) else ""
            for idx in range(len(clean_cols))
        ]
        if len(row) > len(clean_cols):
            padded.extend(_text_value(cell).replace("|", "\\|").replace("\n", " ") for cell in row[len(clean_cols):])
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines).strip()


def _element(element_id: str, element_type: str, text: str, **metadata: Any) -> dict[str, Any]:
    clean_text = _normalize_whitespace(text) if element_type != "code" else text.strip()
    record: dict[str, Any] = {
        "element_id": element_id,
        "type": element_type,
        "text": clean_text,
        "main_content": clean_text,
    }
    for key, value in metadata.items():
        if value is not None:
            record[key] = value
    if element_type == "table":
        columns = record.get("columns") or (record.get("table_data") or {}).get("columns") or []
        rows = record.get("rows") or (record.get("table_data") or {}).get("rows") or []
        table_name = str(record.get("table_name") or record.get("sheet_name") or "Table")
        if columns or rows:
            record["columns"] = [_text_value(c) for c in columns]
            record["rows"] = [[_text_value(v) for v in r] for r in rows]
            record["row_count"] = len(record["rows"])
            record["table_data"] = {
                "columns": record["columns"],
                "rows": record["rows"],
            }
            if not clean_text:
                formatted = _format_table_text(
                    table_name,
                    record["columns"],
                    record["rows"],
                    sheet_name=record.get("sheet_name"),
                    start_row=int((record.get("source_location") or {}).get("start_row") or 1),
                )
                record["text"] = formatted
                record["main_content"] = formatted
            else:
                record["main_content"] = clean_text
    return record


def build_main_content(elements: list[dict[str, Any]]) -> str:
    """Assemble the complete, structured main content from extracted or sanitized elements."""
    blocks: list[str] = []
    seen_running_headers: set[str] = set()

    for element in elements:
        el_type = str(element.get("type") or "text").lower()
        text = str(element.get("main_content") or element.get("text") or "").strip()

        if el_type in {"page_header", "page_footer"}:
            norm = text.casefold()
            if not norm or norm in seen_running_headers:
                continue
            seen_running_headers.add(norm)

        if el_type == "table":
            columns = element.get("columns") or (element.get("table_data") or {}).get("columns") or []
            rows = element.get("rows") or (element.get("table_data") or {}).get("rows") or []
            table_name = str(element.get("table_name") or element.get("sheet_name") or "Table")
            if columns or rows:
                blocks.append(
                    _format_table_text(
                        table_name,
                        [str(c) for c in columns],
                        [[_text_value(v) for v in r] for r in rows],
                        sheet_name=element.get("sheet_name"),
                    )
                )
            elif text:
                blocks.append(text)
            continue

        if not text:
            continue

        if el_type in _HEADING_LABELS:
            raw_level = element.get("heading_level")
            level = int(raw_level) if str(raw_level or "").isdigit() else (1 if el_type == "title" else 2)
            level = max(1, min(6, level))
            prefix = "#" * level
            clean_heading = re.sub(r"^#{1,6}\s+", "", text).strip()
            blocks.append(f"{prefix} {clean_heading}")
        elif el_type in _LIST_LABELS:
            if not text.startswith(("- ", "* ", "+ ")) and not re.match(r"^\d+[.)]\s+", text):
                blocks.append(f"- {text}")
            else:
                blocks.append(text)
        elif el_type in _PICTURE_LABELS:
            blocks.append(f"[Figure: {text}]")
        else:
            blocks.append(text)

    return "\n\n".join(block for block in blocks if block.strip()).strip()


def _finalize_result(result: ExtractionResult) -> ExtractionResult:
    if not result.main_content:
        result.main_content = build_main_content(result.elements)
    return result


def _parse_markdown_table_lines(text: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip() and "|" in line]
    parsed_rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        if any(cells):
            parsed_rows.append(cells)
    if not parsed_rows:
        return None
    columns = parsed_rows[0]
    rows = parsed_rows[1:]
    return columns, rows


def _extract_plain_text(path: Path, markdown: bool) -> ExtractionResult:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    elements: list[dict[str, Any]] = []
    lines = text.splitlines()
    if markdown:
        buffer: list[str] = []
        kind = "paragraph"
        counter = 0
        buffer_start_line = 1
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                if buffer:
                    counter += 1
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            kind,
                            "\n".join(buffer),
                            line_start=buffer_start_line,
                            line_end=line_number - 1,
                            source_location={"line_start": buffer_start_line, "line_end": line_number - 1},
                        )
                    )
                    buffer = []
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading:
                if buffer:
                    counter += 1
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            kind,
                            "\n".join(buffer),
                            line_start=buffer_start_line,
                            line_end=line_number - 1,
                            source_location={"line_start": buffer_start_line, "line_end": line_number - 1},
                        )
                    )
                    buffer = []
                counter += 1
                elements.append(
                    _element(
                        f"el_{counter:06d}",
                        "heading",
                        heading.group(2).strip(),
                        heading_level=len(heading.group(1)),
                        line=line_number,
                        source_location={"line": line_number, "line_start": line_number, "line_end": line_number},
                    )
                )
                continue
            if re.match(r"^([-*+]|\d+[.)])\s+", stripped):
                current_kind = "list_item"
            elif stripped.startswith("|") or (stripped.count("|") >= 2 and kind == "table_markdown"):
                current_kind = "table_markdown"
            else:
                current_kind = "paragraph"
            if buffer and (current_kind != kind or current_kind == "list_item"):
                counter += 1
                elements.append(
                    _element(
                        f"el_{counter:06d}",
                        kind,
                        "\n".join(buffer),
                        line_start=buffer_start_line,
                        line_end=line_number - 1,
                        source_location={"line_start": buffer_start_line, "line_end": line_number - 1},
                    )
                )
                buffer = []
            if not buffer:
                buffer_start_line = line_number
            kind = current_kind
            buffer.append(line.rstrip())
        if buffer:
            counter += 1
            elements.append(
                _element(
                    f"el_{counter:06d}",
                    kind,
                    "\n".join(buffer),
                    line_start=buffer_start_line,
                    line_end=len(lines),
                    source_location={"line_start": buffer_start_line, "line_end": len(lines)},
                )
            )
    else:
        cursor = 0
        for counter, paragraph in enumerate(re.split(r"\n\s*\n", text), start=1):
            if paragraph.strip():
                pos = text.find(paragraph, cursor)
                if pos == -1:
                    pos = cursor
                line_start = text.count("\n", 0, pos) + 1
                line_end = line_start + paragraph.count("\n")
                elements.append(
                    _element(
                        f"el_{counter:06d}",
                        "paragraph",
                        paragraph.strip(),
                        source_location={"line_start": line_start, "line_end": line_end},
                    )
                )
                cursor = pos + len(paragraph)

    if markdown:
        structured_elements: list[dict[str, Any]] = []
        table_counter = 0
        for element in elements:
            if element.get("type") != "table_markdown":
                structured_elements.append(element)
                continue
            parsed = _parse_markdown_table_lines(str(element.get("text") or ""))
            if not parsed:
                structured_elements.append(element)
                continue
            table_counter += 1
            columns, rows = parsed
            structured_elements.append(
                _element(
                    element["element_id"],
                    "table",
                    str(element.get("text") or ""),
                    table_id=f"md_table_{table_counter:06d}",
                    table_name=f"Markdown table {table_counter}",
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    format="markdown",
                    source_location=element.get("source_location"),
                )
            )
        elements = structured_elements

    return _finalize_result(
        ExtractionResult(
            parser_name="python-text",
            parser_version="2",
            elements=elements,
            warnings=[],
        )
    )


def _split_sheet_into_blocks(
    raw_rows: list[tuple[int, list[str]]],
) -> list[tuple[int, list[str], list[list[str]], list[tuple[int, str]]]]:
    """Split a worksheet's non-empty rows into (header_row_num, headers, data_rows, preamble_notes).

    Handles spreadsheets that start with title/metadata banners before the actual header row
    as well as worksheets containing multiple tables separated by blank rows.
    """
    if not raw_rows:
        return []

    # Group contiguous rows (allow at most 1 blank row inside a block; >=2 blank rows split blocks).
    groups: list[list[tuple[int, list[str]]]] = []
    current_group: list[tuple[int, list[str]]] = [raw_rows[0]]
    for prev, curr in zip(raw_rows, raw_rows[1:]):
        gap = curr[0] - prev[0]
        prev_non_empty = sum(1 for c in prev[1] if c)
        curr_non_empty = sum(1 for c in curr[1] if c)
        if gap > 2 and len(current_group) >= 2 and curr_non_empty >= 2:
            groups.append(current_group)
            current_group = [curr]
        else:
            current_group.append(curr)
    if current_group:
        groups.append(current_group)

    blocks: list[tuple[int, list[str], list[list[str]], list[tuple[int, str]]]] = []
    for group in groups:
        max_cols = max((sum(1 for c in r if c) for _, r in group), default=0)
        preamble: list[tuple[int, str]] = []
        header_idx = 0
        if max_cols >= 2 and len(group) >= 2:
            for idx, (r_num, r_vals) in enumerate(group[:-1]):
                non_empty = [c for c in r_vals if c]
                # A single-cell banner row or key:value metadata row above a wider table
                if len(non_empty) == 1 and max_cols >= 2:
                    preamble.append((r_num, non_empty[0]))
                    header_idx = idx + 1
                elif len(non_empty) == 2 and max_cols >= 4 and ":" in non_empty[0]:
                    preamble.append((r_num, f"{non_empty[0].rstrip(':')}: {non_empty[1]}"))
                    header_idx = idx + 1
                else:
                    break

        header_row_num, raw_headers = group[header_idx]
        body_items = group[header_idx + 1 :]
        width = max(
            [len(raw_headers), *(len(r) for _, r in body_items)] if body_items else [len(raw_headers)]
        )
        headers: list[str] = []
        seen_headers: dict[str, int] = {}
        for col_i in range(width):
            candidate = raw_headers[col_i].strip() if col_i < len(raw_headers) else ""
            if not candidate:
                candidate = f"Column {col_i + 1}"
            count = seen_headers.get(candidate, 0)
            seen_headers[candidate] = count + 1
            if count > 0:
                candidate = f"{candidate} ({count + 1})"
            headers.append(candidate)

        body_rows: list[list[str]] = []
        for _, r_vals in body_items:
            normalized = [r_vals[col_i] if col_i < len(r_vals) else "" for col_i in range(width)]
            if any(normalized):
                body_rows.append(normalized)

        blocks.append((header_row_num, headers, body_rows, preamble))

    return blocks


def _extract_delimited_table(path: Path, delimiter: str) -> ExtractionResult:
    elements: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        raw_rows: list[tuple[int, list[str]]] = []
        for index, row in enumerate(reader, start=1):
            if index > MAX_TABULAR_ROWS + 1:
                raise ValueError(f"Tabular input exceeds the {MAX_TABULAR_ROWS:,}-row extraction limit")
            values = [_text_value(cell) for cell in row]
            while values and not values[-1]:
                values.pop()
            if any(values):
                raw_rows.append((index, values))
    if not raw_rows:
        raise ValueError("The tabular document has no rows")

    blocks = _split_sheet_into_blocks(raw_rows)
    el_counter = 0
    table_counter = 0
    for header_row_num, headers, rows, preamble in blocks:
        for line_num, note in preamble:
            el_counter += 1
            elements.append(
                _element(
                    f"el_{el_counter:06d}",
                    "heading" if line_num == 1 else "paragraph",
                    note,
                    heading_level=1 if line_num == 1 else None,
                    source_location={"source_file": path.name, "row": line_num},
                )
            )
        table_counter += 1
        el_counter += 1
        table_id = f"table_{table_counter:06d}"
        table_name = path.stem[:120] if len(blocks) == 1 else f"{path.stem[:100]} (Part {table_counter})"
        elements.append(
            _element(
                table_id,
                "table",
                "",
                table_id=table_id,
                table_name=table_name,
                columns=headers,
                rows=rows,
                row_count=len(rows),
                delimiter="TSV" if delimiter == "\t" else "CSV",
                source_location={"source_file": path.name, "start_row": header_row_num},
            )
        )
    return _finalize_result(ExtractionResult("python-csv", "2", elements))


def _extract_xlsx(path: Path) -> ExtractionResult:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("XLSX extraction requires openpyxl. Install requirements-stage2.txt.") from exc

    workbook = load_workbook(path, read_only=False, data_only=True)
    elements: list[dict[str, Any]] = []
    assets: list[ExtractedAsset] = []
    cell_count = 0
    el_counter = 0
    table_counter = 0
    asset_counter = 0
    warnings: list[str] = []
    try:
        for sheet in workbook.worksheets:
            # Propagate horizontal merged-cell header labels so multi-column grouped headers retain context.
            merged_lookup: dict[tuple[int, int], str] = {}
            try:
                for merged_range in getattr(sheet, "merged_cells", None).ranges if getattr(sheet, "merged_cells", None) else []:
                    top_val = _text_value(sheet.cell(merged_range.min_row, merged_range.min_col).value)
                    if top_val and merged_range.min_row == merged_range.max_row:
                        for col_idx in range(merged_range.min_col, merged_range.max_col + 1):
                            merged_lookup[(merged_range.min_row, col_idx)] = top_val
            except Exception:
                merged_lookup = {}

            raw_rows: list[tuple[int, list[str]]] = []
            for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                values = [
                    _text_value(val) or merged_lookup.get((row_idx, col_idx), "")
                    for col_idx, val in enumerate(row, start=1)
                ]
                while values and not values[-1]:
                    values.pop()
                if any(values):
                    raw_rows.append((row_idx, values))
                cell_count += len(row)
                if cell_count > MAX_SPREADSHEET_CELLS:
                    raise ValueError(f"Workbook exceeds the {MAX_SPREADSHEET_CELLS:,}-cell extraction limit")

            if raw_rows:
                el_counter += 1
                elements.append(
                    _element(
                        f"el_{el_counter:06d}",
                        "heading",
                        f"Worksheet: {sheet.title}",
                        heading_level=1,
                        sheet_name=sheet.title,
                        source_location={"worksheet": sheet.title, "row": 1},
                    )
                )
                blocks = _split_sheet_into_blocks(raw_rows)
                for block_idx, (header_row_num, headers, body_rows, preamble) in enumerate(blocks, start=1):
                    for r_num, note in preamble:
                        el_counter += 1
                        elements.append(
                            _element(
                                f"el_{el_counter:06d}",
                                "heading" if r_num == 1 else "paragraph",
                                note,
                                heading_level=2 if r_num == 1 else None,
                                sheet_name=sheet.title,
                                source_location={"worksheet": sheet.title, "row": r_num},
                            )
                        )
                    table_counter += 1
                    table_id = f"table_{table_counter:06d}"
                    table_name = (
                        sheet.title[:120]
                        if len(blocks) == 1
                        else f"{sheet.title[:100]} - Table {block_idx}"
                    )
                    elements.append(
                        _element(
                            table_id,
                            "table",
                            "",
                            table_id=table_id,
                            table_name=table_name,
                            sheet_name=sheet.title,
                            columns=headers,
                            rows=body_rows,
                            row_count=len(body_rows),
                            source_location={"worksheet": sheet.title, "start_row": header_row_num},
                        )
                    )

            for image in getattr(sheet, "_images", []):
                try:
                    binary = image._data()
                    ext = (getattr(image, "format", None) or "png").lower()
                    if ext not in {"png", "jpeg", "jpg", "gif", "bmp", "tiff"}:
                        ext = "png"
                    anchor = getattr(image, "anchor", None)
                    marker = getattr(anchor, "_from", None)
                    location = {
                        "worksheet": sheet.title,
                        "row": int(marker.row) + 1 if marker else None,
                        "column": int(marker.col) + 1 if marker else None,
                    }
                    asset_counter += 1
                    mime = "image/jpeg" if ext in {"jpeg", "jpg"} else f"image/{ext}"
                    assets.append(
                        ExtractedAsset(
                            asset_id=f"asset_{asset_counter:06d}",
                            filename=f"picture_{asset_counter:06d}.{ext}",
                            content_type=mime,
                            data=binary,
                            source_location=location,
                        )
                    )
                except Exception:
                    warnings.append(f"A picture on worksheet '{sheet.title}' could not be extracted")
    finally:
        workbook.close()

    if not elements and not assets:
        raise ValueError("No readable content was found in the workbook")
    return _finalize_result(
        ExtractionResult("openpyxl", _package_version("openpyxl"), elements, assets, warnings)
    )


def _extract_xls(path: Path) -> ExtractionResult:
    try:
        import xlrd
    except ImportError as exc:
        raise RuntimeError("Legacy XLS extraction requires xlrd. Install requirements-stage2.txt.") from exc

    workbook = xlrd.open_workbook(str(path), on_demand=True)
    elements: list[dict[str, Any]] = []
    warnings = [
        "Legacy XLS embedded image extraction is not supported by the XLS table reader; embedded media was not added to the artifact."
    ]
    el_counter = 0
    table_counter = 0
    try:
        for sheet in workbook.sheets():
            if sheet.nrows * max(sheet.ncols, 1) > MAX_SPREADSHEET_CELLS:
                raise ValueError(f"Worksheet '{sheet.name}' exceeds the extraction cell limit")
            raw_rows: list[tuple[int, list[str]]] = []
            for row_idx in range(sheet.nrows):
                values = [_text_value(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)]
                while values and not values[-1]:
                    values.pop()
                if any(values):
                    raw_rows.append((row_idx + 1, values))
            if not raw_rows:
                continue
            el_counter += 1
            elements.append(
                _element(
                    f"el_{el_counter:06d}",
                    "heading",
                    f"Worksheet: {sheet.name}",
                    heading_level=1,
                    sheet_name=sheet.name,
                    source_location={"worksheet": sheet.name, "row": 1},
                )
            )
            blocks = _split_sheet_into_blocks(raw_rows)
            for block_idx, (header_row_num, headers, body_rows, preamble) in enumerate(blocks, start=1):
                for r_num, note in preamble:
                    el_counter += 1
                    elements.append(
                        _element(
                            f"el_{el_counter:06d}",
                            "heading" if r_num == 1 else "paragraph",
                            note,
                            heading_level=2 if r_num == 1 else None,
                            sheet_name=sheet.name,
                            source_location={"worksheet": sheet.name, "row": r_num},
                        )
                    )
                table_counter += 1
                table_id = f"table_{table_counter:06d}"
                table_name = (
                    sheet.name[:120]
                    if len(blocks) == 1
                    else f"{sheet.name[:100]} - Table {block_idx}"
                )
                elements.append(
                    _element(
                        table_id,
                        "table",
                        "",
                        table_id=table_id,
                        table_name=table_name,
                        sheet_name=sheet.name,
                        columns=headers,
                        rows=body_rows,
                        row_count=len(body_rows),
                        source_location={"worksheet": sheet.name, "start_row": header_row_num},
                    )
                )
    finally:
        workbook.release_resources()
    if not elements:
        raise ValueError("No readable worksheets were found in the XLS file")
    return _finalize_result(ExtractionResult("xlrd", _package_version("xlrd"), elements, [], warnings))


def _extract_docx_openxml(path: Path) -> ExtractionResult:
    """Extract headings, paragraphs, lists, tables, and embedded images from DOCX in body order."""
    elements: list[dict[str, Any]] = []
    assets: list[ExtractedAsset] = []
    warnings: list[str] = []
    counter = 0
    table_counter = 0
    asset_counter = 0

    # First try python-docx if installed, then fallback to direct OpenXML zip parsing.
    try:
        import docx
        from docx.oxml.ns import qn

        doc = docx.Document(str(path))
        para_map = {p._p: p for p in doc.paragraphs}
        table_map = {t._tbl: t for t in doc.tables}

        # Extract header text if non-empty
        seen_headers: set[str] = set()
        for section_idx, section in enumerate(doc.sections, start=1):
            try:
                for hdr_para in section.header.paragraphs:
                    hdr_text = hdr_para.text.strip()
                    if hdr_text and hdr_text.casefold() not in seen_headers:
                        seen_headers.add(hdr_text.casefold())
                        counter += 1
                        elements.append(
                            _element(
                                f"el_{counter:06d}",
                                "page_header",
                                hdr_text,
                                source_location={"section": section_idx, "region": "header"},
                            )
                        )
            except Exception:
                pass

        for child in doc.element.body.iterchildren():
            if child in para_map:
                para = para_map[child]
                text = para.text.strip()
                if not text:
                    continue
                style_name = str(getattr(para.style, "name", "") or "").strip().lower()
                num_pr = child.find(f".//{qn('w:numPr')}")
                counter += 1
                if style_name == "title":
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            "title",
                            text,
                            heading_level=1,
                            source_location={"paragraph_index": counter},
                        )
                    )
                elif style_name == "subtitle":
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            "heading",
                            text,
                            heading_level=2,
                            source_location={"paragraph_index": counter},
                        )
                    )
                elif style_name.startswith("heading"):
                    level_match = re.search(r"(\d+)", style_name)
                    level = int(level_match.group(1)) if level_match else 2
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            "heading",
                            text,
                            heading_level=max(1, min(6, level)),
                            source_location={"paragraph_index": counter},
                        )
                    )
                elif "list" in style_name or num_pr is not None:
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            "list_item",
                            text,
                            source_location={"paragraph_index": counter},
                        )
                    )
                else:
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            "paragraph",
                            text,
                            source_location={"paragraph_index": counter},
                        )
                    )
            elif child in table_map:
                table = table_map[child]
                raw_rows: list[list[str]] = []
                for row in table.rows:
                    cells = [_text_value(cell.text) for cell in row.cells]
                    # Deduplicate horizontally merged cells in python-docx only when identical adjacent objects
                    dedup_cells: list[str] = []
                    prev_tc = None
                    for cell in row.cells:
                        tc = getattr(cell, "_tc", None)
                        if tc is not None and tc is prev_tc:
                            dedup_cells.append("")
                        else:
                            dedup_cells.append(_text_value(cell.text))
                        prev_tc = tc
                    if any(dedup_cells):
                        raw_rows.append(dedup_cells)
                if not raw_rows:
                    continue
                table_counter += 1
                counter += 1
                headers = [h or f"Column {idx + 1}" for idx, h in enumerate(raw_rows[0])]
                body_rows = raw_rows[1:]
                table_id = f"table_{table_counter:06d}"
                elements.append(
                    _element(
                        table_id,
                        "table",
                        "",
                        table_id=table_id,
                        table_name=f"Table {table_counter}",
                        columns=headers,
                        rows=body_rows,
                        row_count=len(body_rows),
                        source_location={"table_index": table_counter, "start_row": 1},
                    )
                )

        # Extract embedded images from package parts
        try:
            for rel in doc.part.rels.values():
                target_ref = str(getattr(rel, "target_ref", "") or "")
                if "media/" in target_ref or "image" in str(getattr(rel, "reltype", "")):
                    part = getattr(rel, "target_part", None)
                    blob = getattr(part, "blob", None)
                    if blob:
                        asset_counter += 1
                        ext = Path(target_ref).suffix.lower().lstrip(".") or "png"
                        mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
                        assets.append(
                            ExtractedAsset(
                                asset_id=f"asset_{asset_counter:06d}",
                                filename=f"picture_{asset_counter:06d}.{ext}",
                                content_type=mime,
                                data=blob,
                                source_location={"part": target_ref},
                            )
                        )
        except Exception:
            warnings.append("One or more embedded DOCX images could not be extracted")

        if not elements and not assets:
            raise ValueError("No readable content was found in the DOCX document")
        return _finalize_result(
            ExtractionResult("python-docx", _package_version("python-docx"), elements, assets, warnings)
        )
    except ImportError:
        pass

    # Pure-Python OpenXML fallback if python-docx is not installed
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path, "r") as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError("Invalid DOCX file: word/document.xml is missing")
        xml_bytes = archive.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        body = root.find("w:body", ns)
        if body is not None:
            for child in list(body):
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "p":
                    texts = [node.text for node in child.findall(".//w:t", ns) if node.text]
                    full_text = "".join(texts).strip()
                    if not full_text:
                        continue
                    style_el = child.find(".//w:pStyle", ns)
                    style_val = (
                        style_el.attrib.get(f"{{{ns['w']}}}val", "").lower()
                        if style_el is not None
                        else ""
                    )
                    num_pr = child.find(".//w:numPr", ns)
                    counter += 1
                    if "title" in style_val:
                        elements.append(
                            _element(
                                f"el_{counter:06d}",
                                "title",
                                full_text,
                                heading_level=1,
                                source_location={"paragraph_index": counter},
                            )
                        )
                    elif style_val.startswith("heading"):
                        m = re.search(r"(\d+)", style_val)
                        lvl = int(m.group(1)) if m else 2
                        elements.append(
                            _element(
                                f"el_{counter:06d}",
                                "heading",
                                full_text,
                                heading_level=max(1, min(6, lvl)),
                                source_location={"paragraph_index": counter},
                            )
                        )
                    elif num_pr is not None or "list" in style_val:
                        elements.append(
                            _element(
                                f"el_{counter:06d}",
                                "list_item",
                                full_text,
                                source_location={"paragraph_index": counter},
                            )
                        )
                    else:
                        elements.append(
                            _element(
                                f"el_{counter:06d}",
                                "paragraph",
                                full_text,
                                source_location={"paragraph_index": counter},
                            )
                        )
                elif tag == "tbl":
                    raw_rows = []
                    for tr in child.findall("./w:tr", ns):
                        row_cells = []
                        for tc in tr.findall("./w:tc", ns):
                            cell_texts = [node.text for node in tc.findall(".//w:t", ns) if node.text]
                            row_cells.append("".join(cell_texts).strip())
                        if any(row_cells):
                            raw_rows.append(row_cells)
                    if raw_rows:
                        table_counter += 1
                        counter += 1
                        headers = [h or f"Column {i + 1}" for i, h in enumerate(raw_rows[0])]
                        body_rows = raw_rows[1:]
                        table_id = f"table_{table_counter:06d}"
                        elements.append(
                            _element(
                                table_id,
                                "table",
                                "",
                                table_id=table_id,
                                table_name=f"Table {table_counter}",
                                columns=headers,
                                rows=body_rows,
                                row_count=len(body_rows),
                                source_location={"table_index": table_counter, "start_row": 1},
                            )
                        )
        for name in archive.namelist():
            if name.startswith("word/media/"):
                data = archive.read(name)
                if data:
                    asset_counter += 1
                    ext = Path(name).suffix.lower().lstrip(".") or "png"
                    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
                    assets.append(
                        ExtractedAsset(
                            asset_id=f"asset_{asset_counter:06d}",
                            filename=f"picture_{asset_counter:06d}.{ext}",
                            content_type=mime,
                            data=data,
                            source_location={"part": name},
                        )
                    )

    if not elements and not assets:
        raise ValueError("No readable content was found in the DOCX document")
    return _finalize_result(ExtractionResult("openxml-docx", "1", elements, assets, warnings))


def _extract_pdf_pypdf(path: Path) -> ExtractionResult:
    """Extract text from text-based PDFs when explicitly selected or as a page-level fallback."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf extraction requires pypdf. Install requirements-stage2.txt.") from exc

    reader = PdfReader(str(path))
    elements: list[dict[str, Any]] = []
    element_count = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for paragraph in re.split(r"\n\s*\n", text):
            cleaned = paragraph.strip()
            if not cleaned:
                continue
            element_count += 1
            # Detect obvious markdown/numbered section headings in plain PDF text
            lines = cleaned.splitlines()
            if (
                len(lines) == 1
                and len(cleaned) <= 100
                and not cleaned.endswith((".", ",", ";", ":"))
                and (cleaned.isupper() or re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+\.)\s+[A-Z]", cleaned))
            ):
                elements.append(
                    _element(
                        f"el_{element_count:06d}",
                        "section_header",
                        cleaned,
                        heading_level=2,
                        source_location={"page": page_number},
                    )
                )
            else:
                elements.append(
                    _element(
                        f"el_{element_count:06d}",
                        "paragraph",
                        cleaned,
                        source_location={"page": page_number},
                    )
                )
    if not elements:
        raise ValueError("pypdf found no selectable text. Use Docling OCR for scanned or image-only PDFs.")
    return _finalize_result(
        ExtractionResult(
            parser_name="pypdf-text-only",
            parser_version=_package_version("pypdf"),
            elements=elements,
            warnings=[
                "PYPDF_TEXT_ONLY: this mode does not extract layout-aware tables or embedded pictures and does not perform OCR."
            ],
        )
    )


def _extract_docling_table(item: Any, document: Any) -> tuple[list[str], list[list[str]], str]:
    """Extract structured columns, rows, and markdown text from a Docling TableItem."""
    content = ""
    try:
        content = str(item.export_to_markdown(doc=document) or "").strip()
    except Exception:
        try:
            content = str(item.export_to_markdown() or "").strip()
        except Exception:
            content = str(getattr(item, "text", "") or "").strip()

    columns: list[str] = []
    rows: list[list[str]] = []

    # 1. Try DataFrame export (using positional row iteration rather than itertuples so
    # duplicate column headers or >255 columns never raise errors).
    try:
        try:
            frame = item.export_to_dataframe(doc=document)
        except TypeError:
            frame = item.export_to_dataframe()
        if frame is not None and not frame.empty:
            raw_cols = list(frame.columns)
            if hasattr(frame.columns, "levels"):  # MultiIndex header
                columns = [
                    " - ".join(_text_value(part) for part in tup if _text_value(part))
                    for tup in raw_cols
                ]
            else:
                columns = [_text_value(col) for col in raw_cols]
            raw_matrix = frame.to_numpy().tolist()
            rows = [[_text_value(val) for val in row] for row in raw_matrix]

            # If Docling assigned default numeric column indices (0, 1, 2, ...) and the first
            # row contains non-empty strings that look like column headers, promote row 0.
            if (
                columns
                and all(col.isdigit() for col in columns)
                and rows
                and any(rows[0])
                and not all(cell.replace(".", "", 1).replace("-", "", 1).isdigit() for cell in rows[0] if cell)
            ):
                columns = [cell or f"Column {idx + 1}" for idx, cell in enumerate(rows[0])]
                rows = rows[1:]
    except Exception:
        columns, rows = [], []

    # 2. Fallback to Docling's native TableData.grid if DataFrame export failed or returned empty
    if not columns and not rows:
        try:
            data_obj = getattr(item, "data", None)
            grid = getattr(data_obj, "grid", None)
            if grid and isinstance(grid, list):
                matrix: list[list[str]] = []
                header_rows: list[list[str]] = []
                body_rows: list[list[str]] = []
                for grid_row in grid:
                    cell_texts = [_text_value(getattr(cell, "text", cell)) for cell in grid_row]
                    if not any(cell_texts):
                        continue
                    is_header = any(bool(getattr(cell, "column_header", False)) for cell in grid_row)
                    if is_header and not body_rows:
                        header_rows.append(cell_texts)
                    else:
                        body_rows.append(cell_texts)
                    matrix.append(cell_texts)
                if header_rows:
                    width = max(len(r) for r in header_rows)
                    columns = [
                        " - ".join(
                            dict.fromkeys(r[col_i] for r in header_rows if col_i < len(r) and r[col_i])
                        )
                        or f"Column {col_i + 1}"
                        for col_i in range(width)
                    ]
                    rows = body_rows
                elif matrix:
                    columns = [c or f"Column {idx + 1}" for idx, c in enumerate(matrix[0])]
                    rows = matrix[1:]
        except Exception:
            pass

    # 3. Fallback to parsing the exported Markdown table
    if not columns and not rows and content:
        parsed = _parse_markdown_table_lines(content)
        if parsed:
            columns, rows = parsed

    # Normalize column labels and filter completely empty rows
    if columns:
        seen: dict[str, int] = {}
        clean_columns: list[str] = []
        for idx, col in enumerate(columns):
            label = col.strip() or f"Column {idx + 1}"
            cnt = seen.get(label, 0)
            seen[label] = cnt + 1
            if cnt > 0:
                label = f"{label} ({cnt + 1})"
            clean_columns.append(label)
        columns = clean_columns
    rows = [row for row in rows if any(_text_value(cell) for cell in row)]
    return columns, rows, content


def _extract_docling_key_value(item: Any, document: Any) -> str:
    """Extract text from Docling KeyValueItem or FormItem."""
    direct_text = str(getattr(item, "text", "") or "").strip()
    if direct_text:
        return direct_text
    try:
        md = str(item.export_to_markdown(doc=document) or "").strip()
        if md:
            return md
    except Exception:
        pass
    graph = getattr(item, "graph", None)
    if graph is not None:
        cells = getattr(graph, "cells", None) or []
        links = getattr(graph, "links", None) or []
        cell_by_id = {getattr(c, "cell_id", idx): _text_value(getattr(c, "text", "")) for idx, c in enumerate(cells)}
        pairs: list[str] = []
        used_ids: set[Any] = set()
        for link in links:
            src = getattr(link, "source_cell_id", None)
            tgt = getattr(link, "target_cell_id", None)
            if src in cell_by_id and tgt in cell_by_id and cell_by_id[src] and cell_by_id[tgt]:
                pairs.append(f"{cell_by_id[src]}: {cell_by_id[tgt]}")
                used_ids.add(src)
                used_ids.add(tgt)
        for cid, text_val in cell_by_id.items():
            if cid not in used_ids and text_val:
                pairs.append(text_val)
        if pairs:
            return "\n".join(pairs)
    return ""


def _extract_with_docling(path: Path, extension: str) -> ExtractionResult:
    """Extract PDF, DOCX, or XLSX using Docling with full table, heading, KV, and OCR support."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("Document extraction requires Docling. Install requirements-stage2.txt.") from exc

    ext = extension.lower()
    warnings: list[str] = []

    if ext == ".pdf":
        options = PdfPipelineOptions()
        options.do_ocr = os.getenv("DOCLING_DO_OCR", "true").strip().lower() != "false"
        options.do_table_structure = True
        options.generate_picture_images = True
        table_opts = getattr(options, "table_structure_options", None)
        if table_opts is not None and hasattr(table_opts, "do_cell_matching"):
            table_opts.do_cell_matching = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        try:
            result = converter.convert(str(path))
        except Exception as first_exc:
            # Retry with OCR disabled if an OCR engine/model runtime error occurred on a digital PDF
            warnings.append(f"Docling OCR pass encountered an error ({type(first_exc).__name__}); retried with standard layout+table pipeline.")
            fallback_options = PdfPipelineOptions()
            fallback_options.do_ocr = False
            fallback_options.do_table_structure = True
            fallback_options.generate_picture_images = False
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=fallback_options)}
            )
            result = converter.convert(str(path))
    else:
        converter = DocumentConverter()
        result = converter.convert(str(path))

    document = result.document
    result_status = str(getattr(result, "status", "SUCCESS")).upper()
    if "FAILURE" in result_status or "ERROR" in result_status:
        raise RuntimeError(f"Docling could not extract the {ext.upper()} document (status={result_status})")

    elements: list[dict[str, Any]] = []
    assets: list[ExtractedAsset] = []
    counter = 0
    table_counter = 0
    asset_counter = 0
    pages_with_text: set[int] = set()

    # Include content inside pictures/groups when supported by the installed Docling version
    try:
        iterator = document.iterate_items(with_groups=False, traverse_pictures=True)
    except TypeError:
        try:
            iterator = document.iterate_items(with_groups=False)
        except TypeError:
            iterator = document.iterate_items()

    for item, iter_level in iterator:
        label_obj = getattr(item, "label", "text")
        label = str(getattr(label_obj, "value", label_obj)).strip().lower()
        provs = getattr(item, "prov", None) or []
        prov = provs[0] if provs else None
        page_no = getattr(prov, "page_no", None) if prov else None
        bbox_obj = getattr(prov, "bbox", None) if prov else None
        try:
            bbox = bbox_obj.model_dump() if hasattr(bbox_obj, "model_dump") else None
        except Exception:
            bbox = None
        location = {k: v for k, v in {"page": page_no, "bbox": bbox}.items() if v is not None}
        self_ref = str(getattr(item, "self_ref", "") or "")

        if "picture" in label or label in _PICTURE_LABELS:
            asset_counter += 1
            caption = ""
            try:
                caption_fn = getattr(item, "caption_text", None)
                if callable(caption_fn):
                    caption = str(caption_fn(document) or "").strip()
            except Exception:
                caption = ""
            if not caption:
                caption = str(getattr(item, "text", "") or "").strip()

            counter += 1
            if caption and page_no is not None:
                pages_with_text.add(int(page_no))
            elements.append(
                _element(
                    f"el_{counter:06d}",
                    "picture",
                    caption,
                    source_ref=self_ref,
                    source_location=location,
                    asset_id=f"asset_{asset_counter:06d}",
                )
            )
            try:
                get_img = getattr(item, "get_image", None)
                pil_image = get_img(document) if callable(get_img) else None
                if pil_image is not None:
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    assets.append(
                        ExtractedAsset(
                            asset_id=f"asset_{asset_counter:06d}",
                            filename=f"picture_{asset_counter:06d}.png",
                            content_type="image/png",
                            data=buffer.getvalue(),
                            source_location=location,
                            description=caption or None,
                        )
                    )
                else:
                    warnings.append(f"Picture {asset_counter} had no extractable image payload")
            except Exception:
                warnings.append(f"Picture {asset_counter} could not be exported as an image asset")
            continue

        if "table" in label or label in _TABLE_LABELS:
            columns, rows, content = _extract_docling_table(item, document)
            if not columns and not rows and not content:
                continue
            table_counter += 1
            counter += 1
            if page_no is not None:
                pages_with_text.add(int(page_no))
            caption = ""
            try:
                caption_fn = getattr(item, "caption_text", None)
                if callable(caption_fn):
                    caption = str(caption_fn(document) or "").strip()
            except Exception:
                caption = ""
            table_name = caption[:120] if caption else f"Table {table_counter}"
            table_id = f"table_{table_counter:06d}"
            elements.append(
                _element(
                    f"el_{counter:06d}",
                    "table",
                    content,
                    table_id=table_id,
                    table_name=table_name,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    table_data={"columns": columns, "rows": rows} if (columns or rows) else None,
                    source_ref=self_ref,
                    source_location=location,
                )
            )
            continue

        if label in _KV_LABELS:
            kv_text = _extract_docling_key_value(item, document)
            if not kv_text:
                continue
            counter += 1
            if page_no is not None:
                pages_with_text.add(int(page_no))
            elements.append(
                _element(
                    f"el_{counter:06d}",
                    "key_value_region",
                    kv_text,
                    source_ref=self_ref,
                    source_location=location,
                )
            )
            continue

        text = getattr(item, "text", None)
        if text is None or not str(text).strip():
            text = getattr(item, "orig", None)
        if text is None:
            continue
        text_str = str(text).strip()
        if not text_str:
            continue

        counter += 1
        if page_no is not None:
            pages_with_text.add(int(page_no))

        # Preserve explicit heading hierarchy from Docling items
        heading_level: int | None = None
        normalized_type = label or "text"
        if label in _HEADING_LABELS:
            raw_item_level = getattr(item, "level", None)
            if isinstance(raw_item_level, int) and raw_item_level >= 1:
                heading_level = min(6, raw_item_level)
            elif label == "title":
                heading_level = 1
            elif isinstance(iter_level, int) and iter_level >= 1:
                heading_level = min(6, iter_level)
            else:
                heading_level = 2
            normalized_type = "title" if label == "title" else "section_header"

        elements.append(
            _element(
                f"el_{counter:06d}",
                normalized_type,
                text_str,
                heading_level=heading_level,
                source_ref=self_ref,
                source_location=location,
            )
        )

    # Fallback 1: If iterate_items produced no text/table elements, check document.export_to_markdown()
    has_substantive_content = any(
        el.get("type") == "table" or (el.get("type") != "picture" and str(el.get("text") or "").strip())
        for el in elements
    )
    if not has_substantive_content:
        try:
            md_export = str(document.export_to_markdown() or "").strip()
        except Exception:
            md_export = ""
        if md_export:
            with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=True) as tmp:
                tmp.write(md_export)
                tmp.flush()
                md_res = _extract_plain_text(Path(tmp.name), markdown=True)
                for md_el in md_res.elements:
                    counter += 1
                    md_el["element_id"] = f"el_{counter:06d}"
                    elements.append(md_el)
            warnings.append("Docling item iterator returned no text blocks; recovered content from Docling markdown export.")

    # Fallback 2 (PDF page-coverage verification): If any PDF page has selectable text in pypdf
    # that yielded zero Docling elements on that page, supplement from pypdf so no page is missed.
    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            supplemented_pages: list[int] = []
            for page_idx, page in enumerate(reader.pages, start=1):
                if page_idx in pages_with_text:
                    continue
                raw_page_text = (page.extract_text() or "").strip()
                if len(raw_page_text) < 10:
                    continue
                supplemented_pages.append(page_idx)
                for para in re.split(r"\n\s*\n", raw_page_text):
                    cleaned = para.strip()
                    if not cleaned:
                        continue
                    counter += 1
                    elements.append(
                        _element(
                            f"el_{counter:06d}",
                            "paragraph",
                            cleaned,
                            source_location={"page": page_idx, "fallback": "pypdf_page_supplement"},
                        )
                    )
            if supplemented_pages:
                warnings.append(
                    f"Supplemented selectable text on PDF page(s) {supplemented_pages} via pypdf where layout blocks were empty."
                )
        except Exception:
            pass

    if "PARTIAL_SUCCESS" in result_status:
        warnings.append("Docling reported partial extraction; review the artifact before production use")
    if not elements and not assets:
        raise ValueError(f"No text, tables or pictures were extracted from the {ext.upper()} document")

    return _finalize_result(
        ExtractionResult("docling", _package_version("docling"), elements, assets, warnings)
    )


def _extract_docx(path: Path) -> ExtractionResult:
    """Extract DOCX using Docling when preferred/available, enriched or backed up by python-docx/OpenXML."""
    docx_parser = os.getenv("DOCX_PARSER", "auto").strip().lower()
    if docx_parser == "docling":
        return _extract_with_docling(path, ".docx")
    if docx_parser in {"python-docx", "openxml"}:
        return _extract_docx_openxml(path)

    # In 'auto' mode, try Docling first when installed, and merge/fallback with OpenXML so
    # tables, headings, and embedded images are guaranteed to be complete.
    docling_result: ExtractionResult | None = None
    try:
        docling_result = _extract_with_docling(path, ".docx")
    except Exception:
        docling_result = None

    openxml_result: ExtractionResult | None = None
    try:
        openxml_result = _extract_docx_openxml(path)
    except Exception:
        openxml_result = None

    if docling_result and openxml_result:
        docling_tables = sum(1 for el in docling_result.elements if el.get("type") == "table")
        openxml_tables = sum(1 for el in openxml_result.elements if el.get("type") == "table")
        docling_chars = len(docling_result.main_content)
        openxml_chars = len(openxml_result.main_content)
        # If OpenXML recovered more tables or significantly more text than Docling, use OpenXML
        if openxml_tables > docling_tables or openxml_chars > docling_chars * 1.15:
            if not openxml_result.assets and docling_result.assets:
                openxml_result.assets = docling_result.assets
            return openxml_result
        if not docling_result.assets and openxml_result.assets:
            docling_result.assets = openxml_result.assets
        return docling_result
    if docling_result:
        return docling_result
    if openxml_result:
        return openxml_result
    raise RuntimeError("Could not extract DOCX document using Docling or python-docx/OpenXML.")


def _extract_doc_legacy(path: Path) -> ExtractionResult:
    """Extract legacy .doc files by converting via LibreOffice to .docx if available, or reading text."""
    # Check if the file is actually a modern DOCX saved with a .doc extension
    if zipfile.is_zipfile(path):
        return _extract_docx(path)

    with tempfile.TemporaryDirectory(prefix="sourcebook-doc-") as tmp_dir:
        try:
            proc = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    tmp_dir,
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            converted = Path(tmp_dir) / f"{path.stem}.docx"
            if proc.returncode == 0 and converted.exists() and converted.stat().st_size > 0:
                res = _extract_docx(converted)
                res.warnings.append("Converted legacy .doc to .docx via headless LibreOffice before extraction.")
                return res
        except Exception:
            pass

    raise ValueError(
        "Legacy binary .doc extraction requires headless LibreOffice or conversion to .docx/.pdf."
    )


def _extract_xlsx_hybrid(path: Path) -> ExtractionResult:
    """Extract XLSX with openpyxl (preserving multi-table blocks, merged headers, and images)
    and optionally Docling when EXCEL_PARSER=docling is configured."""
    excel_parser = os.getenv("EXCEL_PARSER", "openpyxl").strip().lower()
    if excel_parser == "docling":
        try:
            docling_res = _extract_with_docling(path, ".xlsx")
            # Supplement embedded worksheet images and any missing sheets from openpyxl
            try:
                openpyxl_res = _extract_xlsx(path)
                if not docling_res.assets and openpyxl_res.assets:
                    docling_res.assets = openpyxl_res.assets
                docling_tables = sum(1 for el in docling_res.elements if el.get("type") == "table")
                openpyxl_tables = sum(1 for el in openpyxl_res.elements if el.get("type") == "table")
                if openpyxl_tables > docling_tables:
                    return openpyxl_res
            except Exception:
                pass
            return docling_res
        except Exception as exc:
            res = _extract_xlsx(path)
            res.warnings.append(f"Docling XLSX extraction fell back to openpyxl ({type(exc).__name__}).")
            return res
    return _extract_xlsx(path)


def _package_version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


def extract_document(path: Path, extension: str) -> ExtractionResult:
    """Extract supported document types into normalized elements, main_content, and separate binary assets."""
    ext = extension.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'")
    if ext == ".pdf":
        pdf_parser = os.getenv("PDF_PARSER", "docling").strip().lower()
        if pdf_parser == "pypdf":
            return _extract_pdf_pypdf(path)
        try:
            return _extract_with_docling(path, ".pdf")
        except Exception as docling_exc:
            # If Docling is installed but failed on a selectable-text PDF, try pypdf fallback
            # when PDF_FALLBACK_PYPDF is enabled (default true for resilience).
            if os.getenv("PDF_FALLBACK_PYPDF", "true").strip().lower() != "false":
                try:
                    fallback = _extract_pdf_pypdf(path)
                    fallback.warnings.append(
                        f"Docling extraction failed ({type(docling_exc).__name__}); recovered selectable text with pypdf."
                    )
                    return fallback
                except Exception:
                    pass
            raise
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".doc":
        return _extract_doc_legacy(path)
    if ext == ".txt":
        return _extract_plain_text(path, markdown=False)
    if ext == ".md":
        return _extract_plain_text(path, markdown=True)
    if ext == ".csv":
        return _extract_delimited_table(path, ",")
    if ext == ".tsv":
        return _extract_delimited_table(path, "\t")
    if ext == ".xlsx":
        return _extract_xlsx_hybrid(path)
    if ext == ".xls":
        return _extract_xls(path)
    raise ValueError(f"No extractor registered for '{ext}'")
