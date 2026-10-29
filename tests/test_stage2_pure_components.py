"""Offline unit tests for Stage 2 pure parsing, Docling extraction, chunking, main_content and policy."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.ingestion.chunking import MAX_CHUNK_TOKENS, build_chunks, count_tokens
from app.ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    build_main_content,
    extract_document,
)
from app.ingestion.policy import decide_artifact_policy


class ExtractionTests(unittest.TestCase):
    def test_supported_extensions_include_docs_pdf_and_excel(self):
        for ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".tsv", ".md", ".txt"):
            self.assertIn(ext, SUPPORTED_EXTENSIONS)

    def test_txt_is_split_into_paragraph_elements_and_builds_main_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
            result = extract_document(path, ".txt")
        self.assertEqual([element["type"] for element in result.elements], ["paragraph", "paragraph"])
        self.assertEqual(result.elements[0]["text"], "First paragraph.")
        self.assertEqual(result.elements[1]["text"], "Second paragraph.")
        self.assertIn("First paragraph.", result.main_content)
        self.assertIn("Second paragraph.", result.main_content)

    def test_markdown_headings_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.md"
            path.write_text("# Overview\n\nA paragraph.\n", encoding="utf-8")
            result = extract_document(path, ".md")
        self.assertEqual(result.elements[0]["type"], "heading")
        self.assertEqual(result.elements[0]["heading_level"], 1)
        self.assertEqual(result.elements[0]["text"], "Overview")
        self.assertIn("# Overview", result.main_content)

    def test_markdown_table_is_normalized_to_structured_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.md"
            path.write_text("| Name | Email |\n| --- | --- |\n| Alice | alice@example.com |\n", encoding="utf-8")
            result = extract_document(path, ".md")
        table = next(item for item in result.elements if item["type"] == "table")
        self.assertEqual(table["columns"], ["Name", "Email"])
        self.assertEqual(table["rows"], [["Alice", "alice@example.com"]])
        self.assertIn("Alice", result.main_content)

    def test_csv_preserves_headers_rows_and_main_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            path.write_text("Year,Revenue\n2024,100\n2025,120\n", encoding="utf-8")
            result = extract_document(path, ".csv")
        table = next(item for item in result.elements if item["type"] == "table")
        self.assertEqual(table["type"], "table")
        self.assertEqual(table["columns"], ["Year", "Revenue"])
        self.assertEqual(table["rows"], [["2024", "100"], ["2025", "120"]])
        self.assertTrue(bool(table["text"]))
        self.assertTrue(bool(table["main_content"]))
        self.assertIn("| Year | Revenue |", result.main_content)

    def test_xlsx_extracts_preamble_and_multiple_tables_with_main_content(self):
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "financials.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Q3 Summary"
            ws.append(["Q3 Financial Report"])
            ws.append(["Region", "Revenue", "Growth"])
            ws.append(["APAC", 450, "14%"])
            ws.append(["EMEA", 380, "9%"])
            wb.save(path)

            result = extract_document(path, ".xlsx")
        tables = [el for el in result.elements if el["type"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["columns"], ["Region", "Revenue", "Growth"])
        self.assertEqual(tables[0]["rows"], [["APAC", "450", "14%"], ["EMEA", "380", "9%"]])
        self.assertIn("Q3 Financial Report", result.main_content)
        self.assertIn("| Region | Revenue | Growth |", result.main_content)
        self.assertIn("APAC", tables[0]["main_content"])

    def test_docx_extracts_headings_paragraphs_lists_and_tables(self):
        import docx

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "architecture.docx"
            doc = docx.Document()
            doc.add_heading("System Architecture", level=1)
            doc.add_paragraph("Sourcebook ingests PDF, Word, and Excel documents.")
            doc.add_paragraph("Supports OCR and table extraction", style="List Bullet")
            tbl = doc.add_table(rows=2, cols=2)
            tbl.cell(0, 0).text = "Stage"
            tbl.cell(0, 1).text = "Purpose"
            tbl.cell(1, 0).text = "Stage 2"
            tbl.cell(1, 1).text = "Extraction & Chunking"
            doc.save(path)

            result = extract_document(path, ".docx")
        types_found = [el["type"] for el in result.elements]
        self.assertIn("heading", types_found)
        self.assertIn("paragraph", types_found)
        self.assertIn("table", types_found)
        table = next(el for el in result.elements if el["type"] == "table")
        self.assertEqual(table["columns"], ["Stage", "Purpose"])
        self.assertEqual(table["rows"], [["Stage 2", "Extraction & Chunking"]])
        self.assertIn("# System Architecture", result.main_content)
        self.assertIn("Stage 2", result.main_content)

    def test_docling_extraction_handles_tables_kv_headings_and_main_content(self):
        import pandas as pd

        fake_header = MagicMock()
        fake_header.label = "section_header"
        fake_header.level = 2
        fake_header.text = "Financial Highlights"
        fake_header.prov = [MagicMock(page_no=1, bbox=None)]
        fake_header.self_ref = "#/texts/0"

        fake_kv = MagicMock()
        fake_kv.label = "key_value_region"
        fake_kv.text = ""
        fake_kv.export_to_markdown.return_value = "Company: Acme Corp\nFiscal Year: 2026"
        fake_kv.prov = [MagicMock(page_no=1, bbox=None)]
        fake_kv.self_ref = "#/key_value_items/0"

        fake_table = MagicMock()
        fake_table.label = "table"
        # Simulate Docling DataFrame where headers were returned as integer column indices 0, 1
        fake_table.export_to_dataframe.return_value = pd.DataFrame(
            [["Metric", "Value"], ["ARR", "$42M"], ["NDR", "118%"]]
        )
        fake_table.export_to_markdown.return_value = "| Metric | Value |\n| --- | --- |\n| ARR | $42M |\n| NDR | 118% |"
        fake_table.caption_text.return_value = "Key SaaS Metrics"
        fake_table.prov = [MagicMock(page_no=1, bbox=None)]
        fake_table.self_ref = "#/tables/0"

        fake_doc = MagicMock()
        fake_doc.iterate_items.return_value = [
            (fake_header, 2),
            (fake_kv, 2),
            (fake_table, 2),
        ]
        fake_conv_result = MagicMock()
        fake_conv_result.status = "SUCCESS"
        fake_conv_result.document = fake_doc

        fake_converter_cls = MagicMock()
        fake_converter_cls.return_value.convert.return_value = fake_conv_result

        base_models_mod = types.ModuleType("docling.datamodel.base_models")
        base_models_mod.InputFormat = MagicMock(PDF="pdf")
        pipeline_opts_mod = types.ModuleType("docling.datamodel.pipeline_options")
        pipeline_opts_mod.PdfPipelineOptions = MagicMock()
        doc_conv_mod = types.ModuleType("docling.document_converter")
        doc_conv_mod.DocumentConverter = fake_converter_cls
        doc_conv_mod.PdfFormatOption = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "report.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 mock")
            with patch.dict(
                sys.modules,
                {
                    "docling": types.ModuleType("docling"),
                    "docling.datamodel": types.ModuleType("docling.datamodel"),
                    "docling.datamodel.base_models": base_models_mod,
                    "docling.datamodel.pipeline_options": pipeline_opts_mod,
                    "docling.document_converter": doc_conv_mod,
                },
            ):
                result = extract_document(pdf_path, ".pdf")

        self.assertEqual(result.parser_name, "docling")
        self.assertEqual(len(result.elements), 3)
        self.assertEqual(result.elements[0]["type"], "section_header")
        self.assertEqual(result.elements[0]["heading_level"], 2)
        self.assertEqual(result.elements[1]["type"], "key_value_region")
        self.assertIn("Company: Acme Corp", result.elements[1]["text"])
        table_el = result.elements[2]
        self.assertEqual(table_el["type"], "table")
        self.assertEqual(table_el["columns"], ["Metric", "Value"])
        self.assertEqual(table_el["rows"], [["ARR", "$42M"], ["NDR", "118%"]])
        self.assertIn("## Financial Highlights", result.main_content)
        self.assertIn("Company: Acme Corp", result.main_content)
        self.assertIn("ARR", result.main_content)


class ChunkingTests(unittest.TestCase):
    def test_table_becomes_searchable_text_and_keeps_source(self):
        chunks = build_chunks(
            [
                {
                    "element_id": "table_1",
                    "type": "table",
                    "table_name": "Revenue",
                    "table_data": {"columns": ["Year", "Revenue"], "rows": [["2025", "$120M"]]},
                    "source_location": {"worksheet": "Summary", "start_row": 1},
                }
            ]
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_type"], "table")
        self.assertIn("Columns: Year | Revenue", chunks[0]["text"])
        self.assertIn("Revenue: $120M", chunks[0]["text"])
        self.assertIn("Revenue: $120M", chunks[0]["main_content"])
        self.assertEqual(chunks[0]["source_element_id"], "table_1")
        self.assertEqual(chunks[0]["source_location"]["worksheet"], "Summary")
        self.assertEqual(chunks[0]["metadata"]["row_start"], 2)
        self.assertEqual(chunks[0]["source_locations"][0]["location"]["row_start"], 2)

    def test_short_consecutive_paragraphs_merge_and_heading_is_included(self):
        chunks = build_chunks(
            [
                {
                    "element_id": "h1",
                    "type": "heading",
                    "text": "Model Training",
                    "heading_level": 1,
                    "source_location": {"page": 2},
                },
                {
                    "element_id": "p1",
                    "type": "paragraph",
                    "text": "First short statement.",
                    "source_location": {"page": 2},
                },
                {
                    "element_id": "p2",
                    "type": "paragraph",
                    "text": "Second short statement.",
                    "source_location": {"page": 2},
                },
            ]
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("Model Training", chunks[0]["text"])
        self.assertIn("First short statement", chunks[0]["text"])
        self.assertIn("Second short statement", chunks[0]["text"])
        self.assertIn("First short statement", chunks[0]["main_content"])
        self.assertEqual(set(chunks[0]["source_element_ids"]), {"h1", "p1", "p2"})
        self.assertTrue(all(item["location"].get("page") == 2 for item in chunks[0]["source_locations"]))

    def test_sibling_docling_section_headers_without_explicit_level_replace_cleanly(self):
        chunks = build_chunks(
            [
                {"element_id": "h1", "type": "section_header", "text": "Section One", "source_location": {"page": 1}},
                {"element_id": "p1", "type": "paragraph", "text": "Content of section one.", "source_location": {"page": 1}},
                {"element_id": "h2", "type": "section_header", "text": "Section Two", "source_location": {"page": 2}},
                {"element_id": "p2", "type": "paragraph", "text": "Content of section two.", "source_location": {"page": 2}},
            ]
        )
        self.assertEqual(len(chunks), 2)
        self.assertIn("Section One", chunks[0]["text"])
        self.assertNotIn("Section One", chunks[1]["text"])
        self.assertIn("Section Two", chunks[1]["text"])

    def test_standalone_headings_and_key_value_regions_are_never_dropped(self):
        chunks = build_chunks(
            [
                {"element_id": "h_only", "type": "title", "text": "Executive Cover Page", "source_location": {"page": 1}},
                {"element_id": "h_next", "type": "section_header", "text": "Details", "source_location": {"page": 2}},
                {"element_id": "kv_1", "type": "key_value_region", "text": "Account Type: Institutional", "source_location": {"page": 2}},
            ]
        )
        combined = "\n".join(c["text"] for c in chunks)
        self.assertIn("Executive Cover Page", combined)
        self.assertIn("Account Type: Institutional", combined)

    def test_multi_pointer_list_splits_into_focused_concept_chunks(self):
        multi_pointer_block = (
            "- Tokenization: Splitting raw text into words or subword tokens for downstream processing.\n"
            "- POS Tagging: Assigning grammatical part-of-speech labels such as noun or verb to each token.\n"
            "- NER (Named Entity Recognition): Identifying and classifying named entities such as people, organizations, locations, and dates in unstructured text.\n"
            "- Lemmatization: Reducing inflected words to their canonical dictionary base form."
        )
        chunks = build_chunks([
            {"element_id": "h_nlp", "type": "heading", "text": "Core NLP Tasks", "heading_level": 1},
            {"element_id": "p_list", "type": "paragraph", "text": multi_pointer_block},
        ])
        self.assertEqual(len(chunks), 4)
        ner_chunks = [c for c in chunks if "NER (Named Entity Recognition)" in c["main_content"]]
        self.assertEqual(len(ner_chunks), 1)
        self.assertNotIn("Tokenization", ner_chunks[0]["main_content"])
        self.assertNotIn("Lemmatization", ner_chunks[0]["main_content"])

    def test_long_text_is_split_with_token_budget_and_overlap(self):
        chunks = build_chunks([{"element_id": "el_1", "type": "paragraph", "text": ("Useful text. " * 300)}])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(count_tokens(chunk["text"]) <= MAX_CHUNK_TOKENS for chunk in chunks))
        self.assertTrue(
            all(chunk["chunking_config"]["strategy"] == "structure-aware-token-chunks-v2" for chunk in chunks)
        )
        self.assertIn(chunks[0]["text"][-20:], chunks[1]["text"])

    def test_table_rows_are_batched_with_headers_repeated_and_sheet_row_provenance(self):
        rows = [[str(index), "Very useful data value"] for index in range(2, 80)]
        chunks = build_chunks(
            [
                {
                    "element_id": "sheet_table",
                    "type": "table",
                    "sheet_name": "Metrics",
                    "columns": ["Row", "Description"],
                    "rows": rows,
                    "source_location": {"worksheet": "Metrics", "start_row": 1},
                }
            ]
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["chunk_type"] == "table" for chunk in chunks))
        self.assertTrue(all("Columns: Row | Description" in chunk["text"] for chunk in chunks))
        self.assertTrue(all(count_tokens(chunk["text"]) <= MAX_CHUNK_TOKENS for chunk in chunks))
        self.assertTrue(all(chunk["source_location"]["worksheet"] == "Metrics" for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["row_start"] <= chunk["metadata"]["row_end"] for chunk in chunks))


class ArtifactMainContentTests(unittest.TestCase):
    def test_artifact_writer_stores_main_content_in_final_json(self):
        from app.ingestion import pipeline

        uploaded_payloads = []

        def fake_upload_bytes(data, s3_key, content_type, metadata=None):
            uploaded_payloads.append((s3_key, data, content_type, metadata))
            return f"s3://test-bucket/{s3_key}"

        elements = [
            {"element_id": "el_1", "type": "heading", "heading_level": 1, "text": "Overview", "main_content": "Overview"},
            {"element_id": "el_2", "type": "paragraph", "text": "Key document body.", "main_content": "Key document body."},
            {
                "element_id": "table_1",
                "type": "table",
                "table_name": "Metrics",
                "columns": ["Quarter", "Revenue"],
                "rows": [["Q1", "$10M"]],
                "text": "### Table: Metrics\n| Quarter | Revenue |\n| --- | --- |\n| Q1 | $10M |",
                "main_content": "### Table: Metrics\n| Quarter | Revenue |\n| --- | --- |\n| Q1 | $10M |",
            },
        ]
        chunks = build_chunks(elements)
        state = {
            "document": {
                "id": "doc_1",
                "source_id": "src_1",
                "version_number": 1,
                "workspace_id": "wsp_1",
                "session_id": "sess_1",
                "user_id": "usr_1",
                "file_hash": "a" * 64,
                "extension": ".pdf",
                "content_type": "application/pdf",
            },
            "run_id": "ing_1",
            "policy": {
                "status": "SANITIZED",
                "decision": "PII_SCAN_PASSED",
                "embedding_eligible": True,
                "reason_codes": [],
            },
            "sanitized_filename": "report.pdf",
            "parser_name": "docling",
            "parser_version": "2.0",
            "sanitized_elements": elements,
            "main_content": build_main_content(elements),
            "chunks": chunks,
            "entities": [],
            "relationships": [],
            "pii_findings": [],
            "assets": [],
            "sanitized_asset_descriptors": [],
        }
        with patch.object(pipeline, "upload_bytes", side_effect=fake_upload_bytes):
            result = pipeline._artifact_writer(state)

        artifact_json = result["artifact_json"]
        self.assertIn("main_content", artifact_json)
        self.assertIn("# Overview", artifact_json["main_content"])
        self.assertIn("Key document body.", artifact_json["main_content"])
        self.assertIn("| Quarter | Revenue |", artifact_json["main_content"])
        self.assertGreater(artifact_json["summary"]["main_content_char_count"], 0)
        self.assertTrue(all("main_content" in chunk for chunk in artifact_json["chunks"]))


class PolicyTests(unittest.TestCase):
    def test_redacted_sensitive_finding_can_continue_to_indexing(self):
        decision = decide_artifact_policy([{"entity_type": "EMAIL_ADDRESS", "action": "REDACTED"}], [])
        self.assertEqual(decision["status"], "SANITIZED")
        self.assertTrue(decision["embedding_eligible"])

    def test_clean_document_is_eligible_for_next_stage(self):
        decision = decide_artifact_policy([], [])
        self.assertEqual(decision["status"], "SANITIZED")
        self.assertTrue(decision["embedding_eligible"])


if __name__ == "__main__":
    unittest.main()
