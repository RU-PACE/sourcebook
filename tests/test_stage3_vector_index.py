"""Offline tests for Stage 3 artifact validation, version-scoped chunk identities, and Ollama resilience."""

import math
import unittest
from unittest.mock import patch

from app.ingestion import vector_index
from app.ingestion.vector_index import (
    Stage3Error,
    _document_chunk_id,
    _execute_many,
    _generate_document_vectors,
    _sanitize_embedding_text,
    _validate_vector,
    _vector_literal,
    prepare_chunk_records,
)


class Stage3VectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "id": "doc_v1",
            "source_id": "src_logical",
            "user_id": "user_1",
            "workspace_id": "workspace_1",
            "version_number": 1,
            "file_hash": "a" * 64,
        }
        self.artifact_row = {
            "id": "art_1",
            "ingestion_run_id": "run_1",
            "status": "SANITIZED",
            "metadata": {"embedding_eligible": True},
        }
        self.artifact = {
            "document": {
                "document_id": "doc_v1",
                "workspace_id": "workspace_1",
                "document_version_number": 1,
                "sha256": "a" * 64,
            },
            "privacy": {"artifact_status": "SANITIZED", "embedding_eligible": True},
            "chunks": [
                {
                    "chunk_id": "chk_0000001",
                    "chunk_index": 0,
                    "text": "Section: Revenue\n\nRevenue rose by 12%.",
                    "source_element_id": "paragraph_1",
                    "source_element_type": "paragraph",
                    "source_location": {"page": 3},
                    "chunking_config": {"strategy": "structure-aware-token-chunks-v2"},
                },
                {"chunk_id": "chk_0000002", "chunk_index": 1, "text": "  ", "chunk_type": "text"},
            ],
        }

    def test_prepare_records_preserves_citations_and_accepts_older_stage2_chunk_shape(self):
        records = prepare_chunk_records(self.document, self.artifact, self.artifact_row)
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(row["chunk_type"], "text")
        self.assertEqual(row["content"], "Section: Revenue\n\nRevenue rose by 12%.")
        self.assertEqual(row["source_element_id"], "paragraph_1")
        self.assertEqual(row["source_element_ids"], ["paragraph_1"])
        self.assertEqual(row["source_location"], {"page": 3})
        self.assertEqual(row["artifact_id"], "art_1")
        self.assertEqual(row["document_version_number"], 1)
        self.assertEqual(len(row["content_sha256"]), 64)

    def test_chunk_identity_is_stable_for_reprocessing_same_version(self):
        first = _document_chunk_id("doc_v1", "chk_0000001")
        retry = _document_chunk_id("doc_v1", "chk_0000001")
        self.assertEqual(first, retry)

    def test_chunk_identity_differs_across_document_versions(self):
        self.assertNotEqual(
            _document_chunk_id("doc_v1", "chk_0000001"),
            _document_chunk_id("doc_v2", "chk_0000001"),
        )

    def test_quarantined_artifact_is_rejected(self):
        artifact = {**self.artifact, "privacy": {"artifact_status": "QUARANTINED", "embedding_eligible": False}}
        artifact_row = {**self.artifact_row, "status": "QUARANTINED", "metadata": {"embedding_eligible": False}}
        with self.assertRaises(Stage3Error) as raised:
            prepare_chunk_records(self.document, artifact, artifact_row)
        self.assertEqual(raised.exception.code, "ARTIFACT_NOT_SANITIZED")

    def test_identity_mismatch_is_rejected(self):
        artifact = {**self.artifact, "document": {**self.artifact["document"], "document_id": "some_other_doc"}}
        with self.assertRaises(Stage3Error) as raised:
            prepare_chunk_records(self.document, artifact, self.artifact_row)
        self.assertEqual(raised.exception.code, "ARTIFACT_IDENTITY_MISMATCH")

    def test_source_id_mismatch_is_rejected_when_artifact_has_source_id(self):
        self.document["source_id"] = "src_logical"
        artifact = {**self.artifact, "document": {**self.artifact["document"], "document_source_id": "src_other"}}
        with self.assertRaises(Stage3Error) as raised:
            prepare_chunk_records(self.document, artifact, self.artifact_row)
        self.assertEqual(raised.exception.code, "ARTIFACT_IDENTITY_MISMATCH")

    def test_source_checksum_mismatch_is_rejected(self):
        artifact = {**self.artifact, "document": {**self.artifact["document"], "sha256": "b" * 64}}
        with self.assertRaises(Stage3Error) as raised:
            prepare_chunk_records(self.document, artifact, self.artifact_row)
        self.assertEqual(raised.exception.code, "ARTIFACT_SOURCE_CHECKSUM_MISMATCH")

    def test_vector_dimension_must_match_bge_m3_schema(self):
        with self.assertRaises(Stage3Error) as raised:
            _validate_vector([0.1, 0.2], context="unit test")
        self.assertEqual(raised.exception.code, "EMBEDDING_DIMENSION_MISMATCH")

    def test_vector_literal_is_pgvector_compatible(self):
        self.assertEqual(_vector_literal([0.25, -0.5]), "[0.25,-0.5]")

    def test_execute_many_uses_cursor_not_connection_method(self):
        class FakeCursor:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def executemany(self, query, parameter_rows):
                self.calls.append((query, list(parameter_rows)))

        class FakeConnection:
            def __init__(self):
                self.fake_cursor = FakeCursor()

            def cursor(self):
                return self.fake_cursor

        connection = FakeConnection()
        rows = [("chunk-1", "[0.1,0.2]"), ("chunk-2", "[0.3,0.4]")]

        _execute_many(connection, "INSERT INTO test_table VALUES (%s, %s)", rows)

        self.assertEqual(len(connection.fake_cursor.calls), 1)
        self.assertEqual(connection.fake_cursor.calls[0][1], rows)
        self.assertFalse(hasattr(connection, "executemany"))

    def test_focused_excerpt_extracts_exact_pointer_for_simple_question(self):
        from app.ingestion.vector_index import _extract_focused_excerpt
        legacy_big_chunk = (
            "Section: Core NLP Tasks\n\n"
            "- Tokenization: Splitting text into words or subwords.\n"
            "- Stemming: Heuristic suffix stripping.\n"
            "- NER (Named Entity Recognition): Identifying and classifying named entities like people, places, and organizations in text.\n"
            "- Dependency Parsing: Analyzing grammatical structure."
        )
        excerpt = _extract_focused_excerpt(legacy_big_chunk, "what is NER")
        self.assertIn("NER (Named Entity Recognition)", excerpt)
        self.assertNotIn("Tokenization", excerpt)
        self.assertNotIn("Dependency Parsing", excerpt)

    def test_sanitize_embedding_text_strips_nulls_and_dot_leaders(self):
        raw = "Section 1\x00 ......................... Page 4\n--------------\nValue"
        cleaned = _sanitize_embedding_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("..........", cleaned)
        self.assertIn("Section 1", cleaned)

    def test_generate_document_vectors_recovers_from_transient_runner_eof_and_oversized_chunk(self):
        class FlakyOllamaClient:
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                # Simulate transient runner crash on first call
                if self.calls == 1:
                    raise RuntimeError('do embedding request: Post "http://127.0.0.1:54044/embedding": EOF (status code: 500)')
                # Simulate llama.cpp n_ubatch overflow when any single input > 500 chars
                if any(len(t) > 500 for t in texts):
                    raise RuntimeError("llama runner process no longer running: 2 (status code: 500)")
                return [[1.0 / math.sqrt(1024)] * 1024 for _ in texts]

        flaky_client = FlakyOllamaClient()
        texts = [
            "Short first chunk.",
            "Dense table chunk. " * 60,  # ~1,140 chars -> triggers bisection + 350-char sub-segment fallback
            "Short third chunk.",
        ]
        with (
            patch.object(vector_index, "_get_embeddings_client", return_value=flaky_client),
            patch.object(vector_index.time, "sleep", return_value=None),
        ):
            vectors = _generate_document_vectors(texts)

        self.assertEqual(len(vectors), 3)
        self.assertTrue(all(len(v) == 1024 for v in vectors))


if __name__ == "__main__":
    unittest.main()
