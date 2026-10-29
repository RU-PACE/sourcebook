"""Offline contract tests for Stage 3 semantic-search telemetry and Stage 4 reranking + RAG QA."""

import unittest
from unittest.mock import patch

from app.ingestion import rag_answer, vector_index


class FakeCursor:
    def __init__(self, one=None, many=None):
        self.one = one
        self.many = many or []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, _params=None):
        if "SELECT id FROM workspaces" in sql:
            return FakeCursor(one={"id": "workspace-1"})
        if "SELECT dc.id AS chunk_id" in sql:
            return FakeCursor(many=[{
                "chunk_id": "chunk-db-1",
                "source_chunk_id": "chk_000001",
                "document_id": "doc-v1",
                "filename": "research.pdf",
                "version_number": 1,
                "is_current": True,
                "chunk_index": 0,
                "chunk_type": "text",
                "content": "Sourcebook stores approved text chunks.",
                "source_element_id": "element-1",
                "source_element_ids": ["element-1"],
                "source_location": {"page": 2},
                "source_locations": [{"page": 2}],
                "heading_context": ["Overview"],
                "metadata": {},
                "artifact_id": "artifact-1",
                "embedding_model": "bge-m3",
                "similarity": 0.8125,
            }])
        return FakeCursor()


class FakePool:
    def connection(self):
        return FakeConnection()


class FakeEmbeddings:
    def embed_query(self, _query):
        return [0.001] * 1024


class Stage3AndStage4Tests(unittest.TestCase):
    def test_semantic_search_reports_top_k_model_dimensions_and_timings(self):
        with patch.object(vector_index, "_get_pool", return_value=FakePool()), \
             patch.object(vector_index, "_get_embeddings_client", return_value=FakeEmbeddings()):
            response = vector_index.semantic_search(
                user_id="user-1",
                workspace_id="workspace-1",
                query="What is Sourcebook?",
                top_k=5,
            )

        self.assertEqual(response["top_k"], 5)
        self.assertEqual(response["result_count"], 1)
        self.assertEqual(response["embedding_model"], "bge-m3")
        self.assertEqual(response["embedding_dimensions"], 1024)
        self.assertEqual(response["results"][0]["citation"]["source_location"], {"page": 2})
        timings = response["timings_ms"]
        self.assertEqual(set(timings), {"query_embedding_ms", "pgvector_search_ms", "total_ms"})
        self.assertTrue(all(isinstance(value, (int, float)) and value >= 0 for value in timings.values()))

    def test_stage4_rerank_candidates_top_20_to_top_3_isolates_exact_concept(self):
        candidates = []
        for idx in range(20):
            if idx == 14:
                content = (
                    "3. Named Entity Recognition (NER): NER is a token-classification subtask of NLP "
                    "that locates and classifies named entities in unstructured text into predefined "
                    "categories such as persons, organizations, locations, dates, and monetary values."
                )
                heading = ["NLP Tasks", "Named Entity Recognition (NER)"]
                sim = 0.74
            else:
                content = (
                    f"Section {idx}: General overview of machine learning pipelines, data preprocessing, "
                    f"model training, evaluation metrics, and deployment workflows."
                )
                heading = ["General ML Overview"]
                sim = 0.78 - (idx * 0.005)

            candidates.append({
                "rank": idx + 1,
                "chunk_id": f"chk_{idx:04d}",
                "content": content,
                "focused_content": content,
                "full_content": content,
                "similarity": sim,
                "vector_similarity": sim,
                "citation": {
                    "document_id": "doc_nlp",
                    "filename": "nlp_handbook.pdf",
                    "version_number": 1,
                    "chunk_index": idx,
                    "heading_context": heading,
                    "source_location": {"page": idx + 1},
                },
            })

        reranked = rag_answer.rerank_candidates("What is NER?", candidates, top_k=3)
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0]["chunk_id"], "chk_0014")
        self.assertEqual(reranked[0]["rerank_position"], 1)
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])

    def test_stage4_answer_workspace_query_with_ollama_chat_and_fallback(self):
        fake_retrieval = {
            "workspace_id": "ws-1",
            "query": "What is NER?",
            "top_k": 20,
            "result_count": 2,
            "embedding_model": "bge-m3",
            "embedding_dimensions": 1024,
            "timings_ms": {
                "query_embedding_ms": 12.4,
                "pgvector_search_ms": 4.2,
                "total_ms": 16.6,
            },
            "results": [
                {
                    "rank": 1,
                    "chunk_id": "chk_ner",
                    "content": "NER (Named Entity Recognition) identifies and categorizes key entities such as people, organizations, and locations in text.",
                    "focused_content": "NER (Named Entity Recognition) identifies and categorizes key entities such as people, organizations, and locations in text.",
                    "full_content": "NER (Named Entity Recognition) identifies and categorizes key entities such as people, organizations, and locations in text.",
                    "similarity": 0.91,
                    "vector_similarity": 0.86,
                    "citation": {
                        "document_id": "doc_1",
                        "filename": "nlp_guide.pdf",
                        "version_number": 1,
                        "chunk_index": 3,
                        "heading_context": ["Named Entity Recognition"],
                        "source_location": {"page": 4},
                    },
                },
                {
                    "rank": 2,
                    "chunk_id": "chk_pos",
                    "content": "POS Tagging assigns parts of speech to each token.",
                    "focused_content": "POS Tagging assigns parts of speech to each token.",
                    "full_content": "POS Tagging assigns parts of speech to each token.",
                    "similarity": 0.65,
                    "vector_similarity": 0.65,
                    "citation": {
                        "document_id": "doc_1",
                        "filename": "nlp_guide.pdf",
                        "version_number": 1,
                        "chunk_index": 2,
                        "heading_context": ["POS Tagging"],
                        "source_location": {"page": 3},
                    },
                },
            ],
        }

        with patch.object(rag_answer, "_ensure_workspace_indexed", return_value=None), \
             patch.object(rag_answer, "semantic_search", return_value=fake_retrieval), \
             patch.object(rag_answer, "resolve_chat_model", return_value=("llama3.2:latest", ["llama3.2:latest", "bge-m3:latest"])), \
             patch.object(rag_answer, "_call_ollama_chat", return_value="NER (Named Entity Recognition) is an NLP technique that identifies and classifies named entities such as people, organizations, and locations in text [Source 1, p.4]."):
            result = rag_answer.answer_workspace_query(
                user_id="user-1",
                workspace_id="ws-1",
                query="What is NER?",
                candidate_k=20,
                top_k=3,
            )

        self.assertIn("Named Entity Recognition", result["answer"])
        self.assertEqual(result["model_used"], "llama3.2:latest")
        self.assertEqual(result["model_used"], "llama3.2:latest")
        self.assertEqual(len(result["citations"]), 2)
        self.assertEqual(result["citations"][0]["filename"], "nlp_guide.pdf")


if __name__ == "__main__":
    unittest.main()
