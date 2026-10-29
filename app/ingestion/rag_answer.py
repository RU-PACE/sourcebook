"""Stage 4 & Stage 5: Local Ollama Reranking (Top 20 -> Top K) and Stage 5 Ollama LLM Final Answer Generation.

End-to-End Query Workflow:
- Stage 3 (`semantic_search`): Retrieve up to `candidate_k` (default 20) hybrid candidate chunks from
  the authorized workspace (`pgvector` BGE-M3 1,024-D cosine similarity + initial lexical boost).
- Stage 4 (`rerank_candidates` / `stage4_rerank_workspace_query`): Rerank the 20 candidates down to
  the top `top_k` (default 3 or 5) chunks using a multi-signal local Reranker combining:
  1. Fine-grained BGE-M3 cosine similarity on each chunk's pinpoint `focused_answer` excerpt,
  2. Whole-chunk pgvector cosine similarity,
  3. Exact query phrase, acronym, and definition-structure matching (`<Term>: ...`, `<Term> is ...`).
- Stage 5 (`stage5_generate_llm_answer`): Pass the user query + the Top K reranked chunks from Stage 4
  into ANY local Ollama LLM (`llama3.2`, `qwen2.5`, `llama3.1`, `mistral`, `gemma2`, `deepseek-r1`,
  `phi3`, etc. — via `langchain_ollama.ChatOllama` or Ollama's `/api/chat` and `/api/generate` APIs)
  to synthesize the final grounded, citation-backed answer.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from time import perf_counter
from typing import Any
import urllib.error
import urllib.request

try:
    from app.config import (
        GENERATION_PROVIDER,
        OLLAMA_BASE_URL,
        OLLAMA_REQUEST_TIMEOUT_SECONDS,
        OLLAMA_CHAT_MODEL,
        OLLAMA_CHAT_TEMPERATURE,
        OLLAMA_EMBEDDING_MODEL,
        OLLAMA_NUM_CTX,
        OLLAMA_RERANK_MODEL,
        RAG_CANDIDATE_POOL_SIZE,
        RAG_RERANK_TOP_K,
    )
except ModuleNotFoundError:
    GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "ollama").strip().lower()
    OLLAMA_REQUEST_TIMEOUT_SECONDS = max(1.0, float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "90")))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3").strip()
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2").strip()
    OLLAMA_RERANK_MODEL = os.getenv("OLLAMA_RERANK_MODEL", "bge-m3").strip()
    OLLAMA_NUM_CTX = max(512, int(os.getenv("OLLAMA_NUM_CTX", "4096")))
    RAG_CANDIDATE_POOL_SIZE = max(5, min(50, int(os.getenv("RAG_CANDIDATE_POOL_SIZE", "20"))))
    RAG_RERANK_TOP_K = max(1, min(10, int(os.getenv("RAG_RERANK_TOP_K", "3"))))
    OLLAMA_CHAT_TEMPERATURE = float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.1"))

from app.ingestion.vector_index import (
    Stage3Error,
    _OLLAMA_LOCK,
    _extract_focused_excerpt,
    _extract_query_keywords,
    _get_embeddings_client,
    _sanitize_embedding_text,
    _score_text_against_query,
    semantic_search,
)

logger = logging.getLogger("sourcebook_stage4_stage5")

_EMBED_ONLY_PREFIXES = (
    "bge-m3",
    "bge-large",
    "bge-base",
    "bge-small",
    "nomic-embed",
    "mxbai-embed",
    "all-minilm",
    "snowflake-arctic-embed",
    "paraphrase-multilingual",
    "granite-embedding",
)

_PREFERRED_CHAT_MODELS = (
    "llama3.2",
    "qwen2.5",
    "llama3.1",
    "llama3",
    "mistral",
    "gemma2",
    "gemma3",
    "phi4",
    "phi3",
    "deepseek-r1",
    "qwen2",
    "codellama",
)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _list_ollama_models() -> list[str]:
    """Return locally pulled model names from Ollama's /api/tags endpoint."""
    url = f"{OLLAMA_BASE_URL}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            models = payload.get("models") or []
            return [str(m.get("name") or "").strip() for m in models if m.get("name")]
    except Exception:
        return []


def _is_embedding_only_model(name: str) -> bool:
    lower = name.lower().split(":")[0]
    return any(lower.startswith(prefix) or "embed" in lower or "rerank" in lower for prefix in _EMBED_ONLY_PREFIXES)


def resolve_chat_model(requested_model: str | None = None) -> tuple[str | None, list[str]]:
    """Resolve the active Stage 5 local Ollama LLM, supporting any model installed in Ollama."""
    clean_requested = (requested_model or "").strip()
    if clean_requested.lower() in {"", "auto", "default"}:
        target = (OLLAMA_CHAT_MODEL or "llama3.2").strip()
    else:
        target = clean_requested

    installed = _list_ollama_models()
    if not installed:
        return target, []

    chat_candidates = [m for m in installed if not _is_embedding_only_model(m)]
    if not chat_candidates:
        # If the user explicitly named a model not yet listed, allow attempting it directly
        if clean_requested and clean_requested.lower() not in {"auto", "default"} and not _is_embedding_only_model(clean_requested):
            return clean_requested, installed
        return None, installed

    # 1. Exact or tag-prefix match for the requested or configured model
    target_base = target.lower().split(":")[0]
    for model_name in chat_candidates:
        if model_name.lower() == target.lower() or model_name.lower().split(":")[0] == target_base:
            return model_name, installed

    # If the user explicitly requested a specific model name that is not in installed tags yet, try it
    if clean_requested and clean_requested.lower() not in {"auto", "default"} and not _is_embedding_only_model(clean_requested):
        return clean_requested, installed

    # 2. Otherwise pick the highest-priority installed generative LLM from Ollama
    for pref in _PREFERRED_CHAT_MODELS:
        for model_name in chat_candidates:
            if model_name.lower().startswith(pref):
                return model_name, installed

    return chat_candidates[0], installed


def rag_health() -> dict[str, Any]:
    """Inspect local Ollama readiness for Stage 3 embeddings, Stage 4 reranking, and Stage 5 LLM synthesis."""
    from app.ingestion.vector_index import embedding_health

    emb = embedding_health()
    active_chat_model, installed_models = resolve_chat_model()
    chat_models = [m for m in installed_models if not _is_embedding_only_model(m)]
    return {
        "ok": True,
        "provider": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "stage3_embedding_model": emb["model"],
        "embedding_model": emb["model"],
        "embedding_dimensions": emb["dimensions"],
        "stage4_reranker_model": OLLAMA_RERANK_MODEL or emb["model"],
        "reranker_model": OLLAMA_RERANK_MODEL or emb["model"],
        "stage5_configured_llm": OLLAMA_CHAT_MODEL,
        "configured_chat_model": OLLAMA_CHAT_MODEL,
        "stage5_active_llm": active_chat_model,
        "active_chat_model": active_chat_model,
        "chat_model_ready": bool(active_chat_model),
        "installed_models": installed_models,
        "available_chat_models": chat_models,
        "candidate_pool_size": RAG_CANDIDATE_POOL_SIZE,
        "default_rerank_top_k": RAG_RERANK_TOP_K,
    }


def _readable_location(location: dict[str, Any] | None) -> str:
    if not isinstance(location, dict) or not location:
        return "Document"
    parts: list[str] = []
    if location.get("page") is not None:
        parts.append(f"Page {location['page']}")
    if location.get("worksheet") or location.get("sheet_name"):
        parts.append(f"Sheet {location.get('worksheet') or location.get('sheet_name')}")
    if location.get("row_start") is not None and location.get("row_end") is not None:
        if location["row_start"] == location["row_end"]:
            parts.append(f"Row {location['row_start']}")
        else:
            parts.append(f"Rows {location['row_start']}–{location['row_end']}")
    elif location.get("row") is not None:
        parts.append(f"Row {location['row']}")
    elif location.get("line_start") is not None and location.get("line_end") is not None:
        if location["line_start"] == location["line_end"]:
            parts.append(f"Line {location['line_start']}")
        else:
            parts.append(f"Lines {location['line_start']}–{location['line_end']}")
    elif location.get("paragraph_index") is not None:
        parts.append(f"Paragraph {location['paragraph_index']}")
    return " · ".join(parts) if parts else "Document"


# ==============================================================================
# STAGE 4: LOCAL OLLAMA RERANKER (Top 20 Hybrid Chunks -> Top K Ranked Chunks)
# ==============================================================================
def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = RAG_RERANK_TOP_K,
) -> list[dict[str, Any]]:
    """Stage 4 Reranker: Rerank up to 20 hybrid-retrieved chunks down to the top_k most relevant chunks.

    Combines:
    1. BGE-M3 dense cosine similarity on the pinpoint `focused_answer` excerpt (so a short,
       exact definition inside a chunk is not diluted by surrounding text),
    2. Whole-chunk pgvector cosine similarity,
    3. Exact lexical/acronym/definition-pattern precision score.
    """
    if not candidates:
        return []

    keywords, core_phrase = _extract_query_keywords(query)
    clean_query = _sanitize_embedding_text(query)

    # Compute fine-grained excerpt embeddings via Ollama Reranker / BGE-M3 for the candidate pool
    excerpt_similarities: list[float] = []
    try:
        client = _get_embeddings_client()
        excerpts = [
            _sanitize_embedding_text(str(item.get("focused_answer") or item.get("content") or ""))[:900]
            for item in candidates
        ]
        with _OLLAMA_LOCK:
            q_vec = client.embed_query(clean_query[:900])
            if hasattr(client, "embed_documents"):
                e_vecs = client.embed_documents(excerpts)
            else:
                e_vecs = [client.embed_query(ex) for ex in excerpts]
        excerpt_similarities = [_cosine_similarity(q_vec, list(ev)) for ev in e_vecs]
    except Exception:
        excerpt_similarities = [float(item.get("similarity") or 0.0) for item in candidates]

    reranked: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        chunk_sim = float(item.get("similarity") or 0.0)
        excerpt_sim = excerpt_similarities[idx] if idx < len(excerpt_similarities) else chunk_sim
        content_text = str(item.get("content") or "")
        focused = str(item.get("focused_answer") or _extract_focused_excerpt(content_text, query)).strip()

        lex_score = _score_text_against_query(focused, keywords, core_phrase)
        full_lex_score = _score_text_against_query(content_text, keywords, core_phrase)

        token_len = max(10, len(focused.split()))
        conciseness_bonus = 0.04 if 8 <= token_len <= 75 else 0.0

        rerank_score = (
            0.42 * excerpt_sim
            + 0.33 * chunk_sim
            + 0.18 * lex_score
            + 0.07 * full_lex_score
            + conciseness_bonus
        )
        enriched = {
            **item,
            "focused_answer": focused,
            "matched_excerpt": focused,
            "excerpt_similarity": round(float(excerpt_sim), 4),
            "rerank_score": round(float(rerank_score), 4),
            "initial_rank": idx + 1,
        }
        reranked.append(enriched)

    reranked.sort(
        key=lambda entry: (
            entry["rerank_score"],
            entry["excerpt_similarity"],
            entry["similarity"],
            -entry["initial_rank"],
        ),
        reverse=True,
    )
    selected = reranked[: max(1, int(top_k))]
    for rank_idx, item in enumerate(selected, start=1):
        item["rerank_position"] = rank_idx
    return selected


def stage4_rerank_workspace_query(
    *,
    user_id: str,
    workspace_id: str,
    query: str,
    candidate_k: int = RAG_CANDIDATE_POOL_SIZE,
    top_k: int = RAG_RERANK_TOP_K,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Execute Stage 3 (Top-20 Hybrid Retrieval) + Stage 4 (Local Ollama Reranking to Top-K)."""
    _ensure_workspace_indexed(user_id, workspace_id)
    effective_candidate_k = max(int(top_k), min(50, int(candidate_k or RAG_CANDIDATE_POOL_SIZE)))
    effective_top_k = max(1, min(10, int(top_k or RAG_RERANK_TOP_K)))

    retrieval = semantic_search(
        user_id=user_id,
        workspace_id=workspace_id,
        query=query,
        top_k=effective_candidate_k,
        include_superseded=include_superseded,
    )
    candidates = retrieval.get("results") or []
    rerank_started = perf_counter()
    ranked_chunks = rerank_candidates(query, candidates, top_k=effective_top_k)
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    timings = dict(retrieval.get("timings_ms") or {})
    return {
        "workspace_id": workspace_id,
        "query": query.strip(),
        "candidate_k": effective_candidate_k,
        "candidates_retrieved": len(candidates),
        "top_k": effective_top_k,
        "reranked_count": len(ranked_chunks),
        "reranker_model": OLLAMA_RERANK_MODEL,
        "timings_ms": {
            "query_embedding_ms": timings.get("query_embedding_ms", 0.0),
            "pgvector_search_ms": timings.get("pgvector_search_ms", 0.0),
            "rerank_ms": rerank_ms,
        },
        "reranked_chunks": ranked_chunks,
    }


# ==============================================================================
# STAGE 5: LOCAL OLLAMA LLM FINAL ANSWER SYNTHESIS
# ==============================================================================
def _synthesize_extractive_answer(query: str, top_chunks: list[dict[str, Any]]) -> str:
    """High-precision extractive answer synthesis when no generative chat LLM is pulled in Ollama yet."""
    if not top_chunks:
        return "No indexed information matching your question was found in the active workspace."

    primary = top_chunks[0]
    primary_excerpt = str(primary.get("focused_answer") or primary.get("content") or "").strip()

    if len(top_chunks) == 1 or float(primary.get("rerank_score") or 0.0) >= float(top_chunks[1].get("rerank_score") or 0.0) + 0.08:
        return f"{primary_excerpt} [1]"

    lines = [f"{primary_excerpt} [1]"]
    seen_norm = {primary_excerpt.casefold()}
    for idx, chunk in enumerate(top_chunks[1:3], start=2):
        excerpt = str(chunk.get("focused_answer") or "").strip()
        if not excerpt or excerpt.casefold() in seen_norm:
            continue
        if float(chunk.get("rerank_score") or 0.0) < float(primary.get("rerank_score") or 0.0) - 0.15:
            continue
        seen_norm.add(excerpt.casefold())
        lines.append(f"- {excerpt} [{idx}]")
    return "\n\n".join(lines)


def _build_stage5_messages(
    query: str,
    top_chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, str]], str]:
    """Build the system prompt, chat messages array, and combined prompt for Stage 5 Ollama LLM."""
    evidence_blocks: list[str] = []
    for idx, chunk in enumerate(top_chunks, start=1):
        citation = chunk.get("citation") or {}
        filename = chunk.get("filename") or citation.get("filename") or "Document"
        version = chunk.get("version_number") or citation.get("version_number") or 1
        loc_dict = chunk.get("source_location") or citation.get("source_location")
        loc = _readable_location(loc_dict)
        headings = chunk.get("heading_context") or citation.get("heading_context") or []
        section = " > ".join(
            str(h.get("text") if isinstance(h, dict) else h)
            for h in headings
            if (h.get("text") if isinstance(h, dict) else h)
        )
        focused = str(chunk.get("focused_answer") or chunk.get("focused_content") or chunk.get("content") or "").strip()
        full_text = str(chunk.get("content") or chunk.get("full_content") or "").strip()
        block = [
            f"[{idx}] Source: {filename} (v{version}, {loc})",
            f"Section: {section}" if section else "",
            f"Reranked Excerpt: {focused}",
            f"Full Chunk Context: {full_text}" if full_text and full_text != focused else "",
        ]
        evidence_blocks.append("\n".join(part for part in block if part))

    evidence_text = "\n\n---\n\n".join(evidence_blocks)
    system_prompt = (
        "You are Sourcebook Stage 5, an enterprise RAG research assistant powered by a local Ollama LLM. "
        "Answer the user's question directly, accurately, and concisely using ONLY the Stage 4 reranked "
        "workspace sources provided below.\n"
        "Rules:\n"
        "1. Be direct and focused: if the user asks a specific question (e.g. 'What is NER?' or "
        "'What is machine learning?'), give the exact definition/answer for that concept in 2–4 clear "
        "sentences (or a concise bulleted list when appropriate). Do NOT dump unrelated bullet points "
        "or neighboring topics that the user did not ask about.\n"
        "2. Cite every factual claim inline using the bracketed source number, e.g. [1] or [1, 2].\n"
        "3. Never invent facts outside the provided sources. If the sources do not contain the answer, "
        "state clearly what is missing."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-4:]:
        role = str(turn.get("role") or "").lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:1500]})

    user_prompt = (
        f"Stage 4 Top Reranked Workspace Sources:\n\n{evidence_text}\n\n"
        f"User Question: {query}\n\n"
        "Provide a direct, accurate final answer with inline [1] citations:"
    )
    messages.append({"role": "user", "content": user_prompt})
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    return system_prompt, messages, combined_prompt


def _call_ollama_chat(
    model_name: str,
    query: str,
    top_chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Stage 5 LLM execution: generate a grounded answer with the configured local provider.

    Tries LangChain ChatOllama first, then the native `/api/chat` endpoint, and finally
    `/api/generate`. Other generation providers are not implemented in this release.
    """
    if GENERATION_PROVIDER != "ollama":
        raise RuntimeError("Only the Ollama generation provider is currently implemented.")
    _system_prompt, messages, combined_prompt = _build_stage5_messages(query, top_chunks, history)
    raw_output = ""

    # 1. Try LangChain ChatOllama if available
    try:
        from langchain_ollama import ChatOllama

        chat_client = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_CHAT_TEMPERATURE,
            num_ctx=OLLAMA_NUM_CTX,
            client_kwargs={"timeout": OLLAMA_REQUEST_TIMEOUT_SECONDS},
        )
        lc_messages = [(m["role"], m["content"]) for m in messages]
        with _OLLAMA_LOCK:
            ai_msg = chat_client.invoke(lc_messages)
        raw_output = str(getattr(ai_msg, "content", "") or "").strip()
    except Exception:
        raw_output = ""

    # 2. Native Ollama /api/chat endpoint
    if not raw_output:
        payload = json.dumps(
            {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": OLLAMA_CHAT_TEMPERATURE,
                    "num_ctx": OLLAMA_NUM_CTX,
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _OLLAMA_LOCK:
                with urllib.request.urlopen(req, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            raw_output = str((body.get("message") or {}).get("content") or body.get("response") or "").strip()
        except Exception:
            # 3. Fallback to Ollama /api/generate for completion-style models
            gen_payload = json.dumps(
                {
                    "model": model_name,
                    "prompt": combined_prompt,
                    "stream": False,
                    "options": {
                        "temperature": OLLAMA_CHAT_TEMPERATURE,
                        "num_ctx": OLLAMA_NUM_CTX,
                    },
                }
            ).encode("utf-8")
            gen_req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=gen_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _OLLAMA_LOCK:
                with urllib.request.urlopen(gen_req, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            raw_output = str(body.get("response") or "").strip()

    # Strip <think>...</think> reasoning traces if a reasoning model like deepseek-r1 was used
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    return cleaned


def stage5_generate_llm_answer(
    *,
    query: str,
    reranked_chunks: list[dict[str, Any]],
    chat_model: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Stage 5 standalone entry point: takes Stage 4 reranked chunks + user query and calls the Ollama LLM."""
    gen_started = perf_counter()
    active_chat_model, installed_models = resolve_chat_model(chat_model)
    generation_mode = "ollama_chat"
    answer_text = ""

    if active_chat_model and reranked_chunks:
        try:
            answer_text = _call_ollama_chat(
                active_chat_model,
                query.strip(),
                reranked_chunks,
                history=history,
            )
        except Exception as exc:
            logger.warning(
                "Stage 5 Ollama LLM '%s' failed (%s); falling back to pinpoint extractive synthesis.",
                active_chat_model,
                exc,
            )
            answer_text = ""

    if not answer_text:
        generation_mode = "extractive_pinpoint_synthesis"
        answer_text = _synthesize_extractive_answer(query.strip(), reranked_chunks)

    generation_ms = round((perf_counter() - gen_started) * 1000, 2)
    citations = _build_citations(reranked_chunks)
    return {
        "stage": "stage5_ollama_llm",
        "query": query.strip(),
        "answer": answer_text,
        "answer_mode": generation_mode,
        "generation_mode": generation_mode,
        "chat_model": active_chat_model or "extractive-pinpoint",
        "model_used": active_chat_model or "extractive-pinpoint",
        "installed_models": installed_models,
        "generation_ms": generation_ms,
        "citations": citations,
    }


def _build_citations(top_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for idx, chunk in enumerate(top_chunks, start=1):
        cit = chunk.get("citation") or {}
        headings = chunk.get("heading_context") or cit.get("heading_context") or []
        section = " › ".join(
            str(h.get("text") if isinstance(h, dict) else h)
            for h in headings
            if (h.get("text") if isinstance(h, dict) else h)
        )
        loc_dict = chunk.get("source_location") or cit.get("source_location")
        citations.append(
            {
                "index": idx,
                "document_id": chunk.get("document_id") or cit.get("document_id"),
                "filename": chunk.get("filename") or cit.get("filename"),
                "version_number": chunk.get("version_number") or cit.get("version_number"),
                "location": _readable_location(loc_dict),
                "source_location": loc_dict,
                "section": section or None,
                "similarity": chunk.get("similarity"),
                "rerank_score": chunk.get("rerank_score"),
                "excerpt": chunk.get("focused_answer") or chunk.get("focused_content") or chunk.get("content"),
                "source_chunk_id": chunk.get("source_chunk_id") or cit.get("source_chunk_id"),
            }
        )
    return citations


def _ensure_workspace_indexed(user_id: str, workspace_id: str) -> None:
    """Automatically index any SANITIZED, embedding-eligible documents in the workspace that are not yet indexed."""
    try:
        from app.ingestion.vector_index import _get_pool, process_document_embedding, queue_document_index

        pending_docs: list[tuple[str, str]] = []
        with _get_pool().connection() as connection:
            rows = connection.execute(
                """
                SELECT d.id
                FROM documents d
                WHERE d.user_id=%s AND d.workspace_id=%s AND d.deleted_at IS NULL
                  AND d.status IN ('UPLOADED', 'SUPERSEDED')
                  AND d.embedding_status IN ('NOT_INDEXED', 'FAILED')
                ORDER BY d.created_at ASC
                """,
                (user_id, workspace_id),
            ).fetchall()
            for row in rows:
                try:
                    res = queue_document_index(connection, row["id"], user_id, force=False)
                    if res.get("queued") and res.get("artifact_id"):
                        pending_docs.append((row["id"], res["artifact_id"]))
                except Exception:
                    continue
        for doc_id, art_id in pending_docs:
            process_document_embedding(doc_id, user_id, art_id)
    except Exception:
        pass


def answer_workspace_query(
    *,
    user_id: str,
    workspace_id: str,
    query: str,
    candidate_k: int = RAG_CANDIDATE_POOL_SIZE,
    top_k: int = RAG_RERANK_TOP_K,
    include_superseded: bool = False,
    history: list[dict[str, str]] | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    """End-to-end Stage 3 -> Stage 4 -> Stage 5 RAG Pipeline:
    - Stage 3: Hybrid Retrieve Top 20 (`candidate_k`) chunks from pgvector (`bge-m3`).
    - Stage 4: Local Reranker ranks the 20 candidates and selects the Top K (`top_k`, default 3 or 5).
    - Stage 5: Passes the Top K reranked chunks + user query to the local Ollama LLM (`chat_model`)
      to synthesize the final grounded answer with inline citations.
    """
    started_at = perf_counter()
    effective_candidate_k = max(int(top_k), min(50, int(candidate_k or RAG_CANDIDATE_POOL_SIZE)))
    effective_top_k = max(1, min(10, int(top_k or RAG_RERANK_TOP_K)))

    # Ensure any sanitized documents in this workspace are indexed automatically
    _ensure_workspace_indexed(user_id, workspace_id)

    # Stage 3: Retrieve top candidate_k hybrid chunks from pgvector
    retrieval = semantic_search(
        user_id=user_id,
        workspace_id=workspace_id,
        query=query,
        top_k=effective_candidate_k,
        include_superseded=include_superseded,
    )
    candidates = retrieval.get("results") or []
    timings = dict(retrieval.get("timings_ms") or {})

    if not candidates:
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        active_model = chat_model or OLLAMA_CHAT_MODEL
        return {
            "workspace_id": workspace_id,
            "query": query.strip(),
            "answer": (
                "No indexed documents were found in this workspace yet. Upload a document (PDF, Word, Excel, CSV, or Markdown) "
                "and wait for the automatic pipeline on the right to reach **Indexed** before asking questions."
            ),
            "answer_mode": "empty_workspace",
            "generation_mode": "empty_workspace",
            "chat_model": active_model,
            "model_used": active_model,
            "citations": [],
            "stage4_reranked_chunks": [],
            "pipeline_telemetry": {
                "candidates_retrieved": 0,
                "reranked_top_k": 0,
                "embedding_model": retrieval.get("embedding_model", OLLAMA_EMBEDDING_MODEL),
                "reranker_model": OLLAMA_RERANK_MODEL,
                "chat_model": active_model,
                "generation_mode": "empty_workspace",
                "query_embedding_ms": timings.get("query_embedding_ms", 0.0),
                "pgvector_search_ms": timings.get("pgvector_search_ms", 0.0),
                "rerank_ms": 0.0,
                "generation_ms": 0.0,
                "total_ms": total_ms,
            },
        }

    # Stage 4: Rerank candidate pool (20 -> top 3 or 5)
    rerank_started = perf_counter()
    top_chunks = rerank_candidates(query, candidates, top_k=effective_top_k)
    rerank_ms = round((perf_counter() - rerank_started) * 1000, 2)

    # Stage 5: Pass top reranked chunks + user query to local Ollama LLM for final answer
    stage5_result = stage5_generate_llm_answer(
        query=query.strip(),
        reranked_chunks=top_chunks,
        chat_model=chat_model,
        history=history,
    )
    total_ms = round((perf_counter() - started_at) * 1000, 2)

    return {
        "workspace_id": workspace_id,
        "query": query.strip(),
        "answer": stage5_result["answer"],
        "answer_mode": stage5_result["answer_mode"],
        "generation_mode": stage5_result["generation_mode"],
        "chat_model": stage5_result["chat_model"],
        "model_used": stage5_result["model_used"],
        "citations": stage5_result["citations"],
        "stage4_reranked_chunks": [
            {
                "rerank_position": c.get("rerank_position"),
                "initial_rank": c.get("initial_rank"),
                "rerank_score": c.get("rerank_score"),
                "similarity": c.get("similarity"),
                "filename": c.get("filename") or (c.get("citation") or {}).get("filename"),
                "focused_answer": c.get("focused_answer"),
            }
            for c in top_chunks
        ],
        "pipeline_telemetry": {
            "candidates_retrieved": len(candidates),
            "reranked_top_k": len(top_chunks),
            "embedding_model": retrieval.get("embedding_model", OLLAMA_EMBEDDING_MODEL),
            "reranker_model": OLLAMA_RERANK_MODEL,
            "chat_model": stage5_result["chat_model"],
            "generation_mode": stage5_result["generation_mode"],
            "query_embedding_ms": timings.get("query_embedding_ms", 0.0),
            "pgvector_search_ms": timings.get("pgvector_search_ms", 0.0),
            "rerank_ms": rerank_ms,
            "generation_ms": stage5_result["generation_ms"],
            "total_ms": total_ms,
        },
    }
