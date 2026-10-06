"""Ask questions about the Constitution using local retrieval and OpenRouter."""

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
import chromadb
from dotenv import load_dotenv
from create_embeddings import create_embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNKS = "nepal_constitution_chunks.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DENSE_WEIGHT = 0.65
DEFAULT_BM25_WEIGHT = 0.35
DEFAULT_CHROMA_PATH = "nepal_constitution_chroma"
DEFAULT_COLLECTION = "constitution_chunks"


def load_index(collection: Any) -> tuple[List[Dict[str, Any]], str]:
    data = collection.get(include=["documents", "metadatas"])
    chunks = []
    for chunk_id, document, metadata in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        chunk_metadata = dict(metadata)
        parent_chunk_id = chunk_metadata.pop("parent_chunk_id", "")
        if not parent_chunk_id:
            parent_chunk_id = None
        chunks.append(
            {
                "chunk_id": chunk_id,
                "parent_chunk_id": parent_chunk_id,
                "chunk_type": chunk_metadata.pop("chunk_type", ""),
                "text": document,
                "normalized_text": document,
                "metadata": chunk_metadata,
            }
        )
    model_name = collection.metadata.get("model_name") if collection.metadata else None
    if not chunks or not isinstance(model_name, str) or not model_name:
        raise ValueError("ChromaDB collection is missing chunks or model metadata.")
    return chunks, model_name


def tokenize(text: str) -> List[str]:
    """Use simple stable tokens for BM25 without modifying stored source text."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def retrieval_text(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    hierarchy: List[str] = []
    if metadata.get("part_number") is not None:
        hierarchy.append(
            f"Part {metadata['part_number']}: {metadata.get('part_title') or ''}"
        )
    if metadata.get("article_number") is not None:
        hierarchy.append(
            f"Article {metadata['article_number']}: "
            f"{metadata.get('article_title') or ''}"
        )
    if metadata.get("clause_number") is not None:
        hierarchy.append(f"Clause {metadata['clause_number']}")
    if metadata.get("subclause_number") is not None:
        hierarchy.append(f"Subclause {metadata['subclause_number']}")
    if metadata.get("schedule_number") is not None:
        hierarchy.append(
            f"Schedule {metadata['schedule_number']}: "
            f"{metadata.get('schedule_title') or ''}"
        )
    return "\n".join([*hierarchy, str(chunk.get("text") or "")])


def hierarchy_header(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    labels: List[str] = []
    if metadata.get("part_number") is not None:
        labels.append(
            f"Part {metadata['part_number']}: {metadata.get('part_title') or ''}"
        )
    if metadata.get("article_number") is not None:
        labels.append(
            f"Article {metadata['article_number']}: "
            f"{metadata.get('article_title') or ''}"
        )
    if metadata.get("clause_number") is not None:
        labels.append(f"Clause {metadata['clause_number']}")
    if metadata.get("subclause_number") is not None:
        labels.append(f"Subclause {metadata['subclause_number']}")
    if metadata.get("schedule_number") is not None:
        labels.append(
            f"Schedule {metadata['schedule_number']}: "
            f"{metadata.get('schedule_title') or ''}"
        )
    return " | ".join(labels)


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_chunks = [
        tokenize(retrieval_text(chunk))
        for chunk in chunks
    ]
    if not all(tokenized_chunks):
        raise ValueError("Every chunk must contain text for BM25 indexing.")
    return BM25Okapi(tokenized_chunks)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum == minimum:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - minimum) / (maximum - minimum)).astype(np.float32)


def retrieve(
    question: str,
    model: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    bm25: BM25Okapi,
    collection: Any,
    top_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    query_variants = [
        question,
        (
            "Relevant constitutional rights, duties, legal principles, "
            f"articles, clauses, and related provisions concerning: {question}"
        ),
    ]
    query_vectors = model.encode(
        query_variants,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    candidate_count = min(len(chunks), max(100, top_k * 20))
    dense_scores_by_id: Dict[str, float] = {}
    dense_documents_by_id: Dict[str, str] = {}
    for query_vector in query_vectors:
        dense_result = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=candidate_count,
            include=["documents", "distances"],
        )
        for chunk_id, document, distance in zip(
            dense_result["ids"][0],
            dense_result["documents"][0],
            dense_result["distances"][0],
        ):
            score = float(1.0 - distance)
            if score > dense_scores_by_id.get(chunk_id, float("-inf")):
                dense_scores_by_id[chunk_id] = score
                dense_documents_by_id[chunk_id] = document

    bm25_scores_by_id: Dict[str, float] = {}
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    candidate_ids = set(dense_scores_by_id)
    for query in query_variants:
        bm25_scores = np.asarray(
            bm25.get_scores(tokenize(query)), dtype=np.float32
        )
        for index in np.argsort(bm25_scores)[::-1][:candidate_count]:
            chunk_id = str(chunks[int(index)]["chunk_id"])
            bm25_scores_by_id[chunk_id] = max(
                bm25_scores_by_id.get(chunk_id, float("-inf")),
                float(bm25_scores[int(index)]),
            )
    bm25_candidate_count = min(len(chunks), max(100, top_k * 20))
    candidate_ids.update(
        sorted(
            bm25_scores_by_id,
            key=lambda chunk_id: bm25_scores_by_id[chunk_id],
            reverse=True,
        )[:bm25_candidate_count]
    )
    candidate_list = list(candidate_ids)
    dense_scores = np.asarray(
        [dense_scores_by_id.get(chunk_id, 0.0) for chunk_id in candidate_list],
        dtype=np.float32,
    )
    candidate_bm25_scores = np.asarray(
        [bm25_scores_by_id.get(chunk_id, 0.0) for chunk_id in candidate_list],
        dtype=np.float32,
    )
    total_weight = dense_weight + bm25_weight
    hybrid_scores = (
        (dense_weight / total_weight) * normalize_scores(dense_scores)
        + (bm25_weight / total_weight) * normalize_scores(candidate_bm25_scores)
    )
    indices = np.argsort(hybrid_scores)[::-1][:top_k]
    results = []
    for index in indices:
        chunk_id = candidate_list[int(index)]
        result = dict(chunk_by_id[chunk_id])
        if chunk_id in dense_documents_by_id:
            result["text"] = dense_documents_by_id[chunk_id]
        result["_hybrid_score"] = float(hybrid_scores[int(index)])
        result["_dense_score"] = float(dense_scores[int(index)])
        result["_bm25_score"] = float(candidate_bm25_scores[int(index)])
        results.append(result)
    return results


def add_context(results: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> None:
    """Attach hierarchy and parent text without changing source text."""
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    for result in results:
        result["_context_header"] = hierarchy_header(result)
        parent = chunk_by_id.get(str(result.get("parent_chunk_id")))
        if parent and parent.get("chunk_type") in {"article", "schedule"}:
            result["_parent_text"] = str(parent.get("text") or "")
        else:
            result["_parent_text"] = ""


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[Source {index}] {result.get('_context_header', '')}\n"
        f"Parent provision:\n{result.get('_parent_text', '')}\n"
        f"Source text:\n{result['text']}\n"
        f"Chunk: {result['chunk_id']}; "
        f"Pages: {result['metadata'].get('page_start')}-"
        f"{result['metadata'].get('page_end')}"
        for index, result in enumerate(results, start=1)
    )
    return (
        "You are a careful legal-information assistant for the Constitution "
        "of Nepal. Use only the provided source context. Treat each source as "
        "a legal provision with the displayed hierarchy. Do not infer a legal "
        "conclusion from a generic shared word such as treatment, victim, "
        "authority, or service. The subject matter of the cited provision "
        "must match the question.\n\n"
        "Respond in two sections:\n"
        "1. Direct constitutional finding: state what the retrieved text "
        "directly establishes. If the exact facts are not addressed, state "
        "the broader constitutional principle and clearly say what cannot be "
        "concluded.\n"
        "2. Related constitutional provisions: list retrieved provisions that "
        "are thematically or structurally relevant and explain why each is "
        "relevant. Do not present related provisions as direct proof.\n\n"
        "Never invent legal text, penalties, court outcomes, statutes, "
        "regulations, or case law. Mention an external legal framework only "
        "if it appears in the provided context. Distinguish constitutional "
        "text from conclusions requiring other laws or facts. Include source "
        "chunk IDs and page numbers.\n\n"
        f"Question:\n{question}\n\nRetrieved context:\n{context}"
    )


def ask_openrouter(url: str, api_key: str, model_name: str, prompt: str) -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Constitution of Nepal RAG",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter returned HTTP {exc.code}: {detail[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc
    choices = data.get("choices")
    answer = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("OpenRouter returned an empty answer.")
    return answer.strip()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path(DEFAULT_CHUNKS))
    parser.add_argument("--chroma-path", type=Path, default=Path(DEFAULT_CHROMA_PATH))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--openrouter-model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
    )
    parser.add_argument(
        "--openrouter-url",
        default=os.environ.get("OPENROUTER_URL", DEFAULT_OPENROUTER_URL),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT
    )
    parser.add_argument("--bm25-weight", type=float, default=DEFAULT_BM25_WEIGHT)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.dense_weight < 0 or args.bm25_weight < 0:
        parser.error("Retrieval weights cannot be negative.")
    if args.dense_weight + args.bm25_weight <= 0:
        parser.error("At least one retrieval weight must be greater than zero.")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY was not found in the environment or .env.")

    chroma_client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection_names = {item.name for item in chroma_client.list_collections()}
    expected_count = len(json.loads(args.chunks.read_text(encoding="utf-8")))
    collection = (
        chroma_client.get_collection(name=args.collection)
        if args.collection in collection_names
        else None
    )
    sample_metadata: Dict[str, Any] = {}
    if collection is not None and collection.count():
        sample = collection.get(limit=1, include=["metadatas"])
        metadatas = sample.get("metadatas") or []
        if metadatas:
            sample_metadata = dict(metadatas[0] or {})
    collection_ready = (
        collection is not None
        and collection.count() == expected_count
        and isinstance(collection.metadata, dict)
        and bool(collection.metadata.get("model_name"))
        and "article_number" in sample_metadata
    )
    if not collection_ready:
        print("ChromaDB is missing or incomplete; rebuilding the collection...")
        create_embeddings(
            args.chunks,
            Path("nepal_constitution_embeddings.npz"),
            Path("nepal_constitution_embedding_metadata.json"),
            args.chroma_path,
            args.collection,
            args.embedding_model or DEFAULT_MODEL,
            32,
        )
        collection = chroma_client.get_collection(name=args.collection)

    chunks, stored_model = load_index(collection)
    model_name = args.embedding_model or stored_model
    model = SentenceTransformer(model_name)
    bm25 = build_bm25(chunks)
    print("RAG chat ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        results = retrieve(
            question,
            model,
            chunks,
            bm25,
            collection,
            args.top_k,
            args.dense_weight,
            args.bm25_weight,
        )
        add_context(results, chunks)
        print("\nRetrieved sources:")
        for result in results:
            print(
                f"- {result['chunk_id']} "
                f"(hybrid={result['_hybrid_score']:.3f}, "
                f"dense={result['_dense_score']:.3f}, "
                f"bm25={result['_bm25_score']:.3f}, "
                f"pages={result['metadata'].get('page_start')}-"
                f"{result['metadata'].get('page_end')})"
            )
        answer = ask_openrouter(
            args.openrouter_url,
            api_key,
            args.openrouter_model,
            build_prompt(question, results),
        )
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
