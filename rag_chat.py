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
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDINGS = "nepal_constitution_embeddings.npz"
DEFAULT_METADATA = "nepal_constitution_embedding_metadata.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DENSE_WEIGHT = 0.65
DEFAULT_BM25_WEIGHT = 0.35
DEFAULT_CHROMA_PATH = "nepal_constitution_chroma"
DEFAULT_COLLECTION = "constitution_chunks"


def load_index(
    embeddings_path: Path, metadata_path: Path
) -> tuple[np.ndarray, List[Dict[str, Any]], str]:
    with np.load(embeddings_path) as data:
        embeddings = data["embeddings"].astype(np.float32)
        stored_model = str(data["model_name"].item())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(embeddings):
        raise ValueError("Embedding vectors and chunk metadata do not have matching lengths.")
    return embeddings, chunks, stored_model


def tokenize(text: str) -> List[str]:
    """Use simple stable tokens for BM25 without modifying stored source text."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_chunks = [
        tokenize(str(chunk.get("normalized_text") or chunk.get("text") or ""))
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
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)
    bm25_scores = np.asarray(bm25.get_scores(tokenize(question)), dtype=np.float32)
    candidate_count = min(len(chunks), max(100, top_k * 20))
    dense_result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_result["ids"][0]
    dense_documents = dense_result["documents"][0]
    dense_metadatas = dense_result["metadatas"][0]
    dense_distances = np.asarray(dense_result["distances"][0], dtype=np.float32)
    dense_scores_by_id = {
        chunk_id: float(1.0 - distance)
        for chunk_id, distance in zip(dense_ids, dense_distances)
    }
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    candidate_ids = set(dense_ids)
    bm25_candidate_count = min(len(chunks), max(100, top_k * 20))
    candidate_ids.update(
        str(chunks[int(index)]["chunk_id"])
        for index in np.argsort(bm25_scores)[::-1][:bm25_candidate_count]
    )
    candidate_list = list(candidate_ids)
    dense_scores = np.asarray(
        [dense_scores_by_id.get(chunk_id, 0.0) for chunk_id in candidate_list],
        dtype=np.float32,
    )
    candidate_bm25_scores = np.asarray(
        [
            bm25_scores[next(
                index for index, chunk in enumerate(chunks)
                if str(chunk["chunk_id"]) == chunk_id
            )]
            for chunk_id in candidate_list
        ],
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
        if chunk_id in dense_scores_by_id:
            dense_index = dense_ids.index(chunk_id)
            result["text"] = dense_documents[dense_index]
        result["_hybrid_score"] = float(hybrid_scores[int(index)])
        result["_dense_score"] = float(dense_scores[int(index)])
        result["_bm25_score"] = float(candidate_bm25_scores[int(index)])
        results.append(result)
    return results


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[Source {index}] {result['text']}\n"
        f"Chunk: {result['chunk_id']}; "
        f"Pages: {result['metadata'].get('page_start')}-"
        f"{result['metadata'].get('page_end')}"
        for index, result in enumerate(results, start=1)
    )
    return (
        "You answer questions about the Constitution of Nepal using only the "
        "provided source context. Do not invent or amend legal text. If the "
        "context does not answer the question, say that it is not found in the "
        "retrieved context. Include source chunk IDs and page numbers.\n\n"
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
    parser.add_argument("--embeddings", type=Path, default=Path(DEFAULT_EMBEDDINGS))
    parser.add_argument("--metadata", type=Path, default=Path(DEFAULT_METADATA))
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

    embeddings, chunks, stored_model = load_index(args.embeddings, args.metadata)
    model_name = args.embedding_model or stored_model
    model = SentenceTransformer(model_name)
    bm25 = build_bm25(chunks)
    chroma_client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = chroma_client.get_collection(name=args.collection)
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
