"""Create local vector embeddings for Constitution chunks."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INPUT = "nepal_constitution_chunks.json"
DEFAULT_OUTPUT = "nepal_constitution_embeddings.npz"
DEFAULT_METADATA = "nepal_constitution_embedding_metadata.json"
DEFAULT_CHROMA_PATH = "nepal_constitution_chroma"
DEFAULT_COLLECTION = "constitution_chunks"


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of chunks in '{path}'.")
    return data


def create_embeddings(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    chroma_path: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
) -> None:
    chunks = load_chunks(input_path)
    texts = [str(chunk.get("normalized_text") or chunk.get("text") or "") for chunk in chunks]
    if not all(texts):
        raise ValueError("Every chunk must contain non-empty text.")

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        model_name=np.array(model_name),
    )
    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count():
        collection.delete(ids=collection.get()["ids"])
    collection.add(
        ids=[str(chunk["chunk_id"]) for chunk in chunks],
        documents=[str(chunk.get("text") or "") for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "chunk_type": str(chunk.get("chunk_type", "")),
                "page_start": int(chunk["metadata"].get("page_start") or 0),
                "page_end": int(chunk["metadata"].get("page_end") or 0),
            }
            for chunk in chunks
        ],
    )
    print(f"Saved vectors to '{output_path}'.")
    print(f"Saved chunk metadata to '{metadata_path}'.")
    print(f"Saved ChromaDB collection '{collection_name}' to '{chroma_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--metadata", type=Path, default=Path(DEFAULT_METADATA))
    parser.add_argument("--chroma-path", type=Path, default=Path(DEFAULT_CHROMA_PATH))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    create_embeddings(
        args.input,
        args.output,
        args.metadata,
        args.chroma_path,
        args.collection,
        args.model,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
