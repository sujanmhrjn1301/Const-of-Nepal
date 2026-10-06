# Constitution of Nepal RAG

A local retrieval pipeline for asking questions about the Constitution of Nepal.

The project:

1. Extracts and chunks the Constitution PDF.
2. Creates local dense embeddings.
3. Stores the embeddings in NumPy and ChromaDB.
4. Uses hybrid retrieval with ChromaDB and BM25.
5. Uses OpenRouter for answer generation.

## Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```env
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Do not commit `.env`.

## Generate chunks

To extract and chunk the PDF:

```powershell
python .\ingest.py
```

This creates:

```text
nepal_constitution_chunks.json
```

## Create embeddings and ChromaDB

```powershell
python .\create_embeddings.py
```

This creates:

```text
nepal_constitution_embeddings.npz
nepal_constitution_embedding_metadata.json
nepal_constitution_chroma/
```

The embedding model is downloaded and run locally:

```text
all-MiniLM-L6-v2
```

The original chunk text is stored as the ChromaDB document. Metadata is stored separately for source references such as article numbers and page numbers.

## Run the RAG chat

```powershell
python .\rag_chat.py
```

Ask a question at the prompt and type `exit` or `quit` to stop.

Example questions:

```text
- What is the right to communication?
- What does Article 96 provide?
- What is Schedule-1 about?
- I took my cousin to a government hospital after a road accident, but
the staff delayed treatment asking for upfront payment first. Is this
illegal under our constitution?
```

The retriever combines:

- Dense semantic similarity from ChromaDB.
- BM25 keyword matching.

The retrieved text and source page information are sent to the selected OpenRouter model for answer generation.

## Retrieval settings

The default hybrid weights are:

```text
Dense vectors: 65%
BM25:          35%
```

They can be changed at runtime:

```powershell
python .\rag_chat.py --dense-weight 0.6 --bm25-weight 0.4
```

The number of retrieved chunks can also be changed:

```powershell
python .\rag_chat.py --top-k 8
```

## Project files

| File | Purpose |
|---|---|
| `const.pdf` | Source Constitution PDF |
| `ingest.py` | Extracts and chunks the PDF |
| `create_embeddings.py` | Creates embeddings and the ChromaDB index |
| `rag_chat.py` | Runs hybrid retrieval and question answering |
| `requirements.txt` | Python dependencies |

Generated embedding files and local credentials are excluded through `.gitignore` by default.

## Security

Never commit API keys or other secrets. Keep them in `.env`, which is ignored by Git:

```text
.env
OPENROUTER_API_KEY
```
