# Personal RAG

Your own retrieval-augmented generation system. Upload your documents, they get
chunked and embedded into a local vector database, and then you can ask
questions in a chat interface. Every answer is written only from the top
passages retrieved from your own files, and each answer shows you the passages
it used.

Nothing leaves your machine. The language model, the embedding model and the
vector database all run locally.

```
                upload                    ask
                  │                        │
                  ▼                        ▼
           ┌─────────────┐          ┌─────────────┐
           │  extract    │          │  embed the  │
           │  + chunk    │          │  question   │
           └──────┬──────┘          └──────┬──────┘
                  ▼                        ▼
           ┌─────────────┐          ┌─────────────┐
           │  embed with │          │  top 3 most │
           │  BGE-small  │          │  similar    │
           └──────┬──────┘          │  chunks     │
                  ▼                 └──────┬──────┘
           ┌────────────────────┐          ▼
           │   ChromaDB  (disk) │◄──┐ ┌─────────────┐
           └────────────────────┘   └─│ Qwen2.5     │
                                      │ writes the  │
                                      │ answer      │
                                      └─────────────┘
```

## What you get

- A chat interface with streaming answers and inline citations
- A knowledge base panel: drag a file in, see what is stored, delete one file
  or wipe everything
- Top-3 retrieval by default, with the similarity score of every passage shown
- A REST API for all of it, with interactive docs at `/docs`
- PDF, DOCX, TXT, Markdown, CSV and JSON ingestion
- Runs on CPU; uses your GPU automatically if you have one
- Docker setup, tests, and CLI scripts for bulk indexing and resets

## Models

| Role | Model | Size | Why |
|---|---|---|---|
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | ~3.1 GB | The ~2B-class Qwen. Strong instruction following, runs on a laptop CPU. |
| Embeddings | `BAAI/bge-small-en-v1.5` | ~130 MB | 384-dimension vectors, fast, excellent retrieval quality for its size. |

Swap either one with a single line in `.env`. Other Qwen sizes that work
without any code change: `Qwen/Qwen2-1.5B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`.

## Requirements

- Python 3.10 or newer
- 8 GB RAM for the 1.5B model on CPU (16 GB is comfortable)
- About 5 GB of disk for model weights
- A CUDA GPU is optional and makes answers roughly 10 times faster

## Install

```bash
git clone https://github.com/<your-username>/personal-rag.git
cd personal-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first so pip does
not give you the CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Then download the models once, so the first question is not waiting on a
download:

```bash
python scripts/download_models.py
```

## Run

```bash
cp .env.example .env               # optional, the defaults work
uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. The API reference is at
<http://localhost:8000/docs>.

With Docker instead:

```bash
cp .env.example .env
docker compose up --build
```

## Using it

1. Drop a PDF into the left panel. It is chunked, embedded and stored, and the
   file appears in the list with its chunk count.
2. Ask a question in the box on the right. The answer streams in with `[1]`
   style citations, and the passages behind it are listed underneath with their
   similarity scores.
3. Switch to "Passages only" if you want raw retrieval with no model writing on
   top. This is the fastest way to tell whether a bad answer is a retrieval
   problem or a generation problem.
4. Press the `×` next to a file to delete it from the vector database, or
   "Clear everything" to empty the store.

**[GUIDE.md](GUIDE.md) is the detailed manual** — deleting and replacing
documents, tuning chunk size and top-k, API examples for every endpoint,
deployment, and troubleshooting.

## API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/health` | Liveness, whether the model is loaded, document count |
| `GET` | `/api/stats` | Collection name, documents, chunks, model names, storage path |
| `POST` | `/api/warmup` | Load the language model now instead of on first use |
| `GET` | `/api/documents` | List every stored document |
| `POST` | `/api/documents` | Upload and index a file (multipart `file` field) |
| `DELETE` | `/api/documents/{doc_id}` | Delete one document and all its chunks |
| `DELETE` | `/api/documents?confirm=true` | Wipe the whole vector database |
| `POST` | `/api/search` | Top-k passages, no generation |
| `POST` | `/api/ask` | Full answer with sources, returned in one response |
| `POST` | `/api/ask/stream` | The same, streamed as server-sent events |

```bash
# index a file
curl -F "file=@handbook.pdf" http://localhost:8000/api/documents

# ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the notice period?","top_k":3}'

# delete that file again
curl -X DELETE http://localhost:8000/api/documents/4f2a9c1b7e30
```

## Project layout

```
app/
  config.py        every setting, read from the environment
  ingestion.py     PDF/DOCX/CSV/text extraction and sentence-aware chunking
  embeddings.py    the embedding model, loaded once
  vectorstore.py   ChromaDB: add, search, list, delete, reset
  llm.py           Qwen loading, blocking generation and token streaming
  rag.py           retrieve, build the grounded prompt, answer
  schemas.py       request and response models
  main.py          FastAPI routes and static hosting
web/index.html     the whole interface, one file, no build step
scripts/           model download, bulk ingest, database reset
tests/             pytest suite, runs in about a second with models stubbed
```

## Tests

```bash
pytest -q
```

The suite replaces the embedding model with a deterministic hashing vectoriser,
so it exercises upload, retrieval, deletion and the streaming protocol without
downloading multiple gigabytes.

## Configuration

All settings live in `.env`. The ones worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Any Qwen instruct checkpoint |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Changing this needs a rebuild, see GUIDE.md |
| `TOP_K` | `3` | How many passages reach the model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Characters |
| `MIN_SCORE` | `0.15` | Cosine floor for a passage to count |
| `DEVICE` | `auto` | `cuda`, `mps` or `cpu` to force it |
| `API_KEY` | empty | Set it to require `X-API-Key` on write endpoints |
| `LOAD_LLM_ON_STARTUP` | `false` | `true` trades slow boot for a fast first answer |

## License

MIT. See [LICENSE](LICENSE).
