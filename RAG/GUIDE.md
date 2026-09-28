# Guide

Everything you need to run, operate and troubleshoot this system.

1. [Installing the models](#1-installing-the-models)
2. [Starting the server](#2-starting-the-server)
3. [Adding a document](#3-adding-a-document)
4. [Deleting a document from the vector database](#4-deleting-a-document-from-the-vector-database)
5. [Replacing a document with an updated version](#5-replacing-a-document-with-an-updated-version)
6. [Clearing the entire vector database](#6-clearing-the-entire-vector-database)
7. [Asking questions](#7-asking-questions)
8. [Tuning retrieval quality](#8-tuning-retrieval-quality)
9. [Switching models](#9-switching-models)
10. [Running in production](#10-running-in-production)
11. [Backup and restore](#11-backup-and-restore)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Installing the models

Two models are downloaded from Hugging Face: a generator and an embedder.

| | Repository | Download size | Loaded in RAM |
|---|---|---|---|
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` | ~3.1 GB | ~3.5 GB float32 on CPU, ~1.8 GB float16 on GPU |
| Embedder | `BAAI/bge-small-en-v1.5` | ~130 MB | ~150 MB |

### Step 1: install PyTorch for your hardware

Do this **before** `pip install -r requirements.txt`, because the generic wheel
is CPU-only.

```bash
# NVIDIA GPU, CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only (also correct for Apple Silicon, which uses the MPS backend)
pip install torch
```

Check what you got:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 2: install the rest

```bash
pip install -r requirements.txt
```

### Step 3: download the weights

```bash
python scripts/download_models.py
```

This pulls both repositories into the Hugging Face cache
(`~/.cache/huggingface` by default) and prints where they landed. It takes a
few minutes on a normal connection and only happens once.

To keep the weights somewhere else — a larger disk, or a shared drive — set
`HF_HOME` before running it:

```bash
export HF_HOME=/mnt/big-disk/hf          # Windows: set HF_HOME=D:\hf
python scripts/download_models.py
```

Put the same line in `.env` so the server uses the same cache.

If you skip this step entirely, nothing breaks: the model downloads the first
time you ask a question instead, which makes that one question take several
minutes.

### Step 4: verify the model loads

```bash
python - <<'PY'
from app import llm
print(llm.generate([{"role": "user", "content": "Reply with the single word: ready"}]))
PY
```

You should see `ready`. If this works, generation works.

### Gated or private models

Qwen2.5 is open, so no token is needed. If you switch to a gated checkpoint,
log in once:

```bash
huggingface-cli login
```

### Offline machines

Download on a connected machine, then copy the cache folder over and set:

```bash
HF_HOME=/path/to/copied/cache
HF_HUB_OFFLINE=1
```

---

## 2. Starting the server

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Interface: <http://localhost:8000>
- API docs: <http://localhost:8000/docs>

During development, `uvicorn app.main:app --reload --port 8000` restarts on
every code change.

The language model is loaded lazily, on the first question. That keeps startup
instant but makes the first answer slow. Two ways around it:

- Press **Load model** in the bottom left of the interface
- Set `LOAD_LLM_ON_STARTUP=true` in `.env` so the server loads it at boot

---

## 3. Adding a document

### From the interface

Drag the file onto the panel on the left, or press **Choose a file**. While it
indexes you get a toast; when it finishes, the file appears in the list with
its chunk count and its `doc_id`.

```
policy-handbook.pdf
48 chunks · 41.2k chars · 4f2a9c1b7e30          ×
```

That 12-character `doc_id` is the handle you use to delete the file later.

### From the API

```bash
curl -F "file=@policy-handbook.pdf" http://localhost:8000/api/documents
```

```json
{
  "doc_id": "4f2a9c1b7e30",
  "filename": "policy-handbook.pdf",
  "chunks": 48,
  "message": "Indexed policy-handbook.pdf into 48 searchable chunks."
}
```

### From the command line, in bulk

```bash
python scripts/ingest.py ./contracts            # a whole folder, recursively
python scripts/ingest.py a.pdf b.docx notes.md  # specific files
```

### What happens internally

1. The file is saved to `data/uploads/<doc_id><extension>`.
2. Text is extracted. PDFs are extracted page by page so citations can carry a
   page number.
3. The text is split into ~900-character chunks on sentence boundaries, with a
   150-character overlap so a fact that straddles a boundary is not lost.
4. Each chunk is embedded into a 384-dimension vector.
5. The vectors, the chunk text and the metadata (`doc_id`, `filename`, `page`,
   `chunk_index`, `uploaded_at`) are written to ChromaDB in `data/chroma`.

### Supported file types

`.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`. Uploads are capped at 50 MB by
default (`MAX_UPLOAD_MB`).

A scanned PDF is a picture of text, so nothing is extracted and the upload
fails with "No text could be read from this file". Run OCR first:

```bash
ocrmypdf scanned.pdf searchable.pdf
```

Then upload `searchable.pdf`.

---

## 4. Deleting a document from the vector database

Say you uploaded `policy-handbook.pdf` and you no longer want it answering your
questions. Deleting it removes every chunk and every embedding derived from it,
plus the copy of the original file. The rest of your knowledge base is
untouched.

### From the interface

1. Find the file in the left panel.
2. Press the `×` on its row.
3. A dialog explains what will be removed. Press **Delete**.
4. The counters at the top update immediately. The file is gone from retrieval
   from that moment — no restart needed.

### From the API

You need the `doc_id`. List your documents if you do not have it:

```bash
curl http://localhost:8000/api/documents
```

```json
[
  {
    "doc_id": "4f2a9c1b7e30",
    "filename": "policy-handbook.pdf",
    "uploaded_at": "2026-05-04T11:20:31+00:00",
    "chunks": 48,
    "characters": 41207
  }
]
```

Then delete it:

```bash
curl -X DELETE http://localhost:8000/api/documents/4f2a9c1b7e30
```

```json
{
  "doc_id": "4f2a9c1b7e30",
  "chunks_removed": 48,
  "message": "Removed 48 chunks. That document is no longer searchable."
}
```

Deleting an id that does not exist returns `404`.

### Confirming the deletion

```bash
curl http://localhost:8000/api/stats
```

`documents` and `chunks` should both have dropped. A search for a phrase that
only existed in that file now returns nothing:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"a phrase only that pdf contained"}'
```

### Why this is safe

Every chunk carries its `doc_id` in metadata. Deletion fetches the exact ids
belonging to that document and removes only those, so two files that contain
near-identical text are still deleted independently.

---

## 5. Replacing a document with an updated version

There is no in-place update, and that is deliberate: a new version of a
document has different chunk boundaries, so overwriting chunk by chunk would
leave orphans. Delete, then upload.

```bash
# 1. find the old version
curl http://localhost:8000/api/documents

# 2. delete it
curl -X DELETE http://localhost:8000/api/documents/4f2a9c1b7e30

# 3. upload the new one
curl -F "file=@policy-handbook-v2.pdf" http://localhost:8000/api/documents
```

In the interface: press `×` on the old row, then drop the new file in. Takes a
few seconds.

**Do not skip the delete.** If you upload a revised file without removing the
old one, both versions are in the store, retrieval will happily mix passages
from each, and you will get answers that quote a policy you already changed.
Two uploads of the same filename produce two separate `doc_id`s — the system
never assumes same name means same document.

If you want the old copy kept out of retrieval but still on disk, set
`KEEP_ORIGINAL_FILES=true` (the default) and move the file out of
`data/uploads` before deleting, or just keep your originals somewhere else.

---

## 6. Clearing the entire vector database

Three ways, depending on how thorough you want to be.

### From the interface

Press **Clear everything** at the top of the file list, type `DELETE` in the
dialog, confirm. Every document, every embedding and every uploaded file is
removed, and the collection is immediately recreated empty so you can start
uploading again.

### From the API

```bash
curl -X DELETE "http://localhost:8000/api/documents?confirm=true"
```

Without `?confirm=true` this returns `400` and does nothing. That guard exists
so a stray `DELETE /api/documents` cannot wipe your store.

### From the command line

```bash
python scripts/reset_db.py          # prompts for confirmation
python scripts/reset_db.py --yes    # no prompt, for scripts
python scripts/reset_db.py --hard   # also deletes data/chroma from disk
```

Use `--hard` when the store itself is the problem — a corrupted database, a
Chroma version upgrade, or a change of embedding model. Stop the server first.
`--hard` is the only one of these that removes the SQLite files themselves; the
others empty the collection but leave the folder in place.

### After changing the embedding model

Vectors from two different embedding models are not comparable. Changing
`EMBEDDING_MODEL` **requires** a full reset and re-upload:

```bash
# stop the server
python scripts/reset_db.py --hard
# edit EMBEDDING_MODEL in .env
uvicorn app.main:app --port 8000
python scripts/ingest.py ./my-documents
```

Skipping this gives you silently terrible retrieval rather than an error.

---

## 7. Asking questions

### In the interface

Type a question and press Enter. The flow is:

1. Your question is embedded and the 3 closest chunks are fetched.
2. Those chunks appear as the sources under the answer, ranked, with their
   cosine similarity.
3. Qwen receives only those chunks plus your question, and writes the answer
   with `[1]`, `[2]` citations as it streams.

Change **results** in the top bar to retrieve more or fewer than 3 passages.
More passages give the model more to work with but dilute its attention; 3 to 5
is the useful range for a 1.5B model.

**Passages only** mode skips generation and just shows what retrieval found.
When an answer looks wrong, check this first: if the right passage is not in
the list, the problem is retrieval (chunking, embedding, phrasing) and no
amount of prompt tweaking will fix it.

### From the API

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the notice period?","top_k":3,"temperature":0.2}'
```

```json
{
  "answer": "The notice period is 60 days for permanent staff [1].",
  "sources": [
    {
      "rank": 1,
      "score": 0.7412,
      "filename": "policy-handbook.pdf",
      "page": 14,
      "text": "Permanent employees must give 60 days of written notice...",
      "doc_id": "4f2a9c1b7e30",
      "chunk_index": 22
    }
  ],
  "grounded": true
}
```

Streaming uses server-sent events:

```bash
curl -N -X POST http://localhost:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the notice period?"}'
```

```
event: sources
data: [{"rank":1,"score":0.7412,...}]

event: token
data: {"text":"The notice "}

event: done
data: {}
```

### When the answer is "I could not find that in your documents"

That is the system working correctly. The prompt forbids the model from
answering beyond the retrieved passages, so it says this when the passages do
not contain the answer. Either the information is not in your files, or
retrieval missed it — check "Passages only" to find out which.

---

## 8. Tuning retrieval quality

Change one thing at a time and re-test with the same set of questions.

| Symptom | Setting | Try |
|---|---|---|
| Answers cite passages that stop mid-thought | `CHUNK_SIZE` | Raise to 1200–1500 |
| Retrieved passages are topically right but miss the detail | `CHUNK_SIZE` | Lower to 500–700, so each chunk is one idea |
| A fact that spans a chunk boundary is never found | `CHUNK_OVERLAP` | Raise to 250 |
| The model ignores a passage that is clearly relevant | `TOP_K` | Lower to 2–3, less noise |
| Nothing relevant is retrieved for broad questions | `TOP_K` | Raise to 5–8 |
| Irrelevant passages keep appearing | `MIN_SCORE` | Raise to 0.3 |
| Answers wander or invent details | `TEMPERATURE` | Lower to 0.1, or 0 for deterministic |
| Answers are cut off mid-sentence | `MAX_NEW_TOKENS` | Raise to 1024 |

Chunk settings only affect documents indexed **after** the change. To apply
them to everything, reset and re-upload:

```bash
python scripts/reset_db.py --yes
python scripts/ingest.py ./my-documents
```

Reading the similarity scores: above 0.6 is a strong match, 0.4 to 0.6 is
related, below 0.3 usually means retrieval found nothing good and you are
looking at the least-bad option.

A practical habit: keep a text file of 10 questions you know the answers to,
and run them through **Passages only** after any change to chunking or
embedding. It takes two minutes and catches regressions that a single spot
check misses.

---

## 9. Switching models

### A different Qwen size

Edit `.env`:

```bash
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
```

Restart the server. The new weights download on first use, or run
`python scripts/download_models.py` again.

| Model | Download | Answer quality | CPU speed |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 1 GB | Rough, fine for extraction | Fast |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3.1 GB | Good, the default | Usable |
| `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB | Better reasoning | Slow on CPU |
| `Qwen/Qwen2.5-7B-Instruct` | 15 GB | Best here | GPU only, realistically |

`Qwen/Qwen2-1.5B-Instruct` also works unchanged if you specifically want the
Qwen2 generation rather than 2.5.

### Forcing a device or precision

```bash
DEVICE=cuda          # or mps on Apple Silicon, or cpu
DTYPE=float16        # float32 on CPU, bfloat16 or float16 on GPU
```

`auto` picks CUDA if present, then MPS, then CPU, and chooses a sensible dtype
for each. Force `float32` if you see `nan` or garbled output on CPU.

### A different embedding model

```bash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5      # 768-dim, better, ~3x slower
EMBEDDING_MODEL=intfloat/multilingual-e5-base   # many languages
```

If you move to an E5 model, change the query prefix too:

```bash
QUERY_PREFIX=query: 
```

Then reset the store and re-upload everything, as described in section 6.

---

## 10. Running in production

### Process management

Run behind a real process manager. A systemd unit:

```ini
[Unit]
Description=Personal RAG
After=network.target

[Service]
User=rag
WorkingDirectory=/opt/personal-rag
EnvironmentFile=/opt/personal-rag/.env
ExecStart=/opt/personal-rag/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now personal-rag
sudo journalctl -u personal-rag -f
```

### Workers

Keep it to **one worker per machine**. Each worker loads its own copy of the
model, so `--workers 4` means four copies of Qwen in RAM. Generation is
serialised by a lock inside the process; if you need real concurrency, put the
generator behind a dedicated inference server (vLLM, TGI, Ollama) and point
`app/llm.py` at it.

### Authentication

Set an API key in `.env`:

```bash
API_KEY=a-long-random-string
```

Uploads, deletions and warmup then require the header:

```bash
curl -X DELETE http://localhost:8000/api/documents/4f2a9c1b7e30 \
  -H "X-API-Key: a-long-random-string"
```

Reads stay open. Lock those down at the reverse proxy if the machine is
reachable from outside.

### Reverse proxy

Streaming breaks if the proxy buffers. For nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;          # required for server-sent events
    proxy_read_timeout 300s;
    client_max_body_size 60M;     # must exceed MAX_UPLOAD_MB
}
```

### CORS

Default is `*`, which is fine on localhost. If the interface is served from
another origin, list it explicitly:

```bash
CORS_ORIGINS=https://rag.example.com
```

### Monitoring

`GET /api/health` is cheap and returns `llm_loaded`, document and chunk counts.
Point your uptime check at it. Docker Compose already uses it as a healthcheck.

### Capacity notes

- ChromaDB's HNSW index is in memory while the server runs. Roughly 1.5 KB per
  chunk: 100k chunks is about 150 MB, plus the chunk text.
- Embedding throughput on CPU is roughly 200 chunks per second for bge-small,
  so a 300-page PDF indexes in a few seconds.
- Generation on CPU with the 1.5B model runs at roughly 5–15 tokens per second,
  so a 200-token answer takes 15–40 seconds. On a mid-range GPU it is under
  three seconds. This is the slow part of the system, not retrieval.

---

## 11. Backup and restore

Everything that matters is in `data/`:

```
data/
  chroma/     the vector database (SQLite plus index files)
  uploads/    the original files
```

Back up with the server stopped, so the SQLite file is not mid-write:

```bash
sudo systemctl stop personal-rag
tar czf rag-backup-$(date +%F).tar.gz data/
sudo systemctl start personal-rag
```

Restore by stopping the server, replacing `data/`, and starting it again. The
model cache is not worth backing up: it re-downloads.

`data/` is in `.gitignore`. Do not commit it — it contains your documents.

---

## 12. Troubleshooting

**`ModuleNotFoundError: No module named 'app'`**
Run commands from the repository root, with the virtual environment active.

**The first question takes several minutes**
The model is downloading or loading. Watch the logs. Afterwards it stays in
memory. Use **Load model** or `LOAD_LLM_ON_STARTUP=true` to move that cost to
startup.

**`Killed` while loading the model**
The process ran out of RAM. Use a smaller model (`Qwen2.5-0.5B-Instruct`), or
set `DTYPE=float16`, or add swap.

**`CUDA out of memory`**
Set `DTYPE=float16`, lower `MAX_NEW_TOKENS`, drop `TOP_K` to 2, or use a
smaller model. Confirm nothing else is holding the GPU with `nvidia-smi`.

**"No text could be read from this file"**
A scanned or image-only PDF. Run `ocrmypdf input.pdf output.pdf` and upload the
result.

**Answers ignore a document you know you uploaded**
Check `/api/documents` that it is really there and has more than zero chunks,
then use **Passages only** with wording taken from the document itself. If the
passage still does not surface, lower `CHUNK_SIZE`, reset and re-index.

**Retrieval got worse for everything after a change**
Almost always a changed `EMBEDDING_MODEL` without a reset. Old vectors and new
queries live in different spaces. `python scripts/reset_db.py --hard`, then
re-upload.

**A deleted document still appears in answers**
Another copy is in the store under a different `doc_id` — likely uploaded
twice. `curl http://localhost:8000/api/documents` will show both.

**The interface shows "offline"**
The backend is not reachable. Confirm `uvicorn` is running, and that you opened
the page from the server's own address rather than as a local file.

**The answer never streams, it appears all at once**
A buffering proxy. Set `proxy_buffering off;` in nginx, or the equivalent for
your proxy.

**`sqlite3.OperationalError: database is locked`**
Two processes are using the same `data/chroma`. Run one server per store, or
give the second one its own `CHROMA_DIR`.

**Tests fail with a Chroma error after an upgrade**
The on-disk format changed between major versions. Delete the test database
and, for your real store, run `scripts/reset_db.py --hard` and re-index.
