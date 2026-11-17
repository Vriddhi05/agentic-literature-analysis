# Agentic AI Framework for Literature Analysis

A local, interview-ready demo that turns research PDFs or text files into structured, page-cited evidence and a cross-source synthesis. The pipeline is explicit and inspectable: Planner -> Evidence Extractor -> Local Embeddings -> Clusterer -> Synthesizer -> Citation Critic. Uploaded documents are treated as research data, including any imperative text they contain.

## Features

- Local browser UI; no cloud account required
- PDF, TXT, and Markdown uploads (15 MB total request limit)
- Page-level evidence excerpts, per-paper summaries, recurring themes, and research gaps
- Citation integrity review with limitations reported alongside results
- JSON download and Markdown report export
- Deterministic local feature-hash embeddings and cosine clusters
- Saved runs in `data/runs/`, retained across server restarts
- Optional live scholarly lookup using Crossref
- Optional OpenAI-compatible synthesis when `OPENAI_API_KEY` is configured
- Offline deterministic mode; PDF support provided by `pypdf`

## Run locally

Requires Python 3.10+. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:8000>. Choose **Load demo fixtures** for the two included sample papers, then run the analysis. Each run is saved under `data/runs/` and remains available after restart.

Set `PORT` to change the port. Set `HOST=0.0.0.0` only when you intentionally want to expose the local demo to the network.

## Architecture

```text
Browser -> Python HTTP API -> upload parser -> PlannerAgent
                                          -> EvidenceAgent (per source)
                                          -> SynthesisAgent (across sources)
                                          -> CriticAgent (citation checks)
```

The default backend uses Python's standard library plus `pypdf`. Embeddings use signed feature hashing (not a pretrained semantic model); clusters are connected components over cosine similarity. Extraction and synthesis are deterministic and extractive. For optional model synthesis, set `OPENAI_API_KEY`, and optionally `OPENAI_MODEL` and `OPENAI_BASE_URL`. When enabled, extracted summaries and their source references are sent to that configured endpoint. Without a key the workflow skips LLM synthesis. Crossref search sends the search query to Crossref only when the user submits a query. This is a prototype, not a substitute for reading the studies or a systematic review protocol.

## API

- `GET /api/health` returns the service and pipeline mode.
- `POST /api/analyze` accepts multipart form data with one or more `files` fields and persists the run.
- `GET /api/runs` lists saved analyses; `GET /api/runs/{id}` restores one.
- `GET /api/runs/{id}/report.md` and `/report.json` download reports.
- `GET /api/search?q=...` queries Crossref.

## Interview walkthrough

1. Click **Load demo fixtures** and run analysis.
2. Show each 64-dimensional embedding and the resulting cosine clusters.
3. Open a summary reference and trace it to its source page; show recurring themes and gaps.
4. Download Markdown/JSON and open **Saved runs**.
5. Restart the server, refresh saved runs, and open the previous result.
6. Search Crossref live. If configured, discuss the separate LLM synthesis and its external data boundary; otherwise show its explicit skipped state.

## Known limitations

- Text extraction quality depends on the PDF; scanned image-only PDFs need OCR, which is not bundled.
- Offline summaries use sentence ranking and keyword counts, not semantic interpretation.
- PDF page references are preserved, but tables, formulas, and multi-column reading order may be imperfect.
- Feature-hash embeddings encode lexical overlap, not deep semantic meaning; small fixture sets produce illustrative clusters only.
- Run persistence uses local JSON files and is intended for a single-user demo, not concurrent production workloads.
- Crossref and model calls need network access; model synthesis additionally needs an API key.
