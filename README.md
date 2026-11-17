# Agentic AI Framework for Literature Analysis

A local, interview-ready demo that turns research PDFs or text files into structured, page-cited evidence and a cross-source synthesis. The pipeline is explicit and inspectable: Planner ? Evidence Extractor ? Synthesizer ? Citation Critic. Uploaded documents are treated as research data, including any imperative text they contain.

## Features

- Local browser UI; no cloud account required
- PDF, TXT, and Markdown uploads (15 MB total request limit)
- Page-level evidence excerpts, per-paper summaries, recurring themes, and research gaps
- Citation integrity review with limitations reported alongside results
- JSON download and Markdown report export
- Offline deterministic mode; PDF support provided by `pypdf`

## Run locally

Requires Python 3.10+. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:8000>. Upload one or more PDF/TXT/MD research sources. The demo analyzes only uploaded materials and keeps them in memory for the request.

Set `PORT` to change the port. Set `HOST=0.0.0.0` only when you intentionally want to expose the local demo to the network.

## Architecture

```text
Browser -> Python HTTP API -> upload parser -> PlannerAgent
                                          -> EvidenceAgent (per source)
                                          -> SynthesisAgent (across sources)
                                          -> CriticAgent (citation checks)
```

The included backend uses Python's standard library plus `pypdf`. Extraction and synthesis are intentionally deterministic and extractive, which makes a live demo work without credentials. It is a prototype, not a substitute for reading the studies or a systematic review protocol.

## API

- `GET /api/health` returns the service and pipeline mode.
- `POST /api/analyze` accepts multipart form data with one or more `files` fields.

## Interview walkthrough

1. Start the server and upload the supplied research article or a pair of papers.
2. Show the pipeline plan, per-paper evidence excerpts with page numbers, themes, gap prompts, and critic limitations.
3. Download the JSON to discuss the stable schema and how each stage can be swapped for specialized agents.
4. Explain that this demo is extractive; a production extension should add retrieval, domain-aware prompts, stronger PDF layout parsing, and human review.

## Known limitations

- Text extraction quality depends on the PDF; scanned image-only PDFs need OCR, which is not bundled.
- Offline summaries use sentence ranking and keyword counts, not semantic interpretation.
- PDF page references are preserved, but tables, formulas, and multi-column reading order may be imperfect.
