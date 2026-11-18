# Agentic AI Framework for Literature Analysis

Local interview demo for page- or section-cited literature exploration. The standard analysis path uses Sentence Transformers with `sentence-transformers/all-MiniLM-L6-v2`, CPU inference, and semantic document vectors. It also includes extractive synthesis, auditable citations, clustering, saved runs, and live Crossref search.

## Main capabilities

- PDF, TXT, and Markdown uploads (15 MB per request); PDF extraction uses `pypdf`.
- Semantic encoder selection is visible in the UI and every JSON/Markdown report.
- The pretrained model is cached in `data/model_cache/`; one vector is recorded per paper with model ID and dimensions.
- Average-linkage agglomerative clustering uses cosine similarity with threshold `0.48`. Collections smaller than three papers are left unclustered.
- Extractive summaries retain exact source text. References to fixture text identify the abstract excerpt and canonical arXiv URL; uploaded PDFs use page numbers.
- Citation validity and claim support are separate checks. Candidate gap questions include evidence but are explicitly marked as unproven coverage hypotheses unless the source itself states a limitation.
- Runs persist as JSON under `data/runs/` and survive application restarts. Markdown and JSON reports are downloadable.
- Crossref scholarly search is live when network access is available.
- LLM synthesis is unavailable until `OPENAI_API_KEY` is configured. A configured key is validated on first use; invalid keys produce an unavailable status. Source excerpts are sent to the configured OpenAI-compatible endpoint only when enabled.
- If the Sentence Transformer cannot load, the UI shows the load error and offers an explicitly labeled lexical feature-hash fallback. It is never selected silently.

## Start on Windows

Requires Python 3.10+. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PORT = "8000"
python run.py
```

Open <http://127.0.0.1:8000>. On the first semantic analysis, the model is downloaded into `data/model_cache/`; later runs reuse that cache. Set `PORT` to choose another port. Keep the server bound to `127.0.0.1` for a local demonstration.

## Fixture corpus and provenance

`fixtures/papers.json` contains 12 genuine research papers across embeddings, dense retrieval, RAG, scientific claim verification, and evaluation. Each record has the title, cited author string, year, arXiv ID, subtopic, exact source URL, and an abstract excerpt of at most 25 words. The full papers are not bundled; the UI and reports say “Abstract (verbatim excerpt)” and never invent page numbers. See [fixtures/README.md](fixtures/README.md).

## API

- `GET /api/health` reports model selection, load status, cache path, and LLM-key state without exposing credentials.
- `GET /api/fixtures` lists fixture metadata.
- `POST /api/analyze-fixtures` analyzes all 12 curated fixture excerpts.
- `POST /api/analyze` accepts one or more uploaded `files` fields; `?encoder=fallback` selects the explicit fallback.
- `GET /api/runs` lists saved analyses; `GET /api/runs/{id}` restores one.
- `GET /api/runs/{id}/report.md` and `/report.json` download reports.
- `GET /api/search?q=...` queries Crossref.

## Five-minute interview demo

1. Load and analyze the 12-paper fixture; point out the selected model and 384-dimensional vectors.
2. Show the cluster method, threshold, and small-collection behavior.
3. Trace an extractive summary to its linked abstract excerpt; compare reference validity with claim-support status.
4. Explain that candidate gaps are questions, not asserted absences, and inspect the evidence attached to each.
5. Search Crossref, download the report, then restart the server and show the saved run remains available.
6. Show the LLM-unavailable state unless a valid key has been configured. The lexical fallback is available only through explicit selection after a model error.

## Limitations

- Fixture inputs are short abstract excerpts, not full papers. Their provenance and available text are shown per record.
- The summaries are extractive, and the gap candidates are prompts for human review rather than systematic-review conclusions.
- Clusters are illustrative for small collections and should not be interpreted as validated subject taxonomies.
- Saved-run storage is local JSON intended for a single-user demo.
