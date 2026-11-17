"""Small, inspectable multi-stage literature analysis pipeline."""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

STOP = set("a an and are as at be been being by can could did do does for from had has have if in into is it its may might more most of on or our should such than that the their them then there these they this those through to was were what when where which while who will with within without would we you your study paper article research results result analysis using use used based also however therefore indicate suggest show shows found findings participants data method methods study studies experimental using literature framework".split())

@dataclass
class Source:
    name: str
    pages: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.pages)

class PlannerAgent:
    """Plans comparable extraction passes; source text never becomes instructions."""
    def plan(self, sources: list[Source]) -> dict[str, Any]:
        return {"stages": ["extract_metadata", "extract_evidence", "synthesize_themes", "critique_citations"],
                "source_count": len(sources), "analysis_scope": "cross-source literature synthesis"}

class EvidenceAgent:
    def extract(self, source: Source) -> dict[str, Any]:
        text = source.text
        sentences = split_sentences(text)
        ranked = sorted(sentences, key=lambda s: sentence_score(s, text), reverse=True)
        abstract = " ".join(ranked[:3])[:900]
        return {"source": source.name, "title": guess_title(source),
                "summary": abstract or "No extractable text was found in this source.",
                "evidence": [citation(s, source, text) for s in ranked[:6]],
                "word_count": len(re.findall(r"\b\w+\b", text)),
                "pages": len(source.pages), "terms": [w for w, _ in keywords(text, 8)]}

class SynthesisAgent:
    def synthesize(self, sources: list[Source], extracted: list[dict[str, Any]]) -> dict[str, Any]:
        all_text = "\n".join(s.text for s in sources)
        terms = keywords(all_text, 10)
        themes = []
        for term, count in terms[:6]:
            supporting = [e["source"] for e in extracted if term in [x.lower() for x in e["terms"]] or term in e["summary"].lower()]
            themes.append({"theme": term, "mentions": count, "sources": supporting})
        return {"overview": f"Analyzed {len(sources)} source(s) containing {sum(e['word_count'] for e in extracted):,} extracted words. The strongest recurring concepts are {', '.join(t['theme'] for t in themes[:4]) or 'not available'}.",
                "themes": themes,
                "gaps": gap_checks(extracted),
                "cross_source_note": "Themes are based on recurring terms and extractive evidence. Interpret them in the context of each study's methods and population."}

class CriticAgent:
    def review(self, result: dict[str, Any], sources: list[Source]) -> dict[str, Any]:
        known = {s.name for s in sources}
        warnings = []
        for paper in result["papers"]:
            for item in paper["evidence"]:
                if item["source"] not in known:
                    warnings.append(f"Unknown citation source: {item['source']}")
                if not item["quote"].strip():
                    warnings.append(f"Empty evidence excerpt in {item['source']}")
        result["quality_review"] = {"citation_check": "passed" if not warnings else "review_required",
                                    "warnings": warnings,
                                    "limitations": ["Offline analysis is extractive and does not establish causality or study quality.",
                                                    "Automated themes may miss synonyms, negation, and field-specific nuance."]}
        return result

def analyze(sources: list[Source]) -> dict[str, Any]:
    if not sources: raise ValueError("Upload at least one PDF, TXT, or Markdown file.")
    plan = PlannerAgent().plan(sources)
    papers = [EvidenceAgent().extract(source) for source in sources]
    synthesis = SynthesisAgent().synthesize(sources, papers)
    result = {"plan": plan, "papers": papers, "synthesis": synthesis}
    return CriticAgent().review(result, sources)

def split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean) if len(s.split()) >= 6]

def tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text) if w.lower() not in STOP]

def keywords(text: str, limit: int) -> list[tuple[str, int]]:
    return Counter(tokens(text)).most_common(limit)

def sentence_score(sentence: str, full_text: str) -> float:
    counts = Counter(tokens(full_text)); terms = tokens(sentence)
    return sum(counts[t] for t in terms) / max(len(terms), 1) + min(len(terms), 30) / 100

def guess_title(source: Source) -> str:
    for line in source.pages[0].splitlines() if source.pages else []:
        line = re.sub(r"\s+", " ", line).strip()
        if 5 <= len(line) <= 180 and len(line.split()) >= 3:
            return line
    return source.name.rsplit(".", 1)[0]

def citation(sentence: str, source: Source, text: str) -> dict[str, Any]:
    index = text.find(sentence)
    before = text[:index] if index >= 0 else ""
    page = min(before.count("\n") + 1, max(len(source.pages), 1))
    # Recover the actual page by checking each extracted page.
    for n, content in enumerate(source.pages, 1):
        if sentence in content:
            page = n; break
    return {"source": source.name, "page": page, "quote": sentence[:420]}

def gap_checks(papers: list[dict[str, Any]]) -> list[str]:
    gaps = []
    if len(papers) < 2: gaps.append("Add more sources to compare populations, methods, and conflicting findings.")
    missing = [p["source"] for p in papers if not p["evidence"]]
    if missing: gaps.append("No usable evidence statements extracted from: " + ", ".join(missing))
    return gaps or ["Check whether the sources represent different populations, methods, or time periods; this demo does not infer field-wide coverage."]

