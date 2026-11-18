from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from agentic_literature import server
from agentic_literature.pipeline import Source, analyze, cluster_vectors


class PipelineTests(unittest.TestCase):
    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze([], encoder_mode="fallback")

    def test_fixture_provenance_and_excerpt_limits(self):
        corpus = json.loads((Path(__file__).parent.parent / "fixtures" / "papers.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(corpus["papers"]), 10)
        for paper in corpus["papers"]:
            self.assertTrue(paper["source_url"].startswith("https://arxiv.org/abs/"))
            self.assertFalse(paper["full_text_included"])
            self.assertIn("excerpt", paper["text_available"])
            self.assertLessEqual(len(paper["text"].split()), 25)

    def test_explicit_fallback_and_traceable_summaries(self):
        sources = server.load_fixtures()
        result = analyze(sources, encoder_mode="fallback")
        self.assertEqual(result["encoder"]["model_id"], "local-feature-hash-v1")
        self.assertTrue(result["encoder"]["fallback"])
        self.assertEqual(len(result["papers"]), 12)
        self.assertTrue(all(paper["embedding"]["dimensions"] == 128 for paper in result["papers"]))
        self.assertEqual(result["quality_review"]["reference_validity"]["status"], "valid")
        self.assertTrue(all(claim["claim_support_status"] == "directly_supported_by_exact_source_text" for claim in result["quality_review"]["claim_support"]["summary_claims"]))
        self.assertTrue(any(gap["claim_support_status"] == "candidate_question_not_established_by_available_text" for gap in result["synthesis"]["gaps"]))

    def test_small_collection_skips_clustering(self):
        clustered = cluster_vectors([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(clustered["status"], "skipped_too_few_sources")
        self.assertEqual(clustered["groups"], [])

    def test_saved_run_and_report_preserve_encoder_and_refs(self):
        result = analyze(server.load_fixtures(), encoder_mode="fallback")
        with tempfile.TemporaryDirectory() as temp, patch.object(server, "RUNS_DIR", Path(temp)):
            run_id = server.save_run(result)
            restored = server.load_run(run_id)
            report = server.markdown_report(restored)
            self.assertEqual(restored["run_id"], run_id)
            self.assertIn("local-feature-hash-v1", report)
            self.assertIn("Abstract (verbatim excerpt)", report)
            self.assertIn("Reference validity: valid", report)


if __name__ == "__main__":
    unittest.main()
