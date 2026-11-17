from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentic_literature import server
from agentic_literature.pipeline import Source, analyze


class PipelineTests(unittest.TestCase):
    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze([])

    def test_fixture_embeddings_clusters_and_source_references(self):
        fixtures = Path(__file__).parent.parent / "fixtures"
        sources = [
            Source(path.name, [path.read_text(encoding="utf-8")])
            for path in sorted(fixtures.glob("*.txt"))
        ]
        result = analyze(sources)
        self.assertEqual([paper["embedding"]["dimensions"] for paper in result["papers"]], [64, 64])
        self.assertEqual(len(result["clusters"]["groups"]), 2)
        self.assertTrue(all(paper["summary_references"] for paper in result["papers"]))
        self.assertTrue(
            all(
                reference["source"] == paper["source"] and reference["page"] == 1
                for paper in result["papers"]
                for reference in paper["summary_references"]
            )
        )

    def test_citation_review_passes(self):
        source = Source(
            "paper.txt",
            ["A study examined learning outcomes with a controlled method. Results suggest learning outcomes improved for participants."],
        )
        result = analyze([source])
        self.assertEqual(result["quality_review"]["citation_check"], "passed")
        self.assertTrue(all(item["page"] == 1 for item in result["papers"][0]["evidence"]))

    def test_saved_run_and_report(self):
        result = analyze([Source("fixture.txt", ["A useful sentence explains evidence in a reproducible study. Results show repeated evidence across studies."])])
        with tempfile.TemporaryDirectory() as temp, patch.object(server, "RUNS_DIR", Path(temp)):
            run_id = server.save_run(result)
            saved = server.load_run(run_id)
            self.assertEqual(saved["run_id"], run_id)
            self.assertIn("Summary references", server.markdown_report(saved))
            self.assertEqual(server.run_summary(saved)["sources"], ["fixture.txt"])


if __name__ == "__main__":
    unittest.main()
