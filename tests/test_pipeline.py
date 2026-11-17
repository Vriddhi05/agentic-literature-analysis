import unittest
from agentic_literature.pipeline import Source, analyze

class PipelineTests(unittest.TestCase):
    def test_page_citations_and_review(self):
        source=Source("paper.txt", ["A study examined learning outcomes with a controlled method. Results suggest learning outcomes improved for participants."])
        result=analyze([source])
        self.assertEqual(result["plan"]["source_count"], 1)
        self.assertEqual(result["quality_review"]["citation_check"], "passed")
        self.assertTrue(all(x["page"] == 1 for x in result["papers"][0]["evidence"]))
    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError): analyze([])
    def test_multi_source_theme(self):
        data=analyze([Source("a.txt",["Research methods shape results in this study. The results describe the research methods."]),Source("b.txt",["Research methods guide results across this study. The research shows repeated methods."])])
        self.assertEqual(data["plan"]["source_count"],2)
        self.assertTrue(data["synthesis"]["themes"])

if __name__ == "__main__": unittest.main()
