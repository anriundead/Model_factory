import unittest
from types import SimpleNamespace


class TestResearchCitationContract(unittest.TestCase):
    def test_prompt_numbers_evidence_and_accepts_only_known_citations(self):
        from app.model_gateway.research_execution import build_research_messages, validate_research_answer

        evidence = [
            {"file_id": "file-a", "locator": {"kind": "page", "value": 2}, "text": "Verified finding."},
            {"file_id": "file-b", "locator": {"kind": "slide", "value": 4}, "text": "Comparison evidence."},
        ]
        messages = build_research_messages("summary", "Summarize the findings.", evidence)

        self.assertIn("[S1] file-a page 2", messages[1]["content"])
        self.assertIn("[S2] file-b slide 4", messages[1]["content"])
        self.assertEqual(validate_research_answer("The finding is verified [S1].", evidence, True), [1])
        with self.assertRaisesRegex(ValueError, "citation_required"):
            validate_research_answer("The finding is verified.", evidence, True)
        with self.assertRaisesRegex(ValueError, "invalid_citation"):
            validate_research_answer("The finding is verified [S3].", evidence, True)

    def test_internal_model_call_uses_loopback_key_and_returns_content(self):
        from app.model_gateway.research_execution import call_research_model

        captured = {}

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "Verified [S1]."}}]}

        def post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()

        service = SimpleNamespace(vllm_host="127.0.0.1", vllm_port=18001, internal_api_key="internal", served_model_name="qwen")
        answer = call_research_model(service, [{"role": "user", "content": "question"}], post=post)

        self.assertEqual(answer, "Verified [S1].")
        self.assertEqual(captured["url"], "http://127.0.0.1:18001/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer internal")
        self.assertEqual(captured["json"]["model"], "qwen")
        self.assertFalse(captured["json"]["stream"])

    def test_json_output_requires_a_json_object_with_matching_citations(self):
        from app.model_gateway.research_execution import build_research_messages, validate_research_answer

        evidence = [{"file_id": "file-a", "locator": {"kind": "page", "value": 2}, "text": "Verified finding."}]
        messages = build_research_messages("extract", "Extract the finding.", evidence, output_format="json")

        self.assertIn('valid JSON object with "answer" and "citations"', messages[0]["content"])
        self.assertEqual(
            validate_research_answer('{"answer":"Verified [S1].","citations":[1]}', evidence, True, output_format="json"),
            [1],
        )
        with self.assertRaisesRegex(ValueError, "invalid_json_output"):
            validate_research_answer("Verified [S1].", evidence, True, output_format="json")
        with self.assertRaisesRegex(ValueError, "citation_mismatch"):
            validate_research_answer('{"answer":"Verified [S1].","citations":[]}', evidence, True, output_format="json")


if __name__ == "__main__":
    unittest.main()
