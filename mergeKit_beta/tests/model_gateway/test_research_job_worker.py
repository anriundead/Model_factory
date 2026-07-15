import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


class _Redis:
    def __init__(self):
        self.acks = []

    def xack(self, stream, group, message_id):
        self.acks.append((stream, group, message_id))


class TestResearchJobWorker(unittest.TestCase):
    @patch("app.model_gateway.research_job_worker.execute_research_job", return_value="completed")
    @patch("app.model_gateway.research_job_worker.claim_research_job", return_value="claimed")
    def test_claimed_job_executes_once_then_acknowledges(self, claim, execute):
        from app.model_gateway.research_job_worker import RESEARCH_CONSUMER_GROUP, consume_research_message
        from app.model_gateway.queue import RESEARCH_STREAM

        redis = _Redis()
        session = object()
        outcome = consume_research_message(session, redis, "1-0", {b"job_id": b"job-1"}, "worker-a")

        self.assertEqual(outcome, "completed")
        claim.assert_called_once_with(session, "job-1", "worker-a")
        execute.assert_called_once()
        self.assertEqual(redis.acks, [(RESEARCH_STREAM, RESEARCH_CONSUMER_GROUP, "1-0")])

    def test_executor_passes_requested_output_format_to_prompt_and_validation(self):
        from app.model_gateway.research_job_worker import execute_research_job

        with tempfile.TemporaryDirectory() as root:
            payload_path = os.path.join(root, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as handle:
                json.dump({"input": "Extract the finding."}, handle)
            job = SimpleNamespace(
                id="job-1", status="running", lease_owner="worker-a", payload_path=payload_path,
                model_service_id="service-1", api_key_id="key-1", file_ids=["file-1"],
                task_type="extract", output_format="json", require_citations=True,
            )
            service = SimpleNamespace(status="running")

            class Session:
                def get(self, _, row_id):
                    return job if row_id == "job-1" else service

            config = SimpleNamespace(MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT=root)
            with patch("app.model_gateway.research_job_worker._get_encoder"), \
                 patch("app.model_gateway.research_job_worker.retrieve_research_evidence", return_value=[{"file_id": "file-1", "locator": {"kind": "page", "value": 1}, "text": "Finding."}]), \
                 patch("app.model_gateway.research_job_worker.build_research_messages", return_value=[{"role": "user", "content": "prompt"}]) as build, \
                 patch("app.model_gateway.research_job_worker.call_research_model", return_value='{"answer":"Finding [S1].","citations":[1]}'), \
                 patch("app.model_gateway.research_job_worker.validate_research_answer", return_value=[1]) as validate, \
                 patch("app.model_gateway.research_job_worker.complete_research_job", return_value="completed"):
                outcome = execute_research_job(Session(), "job-1", "worker-a", config=config)

            self.assertEqual(outcome, "completed")
            self.assertEqual(build.call_args.kwargs["output_format"], "json")
            self.assertEqual(validate.call_args.kwargs["output_format"], "json")


if __name__ == "__main__":
    unittest.main()
