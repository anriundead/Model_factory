import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import requests


class _Redis:
    def __init__(self):
        self.acks = []

    def xack(self, stream, group, message_id):
        self.acks.append((stream, group, message_id))


class TestResearchJobWorker(unittest.TestCase):
    def test_executor_pauses_job_and_marks_service_offline_when_model_connection_fails(self):
        from app.model_gateway.research_job_worker import execute_research_job

        with tempfile.TemporaryDirectory() as root:
            payload_path = os.path.join(root, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as handle:
                json.dump({"input": "Summarize the evidence."}, handle)
            job = SimpleNamespace(
                id="job-1", status="running", lease_owner="worker-a", payload_path=payload_path,
                model_service_id="service-1", api_key_id="key-1", file_ids=["file-1"],
                task_type="summary", output_format="markdown", require_citations=True,
            )
            service = SimpleNamespace(status="running", vllm_pid=44, vllm_pgid=44, last_error=None)

            class Session:
                def get(self, _, row_id):
                    return job if row_id == "job-1" else service

                def add(self, _):
                    pass

                def commit(self):
                    pass

            config = SimpleNamespace(MERGEKIT_MODEL_GATEWAY_RESEARCH_ROOT=root)
            with patch("app.model_gateway.research_job_worker._get_encoder"), \
                 patch("app.model_gateway.research_job_worker.retrieve_research_evidence", return_value=[{"file_id": "file-1", "locator": {"kind": "page", "value": 1}, "text": "Finding."}]), \
                 patch("app.model_gateway.research_job_worker.call_research_model", side_effect=requests.ConnectionError("offline")), \
                 patch("app.model_gateway.research_job_worker.pause_research_job", return_value="paused_model_offline") as pause:
                outcome = execute_research_job(Session(), "job-1", "worker-a", config=config)

            self.assertEqual(outcome, "paused_model_offline")
            self.assertEqual(service.status, "stopped")
            self.assertIsNone(service.vllm_pid)
            self.assertIsNone(service.vllm_pgid)
            self.assertEqual(service.last_error, "model_runtime_unavailable")
            pause.assert_called_once_with(unittest.mock.ANY, "job-1", "worker-a", "model_runtime_unavailable")

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
                 patch(
                     "app.model_gateway.research_job_worker.call_research_model",
                     return_value=('{"answer":"Finding [S1].","citations":[1]}', {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}),
                 ), \
                 patch("app.model_gateway.research_job_worker.validate_research_answer", return_value=[1]) as validate, \
                 patch("app.model_gateway.research_job_worker.complete_research_job", return_value="completed") as complete:
                outcome = execute_research_job(Session(), "job-1", "worker-a", config=config)

            self.assertEqual(outcome, "completed")
            self.assertEqual(build.call_args.kwargs["output_format"], "json")
            self.assertEqual(validate.call_args.kwargs["output_format"], "json")
            self.assertEqual(
                complete.call_args.kwargs["usage"],
                {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            )


if __name__ == "__main__":
    unittest.main()
