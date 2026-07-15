import os
import unittest

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class TestResearchQueuePayload(unittest.TestCase):
    def test_enqueue_only_emits_job_identity_fields(self):
        from app.model_gateway.queue import enqueue_research_job

        captured = {}

        class FakeRedis:
            def xadd(self, stream, fields):
                captured["stream"] = stream
                captured["fields"] = fields
                return b"1-0"

        enqueue_research_job(FakeRedis(), "job-1", "svc-1", "document_qa")

        self.assertEqual(captured["stream"], "model_gateway:research")
        self.assertEqual(captured["fields"], {"job_id": "job-1", "service_id": "svc-1", "task_type": "document_qa"})

    def test_file_queue_emits_only_file_identity(self):
        from app.model_gateway.queue import FILE_STREAM, enqueue_research_file

        captured = {}

        class FakeRedis:
            def xadd(self, stream, fields):
                captured["stream"] = stream
                captured["fields"] = fields
                return b"2-0"

        enqueue_research_file(FakeRedis(), "file-1")

        self.assertEqual(captured["stream"], FILE_STREAM)
        self.assertEqual(captured["fields"], {"file_id": "file-1"})


if __name__ == "__main__":
    unittest.main()
