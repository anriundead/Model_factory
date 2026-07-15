import os
import unittest

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class TestGatewayStorageBoundary(unittest.TestCase):
    def test_gateway_models_use_dedicated_bind_without_core_model_foreign_key(self):
        from app.model_gateway.models import ServingModelService

        self.assertEqual(ServingModelService.__bind_key__, "model_gateway")
        self.assertFalse(ServingModelService.__table__.c.model_id.foreign_keys)

    def test_research_records_share_gateway_bind_and_keep_ttl_fields(self):
        from app.model_gateway.models import ResearchChunk, ResearchFile, ResearchJob

        self.assertEqual(ResearchFile.__bind_key__, "model_gateway")
        self.assertEqual(ResearchJob.__bind_key__, "model_gateway")
        self.assertIn("expires_at", ResearchFile.__table__.c)
        self.assertIn("expires_at", ResearchJob.__table__.c)
        self.assertEqual(ResearchChunk.__bind_key__, "model_gateway")
        self.assertIn("locator", ResearchChunk.__table__.c)


if __name__ == "__main__":
    unittest.main()
