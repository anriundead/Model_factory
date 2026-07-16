"""Administrator publication portal DOM contracts."""
import os
import unittest


class ModelPublicationPortalTestCase(unittest.TestCase):
    @staticmethod
    def _read(*parts):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(root, *parts), encoding="utf-8") as handle:
            return handle.read()

    def test_model_repository_exposes_publication_controls(self):
        page = self._read("templates", "model_repo.html")

        self.assertIn('id="publication-create-form"', page)
        self.assertIn('id="publication-status-list"', page)
        self.assertIn("/api/model-publications", page)

    def test_gateway_uses_formal_published_model_selector(self):
        page = self._read("templates", "model_gateway", "console.html")
        script = self._read("static", "model_gateway", "console.js")

        self.assertIn('id="gateway-published-model"', page)
        self.assertNotIn('id="gateway-model-path"', page)
        self.assertIn("/api/model-gateway/admin/publishable-models", script)
        self.assertIn("blocked_reason", script)


if __name__ == "__main__":
    unittest.main()
