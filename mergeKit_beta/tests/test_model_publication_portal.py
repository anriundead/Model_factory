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
        self.assertIn(".sticky-header { margin: 0 0 40px; }", page)
        self.assertIn("#publication-title, #published-assets-title { scroll-margin-top: 116px; }", page)

    def test_publication_validation_dialog_has_accessible_modal_contract(self):
        page = self._read("templates", "model_repo.html")

        self.assertIn('role="dialog"', page)
        self.assertIn('aria-modal="true"', page)
        self.assertIn('aria-labelledby="publication-validate-title"', page)
        self.assertIn('id="publication-validate-title"', page)
        self.assertIn("publicationValidationTrigger", page)
        self.assertIn("publicationTrapFocus", page)
        self.assertIn("event.key === 'Escape'", page)
        self.assertIn("publicationBackground.inert = true", page)
        self.assertIn("publicationBackground.inert = false", page)
        self.assertIn("max-height:calc(100dvh - 24px)", page)

    def test_publication_requests_use_token_generation_and_guarded_storage(self):
        page = self._read("templates", "model_repo.html")

        self.assertIn("PublicationUI.createGenerationGate", page)
        self.assertIn("publicationGeneration.snapshot()", page)
        self.assertIn("publicationGeneration.isCurrent", page)
        self.assertIn("!error.stale && publicationGeneration.isCurrent(snapshot)", page)
        self.assertIn("PublicationUI.createPollScheduler", page)
        self.assertIn("PublicationUI.safeStorage", page)
        self.assertIn("本地任务记录不可用", page)

    def test_publication_terminal_states_have_explicit_styling(self):
        page = self._read("templates", "model_repo.html")

        self.assertIn("PublicationUI.lifecycleForStatus", page)
        self.assertIn("is-terminal-failed", page)
        self.assertIn("is-terminal-canceled", page)
        self.assertIn("is-terminal-unknown", page)
        self.assertIn(".publication-rail span.is-gateway", page)
        self.assertIn("background:#e5e5ea", page)

    def test_gateway_uses_formal_published_model_selector(self):
        page = self._read("templates", "model_gateway", "console.html")
        css = self._read("static", "model_gateway", "console.css")
        script = self._read("static", "model_gateway", "console.js")

        self.assertIn('id="gateway-published-model"', page)
        self.assertNotIn('id="gateway-model-path"', page)
        self.assertIn('id="gateway-published-summary"', page)
        self.assertNotIn("<datalist", page)
        self.assertIn(".gateway-admin-layout {", css)
        self.assertIn("grid-template-columns: minmax(360px, 1fr) minmax(320px, 1fr);", css)
        self.assertIn(".gateway-inline-field {", css)
        self.assertIn("grid-template-columns: minmax(0, 1fr) auto;", css)
        self.assertIn("overflow-wrap: anywhere;", css)
        self.assertIn("/api/model-gateway/admin/publishable-models", script)
        self.assertIn("blocked_reason", script)
        self.assertIn("model_path", script)
        self.assertIn("textContent", script)
        self.assertIn("replaceChildren", script)
        self.assertNotIn("gateway-model-path", script)
        self.assertIn("Promise.allSettled", script)
        self.assertIn("renderServicesResult", script)
        self.assertIn("renderKeysResult", script)
        self.assertIn("renderPublishedModelsResult", script)

    def test_user_research_portal_has_no_formal_model_path_surface(self):
        page = self._read("templates", "model_gateway", "research.html")
        script = self._read("static", "model_gateway", "research.js")

        self.assertNotIn("model_path", page)
        self.assertNotIn("model_path", script)


if __name__ == "__main__":
    unittest.main()
