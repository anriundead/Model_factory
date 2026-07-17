"""Published-model gateway portal page tests."""
import os
import unittest

os.environ["MERGEKIT_CLI_SCRIPT"] = "1"


class ModelGatewayPortalPageTestCase(unittest.TestCase):
    @staticmethod
    def _research_page():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "templates", "model_gateway", "research.html"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _research_script():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "static", "model_gateway", "research.js"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _research_archive_script():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "static", "model_gateway", "research_archive.js"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _research_css():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "static", "model_gateway", "research.css"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _console_page():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "templates", "model_gateway", "console.html"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _console_css():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "static", "model_gateway", "console.css"), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _console_script():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(root, "static", "model_gateway", "console.js"), encoding="utf-8") as handle:
            return handle.read()

    def test_research_page_keeps_admin_controls_out_of_user_workspace(self):
        from flask import Flask
        from app.routes import register_routes

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        app = Flask(
            __name__,
            static_folder=os.path.join(root, "static"),
            template_folder=os.path.join(root, "templates"),
        )
        register_routes(app, state=object(), services=object(), dataset_service=object())

        resp = app.test_client().get("/research")

        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("RESEARCH ARCHIVE", body)
        self.assertIn("research.js", body)
        self.assertNotIn("Admin Token", body)

    def test_research_page_has_no_administrator_navigation(self):
        page = self._research_page()

        self.assertNotIn('href="/model-gateway"', page)
        self.assertNotIn("管理控制台", page)
        self.assertIn('class="research-archive-brand" href="/research"', page)

    def test_research_page_exposes_one_composer_with_tools_and_settings_drawer(self):
        page = self._research_page()

        for token in (
            'id="research-conversation"',
            'id="research-composer-tools"',
            'id="research-add-source-menu"',
            'id="research-url-panel"',
            'id="research-attachment-strip"',
            'id="research-session-settings"',
            'id="research-settings-drawer"',
            'id="research-file-input"',
            'id="research-source-url"',
        ):
            self.assertIn(token, page)
        self.assertNotIn('id="research-mode-switcher"', page)
        self.assertNotIn('id="research-sidebar"', page)

    def test_research_page_versions_composer_static_assets_together(self):
        page = self._research_page()

        self.assertIn('/static/model_gateway/research.css?v=research-command-desk-v1', page)
        self.assertIn('/static/model_gateway/research_archive.js?v=research-command-desk-v1', page)
        self.assertIn('/static/model_gateway/research.js?v=research-command-desk-v1', page)

    def test_research_url_tool_avoids_nested_forms(self):
        page = self._research_page()
        script = self._research_script()

        self.assertIn('<div id="research-url-form"', page)
        self.assertIn('id="research-submit-url"', page)
        self.assertIn('$("research-submit-url").addEventListener("click"', script)

    def test_research_client_declares_bounded_session_and_source_aware_routing(self):
        script = self._research_script()

        for token in (
            'const CONVERSATION_STORAGE_KEY = "mergeneticResearchConversationV1"',
            "const MAX_SESSION_ENTRIES = 60",
            "const MAX_SESSION_BYTES = 256 * 1024",
            "const MAX_MODEL_CONTEXT_MESSAGES = 16",
            "function persistConversation",
            "function restoreConversation",
            "function selectedSourceIds",
            "async function submitSelectedSources",
            '"Idempotency-Key"',
        ):
            self.assertIn(token, script)

    def test_research_client_projects_completed_results_as_text_only_chat_entries(self):
        script = self._research_script()

        self.assertIn("function appendChatEntry", script)
        self.assertIn("entry.textContent", script)
        self.assertIn("function pollResearchTurn", script)

    def test_research_client_uses_existing_openai_chat_endpoint_without_sources(self):
        script = self._research_script()

        self.assertIn("async function submitDirectChat(turn)", script)
        self.assertIn('api("/v1/chat/completions"', script)
        self.assertIn("MAX_MODEL_CONTEXT_MESSAGES", script)
        self.assertNotIn("/api/model-publications", script)
        self.assertNotIn("/api/model-gateway/admin", script)

    def test_research_client_uses_selected_sources_or_direct_chat_without_fallback(self):
        script = self._research_script()

        self.assertIn("function selectedSourceIds()", script)
        self.assertIn("async function submitSelectedSources(turn)", script)
        self.assertIn("require_citations: true", script)
        self.assertIn('"Idempotency-Key": turn.idempotencyKey', script)
        self.assertIn("await submitDirectChat(turn)", script)
        self.assertIn("async function cancelResearchTurn(turnId)", script)
        self.assertIn("资料处理失败，请移除后重新发送", script)

    def test_research_styles_keep_tools_attachments_and_drawer_non_overlapping(self):
        css = self._research_css()

        for token in (
            ".research-add-source-menu",
            ".research-attachment-strip",
            ".research-settings-drawer",
            ".research-pending-entry",
            "min-width: 0",
            "overflow-wrap: anywhere",
            ":focus-visible",
            "@media (prefers-reduced-motion: reduce)",
        ):
            self.assertIn(token, css)

    def test_conversation_client_keeps_existing_secure_file_and_job_contracts(self):
        script = self._research_script()

        for token in (
            '"/api/model-gateway/files"',
            '"/api/model-gateway/sources/url"',
            '"/api/model-gateway/research/jobs"',
            '"/v1/chat/completions"',
            "async function cancelResearchTurn(turnId)",
            "require_citations: true",
            "资料处理失败，请移除后重新发送",
            '"processing"',
        ):
            self.assertIn(token, script)

    def test_research_client_uses_concise_chinese_source_status_labels(self):
        script = self._research_script()

        for token in (
            "function sourceStatusLabel(status)",
            'processing: "处理中"',
            'ready: "已就绪"',
            "status.textContent = sourceStatusLabel(file.status)",
        ):
            self.assertIn(token, script)

    def test_research_archive_storage_is_bounded_and_secret_free(self):
        script = self._research_archive_script()

        for token in (
            'const DB_NAME = "mergeneticResearchArchiveV1"',
            'const ACTIVE_SESSION_KEY = "mergeneticResearchArchiveActiveSessionId"',
            "const MAX_SESSIONS = 20",
            "const MAX_TURNS = 60",
            "const MAX_SESSION_BYTES = 256 * 1024",
            "function sanitizeSession(record)",
            "async function migrateLegacy(legacySnapshot)",
            "async function invalidateSources()",
            "window.MergeneticResearchArchive",
            "map(Number).filter(Number.isInteger)",
        ):
            self.assertIn(token, script)
        self.assertNotIn("Authorization", script)
        self.assertNotIn("mergeneticResearchKey", script)

    def test_research_page_exposes_archive_navigation_without_admin_controls(self):
        page = self._research_page()

        for token in (
            'id="research-session-rail"',
            'id="research-new-session"',
            'id="research-session-search"',
            'id="research-session-list"',
            'id="research-session-drawer-toggle"',
            'id="research-active-session-title"',
            'id="research-session-limit-dialog"',
            'id="research-session-delete-dialog"',
            'research_archive.js?v=research-command-desk-v1',
            'research.js?v=research-command-desk-v1',
        ):
            self.assertIn(token, page)
        self.assertNotIn('href="/model-gateway"', page)

    def test_research_client_uses_archive_without_persisting_credentials_or_bypassing_sources(self):
        script = self._research_script()

        for token in (
            "MergeneticResearchArchive.migrateLegacy",
            "MergeneticResearchArchive.save",
            "MergeneticResearchArchive.invalidateSources",
            "async function switchArchiveSession(sessionId)",
            "async function createArchiveSession()",
            "async function revalidateArchiveSources()",
            "async function persistActiveArchiveSession()",
            "资料已过期，无法用于新的研究请求",
            "require_citations: true",
            'api("/v1/chat/completions"',
        ):
            self.assertIn(token, script)
        self.assertNotIn("Authorization: state", script)

    def test_research_archive_styles_define_rail_signal_and_reduced_motion_boundaries(self):
        css = self._research_css()

        for token in (
            ".research-session-rail",
            ".research-stage",
            ".research-session-row.is-active",
            ".research-signal-cut",
            "--rcd-amber: #E6B422",
            "grid-template-columns: 286px minmax(0, 1fr)",
            "@media (max-width: 1080px)",
            "@media (prefers-reduced-motion: reduce)",
            "min-width: 0",
            "overflow-wrap: anywhere",
        ):
            self.assertIn(token, css)
        self.assertNotIn("linear-gradient", css)

    def test_research_archive_motion_is_reduced_motion_safe_and_cancellable(self):
        script = self._research_script()

        for token in (
            'window.matchMedia("(prefers-reduced-motion: reduce)")',
            "function runArchiveEntranceMotion()",
            "function runSessionSwitchMotion()",
            "function killResearchMotion()",
            "window.gsap.timeline",
            "autoAlpha",
            'overwrite: "auto"',
        ):
            self.assertIn(token, script)

    def test_research_archive_uses_named_rename_dialog_instead_of_browser_prompt(self):
        page = self._research_page()
        script = self._research_script()

        self.assertIn('id="research-session-rename-dialog"', page)
        self.assertIn("function openRenameSessionDialog(sessionId)", script)
        self.assertNotIn("window.prompt", script)

    def test_research_command_desk_uses_cold_material_tokens_without_prohibited_effects(self):
        css = self._research_css()

        for token in (
            "--rcd-void: #171A1F",
            "--rcd-paper: #FAFBFB",
            "--rcd-amber: #E6B422",
            ".research-command-strip",
            ".research-evidence-index",
            ".research-composer-deck",
            ".research-signal-cut",
            "@media (prefers-reduced-motion: reduce)",
        ):
            self.assertIn(token, css)
        for forbidden in ("linear-gradient", "radial-gradient", "backdrop-filter", "animation: infinite"):
            self.assertNotIn(forbidden, css)

    def test_research_command_desk_keeps_one_composer_and_real_state_markers(self):
        page = self._research_page()
        script = self._research_script()

        for token in (
            "research-command-strip",
            'id="research-local-archive-state"',
            'class="research-composer research-composer-deck"',
        ):
            self.assertIn(token, page)
        for token in (
            "research-evidence-index",
            "data-research-empty-action",
            "research-local-archive-state",
            'api("/v1/chat/completions"',
            "require_citations: true",
        ):
            self.assertIn(token, script)
        self.assertNotIn('href="/model-gateway"', page)

    def test_model_gateway_page_renders_portal_shell(self):
        from flask import Flask
        from app.routes import register_routes

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        app = Flask(
            __name__,
            static_folder=os.path.join(root, "static"),
            template_folder=os.path.join(root, "templates"),
        )
        register_routes(app, state=object(), services=object(), dataset_service=object())

        resp = app.test_client().get("/model-gateway")

        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("模型访问网关", body)
        self.assertIn("model_gateway/console.css", body)
        self.assertIn("model_gateway/console.js", body)
        self.assertIn("gateway-admin-panel", body)
        self.assertIn("gateway-playground", body)
        self.assertIn("gateway-cancel-request-form", body)
        self.assertIn("/v1/requests/", body)
        self.assertIn("gateway-check-request-status", body)
        self.assertIn("gateway-request-status-output", body)

    def test_console_keeps_operational_controls_and_visible_delivery_guidance(self):
        page = self._console_page()

        for token in (
            "gateway-create-service-form",
            "gateway-create-key-form",
            "gateway-chat-form",
            "gateway-cancel-request-form",
            "gateway-services-list",
            "gateway-api-keys-list",
            "gateway-copy-curl",
            "/v1/chat/completions",
            "/v1/requests/",
            "选择正式资产",
            "管理员手动启动",
            "API Key 验证",
        ):
            self.assertIn(token, page)

    def test_console_styles_protect_layout_and_interaction_feedback(self):
        css = self._console_css()

        for token in (
            ".gateway-publication-flow",
            ":focus-visible",
            ":active",
            "minmax(0,",
            "min-width: 0",
            "overflow-wrap: anywhere",
            "@media (prefers-reduced-motion: reduce)",
        ):
            self.assertIn(token, css)

    def test_console_uses_existing_reduced_motion_safe_gsap_entry(self):
        script = self._console_script()

        self.assertIn('window.matchMedia("(prefers-reduced-motion: reduce)")', script)
        self.assertIn("window.gsap.matchMedia()", script)
        self.assertIn("autoAlpha", script)
        self.assertIn("function runEntranceAnimation", script)


if __name__ == "__main__":
    unittest.main()
