# Conversational Research Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the multi-mode research workspace with one conversational interface where composer attachments automatically select direct chat or citation-backed research while remaining reusable for the current browser session.

**Architecture:** Reuse the existing file upload, URL import, status polling, research-job, cancellation and direct-chat APIs. `research.js` owns a versioned bounded session snapshot, renders the conversation and attachment strip, and decides one execution path at send time. The Flask backend, queues, database schema, model runtime and GPU behavior remain unchanged.

**Tech Stack:** Flask template, vanilla JavaScript, vanilla CSS, existing GSAP CDN, existing Remix Icon CDN, Python `unittest`, Node syntax checking, Firefox WebDriver screenshot/layout checks, Docker Compose.

## Execution Status (2026-07-14)

Tasks 1-5 were implemented and verified by
`ACCEPTANCE_20260714_CONVERSATIONAL_COMPOSER.md`. That record also identifies
the intentionally deferred live-model exercise and the accepted-state rollback
archive for future changes.

## Global Constraints

- Do not alter `/api/model-gateway/files`, `/api/model-gateway/sources/url`, `/api/model-gateway/research/jobs`, `/v1/chat/completions`, queue, parser, TTL, SSRF, ClamAV, retrieval, citation, cancellation, database, model service, vLLM, Ray, Compose or GPU behavior.
- The user portal has one timeline and one composer. Remove `chat`, `research` and `focus` mode controls, permanent source sidebar, mode-local storage and mode transition code.
- The composer routes to direct chat only when its selected source snapshot is empty. If any source is selected, it creates a citation-required research job; it never silently falls back to uncited direct chat after a source failure.
- Attachments are local-session selections. Removing an attachment only removes its local session metadata and selection; the backend retains the owned source until its existing TTL cleanup.
- Store only source IDs/display metadata, selected state, selected model, user/assistant/result text, citation locators, job IDs and idempotency keys in `sessionStorage` under `mergeneticResearchConversationV1`. Never persist an API Key beyond its existing session/local policy, source body, parsed text, raw response payload or a new server-side history.
- Safety values: retain at most `60` rendered entries and `256 KiB` of serialized conversation snapshot; send at most the latest `16` model-context messages to `/v1/chat/completions`. These bounds preserve a useful working session without turning browser storage or model prompt length into unbounded state.
- Pending source-backed turns receive a generated idempotency key and start exactly once. Reuse the same key across reloads until terminal status. A direct-chat request interrupted by reload is shown as interrupted and is not replayed.
- Explicitly clearing or changing an API Key clears local source metadata, selections and pending job references before another Key can use the page. It does not delete backend data or local message entries.
- The toolbar contains exactly upload, public URL import and session materials. Settings drawer contains API Key, model selection, trusted-device Key choice and full attachment management.
- Use native HTML/CSS and existing dependencies. No framework, component library, asset download, hidden drawer navigation, continuous animation, visual clone or GPU/model test.
- All elements must use `min-width: 0`, wrapping and responsive layout so no attachment, toolbar, drawer, result or action overlaps. GSAP may animate only `transform` and `opacity`; reduced motion uses no positional animation.
- Do not commit or reset the shared dirty worktree.

## File Responsibilities

| File | Responsibility |
| --- | --- |
| `templates/model_gateway/research.html` | Single-timeline shell, compact header settings trigger, attachment strip, composer tools, URL panel, settings drawer and accessibility labels. |
| `static/model_gateway/research.css` | Responsive conversation layout, normal-flow tools, attachment chips, drawer, pending/terminal message states and interaction feedback. |
| `static/model_gateway/research.js` | Versioned session snapshot, safe DOM rendering, source selection, pending-turn resume, automatic execution routing and existing API calls. |
| `tests/model_gateway/test_gateway_portal.py` | Template/CSS/JavaScript static contracts for the new UI boundary and session/execution invariants. |
| `docs/model_gateway/CONVERSATIONAL_RESEARCH_COMPOSER_UX.md` | Approved product design; update only if implementation reveals a verified design correction. |
| `docs/model_gateway/ACCEPTANCE_20260714_CONVERSATIONAL_COMPOSER.md` | Fresh validation outcomes, screenshots, session recovery limits, GPU snapshot and rollback evidence. |

## Task 1: Lock the New Portal and Routing Contracts

**Files:**
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Produces:** Failing tests defining the single-composer shell, source-aware routing, session snapshot boundary and no-regression backend endpoint expectations.

- [ ] **Step 1: Write failing template-contract tests**

Add these tests beside the existing research portal tests:

```python
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

def test_research_client_declares_bounded_session_and_source_aware_routing(self):
    script = self._research_script()
    for token in (
        'const CONVERSATION_STORAGE_KEY = "mergeneticResearchConversationV1"',
        'const MAX_SESSION_ENTRIES = 60',
        'const MAX_SESSION_BYTES = 256 * 1024',
        'const MAX_MODEL_CONTEXT_MESSAGES = 16',
        'function persistConversation',
        'function restoreConversation',
        'function selectedSourceIds',
        'async function submitSelectedSources',
        '"Idempotency-Key"',
    ):
        self.assertIn(token, script)
```

- [ ] **Step 2: Verify the red state**

Run:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest \
  tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_page_exposes_one_composer_with_tools_and_settings_drawer \
  tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_client_declares_bounded_session_and_source_aware_routing
```

Expected: both tests fail because the current page has mode controls/sidebar and no conversation snapshot/routing helpers.

- [ ] **Step 3: Add CSS contract test before CSS implementation**

```python
def test_research_styles_keep_tools_attachments_and_drawer_non_overlapping(self):
    css = self._research_css()
    for token in (
        '.research-add-source-menu',
        '.research-attachment-strip',
        '.research-settings-drawer',
        '.research-pending-entry',
        'min-width: 0',
        'overflow-wrap: anywhere',
        ':focus-visible',
        '@media (prefers-reduced-motion: reduce)',
    ):
        self.assertIn(token, css)
```

- [ ] **Step 4: Verify the CSS red state**

Run the new CSS test alone. Expected: fail because the new composer components and overflow rules do not exist.

## Task 2: Replace the Research Shell with a Single Conversation Surface

**Files:**
- Modify: `templates/model_gateway/research.html`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** Task 1 template contract.

**Produces:** Stable DOM IDs for the timeline, attachment strip, right-side tools, URL panel and settings drawer. Existing backend API endpoints are unchanged.

- [ ] **Step 1: Implement the compact header and drawer shell**

Keep the `/research` brand link. Replace the current header status with:

```html
<button id="research-session-settings" class="research-header-action" type="button" aria-expanded="false" aria-controls="research-settings-drawer">
  <i class="ri-equalizer-line" aria-hidden="true"></i><span>会话设置</span>
</button>
```

Add an initially hidden drawer and overlay after `<main>`:

```html
<div id="research-settings-overlay" class="research-settings-overlay" hidden></div>
<aside id="research-settings-drawer" class="research-settings-drawer" aria-label="会话设置" aria-hidden="true" hidden>
  <div class="research-drawer-head"><h2>会话设置</h2><button id="research-close-settings" class="research-icon-button" type="button" aria-label="关闭会话设置"><i class="ri-close-line"></i></button></div>
  <section class="research-drawer-section">
    <label class="research-field"><span>API Key</span><input id="research-api-key" type="password" autocomplete="off" placeholder="mk_live_..."></label>
    <label class="research-remember"><input id="research-remember-key" type="checkbox"><span>在此可信设备保存</span></label>
    <button id="research-load-models" class="research-secondary-button" type="button"><i class="ri-radar-line"></i>连接模型</button>
    <label class="research-select"><span>模型</span><select id="research-model" required><option value="">连接 API Key 后选择</option></select></label>
  </section>
  <section class="research-drawer-section">
    <div class="research-section-title"><span>本会话资料</span><span id="research-source-count">0 / 10</span></div>
  <div id="research-source-list" class="research-source-list" aria-live="polite" tabindex="-1"></div>
  </section>
</aside>
```

Move existing `research-api-key`, `research-remember-key`, `research-load-models`, `research-model` and `research-source-list` into this drawer. Keep hidden `research-file-input` and the existing URL endpoint behavior in the composer tool surfaces, preserving their API behavior. Remove the permanent `research-sidebar`.

- [ ] **Step 2: Implement the one-timeline workspace**

Replace the mode header, `research-result` and mode-specific timeline with:

```html
<main id="research-conversation" class="research-conversation" aria-live="polite">
  <section id="research-chat-timeline" class="research-chat-timeline" aria-label="对话记录">
    <p class="research-chat-empty">提出问题，或在发送前添加资料。</p>
  </section>
  <form id="research-job-form" class="research-composer">
    <div id="research-attachment-strip" class="research-attachment-strip" hidden aria-label="本次使用的资料"></div>
    <div id="research-add-source-menu" class="research-add-source-menu" hidden>
      <label for="research-file-input" class="research-tool-action"><i class="ri-file-upload-line"></i><span>上传文件</span></label>
      <button id="research-open-url-panel" class="research-tool-action" type="button"><i class="ri-link"></i><span>导入链接</span></button>
      <button id="research-open-materials" class="research-tool-action" type="button"><i class="ri-folder-open-line"></i><span>查看资料</span></button>
    </div>
    <div id="research-url-panel" class="research-url-panel" hidden>
      <div id="research-url-form" class="research-url-form">
        <input id="research-source-url" type="url" placeholder="公开 HTML 或 PDF URL">
        <button id="research-submit-url" type="button" class="research-icon-button" title="导入链接"><i class="ri-link"></i></button>
      </div>
    </div>
    <textarea id="research-question" rows="4" required placeholder="输入问题，或添加资料后进行研究。"></textarea>
    <div class="research-composer-footer">
      <span id="research-hint">无资料时直接对话；添加资料后自动提供引用。</span>
      <div class="research-composer-actions">
        <button id="research-toggle-tools" class="research-icon-button" type="button" aria-expanded="false" aria-controls="research-add-source-menu" title="添加资料"><i class="ri-add-line"></i></button>
        <button class="research-primary-button" type="submit"><i class="ri-arrow-up-line"></i><span>发送</span></button>
      </div>
    </div>
  </form>
</main>
```

Create cancellation controls dynamically in pending research timeline entries with `data-cancel-turn="<turn-id>"`; do not retain the obsolete singleton `research-cancel-job`. Set the file input to one-file-at-a-time, preserving the current backend limit and avoiding unsupported multi-upload behavior.

- [ ] **Step 3: Verify template green state**

Run all portal tests. Expected: Task 1 template test passes; old three-mode tests will be intentionally updated in Task 4, so record their expected failures without changing unrelated tests yet.

## Task 3: Implement Session-Safe Conversation State and Automatic Routing

**Files:**
- Modify: `static/model_gateway/research.js`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** Task 2 DOM IDs and existing APIs `GET /api/model-gateway/files/<id>`, `POST /api/model-gateway/files`, `POST /api/model-gateway/sources/url`, `POST /api/model-gateway/research/jobs`, `GET/POST /api/model-gateway/research/jobs/<id>[/cancel]`, `/v1/models` and `/v1/chat/completions`.

**Produces:** `persistConversation()`, `restoreConversation()`, `selectedSourceIds()`, `submitSelectedSources(turn)`, `resumePendingTurns()` and safe timeline rendering.

- [ ] **Step 1: Replace obsolete mode tests with failing conversation-state tests**

Delete tests that require `WORKSPACE_STORAGE_KEY`, `WORKSPACE_MODES`, `setWorkspaceMode`, three mode IDs and `research-mode-transitioning`. Add:

```python
def test_research_client_persists_only_bounded_session_safe_fields(self):
    script = self._research_script()
    self.assertIn('const CONVERSATION_STORAGE_KEY = "mergeneticResearchConversationV1"', script)
    self.assertIn('const MAX_SESSION_ENTRIES = 60', script)
    self.assertIn('const MAX_SESSION_BYTES = 256 * 1024', script)
    self.assertIn('sessionStorage.setItem(CONVERSATION_STORAGE_KEY', script)
    self.assertIn('sessionStorage.removeItem(CONVERSATION_STORAGE_KEY)', script)
    self.assertNotIn('localStorage.setItem(CONVERSATION_STORAGE_KEY', script)

def test_research_client_uses_selected_sources_or_direct_chat_without_fallback(self):
    script = self._research_script()
    self.assertIn('function selectedSourceIds()', script)
    self.assertIn('async function submitSelectedSources(turn)', script)
    self.assertIn('require_citations: true', script)
    self.assertIn('"Idempotency-Key": turn.idempotencyKey', script)
    self.assertIn('await submitDirectChat(turn)', script)
    self.assertIn('async function cancelResearchTurn(turnId)', script)
    self.assertIn('资料处理失败，请移除后重新发送', script)
```

- [ ] **Step 2: Verify red state**

Run the two tests. Expected: fail because session snapshot and unified routing do not exist.

- [ ] **Step 3: Define the bounded state model**

Replace mode state with:

```javascript
const CONVERSATION_STORAGE_KEY = "mergeneticResearchConversationV1";
const MAX_SESSION_ENTRIES = 60;
const MAX_SESSION_BYTES = 256 * 1024;
const MAX_MODEL_CONTEXT_MESSAGES = 16;
const state = {
    key: sessionStorage.getItem("mergeneticResearchKey") || localStorage.getItem("mergeneticResearchKey") || "",
    files: [],
    selectedFileIds: new Set(),
    turns: [],
    activeJobIds: new Set(),
    fileTimer: null,
    jobTimers: new Map(),
    lastSettingsTrigger: null,
};
```

Each `turn` is a plain serializable object:

```javascript
{
  id: crypto.randomUUID(),
  kind: "user" | "assistant" | "research_pending" | "research_result" | "error",
  content: "...",
  fileIds: ["..."],
  jobId: null,
  idempotencyKey: null,
  citations: [],
  sources: [],
  status: "ready" | "preparing" | "queued" | "running" | "completed" | "failed" | "canceled"
}
```

`persistConversation()` serializes `{version: 1, files, selectedFileIds, turns, selectedModel}` after removing terminal entries beyond 60. If JSON exceeds 256 KiB, remove oldest completed assistant/result turns until it fits. If a single retained record still exceeds the limit, store a short error/status entry instead. It must never write `state.key`.

`restoreConversation()` parses only version `1`, validates arrays/string IDs, removes invalid records, restores the model option after models load, and clears the storage key on malformed JSON. It calls `rehydrateFiles()` when `state.key` exists; that function requests each stored file ID, removes 404/expired sources locally, and restarts `pollFiles()` for pending sources.

`clearSourceSessionState()` clears `files`, `selectedFileIds`, `activeJobIds` and pending/research turn references, then persists. Call it before an explicit API Key clear and before accepting a different nonempty API Key in `loadModels()`. It does not delete server files or erase rendered local conversation text.

- [ ] **Step 4: Render files and attachment strip safely**

Keep DOM construction through `document.createElement` and `textContent`. `renderAttachments()` shows each selected file as a chip with name, visible status text and `data-remove-source`. A compact overflow chip may show `+N` after three visible sources, but every source remains accessible in the drawer. Ready files default into `selectedFileIds`; user removal deletes local metadata and does not call a delete endpoint.

- [ ] **Step 5: Implement deterministic submit routing**

`createTurnFromComposer()` snapshots the question, model and `selectedSourceIds()` before clearing the textarea. It appends and persists the user turn.

- With no source IDs, `submitDirectChat(turn)` sends only the latest 16 user/assistant contents from terminal direct/research turns to `/v1/chat/completions`. It appends a terminal assistant/error turn. A page reload marks an in-flight direct turn as `failed` with `页面已刷新，请重新发送` and never replays it.
- With ready source IDs, `submitSelectedSources(turn)` sets `turn.idempotencyKey = crypto.randomUUID()` if absent, posts the existing research payload with `require_citations: true` and header `"Idempotency-Key": turn.idempotencyKey`, stores `jobId`, then calls `pollResearchTurn(turn.id)`.
- With any pending source, set the turn to `preparing`, show `正在准备资料`, persist it and let `resumePendingTurns()` start it only when every snapshot source is `ready`.
- With a failed/missing/canceled selected source, make the turn terminal `failed` with exactly `资料处理失败，请移除后重新发送`; do not call direct chat.

`pollResearchTurn(turnId)` gets the existing job status, updates the same timeline turn, includes answer/citations/sources on completion, shows a `data-cancel-turn` button only for a cancelable job, and removes its timer on every terminal state. `cancelResearchTurn(turnId)` uses the existing job cancel endpoint for that turn only. Both functions must reuse `jobId` after reload and never issue a second `POST` for that turn.

- [ ] **Step 6: Verify JavaScript green state**

Run:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal
node --check mergeKit_beta/static/model_gateway/research.js
```

Expected: portal contract tests and syntax check pass.

## Task 4: Build Accessible Tools, Drawer and Responsive Conversation Layout

**Files:**
- Modify: `static/model_gateway/research.css`
- Modify: `static/model_gateway/research.js`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** Task 2 IDs and Task 3 rendering/state functions.

**Produces:** Normal-flow composer tool surface, attachment strip, keyboard-safe drawer and responsive desktop/mobile layout.

- [ ] **Step 1: Implement tool and drawer behavior in JavaScript**

Add `setToolsOpen(open)` and `setSettingsOpen(open, trigger)`:

```javascript
function setToolsOpen(open) {
    $("research-add-source-menu").hidden = !open;
    $("research-toggle-tools").setAttribute("aria-expanded", String(open));
    if (!open) $("research-url-panel").hidden = true;
}

function setSettingsOpen(open, trigger) {
    const drawer = $("research-settings-drawer");
    const overlay = $("research-settings-overlay");
    if (open) state.lastSettingsTrigger = trigger || document.activeElement;
    drawer.hidden = !open;
    overlay.hidden = !open;
    drawer.setAttribute("aria-hidden", String(!open));
    $("research-session-settings").setAttribute("aria-expanded", String(open));
    if (open) $("research-close-settings").focus();
    else state.lastSettingsTrigger?.focus();
}
```

Escape closes the tools first, then the drawer. Overlay click closes the drawer. The URL panel opens only from the link action and focuses `research-source-url`. The materials action opens the drawer and focuses its material list. Reuse existing file-change and URL-form event handlers.

- [ ] **Step 2: Implement CSS without overlap**

Replace sidebar/mode layout rules with a centered `.research-conversation` grid. Keep the timeline flexible with a stable composer at its end, but do not use `position: fixed` for the composer. `.research-add-source-menu` is a grid surface positioned in the composer action area using a containing block; its `max-width` is the composer width and it opens upward. It must be hidden with `[hidden]`, not opacity-only.

Use these layout rules:

```css
.research-conversation, .research-composer, .research-attachment-strip,
.research-settings-drawer, .research-add-source-menu { min-width: 0; }
.research-attachment-strip { display: flex; flex-wrap: wrap; gap: 8px; }
.research-attachment-chip { max-width: 100%; overflow-wrap: anywhere; }
.research-composer-actions { display: inline-flex; gap: 8px; flex: 0 0 auto; }
```

At `max-width: 720px`, stack composer footer actions only when necessary, make the drawer full width, retain a minimum 44px touch target for tool/settings/send controls and ensure attachment chips wrap above the textarea. Keep existing teal/neutral palette, status text, hover/active/focus feedback and reduced-motion override. Do not add gradients, decorative graphics, viewport-scaled fonts or continuous animation.

- [ ] **Step 3: Verify CSS and interaction green state**

Run all portal tests and `node --check`. Inspect via browser at 1440px, 1024px, 720px and Firefox's available narrow viewport. Confirm no horizontal overflow, no covered textarea/send button, keyboard focus restoration and Escape behavior.

## Task 5: Restore Existing Jobs, Validate End-to-End Contracts and Record Evidence

**Files:**
- Modify: `tests/model_gateway/test_gateway_portal.py`
- Modify: `docs/model_gateway/ACCEPTANCE_20260714_CONVERSATIONAL_COMPOSER.md`

**Consumes:** Tasks 1-4.

**Produces:** Regression coverage and final acceptance record; no persistent backend change.

- [ ] **Step 1: Add terminal-state and no-regression static tests**

Add a test that asserts the script keeps existing endpoint strings and cancellation behavior:

```python
def test_conversation_client_keeps_existing_secure_file_and_job_contracts(self):
    script = self._research_script()
    for token in (
        '"/api/model-gateway/files"',
        '"/api/model-gateway/sources/url"',
        '"/api/model-gateway/research/jobs"',
        '"/v1/chat/completions"',
        'async function cancelResearchTurn(turnId)',
        'require_citations: true',
        '资料处理失败，请移除后重新发送',
    ):
        self.assertIn(token, script)
```

Keep existing backend route tests, especially `test_repeated_idempotency_key_returns_original_job_without_requeue` and `test_idempotency_key_rejects_different_research_payload`; do not rewrite them.

- [ ] **Step 2: Run focused red/green validation**

Run portal tests and the two existing research idempotency route tests before/after the implementation:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest \
  tests.model_gateway.test_gateway_portal \
  tests.model_gateway.test_gateway_routes.TestResearchRoutes.test_repeated_idempotency_key_returns_original_job_without_requeue \
  tests.model_gateway.test_gateway_routes.TestResearchRoutes.test_idempotency_key_rejects_different_research_payload
```

Expected: all pass after implementation; the two backend idempotency tests prove the UI's reuse of an idempotency key maps to an existing protected backend contract.

- [ ] **Step 3: Run full non-GPU acceptance**

```bash
docker compose config --quiet
docker compose ps
for endpoint in /healthz /readyz /api/models /api/testset/list /api/history /research /model-gateway; do
  curl -fsS -o /dev/null -w "$endpoint %{http_code}\n" "http://127.0.0.1:5000$endpoint"
done
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/research.js
git diff --check
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
```

Expected: all HTTP calls return 200, the full test suite has no new failure, syntax/diff checks pass and no model/GPU compute process begins.

- [ ] **Step 4: Browser acceptance**

Use Firefox WebDriver, wait three seconds after navigation, and capture desktop/mobile screenshots in ignored `logs/model_gateway/acceptance/<timestamp>/`. Verify:

1. Initial page has one timeline/composer and no mode switcher/permanent sidebar.
2. Settings drawer opens/closes through button, overlay and Escape, with focus restored.
3. Tools open above the composer and do not overlap textarea/send; URL panel opens from the link action.
4. Attachment chips wrap, show ready/processing/failed text, remove locally, and drawer shows the full list.
5. No selected source uses direct-chat routing; selected ready sources use citation-required routing; pending source sends show `正在准备资料`; failed source sends show the fixed error with no direct-chat fallback.
6. Reload restores local messages, selected source metadata and model selection only. It does not restore an API Key beyond the existing session/local Key policy.
7. Desktop/mobile DOM `scrollWidth` does not exceed `innerWidth`; reduced motion leaves controls operable without transition movement.

Document Firefox's actual narrow viewport if it clamps the requested value; do not claim an unobserved 390px viewport.

- [ ] **Step 5: Record rollback**

Record exact outcomes, screenshot names, GPU pre/post snapshots and any pre-existing warnings in the acceptance file. Before a rollback in the shared dirty worktree, export only this batch's patch. Restore only `research.html`, `research.css`, `research.js`, portal tests and the two conversational-composer documents. Do not delete files, jobs, indexes, database rows or queues created before the batch.

## Plan Self-Review

- Tasks 1-2 replace the visual shell and lock all required DOM boundaries.
- Task 3 covers session storage limits, reload, source rehydration, automatic routing, pending wait, failure behavior, cancellation and existing idempotency.
- Task 4 covers keyboard drawer behavior, normal-flow no-overlap layout, attachment chips and reduced motion.
- Task 5 covers preserved backend contracts, full validation, screenshots, GPU non-use and rollback.
- The plan uses existing APIs only; source removal is explicitly local selection removal, not an unimplemented server delete operation.

## Implementation Notes (2026-07-14)

- The URL input panel intentionally uses a `div`, not a nested `form`, inside
  `research-job-form`. HTML form parsing otherwise removes the inner form from
  its expected location and prevents the URL action listener from binding.
  `research-submit-url` is an explicit `type="button"` control.
- `processing` is an actual file-worker state and is treated as pending by the
  client alongside download, scan and parse states. It cannot be routed as a
  direct, uncited chat request.
- Source statuses are localised for the Chinese user portal (`已就绪`, `处理中`,
  and related concise labels) only at render time. The original backend values
  remain the state-machine and API contract values.
