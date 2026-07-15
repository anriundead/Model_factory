# Research Portal Three-Mode UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/research` chat-first with chat, research and focus layouts, clear interaction feedback, and accessible discrete transitions.

**Architecture:** Flask and Gateway APIs stay unchanged. Browser state retains only the selected layout under `mergeneticResearchWorkspace`; source/job polling remains shared and chat entries exist only in memory. Chat mode calls the existing API-Key-authenticated `/v1/chat/completions` endpoint without source files, while research mode retains the existing job/citation path. CSS provides control feedback; one GSAP timeline handles a mode transition only when reduced motion is off.

**Tech Stack:** Flask template, vanilla JavaScript, CSS, existing GSAP CDN, Python `unittest`, Node syntax checking.

## Global Constraints

- Do not change API, schema, queue, TTL, model runtime or GPU scheduling.
- Default desktop mode is `chat`; sidebar remains expanded.
- Persist only `chat`, `research`, or `focus` in `mergeneticResearchWorkspace`.
- Do not persist messages, answers, source text, or any new API Key data; chat history exists only until page refresh.
- Preserve file/job polling and cancellation across every mode switch.
- GSAP may animate only `transform` and `opacity`; do not add a continuous animation.
- Every control needs hover, active, and visible keyboard-focus feedback.
- Do not commit from the shared dirty worktree.

---

### Task 1: Add Three-Mode Markup

**Files:**
- Modify: `templates/model_gateway/research.html:17-52`
- Modify: `tests/model_gateway/test_gateway_portal.py:8-50`

**Produces:** `research-mode-switcher`, `research-mode-chat`, `research-mode-research`, `research-mode-focus`, `research-chat-timeline`; shell classes `research-mode-chat`, `research-mode-research`, `research-mode-focus`.

- [ ] **Step 1: Write the failing template test**

```python
def test_research_workspace_exposes_three_accessible_modes(self):
    page = self._research_page()
    self.assertIn('id="research-mode-switcher"', page)
    self.assertIn('id="research-mode-chat"', page)
    self.assertIn('id="research-mode-research"', page)
    self.assertIn('id="research-mode-focus"', page)
    self.assertIn('id="research-chat-timeline"', page)
    self.assertIn('aria-pressed="true"', page)
```

Add `_research_page()` to read the template once.

- [ ] **Step 2: Verify red state**

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_workspace_exposes_three_accessible_modes`

Expected: `FAIL` because the mode switcher does not exist.

- [ ] **Step 3: Implement the template skeleton**

Add a labelled `role="group"` in the workspace header:

```html
<div id="research-mode-switcher" class="research-mode-switcher" role="group" aria-label="工作区布局">
  <button id="research-mode-chat" type="button" data-mode="chat" aria-pressed="true"><i class="ri-message-3-line"></i><span>聊天</span></button>
  <button id="research-mode-research" type="button" data-mode="research" aria-pressed="false"><i class="ri-flask-line"></i><span>研究</span></button>
  <button id="research-mode-focus" type="button" data-mode="focus" aria-pressed="false"><i class="ri-focus-3-line"></i><span>专注</span></button>
</div>
<section id="research-chat-timeline" class="research-chat-timeline" aria-live="polite"></section>
```

Set the shell default class to `research-mode-chat`. Preserve existing upload, result, cancellation and composer IDs.

- [ ] **Step 4: Verify green state**

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal`

Expected: portal tests pass.

### Task 2: Add Local Mode Memory, Direct Chat, and Research Result Projection

**Files:**
- Modify: `static/model_gateway/research.js:1-246`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Produces:** `setWorkspaceMode(mode, animate)`, `appendChatEntry(kind, content, citations, sources)`, and `submitDirectChat(question)`.

- [ ] **Step 1: Write failing client-contract tests**

```python
def test_research_client_persists_only_known_workspace_modes(self):
    script = self._research_script()
    self.assertIn('const WORKSPACE_STORAGE_KEY = "mergeneticResearchWorkspace"', script)
    self.assertIn('const WORKSPACE_MODES = new Set(["chat", "research", "focus"])', script)
    self.assertIn("function setWorkspaceMode", script)
    self.assertIn("localStorage.setItem(WORKSPACE_STORAGE_KEY, mode)", script)

def test_research_client_projects_completed_results_as_text_only_chat_entries(self):
    script = self._research_script()
    self.assertIn("function appendChatEntry", script)
    self.assertIn("entry.textContent", script)
    self.assertIn('appendChatEntry("assistant"', script)

def test_research_client_uses_existing_openai_chat_endpoint_without_sources(self):
    script = self._research_script()
    self.assertIn("async function submitDirectChat", script)
    self.assertIn('api("/v1/chat/completions"', script)
    self.assertIn("messages: state.chatMessages", script)
```

Add `_research_script()` to load the JavaScript file.

- [ ] **Step 2: Verify red state**

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_client_persists_only_known_workspace_modes tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_client_projects_completed_results_as_text_only_chat_entries`

Expected: `FAIL` because mode state, direct chat and chat projection do not exist.

- [ ] **Step 3: Implement minimal client state**

```javascript
const WORKSPACE_STORAGE_KEY = "mergeneticResearchWorkspace";
const WORKSPACE_MODES = new Set(["chat", "research", "focus"]);
const state = {
    key: sessionStorage.getItem("mergeneticResearchKey") || localStorage.getItem("mergeneticResearchKey") || "",
    files: [], activeJobId: null, fileTimer: null, jobTimer: null,
    workspaceMode: "chat", chatJobIds: new Set(),
};
```

`setWorkspaceMode` rejects unknown values, updates the shell class and every button's `aria-pressed`, then persists only `mode`. On load restore a valid value or `chat`. `appendChatEntry` builds DOM nodes via `textContent`, never `innerHTML`. `submitDirectChat(question)` appends the user message, sends the in-memory `state.chatMessages` to `/v1/chat/completions` with the selected model, then appends the returned assistant message. It must not send source text, create a research job, or save message history. `renderResearchResult` projects a completed result once per job ID. Switching layouts must not clear files, timers, job state, or chat entries.

- [ ] **Step 4: Verify green state**

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal`

Run: `node --check static/model_gateway/research.js`

Expected: both commands exit `0`.

### Task 3: Add Mode Transitions and Control Feedback

**Files:**
- Modify: `static/model_gateway/research.js:211-246`
- Modify: `static/model_gateway/research.css:1-19`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** `setWorkspaceMode` from Task 2.

- [ ] **Step 1: Write failing motion/accessibility tests**

```python
def test_research_client_uses_reduced_motion_safe_mode_transition(self):
    script = self._research_script()
    self.assertIn('window.matchMedia("(prefers-reduced-motion: reduce)")', script)
    self.assertIn("research-mode-transitioning", script)
    self.assertIn("window.gsap.timeline", script)
    self.assertIn("autoAlpha", script)

def test_research_styles_define_mode_layouts_and_interactive_feedback(self):
    css = self._research_css()
    self.assertIn(".research-mode-chat", css)
    self.assertIn(".research-mode-research", css)
    self.assertIn(".research-mode-focus", css)
    self.assertIn(":focus-visible", css)
    self.assertIn(":active", css)
```

Add `_research_css()` to read the CSS file.

- [ ] **Step 2: Verify red state**

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_client_uses_reduced_motion_safe_mode_transition tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_styles_define_mode_layouts_and_interactive_feedback`

Expected: `FAIL` because the layout and motion definitions do not exist.

- [ ] **Step 3: Implement motion and CSS**

Use one GSAP timeline when GSAP exists and reduced motion is off: fade/translate the active surface out, apply the layout class, then enter the new surface with `y: 10`, `autoAlpha` and total duration `0.22-0.28`. Remove `research-mode-transitioning` on completion; otherwise update immediately.

CSS requirements: chat shows timeline with composer emphasis; research shows task/result and source status; focus hides sidebar and secondary controls but preserves answer, citations, cancellation and composer. Mode/action/source/citation controls receive hover lift `translateY(-1px)` or `translateY(-2px)`, press `scale(.98)`, and teal focus ring. Existing reduced-motion query neutralizes transforms and transitions.

- [ ] **Step 4: Verify green state**

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal`

Run: `node --check static/model_gateway/research.js`

Run: `git diff --check -- templates/model_gateway/research.html static/model_gateway/research.js static/model_gateway/research.css tests/model_gateway/test_gateway_portal.py`

Expected: every command exits `0`.

### Task 4: Live Acceptance and Documentation

**Files:**
- Modify: `docs/model_gateway/ACCEPTANCE_20260713_STAGE1_BATCH1.md`
- Modify: `docs/model_gateway/TEXT_RESEARCH_STAGE1.md`

- [ ] **Step 1: Run full non-GPU validation**

Run: `docker compose config --quiet`

Run: `curl -fsS http://127.0.0.1:5000/healthz`

Run: `curl -fsS http://127.0.0.1:5000/readyz`

Run: `curl -fsS http://127.0.0.1:5000/research`

Run: `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests`

Run: `node --check mergeKit_beta/static/model_gateway/research.js`

Run: `git -C mergeKit_beta diff --check`

Run: `nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits`

Expected: HTTP endpoints return success, tests and checks pass, and no model/fusion process starts.

- [ ] **Step 2: Browser acceptance**

Open `http://127.0.0.1:5000/research` at desktop and 390 px mobile widths. Verify first-load chat mode, expanded desktop sidebar, persisted selection after reload, keyboard focus, hover/press feedback, focus reading, reduced-motion behavior, and task/source state preservation through all modes.

- [ ] **Step 3: Record evidence and rollback**

Append exact test count, HTTP outcomes, GPU snapshot and viewport checks. State rollback restores only template, CSS, JavaScript and portal tests, then clears:

```javascript
localStorage.removeItem("mergeneticResearchWorkspace");
```

- [ ] **Step 4: Check live assets**

Run: `curl -fsS http://127.0.0.1:5000/research | grep -q 'research-mode-switcher'`

Run: `curl -fsS http://127.0.0.1:5000/static/model_gateway/research.js | grep -q 'setWorkspaceMode'`

Run: `docker compose ps --format 'table {{.Name}} {{.Status}}'`

Expected: live page and JavaScript contain the new behavior; services remain healthy. No recreation is required because Flask serves bind-mounted assets.

## Plan Self-Review

- Tasks 1-3 cover source-free direct chat, chat default, three layouts, expanded sidebar, local-only preference, research result projection, shared task state, feedback, GSAP limits and reduced motion.
- Task 4 covers live verification, documentation and rollback.
- Storage key, mode values, DOM IDs and rollback boundary are consistent across tasks.
- No API, queue, database, model or GPU change is planned.
