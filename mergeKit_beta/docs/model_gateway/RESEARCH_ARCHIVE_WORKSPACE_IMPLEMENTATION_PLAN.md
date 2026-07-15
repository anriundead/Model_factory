# Research Archive Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `/research` into a distinctive Mergenetic Research Archive with persistent browser-local conversation management, a stable session rail and restrained Persona-inspired interaction feedback, while preserving every accepted Gateway execution contract.

**Architecture:** Add one dependency-free browser module, `research_archive.js`, which owns bounded IndexedDB records and exposes a small asynchronous API on `window.MergeneticResearchArchive`. Keep `research.js` as the only caller of Gateway APIs; it consumes the archive API for active-session state and the existing source/job routing functions stay intact. Replace the current centered layout with a CSS Grid rail/stage design and use optional GSAP timelines only for bounded `transform`/`opacity` motion.

**Tech Stack:** Flask templates, vanilla JavaScript, browser-native IndexedDB and localStorage, existing GSAP CDN, existing Remix Icon CDN, vanilla CSS Grid, Python `unittest`, Node syntax check, Firefox WebDriver, Docker Compose.

## Execution Status (2026-07-14)

Tasks 1-7 were completed and accepted. Fresh command output, browser evidence,
storage privacy checks, GPU snapshots and rollback archives are recorded in
`ACCEPTANCE_20260714_RESEARCH_ARCHIVE.md`.

## Global Constraints

- Scope is the user-facing `/research` template, its CSS/JS, portal tests and `docs/model_gateway/` records only.
- Do not change `/api/model-gateway/files`, `/api/model-gateway/sources/url`, `/api/model-gateway/research/jobs`, `/v1/chat/completions`, queue, parser, TTL, SSRF, ClamAV, retrieval, citations, cancellation, database, model service, vLLM, Ray, Compose or GPU behaviour.
- Direct chat is allowed only when the selected ready-source list is empty. Any selected source requires the existing citation-required research job; never downgrade a failed or expired source-backed request to uncited chat.
- Persist at most 20 browser-local sessions, and at most 60 rendered turns / 256 KiB per session. Never silently delete a 21st session candidate; ask the user to delete an existing local session or cancel.
- IndexedDB records may store only session title, timestamps, message text/status/route/model/citations/locators, source display metadata and selected model. They must never store API Keys, authorization headers, source bytes, parsed text, chunks, embeddings, raw HTTP payloads or unfiltered model/task metadata.
- `localStorage` may store only the non-secret `mergeneticResearchArchiveActiveSessionId`; API Key storage keeps its existing explicit session/trusted-device behaviour.
- A source ID restored from IndexedDB must be revalidated with the existing owned-source endpoint before it can be selected for a new job. Missing/expired sources display as unavailable and cannot submit.
- Replace the old `sessionStorage` conversation snapshot only after a successful IndexedDB transaction. Retain it on migration failure and continue the current session-only behaviour.
- No copied Gemini, Grok, Persona or other third-party assets, fonts, names, layout or copy. Persona is an abstract reference for strong selection hierarchy, cut markers and short feedback only.
- Do not add a frontend framework, client-side database library, new CDN, continuous animation, gradient/orb decoration, hidden desktop rail or model/GPU runtime test.
- Motion may animate only `transform`, `opacity` or GSAP `autoAlpha`, honours `prefers-reduced-motion`, is cancellable on rapid switching and must not alter layout dimensions.
- The worktree is shared and dirty. Do not commit, reset, clean, restore unrelated files or stage unrelated changes.

## File Map

| File | Responsibility |
| --- | --- |
| `templates/model_gateway/research.html` | Archive rail, mobile session trigger, active-session bar, limit/delete dialogs and cache-versioned script loading. |
| `static/model_gateway/research_archive.js` | Native IndexedDB schema, record sanitisation, migration, bounded CRUD, source invalidation and local search. No Gateway network calls. |
| `static/model_gateway/research.js` | Active-session controller, current request routing, rendering, archive integration, keyboard interactions and optional GSAP lifecycle. |
| `static/model_gateway/research.css` | Research Archive layout, visual tokens, rail/drawer states, readable timeline, interaction states and reduced-motion overrides. |
| `tests/model_gateway/test_gateway_portal.py` | Static contract tests for the archive shell, persistence whitelist, retention guard, request invariants and motion boundaries. |
| `docs/model_gateway/RESEARCH_ARCHIVE_WORKSPACE_DESIGN.md` | Approved product/design record. |
| `docs/model_gateway/ACCEPTANCE_20260714_RESEARCH_ARCHIVE.md` | New execution evidence, screenshots, storage checks, GPU/process snapshots and rollback proof. |

---

## Task 1: Establish a Baseline and an Isolated Rollback Point

**Files:**
- Create: `docs/model_gateway/ACCEPTANCE_20260714_RESEARCH_ARCHIVE.md`
- Modify later: files listed in the File Map only

**Consumes:** The approved `RESEARCH_ARCHIVE_WORKSPACE_DESIGN.md`.

**Produces:** A repeatable baseline and a local rollback archive before any source change.

- [ ] **Step 1: Record baseline before changes**

Run:

```bash
docker compose config --quiet
docker compose ps
for endpoint in /healthz /readyz /api/models /api/testset/list /api/history /research /model-gateway; do
  curl -fsS -o /dev/null -w "$endpoint %{http_code}\n" "http://127.0.0.1:5000$endpoint"
done
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/research.js
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
```

Expected: Compose parses, all seven endpoints are `200`, the current full test
suite passes and no process is started by the check. Record existing external
GPU work as baseline, especially GPU 2 if it is occupied.

- [ ] **Step 2: Create a local rollback archive**

Run before editing:

```bash
tar -czf /tmp/mergenetic_research_archive_prechange.tar.gz \
  mergeKit_beta/templates/model_gateway/research.html \
  mergeKit_beta/static/model_gateway/research.css \
  mergeKit_beta/static/model_gateway/research.js \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/docs/model_gateway/RESEARCH_ARCHIVE_WORKSPACE_DESIGN.md \
  mergeKit_beta/docs/model_gateway/RESEARCH_ARCHIVE_WORKSPACE_IMPLEMENTATION_PLAN.md
sha256sum /tmp/mergenetic_research_archive_prechange.tar.gz
```

Expected: the archive exists outside the repository and its SHA-256 is written
to the acceptance record. It is a restoration point for this UX batch only.

- [ ] **Step 3: Define the initial acceptance record**

Create the record with these headings before implementation:

```markdown
# Research Archive Acceptance (2026-07-14)

## Baseline
## Storage Privacy Checks
## Automated Gates
## Browser Evidence
## GPU And Process Safety
## Rollback
```

Do not write success claims until the commands in later tasks produce fresh
evidence.

---

## Task 2: Add Tested, Bounded Browser-Local Archive Storage

**Files:**
- Create: `static/model_gateway/research_archive.js`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** Browser-native IndexedDB only.

**Produces:** `window.MergeneticResearchArchive` with `open`, `list`, `read`,
`save`, `remove`, `clear`, `search`, `migrateLegacy` and `invalidateSources`.

- [ ] **Step 1: Write failing archive module tests**

Add a reader and static contract test:

```python
@staticmethod
def _research_archive_script():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(root, "static", "model_gateway", "research_archive.js"), encoding="utf-8") as handle:
        return handle.read()

def test_research_archive_storage_is_bounded_and_secret_free(self):
    script = self._research_archive_script()
    for token in (
        'const DB_NAME = "mergeneticResearchArchiveV1"',
        'const ACTIVE_SESSION_KEY = "mergeneticResearchArchiveActiveSessionId"',
        'const MAX_SESSIONS = 20',
        'const MAX_TURNS = 60',
        'const MAX_SESSION_BYTES = 256 * 1024',
        'function sanitizeSession(record)',
        'async function migrateLegacy(legacySnapshot)',
        'async function invalidateSources()',
        'window.MergeneticResearchArchive',
    ):
        self.assertIn(token, script)
    self.assertNotIn("Authorization", script)
    self.assertNotIn("mergeneticResearchKey", script)
```

- [ ] **Step 2: Verify the test fails before implementation**

Run:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest \
  tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_archive_storage_is_bounded_and_secret_free
```

Expected: fail because `research_archive.js` does not exist.

- [ ] **Step 3: Create the archive module**

Create `research_archive.js` as a standalone IIFE. Its public interface must
be exactly:

```javascript
window.MergeneticResearchArchive = {
    open,
    list,
    read,
    save,
    remove,
    clear,
    search,
    migrateLegacy,
    invalidateSources,
};
```

Use one `sessions` object store with `id` as its key path and an `updatedAt`
index. Implement the record boundary as follows:

```javascript
const DB_NAME = "mergeneticResearchArchiveV1";
const ACTIVE_SESSION_KEY = "mergeneticResearchArchiveActiveSessionId";
const MAX_SESSIONS = 20;
const MAX_TURNS = 60;
const MAX_SESSION_BYTES = 256 * 1024;

function sanitizeSession(record) {
    const messages = Array.isArray(record.messages) ? record.messages.slice(-MAX_TURNS).map(sanitizeMessage) : [];
    const sources = Array.isArray(record.sources) ? record.sources.map(sanitizeSource) : [];
    const session = {
        id: String(record.id || ""),
        title: String(record.title || "新建研究").slice(0, 48),
        createdAt: Number(record.createdAt) || Date.now(),
        updatedAt: Number(record.updatedAt) || Date.now(),
        messages,
        sources,
        selectedModel: String(record.selectedModel || ""),
    };
    return trimToByteLimit(session, MAX_SESSION_BYTES);
}

function sanitizeSource(source) {
    return {
        id: String(source.id || ""),
        name: String(source.name || "未命名资料").slice(0, 256),
        sourceKind: String(source.sourceKind || "upload"),
        status: String(source.status || "expired"),
        selected: Boolean(source.selected),
    };
}
```

`save()` must reject a new twenty-first record with `{ ok: false,
reason: "session_limit" }`; it must not evict an older record. `search()`
performs case-insensitive local matching on `title` and message `content`.
`invalidateSources()` preserves only display name/source kind/status, clears
source IDs and selection on every stored session, and returns the affected
session IDs. It must not modify messages.

- [ ] **Step 4: Implement safe legacy migration**

Use the existing snapshot only as an input and remove it only after `save()`
returns `{ ok: true }`:

```javascript
async function migrateLegacy(legacySnapshot) {
    if (!legacySnapshot || !Array.isArray(legacySnapshot.turns)) return null;
    const record = snapshotToSession(legacySnapshot);
    const result = await save(record, { allowAtLimit: false });
    return result.ok ? record.id : null;
}
```

The caller, not this module, removes the old `sessionStorage` key. IndexedDB
open or transaction failure must reject so the caller can retain the legacy
session behaviour.

- [ ] **Step 5: Verify green state**

Run:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest \
  tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_archive_storage_is_bounded_and_secret_free
node --check mergeKit_beta/static/model_gateway/research_archive.js
```

Expected: test passes and Node reports no syntax error.

---

## Task 3: Add the Archive Rail and Keep Existing Composer Contracts Stable

**Files:**
- Modify: `templates/model_gateway/research.html`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** The archive module loaded before `research.js`.

**Produces:** Stable rail/drawer/dialog DOM IDs without changing existing
composer/source/settings IDs.

- [ ] **Step 1: Write failing shell contract tests**

Add this test:

```python
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
        'research_archive.js?v=research-archive-v1',
        'research.js?v=research-archive-v1',
    ):
        self.assertIn(token, page)
    self.assertNotIn('href="/model-gateway"', page)
```

- [ ] **Step 2: Verify red state**

Run the named test. Expected: fail because the archive rail IDs and versioned
archive script are absent.

- [ ] **Step 3: Build the semantic shell**

Use this hierarchy, keeping the existing `research-job-form`, source controls,
settings drawer and toast IDs unchanged:

```html
<aside id="research-session-rail" class="research-session-rail" aria-label="本机会话">
  <a class="research-archive-brand" href="/research" aria-label="Mergenetic Research Archive">...</a>
  <button id="research-new-session" type="button">...</button>
  <label class="research-session-search"><span>搜索会话</span><input id="research-session-search" type="search"></label>
  <nav id="research-session-list" aria-label="最近会话"></nav>
  <button id="research-clear-local-history" type="button">清除本机记录</button>
</aside>
<main class="research-stage">
  <header class="research-stage-header">
    <button id="research-session-drawer-toggle" type="button">会话</button>
    <p>RESEARCH ARCHIVE</p><h1 id="research-active-session-title">新建研究</h1>
    <button id="research-session-settings" type="button">...</button>
  </header>
  <!-- existing timeline and composer -->
</main>
<dialog id="research-session-limit-dialog" aria-labelledby="research-session-limit-title">...</dialog>
<dialog id="research-session-delete-dialog" aria-labelledby="research-session-delete-title">...</dialog>
```

Use `dialog.showModal()` / `dialog.close()` rather than a custom confirmation
overlay. Buttons must name the specific session in the confirmation text and
provide an explicit cancel action.

- [ ] **Step 4: Load scripts in dependency order**

Replace the two script tags at the page end with:

```html
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
<script src="/static/model_gateway/research_archive.js?v=research-archive-v1"></script>
<script src="/static/model_gateway/research.js?v=research-archive-v1"></script>
```

Do not add another CDN or use ES modules; the archive global must exist before
the portal script's `DOMContentLoaded` handler runs.

- [ ] **Step 5: Verify green state**

Run the new test and the complete `test_gateway_portal.py` module. Expected:
all portal contracts pass and the existing URL tool / admin-navigation guards
remain green.

---

## Task 4: Integrate Local Sessions Without Weakening Request Safety

**Files:**
- Modify: `static/model_gateway/research.js`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** `window.MergeneticResearchArchive` and Task 3 DOM IDs.

**Produces:** Safe active-session switching, local CRUD/search, migration and
source revalidation while keeping request functions as the only network layer.

- [ ] **Step 1: Write failing integration-contract tests**

Add:

```python
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
        'require_citations: true',
        'api("/v1/chat/completions"',
    ):
        self.assertIn(token, script)
    self.assertNotIn("Authorization: state", script)
```

- [ ] **Step 2: Verify red state**

Run the named test. Expected: fail because archive controller functions do not
exist yet.

- [ ] **Step 3: Add active-session state and persistence**

Extend state with only browser-local IDs and UI state:

```javascript
archive: window.MergeneticResearchArchive,
activeSessionId: localStorage.getItem("mergeneticResearchArchiveActiveSessionId") || "",
archiveAvailable: Boolean(window.MergeneticResearchArchive),
pendingDeleteSessionId: "",
```

Implement these responsibilities:

```javascript
async function persistActiveArchiveSession() {
    if (!state.archiveAvailable || !state.activeSessionId) return;
    const result = await state.archive.save(sessionRecordFromState());
    if (!result.ok && result.reason === "session_limit") openSessionLimitDialog();
}

async function switchArchiveSession(sessionId) {
    const record = await state.archive.read(sessionId);
    if (!record) return;
    cancelUiOnlyTimers();
    restoreStateFromSession(record);
    state.activeSessionId = record.id;
    localStorage.setItem("mergeneticResearchArchiveActiveSessionId", record.id);
    await revalidateArchiveSources();
    renderAll();
    runSessionSwitchMotion();
}

async function createArchiveSession() {
    const record = emptyArchiveSession();
    const result = await state.archive.save(record);
    if (!result.ok) return openSessionLimitDialog();
    await switchArchiveSession(record.id);
}
```

`sessionRecordFromState()` must reuse existing `normalizeFile`, `normalizeTurn`
and bounded conversation rules. It must never read `state.key`. All existing
state changes that currently call `persistConversation()` must call one new
`persistResearchState()` wrapper instead:

```javascript
function persistResearchState() {
    if (state.archiveAvailable) {
        persistActiveArchiveSession().catch(() => toast("本机记录暂时无法保存"));
    } else {
        persistConversation();
    }
}
```

- [ ] **Step 4: Migrate and preserve legacy fallback**

At startup, first try the archive; if no active record exists, pass the current
legacy snapshot to `migrateLegacy`. Remove
`mergeneticResearchConversationV1` only after it returns a new session ID:

```javascript
const legacyRaw = sessionStorage.getItem(CONVERSATION_STORAGE_KEY);
const migratedId = legacyRaw ? await state.archive.migrateLegacy(JSON.parse(legacyRaw)) : null;
if (migratedId) {
    sessionStorage.removeItem(CONVERSATION_STORAGE_KEY);
    await switchArchiveSession(migratedId);
} else if (!state.activeSessionId) {
    await createArchiveSession();
}
```

Wrap parsing/open errors. On error, set `archiveAvailable = false`, retain the
legacy snapshot and show exactly `本机记录不可用，本次会话不会长期保存`.

- [ ] **Step 5: Revalidate sources and handle Key changes**

`revalidateArchiveSources()` must call the existing file read endpoint only
when a Key is present. For `404`, retain a display-only source with
`status: "expired"`, clear its ID and selection, and show no stale ID in
`selectedSourceIds()`. Before changing or clearing a Key, call:

```javascript
await state.archive.invalidateSources();
state.files = [];
state.selectedFileIds.clear();
```

Keep local text turns and their historical citation locators. Do not use a
source from the prior Key for a new request. A user-facing expired selection
must produce `资料已过期，无法用于新的研究请求` and must not call direct chat.

- [ ] **Step 6: Implement archive UI controls**

Implement local-only handlers:

- Search list with `archive.search(query)` and do not mutate stored records.
- Derive a title from the first user prompt, limited to 48 characters, until a
  user explicitly renames it.
- Render rename/delete buttons per session row; use event delegation on
  `research-session-list`.
- Open the native delete dialog before `archive.remove(id)`. If it deletes the
  active session, switch to the most recently updated remaining session or
  create one empty session.
- `research-clear-local-history` opens the same dialog in `clear` mode. It
  invokes `archive.clear()` and then creates one empty active session.
- The session-limit dialog presents `取消` and `管理会话`; the latter focuses
  `research-session-list`. It must not auto-delete anything.

- [ ] **Step 7: Verify green state**

Run:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest \
  tests.model_gateway.test_gateway_portal
node --check mergeKit_beta/static/model_gateway/research_archive.js
node --check mergeKit_beta/static/model_gateway/research.js
```

Expected: all portal tests pass. Existing direct chat, research-job,
idempotency, cancellation and Key-clearing strings remain present.

---

## Task 5: Build the Mergenetic Research Archive Visual System

**Files:**
- Modify: `static/model_gateway/research.css`
- Modify: `templates/model_gateway/research.html`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** Archive rail/stage DOM from Task 3.

**Produces:** Responsive, original high-contrast visual hierarchy with no
overlap, no generic dashboard cards and no third-party visual copying.

- [ ] **Step 1: Write failing visual contract tests**

Add:

```python
def test_research_archive_styles_define_rail_signal_and_reduced_motion_boundaries(self):
    css = self._research_css()
    for token in (
        ".research-session-rail",
        ".research-stage",
        ".research-session-row.is-active",
        ".research-signal-cut",
        "--ra-signal-red: #D9362B",
        "grid-template-columns: 272px minmax(0, 1fr)",
        "@media (max-width: 980px)",
        "@media (prefers-reduced-motion: reduce)",
        "min-width: 0",
        "overflow-wrap: anywhere",
    ):
        self.assertIn(token, css)
    self.assertNotIn("linear-gradient", css)
```

- [ ] **Step 2: Verify red state**

Run the named test. Expected: fail because the current CSS has no archive rail
or red signal token.

- [ ] **Step 3: Replace visual tokens and layout deliberately**

Use the approved variables and document their role in CSS:

```css
:root {
  --ra-ink: #17191D;
  --ra-paper: #F6F7F4;
  --ra-signal-red: #D9362B;
  --ra-graphite: #34383F;
  --ra-steel: #9AA2AC;
  --ra-verified: #14805E;
  --ra-control-radius: 4px;
  --ra-surface-radius: 8px;
}

.research-shell { min-height: 100dvh; display: grid; grid-template-columns: 272px minmax(0, 1fr); background: var(--ra-paper); }
.research-session-rail { min-width: 0; display: grid; grid-template-rows: auto auto auto minmax(0, 1fr) auto; background: var(--ra-ink); color: #fff; }
.research-stage { min-width: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); }
.research-session-row.is-active::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; background: var(--ra-signal-red); transform: skewY(-28deg); }
```

Retain white/light reading surfaces for message content. Use red for current
session, active command and selection, and green only for ready/success.
Every status retains text. Avoid gradients, absolute content positioning,
background effects and nested cards.

- [ ] **Step 4: Implement responsive and interaction rules**

At `max-width: 980px`, hide the desktop rail visually but retain it in a
native-dialog-style drawer controlled by `research-session-drawer-toggle`.
The stage must occupy full width and the composer must stay in normal document
flow. At `max-width: 720px`, stack source actions, use 44px minimum targets,
wrap long titles/files and prevent send controls from shrinking.

Provide `:hover`, `:active` and `:focus-visible` for rail rows, tool buttons,
dialogs, source controls and message citations. Preserve a clear keyboard
focus ring on dark rail surfaces.

- [ ] **Step 5: Verify green state**

Run the new visual test and complete portal tests. Then run `node --check` on
both research JavaScript files. Expected: all pass.

---

## Task 6: Add Controlled Motion, Keyboard Behaviour and Focus Safety

**Files:**
- Modify: `static/model_gateway/research.js`
- Modify: `tests/model_gateway/test_gateway_portal.py`

**Consumes:** Existing GSAP CDN and archive DOM/state.

**Produces:** Optional, cancellable motion that is cosmetic only and retains
complete function when GSAP is unavailable.

- [ ] **Step 1: Write failing motion contract tests**

Add:

```python
def test_research_archive_motion_is_reduced_motion_safe_and_cancellable(self):
    script = self._research_script()
    for token in (
        'window.matchMedia("(prefers-reduced-motion: reduce)")',
        "function runArchiveEntranceMotion()",
        "function runSessionSwitchMotion()",
        "function killResearchMotion()",
        "window.gsap.timeline",
        "autoAlpha",
        "overwrite: \"auto\"",
    ):
        self.assertIn(token, script)
```

- [ ] **Step 2: Verify red state**

Run the named test. Expected: fail because the research portal currently has
no GSAP lifecycle functions.

- [ ] **Step 3: Implement motion controller**

Add a small local controller in `research.js`:

```javascript
let researchTimeline = null;

function killResearchMotion() {
    researchTimeline?.kill();
    researchTimeline = null;
}

function motionAllowed() {
    return Boolean(window.gsap) && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function runSessionSwitchMotion() {
    killResearchMotion();
    if (!motionAllowed()) return;
    const entries = [...document.querySelectorAll(".research-chat-entry")].slice(-8);
    researchTimeline = window.gsap.timeline({ defaults: { ease: "power2.out", overwrite: "auto" } });
    researchTimeline.fromTo(entries, { y: 12, autoAlpha: 0 }, { y: 0, autoAlpha: 1, duration: 0.18, stagger: 0.025 });
}
```

Use the same guard for initial entrance and open/close surfaces. CSS remains
the fallback for hover/press. Never animate width, height, top, left, margin,
padding, CSS grid tracks or scroll position. Call `killResearchMotion()` before
session switch, on `pagehide`, and when reduced-motion changes.

- [ ] **Step 4: Complete keyboard and dialog behaviour**

- `Escape` closes the source tools first, then settings dialog, then session
  drawer; it must not close a destructive confirmation accidentally.
- Opening a session drawer or delete/limit dialog moves focus into it; closing
  restores focus to the originating control.
- Enter activates a focused session row; Delete/Backspace never deletes a row
  without the named dialog confirmation.
- Search results use `aria-live="polite"` and report only result count.

- [ ] **Step 5: Verify green state**

Run the motion test, complete portal test module and Node checks. Expected:
all pass with no new dependency.

---

## Task 7: Browser Acceptance, Failure Exercises and Final Record

**Files:**
- Modify: `docs/model_gateway/ACCEPTANCE_20260714_RESEARCH_ARCHIVE.md`
- Verify: all implementation files above

**Consumes:** Completed Tasks 1-6.

**Produces:** Fresh evidence for function, visual quality, privacy boundary,
resource safety and rollback.

- [ ] **Step 1: Run non-GPU automated gates**

Run:

```bash
docker compose config --quiet
for endpoint in /healthz /readyz /api/models /api/testset/list /api/history /research /model-gateway; do
  curl -fsS -o /dev/null -w "$endpoint %{http_code}\n" "http://127.0.0.1:5000$endpoint"
done
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/research_archive.js
node --check mergeKit_beta/static/model_gateway/research.js
git diff --check
```

Expected: all HTTP routes return `200`, full tests pass and no syntax/diff
check fails.

- [ ] **Step 2: Execute Firefox WebDriver storage and layout acceptance**

Use the already-installed `/snap/bin/geckodriver`, synthetic browser records
and an explicit cleanup trap. Save only screenshots and redacted result JSON
to ignored `logs/model_gateway/acceptance/20260714_research_archive/`.

Verify all of these conditions in the actual browser:

1. First load creates an empty local session with no stored API Key.
2. Seeded 20 records render in recency groups; local search finds a message
   term and does not mutate the record list.
3. Creating the 21st record opens the limit dialog and does not delete any
   record; after deleting a named record, creation succeeds.
4. Rename, switch, delete active session, clear local history and refresh
   retain the expected local state only.
5. Seeded source metadata displays, but a synthetic missing source becomes
   expired/unselected before a request; no direct-chat route is invoked.
6. The active rail marker, stage, composer, tools, settings drawer and mobile
   session drawer do not overlap. `scrollWidth <= innerWidth` at 1440px,
   1024px, 720px and Firefox's actual narrow viewport.
7. Normal motion has no active GSAP timeline after rapid session changes;
   reduced-motion run has no positional animation and all controls work.
8. No geckodriver, Firefox or Playwright process remains after cleanup.

- [ ] **Step 3: Verify storage privacy directly**

In WebDriver execute script, read the IndexedDB record and assert it does not
have any of these keys or values:

```javascript
["key", "authorization", "headers", "payload", "fileBytes", "parsedText", "chunks", "embeddings"]
```

Set a sentinel API Key in existing session storage before save and assert the
sentinel is absent from `JSON.stringify(record)`. Check that the active-session
localStorage pointer contains only a UUID-like session ID.

- [ ] **Step 4: Record GPU/process safety**

Before and after browser tests run:

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
pgrep -af 'geckodriver|playwright_firefox|firefox.*headless' || true
```

Expected: no new GPU process, GPU 2 external workload unchanged, and no
temporary browser process remains. Do not start a model, fusion, Ray, vLLM,
file upload, URL fetch or research job for this UX batch.

- [ ] **Step 5: Fill the acceptance record and create a post-change archive**

Record exact command outputs, test counts, browser viewport values, screenshot
names, known external GPU processes, test omissions and both archive hashes.
Then create:

```bash
tar -czf /tmp/mergenetic_research_archive_accepted.tar.gz \
  mergeKit_beta/templates/model_gateway/research.html \
  mergeKit_beta/static/model_gateway/research_archive.js \
  mergeKit_beta/static/model_gateway/research.js \
  mergeKit_beta/static/model_gateway/research.css \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/docs/model_gateway/RESEARCH_ARCHIVE_WORKSPACE_DESIGN.md \
  mergeKit_beta/docs/model_gateway/RESEARCH_ARCHIVE_WORKSPACE_IMPLEMENTATION_PLAN.md
sha256sum /tmp/mergenetic_research_archive_accepted.tar.gz
```

Rollback after acceptance restores this archive with `tar -xzf ... -C
/home/a/Workspace/Model_factory`. It must never use `git reset`, delete server
sources/jobs or stop external GPU processes.

## Plan Self-Review

- Approved visual and product requirements map to Tasks 3, 5 and 6.
- Long-lived local history, bounded storage, migration, no silent deletion and
  source expiry are isolated in Task 2 and integrated safely in Task 4.
- Existing direct/research routing, idempotency, cancellation and Key boundary
  remain explicit in Tasks 4 and 7.
- Privacy is enforced by `sanitizeSession`, a static whitelist contract and a
  browser IndexedDB inspection.
- Motion has a documented property budget, cancellation rule and reduced-motion
  test instead of decorative background effects.
- The plan changes no backend/runtime component and explicitly prevents model
  or GPU work during UI acceptance.
