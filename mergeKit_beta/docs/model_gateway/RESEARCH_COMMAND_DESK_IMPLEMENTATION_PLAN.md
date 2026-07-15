# Research Command Desk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle `/research` as a cold-white research workspace with graphite archive structure, limited amber signal states, a functional evidence index, and reduced-motion-safe feedback.

**Architecture:** Gateway calls, source routing, local IndexedDB records, and model behavior remain unchanged. The template adds semantic markers, CSS replaces the old red visual system, and existing vanilla JavaScript gains only presentational controls: empty-workspace shortcuts, evidence disclosure, and bounded GSAP motion.

**Tech Stack:** Flask/Jinja, vanilla CSS/JavaScript, existing GSAP and Remix Icon CDNs, Python `unittest`, Node syntax checks, Firefox WebDriver, Docker Compose.

## Global Constraints

- Change only `research.html`, `research.css`, `research.js`, the portal test, and `docs/model_gateway/` records.
- Do not change Gateway APIs, queues, TTL, PostgreSQL, Redis, source ownership, IndexedDB data shape, model service, vLLM, Ray, Compose, or GPU configuration.
- Do not add dependencies, external assets, fonts, copied layouts, gradients, blur, bokeh, particles, fake telemetry, or continuous animation.
- Preserve direct-chat versus citation-required source routing, source-expiry behavior, local-only sessions, and the no-admin-link guarantee.
- Keep Remix Icons. Icon-only controls need `aria-label`, native `title`, and visible focus.
- GSAP may animate only `opacity`, `x`, `y`, or `autoAlpha`; kill old timelines and obey `prefers-reduced-motion`.
- Do not start models, merges, evaluations, Ray/vLLM, or any GPU workload. GPU 2 is external work and must not change.
- The worktree is shared and dirty. Do not commit, reset, clean, stage, restore, or alter unrelated changes.

## Skills Record

Before implementation, append this batch to `docs/model_gateway/IMPLEMENTATION_SKILLS.md` with these actual local paths:

| Purpose | Path |
| --- | --- |
| Design boundary | `/home/a/.codex/skills/brainstorming/SKILL.md` |
| Frontend direction | `/home/a/.codex/skills/frontend-design/SKILL.md` |
| Visual review | `/home/a/.codex/skills/taste-skill/SKILL.md` |
| Motion and performance | `/home/a/.codex/skills/gsap-core/SKILL.md`, `/home/a/.codex/skills/gsap-performance/SKILL.md` |
| Minimal implementation | `/home/a/.codex/skills/ponytail/SKILL.md` |
| Test-first workflow | `/home/a/.codex/skills/test-driven-development/SKILL.md` |
| Completion evidence | `/home/a/.codex/skills/verification-before-completion/SKILL.md` |

## File Map

| File | Responsibility |
| --- | --- |
| `templates/model_gateway/research.html` | Command-strip and composition-deck markers; aligned cache version. |
| `static/model_gateway/research.css` | Material tokens, layout, component states, responsive rules. |
| `static/model_gateway/research.js` | Evidence index, empty shortcuts, true local-state label, bounded motion. |
| `tests/model_gateway/test_gateway_portal.py` | Static safety contracts for visuals and existing routing invariants. |
| `docs/model_gateway/RESEARCH_COMMAND_DESK_DESIGN.md` | Approved design. |
| `docs/model_gateway/ACCEPTANCE_<date>_RESEARCH_COMMAND_DESK.md` | Fresh test, screenshot, GPU, and rollback evidence. |

## Baseline and Rollback Gate

Before editing, create the dated acceptance record with `Baseline`, `Automated Gates`, `Browser Evidence`, `GPU And Process Safety`, and `Rollback` headings. Record these command results:

```bash
docker compose config --quiet
docker compose ps
curl -fsS -o /dev/null -w '/healthz %{http_code}\n' http://127.0.0.1:5000/healthz
curl -fsS -o /dev/null -w '/readyz %{http_code}\n' http://127.0.0.1:5000/readyz
curl -fsS -o /dev/null -w '/api/models %{http_code}\n' http://127.0.0.1:5000/api/models
curl -fsS -o /dev/null -w '/api/testset/list %{http_code}\n' http://127.0.0.1:5000/api/testset/list
curl -fsS -o /dev/null -w '/api/history %{http_code}\n' http://127.0.0.1:5000/api/history
curl -fsS -o /dev/null -w '/research %{http_code}\n' http://127.0.0.1:5000/research
curl -fsS -o /dev/null -w '/model-gateway %{http_code}\n' http://127.0.0.1:5000/model-gateway
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
node --check mergeKit_beta/static/model_gateway/research.js
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
```

Create the rollback archive before visual edits:

```bash
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p "/tmp/mergenetic_command_desk_$stamp"
cp mergeKit_beta/templates/model_gateway/research.html "/tmp/mergenetic_command_desk_$stamp/"
cp mergeKit_beta/static/model_gateway/research.css "/tmp/mergenetic_command_desk_$stamp/"
cp mergeKit_beta/static/model_gateway/research.js "/tmp/mergenetic_command_desk_$stamp/"
cp mergeKit_beta/tests/model_gateway/test_gateway_portal.py "/tmp/mergenetic_command_desk_$stamp/"
cp mergeKit_beta/docs/model_gateway/IMPLEMENTATION_SKILLS.md "/tmp/mergenetic_command_desk_$stamp/"
tar -czf "/tmp/mergenetic_command_desk_$stamp.tar.gz" "/tmp/mergenetic_command_desk_$stamp"
sha256sum "/tmp/mergenetic_command_desk_$stamp.tar.gz"
```

Any new baseline failure stops the work. A later failure restores only these files from the archive, then repeats the baseline. Never use destructive Git commands.

## Task 1: Write and Prove Failing Contracts

**Files:** `tests/model_gateway/test_gateway_portal.py`

- [ ] Add this failing style contract:

```python
def test_research_command_desk_uses_cold_material_tokens_without_prohibited_effects(self):
    css = self._research_css()
    for token in (
        "--rcd-void: #171A1F", "--rcd-paper: #FAFBFB", "--rcd-amber: #E6B422",
        ".research-command-strip", ".research-evidence-index", ".research-composer-deck",
        ".research-signal-cut", "@media (prefers-reduced-motion: reduce)",
    ):
        self.assertIn(token, css)
    for forbidden in ("linear-gradient", "radial-gradient", "backdrop-filter", "animation: infinite"):
        self.assertNotIn(forbidden, css)
```

- [ ] Add this failing template/route contract:

```python
def test_research_command_desk_keeps_one_composer_and_real_state_markers(self):
    page, script = self._research_page(), self._research_script()
    for token in (
        'class="research-command-strip"', 'id="research-local-archive-state"',
        'class="research-composer research-composer-deck"',
        'data-research-empty-action="question"', 'data-research-empty-action="file"',
        'data-research-empty-action="url"',
    ):
        self.assertIn(token, page)
    for token in ("research-evidence-index", "data-research-empty-action", "research-local-archive-state", 'api("/v1/chat/completions"', "require_citations: true"):
        self.assertIn(token, script)
    self.assertNotIn('href="/model-gateway"', page)
```

- [ ] Run both tests. Expected: FAIL because Command Desk markers are absent.

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_command_desk_uses_cold_material_tokens_without_prohibited_effects tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_command_desk_keeps_one_composer_and_real_state_markers
```

## Task 2: Add Semantic Template Regions

**Files:** `templates/model_gateway/research.html`, `tests/model_gateway/test_gateway_portal.py`

- [ ] Add `research-command-strip` to the existing stage header without changing any current IDs. Beside the source count, add:

```html
<span id="research-local-archive-state" class="research-command-state"><i class="ri-computer-line" aria-hidden="true"></i>本机存档</span>
```

- [ ] Change only the existing form class to:

```html
<form id="research-job-form" class="research-composer research-composer-deck">
```

- [ ] Update all three static URLs together to `research-command-desk-v1` and adjust the current cache-version test. Do not add model metrics, a second composer, or an administrator link.
- [ ] Run the Task 1 template/route test. Expected: PASS.

## Task 3: Implement the Cold-Material Visual System

**Files:** `static/model_gateway/research.css`

- [ ] Replace all `--ra-*` uses with approved `--rcd-*` tokens. Delete the old primary red `#D9362B`; amber is used only for selected session, pending attention, primary send, and focus. Ready and danger colors remain semantic.
- [ ] Implement the material boundaries below. Use only 1px cool rules and background-tinted shadows; retain 4px control radius and 8px surface radius.

```css
.research-shell { grid-template-columns: 286px minmax(0, 1fr); }
.research-session-rail { background: var(--rcd-void); }
.research-command-strip { background: var(--rcd-void); color: var(--rcd-paper); }
.research-stage { background: var(--rcd-fog); }
.research-composer-deck, .research-chat-entry { background: var(--rcd-paper); }
```

- [ ] Give session rows an amber diagonal signal and `:focus-within` action reveal. Compose source chips, tool tray, and send control as one deck. Reserve `.research-evidence-index` for Task 4; it uses rules and locator tags, never a card inside a card.
- [ ] At `max-width: 1080px` preserve the current rail drawer. At `max-width: 720px`, prevent title/action overlap while keeping compact local-archive and source-count state visible. Reduced motion disables visual transition duration but preserves focus and selected/expanded state.
- [ ] Pass the style contract and this scan, which must produce no output:

```bash
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal.ModelGatewayPortalPageTestCase.test_research_command_desk_uses_cold_material_tokens_without_prohibited_effects
grep -nE 'linear-gradient|radial-gradient|backdrop-filter|animation: infinite|#D9362B' mergeKit_beta/static/model_gateway/research.css && exit 1 || true
```

## Task 4: Add Evidence Index and True Empty-Workspace Shortcuts

**Files:** `static/model_gateway/research.js`

- [ ] In `appendChatEntry(turn)`, retain the existing citation/source condition but replace the plain block with a `section.research-evidence-index`. It uses a `证据索引` heading and a native `<details>` per source. Its summary includes the existing `locatorLabel(source.locator)` result. All dynamic source content uses `textContent`, never model-output `innerHTML`.
- [ ] In `renderConversation()`, replace the empty paragraph with a `section.research-chat-empty` containing three real buttons:

```html
<button type="button" data-research-empty-action="question">直接提问</button>
<button type="button" data-research-empty-action="file">添加资料</button>
<button type="button" data-research-empty-action="url">导入链接</button>
```

- [ ] In the existing timeline click listener: `question` focuses `research-question`; `file` invokes `research-file-input.click()`; `url` opens existing tools and URL panel then focuses `research-source-url`. Keep cancellation in this same listener and create no new upload or request path.
- [ ] In `renderAll()`, set `research-local-archive-state` to `本机存档` when `state.archiveAvailable` and `临时会话` otherwise. It must not read/store credentials.
- [ ] Retain and refine current `motionAllowed`, `killResearchMotion`, `runArchiveEntranceMotion`, and `runSessionSwitchMotion`; include only command strip/evidence selectors within the existing 160-260ms budget.
- [ ] Verify syntax and portal contracts:

```bash
node --check mergeKit_beta/static/model_gateway/research.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest tests.model_gateway.test_gateway_portal
```

Expected: both pass. Any failure in direct chat, citation routing, archive privacy, or source expiry tests stops the batch.

## Task 5: Browser and Runtime Acceptance

**Files:** `docs/model_gateway/ACCEPTANCE_<date>_RESEARCH_COMMAND_DESK.md`

- [ ] Run `git diff --check`, both Node checks, and the full container suite:

```bash
git diff --check
node --check mergeKit_beta/static/model_gateway/research.js
node --check mergeKit_beta/static/model_gateway/research_archive.js
docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests
```

- [ ] In a temporary Firefox WebDriver profile with a shell cleanup trap, seed only synthetic IndexedDB sessions through `MergeneticResearchArchive`. Do not set API keys or request a model. Verify desktop `1440x960` has a 286px rail, visible command strip, no horizontal overflow, and fitting composer controls. Verify each empty shortcut performs its specified focus/input action; evidence `<details>` opens by keyboard; at `390x844` the rail opens, Escape closes it, and focus returns to `research-session-drawer-toggle`; reduced motion reports `motionAllowed() === false`.
- [ ] Save desktop/mobile screenshots under ignored `logs/model_gateway/acceptance/<timestamp>/`. Repeat the seven endpoint checks and GPU/process snapshot from Gate 0. All endpoints must remain `200`, GPU 2 ownership must remain unchanged, and no new GPU process may exist.
- [ ] Record exact outputs, screenshot paths, expected Git changes, rollback archive checksum, and any pre-existing warning. Mark accepted only when every gate passes.

## Stop Conditions and Completion

Stop and roll back the current batch on: new unit failure; endpoint not `200`; changed direct-chat/citation/source-expiry behavior; prohibited CSS effect; external asset/dependency; GPU process change; keyboard-focus failure; text/control overlap at tested widths; or manifest-external Git change.

The work is complete only when the cold-grey/amber palette, material planes, evidence index, composition deck, responsive behavior, and all acceptance evidence match `RESEARCH_COMMAND_DESK_DESIGN.md` without changing any research system contract.
