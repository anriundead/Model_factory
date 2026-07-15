# Conversational Research Composer Acceptance (2026-07-14)

## Scope

This record covers the user-facing `/research` conversational-composer batch
specified by `CONVERSATIONAL_RESEARCH_COMPOSER_UX.md` and
`CONVERSATIONAL_RESEARCH_COMPOSER_IMPLEMENTATION_PLAN.md`.

- One conversation timeline and one composer replace the visible research-mode
  split and permanent source sidebar.
- The composer `+` menu provides upload, public URL import and session
  materials. Selected sources are reused in the current browser session.
- No selected source keeps the existing direct OpenAI-compatible chat route.
  Selected sources keep the citation-required research-job route.
- This batch changes no Gateway API, source-security policy, database, queue,
  worker, model process, vLLM, Ray, Docker Compose, or GPU scheduling.

## Implementation Evidence

- `processing`, the actual file-worker state, is handled as pending. A waiting
  research turn therefore remains citation-bound until its sources are ready.
- Source state labels are rendered in concise Chinese (`已就绪`, `处理中`, and
  related labels) without changing the backend state values.
- The link panel no longer nests a form inside the main composer form. The
  explicit `research-submit-url` button prevents HTML parser relocation and
  keeps the event listener bound.
- `research-source-list` has a programmatic focus target for the keyboard
  "查看资料" action.

## Automated Gates

| Gate | Evidence | Result |
| --- | --- | --- |
| Compose configuration | `docker compose config --quiet` | Passed |
| Running services | `docker compose ps` | Main app healthy; ClamAV, PostgreSQL and Redis healthy; research worker running |
| HTTP smoke | `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`, `/research`, `/model-gateway` | All HTTP 200 |
| Composer portal contracts | `python -m unittest tests.model_gateway.test_gateway_portal` | 16 passed |
| Idempotency regression | Two `TestResearchRoutes` idempotency tests with portal contracts | 18 passed |
| Full container suite | `python -m unittest discover -s tests` | 131 passed in 9.189s |
| JavaScript syntax | `node --check static/model_gateway/research.js` | Passed |
| Whitespace/diff check | `git diff --check` on changed tracked paths and trailing-whitespace scan of composer files | Passed |
| Screenshot ignore rule | `git check-ignore -v logs/model_gateway/acceptance/...` | Passed via root `.gitignore` |

The full suite emitted an existing CUDA-platform detection message and Python
Swig deprecation warnings, but finished successfully. No model request,
source upload, URL fetch, research job, fusion, Ray, or vLLM process was
started by this batch.

## Browser Evidence

Firefox WebDriver used synthetic, non-sensitive `sessionStorage` values only.
It did not attach an API Key or call a model or source endpoint. Evidence is
stored locally and ignored by Git:

`logs/model_gateway/acceptance/20260714_conversational_composer/`

- `composer-desktop-tools.png`: restored conversation, attachment strip and
  open source tools at 1440px.
- `composer-mobile-drawer.png`: drawer with session sources at Firefox's
  actual narrow viewport.

Observed DOM checks:

- Desktop: `innerWidth=1440`, `scrollWidth=1440`.
- Requested 390px narrow window: Firefox headless clamped to
  `innerWidth=500`; `scrollWidth=488`.
- First Escape closed the source tools while retaining the open drawer;
  second Escape closed the drawer and restored focus to
  `research-session-settings`.
- Reload restored synthetic message summaries and source metadata, including
  `已就绪` and `处理中` labels.

## GPU And Process Safety

Pre- and post-test snapshots were unchanged for GPUs 0, 1 and 3: each used
18 MiB and had no compute process. GPU 2 remained occupied by pre-existing
external processes:

- `/usr/bin/python3`, 1116 MiB
- `VLLM::EngineCore`, 12484 MiB

The batch did not start, stop or send work to either process. No temporary
Firefox, geckodriver, or Playwright process remained after browser checks.

## Remaining Runtime Boundary

The UI routing, persistence, cancellation contract and browser interaction
are accepted. A real source-backed answer remains a separate Gate 5 runtime
exercise because it requires an administrator-started published model and a
valid invited-user API Key. It must use an approved idle GPU only and must not
affect the existing GPU 2 workload.

## Rollback

The shared worktree contains untracked Gateway portal files, so Git cannot
mechanically restore a pre-batch version of those files. The accepted-state
archive below is the rollback point for changes made after this acceptance:

`/tmp/mergenetic_research_composer_20260714_accepted.tar.gz`

It contains only `.gitignore`, the research template/CSS/JS, the portal test,
and the UX, implementation-plan and skill records. The acceptance report is
not archived, so its audit notes cannot change the rollback payload. SHA-256:

`ef583153396d83ab23ea353498a6013568d9c28baa02cf086f76d4639571e44d`

To restore this accepted state later:

```bash
tar -xzf /tmp/mergenetic_research_composer_20260714_accepted.tar.gz \
  -C /home/a/Workspace/Model_factory
```

Do not use `git reset`, delete source data, modify Gateway databases, or stop
the external GPU 2 processes as part of this UI rollback.
