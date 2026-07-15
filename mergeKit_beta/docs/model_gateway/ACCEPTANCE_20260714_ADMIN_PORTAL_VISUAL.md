# Admin Portal Visual Acceptance (2026-07-14)

## Scope

This record covers the visual-only Gateway portal batch described in
`ADMIN_PORTAL_VISUAL_IMPLEMENTATION_PLAN.md`.

- `/research` is a user workspace and no longer links to `/model-gateway`.
- `/model-gateway` keeps its existing publication, API Key, direct-call,
  request cancellation, and OpenAI-compatible integration controls.
- No API, database, Redis, worker, model runtime, vLLM, Ray, Compose, or GPU
  behavior was changed.

## Automated Evidence

| Gate | Command | Result |
| --- | --- | --- |
| Compose | `docker compose config --quiet` | Passed |
| Service state | `docker compose ps` | Main app and ClamAV/PostgreSQL/Redis healthy; research worker running |
| Health | `curl -fsS http://127.0.0.1:5000/healthz` | `{"ok": true}` |
| Readiness | `curl -fsS http://127.0.0.1:5000/readyz` | Database and merge directory `ok` |
| Portal HTTP | `/research`, `/model-gateway` | Both returned HTTP 200 |
| Python tests | `docker compose exec -T mergekit-beta /opt/conda/envs/mergenetic/bin/python -m unittest discover -s tests` | 129 tests passed |
| JavaScript syntax | `node --check static/model_gateway/{research,console}.js` | Passed |
| Diff whitespace | `git diff --check` for touched portal/docs paths | Passed |
| GPU | `nvidia-smi --query-gpu=...` | GPUs 0-3 each at 18 MiB; no compute applications |

The full suite emitted pre-existing dependency warnings while importing CUDA
capability code, but completed successfully. No inference, fusion, vLLM, Ray,
or model process was started by this batch.

## Browser Evidence

Settled screenshots are stored in the ignored directory:

`logs/model_gateway/acceptance/20260714_admin_portal_visual/`

Files:

- `admin-desktop-settled.png`
- `admin-mobile-settled.png`
- `research-desktop-settled.png`
- `research-mobile-settled.png`

Firefox WebDriver waited three seconds after navigation before capture. The
first command-line screenshot path form was ignored by Firefox 152, so the
accepted screenshots use WebDriver rather than the browser's `--screenshot
<path>` form. The WebDriver and Firefox processes were explicitly stopped.

Visual inspection found no overlap in the captured desktop/mobile layouts:

- The administrator status board, connection panel, release flow, form and
  Key panel remain in normal document flow.
- Release steps become a single column below the mobile breakpoint.
- Long-form fields and generated status content have wrapping/overflow rules.
- The research workspace keeps only user-facing navigation and a concise
  direct-question/source-backed-research instruction.

DOM width checks passed without horizontal overflow at 1440px, 1024px and
720px for both pages. Firefox headless clamps a requested 390px window to a
500px viewport; the captured mobile screenshot therefore represents the
engine's smallest viewport. The `max-width: 720px` CSS contract is covered by
unit tests and the 720px DOM check.

## Accessibility And Motion

- Hover, active and `:focus-visible` feedback are provided for navigation,
  buttons, form controls, dynamic list rows and the copy control.
- Status labels retain text in addition to colour.
- Existing GSAP entry motion is restricted to `transform` and `autoAlpha` and
  skipped for `prefers-reduced-motion: reduce`; CSS also disables transitions
  and hover transforms for that preference.

## Rollback

No persisted state changed. Roll back only the current batch's portal files
and documentation. Before any rollback in the shared dirty worktree, export
the current patch so unrelated work remains intact:

```bash
git diff -- \
  mergeKit_beta/templates/model_gateway/research.html \
  mergeKit_beta/static/model_gateway/research.css \
  mergeKit_beta/templates/model_gateway/console.html \
  mergeKit_beta/static/model_gateway/console.css \
  mergeKit_beta/tests/model_gateway/test_gateway_portal.py \
  mergeKit_beta/docs/model_gateway \
  > /tmp/mergenetic_admin_portal_visual_rollback.patch
```

Then revert only the files that were changed by this batch. Do not touch
Gateway data, SQLite/PostgreSQL, Compose, Redis, ClamAV, models, or GPU
configuration.
