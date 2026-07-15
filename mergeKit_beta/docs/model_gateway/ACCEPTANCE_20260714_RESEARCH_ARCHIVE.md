# Research Archive Acceptance (2026-07-14)

## Scope

This record covers the `/research` Research Archive UX batch.

- Browser-local persistent conversations use `IndexedDB` only.
- The desktop session rail and mobile session drawer manage at most 20 local
  conversations.
- Direct chat, citation-required research, source status, idempotency,
  cancellation and API Key boundaries retain their existing Gateway routes.
- No backend API, queue, database, source TTL, model runtime, vLLM, Ray,
  Compose or GPU scheduling code changed.

## Baseline And Rollback Point

Pre-change archive:

`/tmp/mergenetic_research_archive_prechange.tar.gz`

SHA-256:

`626d915aa71bc105e0a243995cee62a817717cf7cee92c788ef158c1ace77cf4`

It contains only the original research portal template/CSS/JS, its portal
test and the approved archive design/plan documents. It does not contain
server data.

## Implementation Evidence

- `static/model_gateway/research_archive.js` is a dependency-free IndexedDB
  module. It uses `mergeneticResearchArchiveV1`, stores only whitelisted
  session/message/source display metadata, limits a session to 60 turns and
  256 KiB, and rejects a new twenty-first session without deleting another.
- It does not contain API-Key or authorization handling. `localStorage` keeps
  only `mergeneticResearchArchiveActiveSessionId`; existing Key policy remains
  in the portal script.
- Legacy bounded `sessionStorage` conversation state migrates only after a
  successful archive save. Archive failure keeps session-only behaviour.
- The user portal now has a stable archive rail, local search, new/rename/
  delete/clear controls, native named confirmation dialogs and a mobile rail
  drawer. The original source toolbar, settings drawer and request IDs remain.
- Missing sources render as expired, are unselected and show
  `资料已过期，无法用于新的研究请求` when the user attempts to reuse them.
- Visual hierarchy uses Mergenetic's original graphite/paper/signal-red token
  system and cut-marker selection state. It does not include copied game or
  third-party product assets, gradients or continuous effects.
- GSAP is optional and limited to cancellable `transform`/`autoAlpha` entry
  and session-switch timelines. Reduced motion disables the timelines.

## Automated Gates

| Gate | Command / Evidence | Result |
| --- | --- | --- |
| Compose | `docker compose config --quiet` | Passed |
| Services | `docker compose ps` | Main app healthy; ClamAV, PostgreSQL and Redis healthy; research worker running |
| HTTP | `/healthz`, `/readyz`, `/api/models`, `/api/testset/list`, `/api/history`, `/research`, `/model-gateway` | All HTTP 200 |
| Portal contracts | `python -m unittest tests.model_gateway.test_gateway_portal` | 22 passed |
| Full suite | `python -m unittest discover -s tests` | 137 passed in 9.103 seconds |
| JavaScript syntax | `node --check research_archive.js` and `research.js` | Passed |
| Whitespace/diff | `git diff --check` on tracked path plus changed-file whitespace scan | Passed |
| Acceptance artifacts | `git check-ignore -v logs/model_gateway/acceptance/...` | Passed via root `.gitignore` |

The full suite emitted existing CUDA-platform detection and Python Swig
deprecation warnings but completed successfully.

## Browser Evidence

Firefox WebDriver used only synthetic local session data. Accepted evidence is
stored in the ignored directory:

`logs/model_gateway/acceptance/20260714_research_archive/`

- `research-archive-desktop.png`: desktop rail, active cut-marker, renamed
  session, research stage and source count.
- `research-archive-mobile-rail.png`: mobile session drawer, active session,
  source count and normal-flow composer behind the drawer.

Verified in the browser:

1. A fresh archive created an active local session; no API Key was present.
2. Twenty synthetic sessions rendered in recency groups. Local message search
   returned one matching session without mutating the archive.
3. Creating a twenty-first session opened the limit dialog. After deleting a
   named session, new-session creation returned the archive to exactly 20
   records.
4. Native rename dialog changed both the rail row and active title.
5. A sentinel Key was absent from serialized IndexedDB records before reload;
   the sentinel was removed before the accepted browser reload, so the final
   scenario had no API Key and did not make a model request.
6. An expired synthetic source stayed visible but could not be selected for a
   new research request; the explicit expired-source message appeared.
7. Desktop measured `innerWidth=1440`, `scrollWidth=1428`. Firefox headless
   clamped a requested 390px window to `innerWidth=500`, where
   `scrollWidth=488`; the mobile rail opened and Escape restored focus to
   `research-session-drawer-toggle`.
8. A separate Firefox session with `ui.prefersReducedMotion=1` reported the
   reduced-motion media query as active.

One preliminary browser smoke run set a synthetic Key before reload and
received an expected invalid-Key response from the existing model-list route.
It started no model or GPU work and is not used as acceptance evidence. The
accepted browser run removed that synthetic Key before reload.

## GPU And Process Safety

Before and after acceptance:

- GPUs 0, 1 and 3 each used 18 MiB and had no compute process.
- GPU 2 retained the same external `/usr/bin/python3` process (1116 MiB) and
  `VLLM::EngineCore` process (12484 MiB), for about 13.63 GiB total.
- No model, fusion, Ray, vLLM, upload, URL fetch or research job was started
  by this UX batch.
- No geckodriver, Firefox headless or Playwright process remained after the
  browser checks.

## Deferred Runtime Gate

The portal's routing and source-expiry contracts are accepted with synthetic
browser state and existing route tests. A real file-backed answer remains a
separate administrator-approved runtime gate: it requires a published model,
a valid invited-user Key and an idle non-GPU-2 device. It is intentionally not
part of this UI acceptance.

## Post-Change Rollback

Accepted-state archive:

`/tmp/mergenetic_research_archive_accepted.tar.gz`

SHA-256:

`a87edae6ac1e2244ab5a755b82cf2209213eaf2d14151b2235a9dbeed9f8afb0`

It contains only the research archive implementation, tests and design/plan
records, not this acceptance report. Restore it with:

```bash
tar -xzf /tmp/mergenetic_research_archive_accepted.tar.gz \
  -C /home/a/Workspace/Model_factory
```

Do not use `git reset`, delete Gateway source/job data, change databases or
stop the external GPU 2 processes as part of rollback.
