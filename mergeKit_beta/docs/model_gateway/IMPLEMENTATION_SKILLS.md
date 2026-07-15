# Model Gateway Implementation Skills

| Skill | Local path | Required use |
| --- | --- | --- |
| Brainstorming | `/home/a/.codex/skills/brainstorming/SKILL.md` | Scope approval before new capability batches. |
| Ponytail | `/home/a/.codex/skills/ponytail/SKILL.md` | Reuse Flask, SQLAlchemy and stdlib patterns first. |
| Systematic debugging | `/home/a/.codex/skills/systematic-debugging/SKILL.md` | Root-cause investigation for failed gates. |
| Test-driven development | `/home/a/.codex/skills/test-driven-development/SKILL.md` | Failing behavior test before code. |
| Verification before completion | `/home/a/.codex/skills/verification-before-completion/SKILL.md` | Fresh verification evidence for each gate. |
| Frontend design | `/home/a/.codex/skills/frontend-design/SKILL.md` | Research workspace hierarchy and Mergenetic visual language. |
| Design taste | `/home/a/.codex/skills/taste-skill/SKILL.md` | Anti-template visual review. |
| GSAP core/performance | `/home/a/.codex/skills/gsap-core/SKILL.md`, `/home/a/.codex/skills/gsap-performance/SKILL.md` | Compositor-safe motion and reduced motion. |

## Admin Portal Visual Batch (2026-07-14)

- Active skills: `brainstorming`, `writing-plans`, `executing-plans`, `ponytail`,
  `test-driven-development`, `frontend-design`, `design-taste-frontend`,
  `gsap-core`, `gsap-performance`, `verification-before-completion`, and
  `requesting-code-review` from the paths listed above.
- Boundary: only `/research` and `/model-gateway` presentation, their static
  portal contracts, and documentation may change. Gateway APIs, database
  state, Redis, model runtime, vLLM, Ray, Docker Compose, and GPU scheduling
  are out of scope.
- Runtime safety: no model, fusion, vLLM, Ray, or GPU task is started for this
  batch. Existing Compose services remain running and GPU snapshots are
  recorded only during final acceptance.

## Conversational Research Composer Batch (2026-07-14)

- Active skills: `brainstorming`, `writing-plans`, `executing-plans`,
  `ponytail`, `systematic-debugging`, `test-driven-development`,
  `frontend-design`, `design-taste-frontend`, `gsap-core`,
  `gsap-performance`, `verification-before-completion`, and
  `requesting-code-review` from the paths listed above.
- Scope: only the user-facing `/research` template, static assets, portal
  contract tests and the conversational-composer records. Existing Gateway
  APIs, source security pipeline, queues, data retention, model runtime, Ray,
  vLLM, Compose and GPU scheduling remain unchanged.
- Runtime safety: browser tests inject only synthetic `sessionStorage` data.
  They do not upload sources, fetch URLs, invoke a model, start a research job,
  or allocate GPU resources.

## Research Archive Workspace Batch (2026-07-14)

- Active skills: `brainstorming`, `writing-plans`, `executing-plans`,
  `ponytail`, `systematic-debugging`, `test-driven-development`,
  `frontend-design`, `design-taste-frontend`, `gsap-core`,
  `gsap-performance`, `verification-before-completion`, and
  `requesting-code-review` from the paths listed above.
- Scope: browser-local conversation archive, archive rail/drawer, original
  visual system and motion in `/research`. Gateway routes, source security,
  queues, databases, model process control, vLLM, Ray, Docker and GPU policy
  remain unchanged.
- Runtime safety: Firefox tests use synthetic IndexedDB records only. The final
  accepted run has no API Key and does not upload a source, import a URL,
  create a research job, start a model or allocate GPU work.

## Research Command Desk Batch (2026-07-14)

- Active skills: `brainstorming`, `writing-plans`, `executing-plans`,
  `ponytail`, `test-driven-development`, `frontend-design`,
  `design-taste-frontend`, `gsap-core`, `gsap-performance`, and
  `verification-before-completion` from the paths listed above.
- Scope: the `/research` presentation, its static portal contracts and the
  Command Desk records only. Gateway APIs, local archive storage shape, source
  security, queues, databases, model runtime, Ray, vLLM, Compose and GPU
  scheduling remain unchanged.
- Runtime safety: browser acceptance may use synthetic IndexedDB data only. It
  must not supply an API Key, upload a source, import a URL, create a research
  task, start a model or allocate GPU work. GPU 2 remains external work.

## Legacy Office Parser Batch (2026-07-15)

- Active skills: `brainstorming`, `writing-plans`, `executing-plans`,
  `ponytail`, `systematic-debugging`, `test-driven-development`, and
  `verification-before-completion` from the paths listed above.
- Scope: the no-GPU legacy DOC/PPT text parsing boundary, its Worker
  integration, Compose isolation and tests only. User APIs, Gateway storage,
  retrieval, model runtime, Ray, vLLM, fusion and evaluation remain unchanged.
- Runtime safety: dependency download uses a temporary proxy that is stopped
  at build completion. The parser has no GPU, no published port and only the
  private Worker network. Real fixture checks use disposable files and do not
  start a model, research task, fusion, evaluation or GPU workload.
