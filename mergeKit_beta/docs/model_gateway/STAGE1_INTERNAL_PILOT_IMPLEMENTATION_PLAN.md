# Stage 1 Internal Pilot Implementation Plan

> Use the approved hardening design in `STAGE1_INTERNAL_PILOT_HARDENING.md`.
> Every task is independently verified before the next begins.

## Task 1: Baseline and Documentation

- Record Git revision, Compose service state, HTTP smoke responses and GPU
  snapshot under ignored `logs/model_gateway/acceptance/<timestamp>/`.
- Update `TEXT_RESEARCH_STAGE1.md` so legacy DOC/PPT support names the private
  Apache POI parser instead of the removed conversion placeholder.
- Rollback: revert only the documentation commit or restore the baseline tag.

## Task 2: Secure Web Sources

- Change URL submission from a passive `queued_download` record to a queued
  web-source delivery. The existing no-GPU Worker performs bounded HTTP(S)
  acquisition and revalidates every redirect before following it.
- Accept only HTML and PDF. Scan the staged response with ClamAV, then reuse
  the existing PDF parser or a standard-library HTML text extractor. HTML
  produces title, visible paragraph, table and image-alt/caption sections with
  stable web locators.
- Add focused tests first for redirect rejection, HTML locator extraction,
  image-context extraction and the same ready-state transition used by uploads.
- Rollback: reject URL sources again; uploads and existing research data stay
  unaffected.

## Task 3: Durable API-Key Quotas

- Add a Gateway-bound quota bucket model with a unique Key, scope and window
  start. Counters are updated inside the request transaction.
- Add configuration values for the four approved limits and human-readable
  comments adjacent to each value.
- Enforce chat requests, research job creation, and file/URL import admission.
  Return `429` plus `Retry-After` and a stable code.
- Add focused tests first for window exhaustion, active-job cancellation release,
  daily byte rejection and owner isolation; observe each test fail before code.
- Rollback: revert this batch. Existing Key and research tables remain valid.

## Task 4: Expiry Cleanup

- Add one cleanup unit that finds expired sources, jobs and chunks, unlinks only
  contained runtime files, then deletes the rows in a safe dependency order.
- Call it periodically from the existing no-GPU file Worker, with a configurable
  reconciliation interval.
- Add focused tests first for expired content removal, runtime path containment
  and preservation of non-expired/foreign-Key records.
- Rollback: stop the file Worker and revert this batch; no core model-factory
  process is affected.

## Task 5: Regression and Real Acceptance

- Run all container tests and Compose/HTTP smoke.
- Create a temporary invited Key through the real administrator API. Exercise
  rate/quota errors through real HTTP routes using temporarily reduced limits,
  then restore defaults.
- Start the verified Qwen 7B only on a confirmed-free GPU 0 or 1. Submit a
  neutral public scientific web page and upload a public scientific PDF, wait
  for both to become `ready`, submit research, verify web/PDF citations against
  their original sources, then submit/cancel a second task.
- Temporarily lower TTL, wait for the real cleanup timer, and verify source,
  chunk, payload and result disappearance through the owner APIs and database.
- Stop the model, revoke the Key, remove all temporary records, recheck GPU
  state, write acceptance evidence and commit/tag/push the batch.
