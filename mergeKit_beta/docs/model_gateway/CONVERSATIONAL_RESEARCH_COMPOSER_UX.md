# Conversational Research Composer UX

## Purpose

Make `/research` feel like one natural conversation instead of a collection
of separate modes and side panels. Invited users can ask ordinary questions,
add documents or public URLs from the composer, and receive source-grounded
research results in the same message timeline.

## Confirmed Product Decisions

- The user portal has one conversation surface. It does not expose separate
  chat, research, or focus modes.
- The composer decides execution automatically:
  - no selected ready source: existing direct `/v1/chat/completions` request;
  - one or more selected ready sources: existing source-backed research job
    with citations.
- File and URL imports belong to the current browser session and are reusable
  for later turns. Every send selects all ready sources by default; a user may
  temporarily exclude an individual source from that send.
- A right-side composer tool icon opens a small anchored toolbar with exactly
  three first-version actions: upload file, import public URL, and view current
  session materials.
- API Key, model selection, and complete material management move to an
  on-demand session-settings drawer. They are not permanently visible beside
  the conversation.
- If a user sends while a selected source is still processing, the user message
  appears immediately with a visible preparing state. The queued turn starts
  automatically after all selected sources are ready.
- Message text, rendered answer/result summaries and selected source IDs are
  retained in browser `sessionStorage` for reload within the same browser
  session. Closing the browser clears this local state. No new server-side
  conversation history is created; existing server TTL rules remain
  authoritative for files, chunks, indexes, jobs, and results.
- Explicitly clearing or replacing an API Key clears local source selections,
  source metadata and pending job references before the new Key is used. Text
  conversation entries remain local to the browser session, but a later Key
  cannot inspect or submit the previous Key's source IDs.

## Interaction Structure

```text
Header: Mergenetic | session settings

Timeline
  user question
  assistant answer or research result + citations
  preparing / queued / failed state where applicable

Selected materials strip (only when a source exists)
  [report.pdf ready x] [paper URL processing x] [2 more]

Composer
  question textarea                         [tools] [send]
                                     tools: upload | link | materials
```

The main timeline remains the source of truth for user-facing task state.
The materials strip gives a compact current-selection view; the drawer holds
the complete session list, model choice, API Key controls, removal actions and
status detail.

## Data And Execution Flow

1. The user chooses upload or URL from the composer toolbar.
2. Existing `addUpload()` or `addUrl()` APIs create a short-lived owned source
   and the existing polling path reports its status.
3. The client stores source IDs, source display metadata, selected state, model
   selection, user message text and rendered assistant/result summaries in
   `sessionStorage`. It never stores source body, parsed text, API Key plaintext
   beyond the existing session-key policy, raw API responses or new server-side
   history.
4. On send, the client snapshots selected source IDs. With none, it uses
   existing direct chat. With any ready source, it creates the existing
   citation-required research job.
5. With pending selected sources, the client appends a queued message state and
   waits through the existing file polling mechanism. It starts exactly once
   when the snapshot becomes ready; failed/canceled sources convert that turn
   to a readable failure state and do not silently fall back to uncited chat.
6. Research completion, failure and cancellation are rendered in the same
   timeline. Existing result projection and citation DOM safety remain in use.

## Error Handling

- Unsupported, oversize, unsafe, password-protected, malformed or blocked URL
  sources continue to be rejected by existing backend validation. The toolbar
  reports the returned error and does not add a fake attachment chip.
- A selected source that fails processing blocks only the affected waiting
  turn. The user can remove it and resend; no direct-chat fallback is made
  because that could produce a response without the requested evidence.
- Duplicate click or reload cannot create duplicate jobs: the pending turn has
  a browser-generated idempotency key, persisted with its snapshot until it
  reaches a terminal UI state.
- On reload, missing/expired server sources are removed from local session
  metadata and displayed as expired in the timeline; no stale ID is sent.

## Visual And Accessibility Rules

- The toolbar is anchored above the composer action area in normal document
  flow, uses a small surface with an explicit close action, and never covers
  the send button or text input.
- Attachments are concise, text-labelled chips with status text and a remove
  button. They wrap before overflowing. Colour supplements, never replaces,
  ready/processing/failed state text.
- The session drawer is modal on small screens and a side sheet on desktop;
  focus is moved into it when opened and restored to the triggering settings
  button when closed.
- Tool, attachment, drawer and send controls have hover, active and
  `:focus-visible` states. Reduced-motion mode uses immediate state changes.
- GSAP remains optional and may animate only opening/closing surfaces with
  `transform` and `opacity`; no continuous effect or layout-property animation
  is introduced.

## Scope And Non-Goals

- Reuse the existing upload, URL import, SSRF protection, ClamAV, parser,
  queue, retrieval, citation and cancellation APIs. Do not change their limits
  or security policy in this UX batch.
- Do not implement a new chat-history database, long-term server persistence,
  billing, tool-calling loop, remote image input, OCR, a new frontend framework
  or model/GPU behavior.
- Do not create a Gemini visual clone. The interaction is conversational, but
  typography, colours, layout and Mergenetic identity remain local.

## Acceptance And Rollback

1. A first visit exposes one timeline and one composer, not a mode switcher or
   permanent source sidebar.
2. Upload, public URL import, material selection/removal, direct chat and
   citation-backed research all remain reachable by keyboard and mouse.
3. Ready sources lead to citation-backed jobs; no sources lead to direct chat;
   pending sources create one waiting turn and start once after readiness.
4. Reload restores only session-scoped metadata. Closing the browser removes
   local state; existing server expiration remains unchanged.
5. API contract, portal unit tests, Node syntax checks, browser desktop/mobile
   checks and the full container suite pass without starting a model, fusion,
   Ray, vLLM or GPU job.
6. Rollback is limited to the research template, CSS, JavaScript, relevant
   portal tests and this documentation. It does not delete server data or alter
   queues, databases, Compose configuration, model service state or GPU use.

## Design Self-Review

- The automatic routing rule is explicit and never substitutes uncited chat for
  a requested source-backed turn.
- Browser-session persistence has an explicit message/summary-only boundary
  for conversation content and a reload-expiry path for sources.
- Waiting turns have a single-start and idempotency requirement, avoiding
  duplicate jobs after polling or reload.
- The design is one focused UX batch and reuses existing backend contracts.
