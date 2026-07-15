# Research Archive Workspace Design

> Status: proposed design for review
> Date: 2026-07-14
> Scope: user-facing `/research` workspace only

## Decision

Replace the current sparse single-column research surface with the
**Mergenetic Research Archive**: a persistent conversation workspace with a
desktop session rail, a focused research timeline and a compact context rail.
It keeps the existing automatic execution rule:

- no selected ready source: direct OpenAI-compatible chat;
- selected source: citation-required research job;
- pending source: visible waiting turn, never an uncited fallback.

The design borrows interaction patterns found in mature AI conversation tools:
stable recent-conversation navigation, explicit creation, visible active
context, and a composer that keeps source actions close to the prompt. It does
not copy Gemini, Grok, Persona, or any other product's assets, fonts, wording
or layout.

## Product Boundaries

### Included

- Local, long-lived session history on the user's browser only.
- A desktop left session rail and a mobile session drawer.
- At most 20 locally stored sessions, grouped by recency and searchable by
  locally stored title and message text.
- New, rename, switch, delete and clear-local-history controls.
- Restored active-session history after reload and browser restart.
- A redesigned research header, composer, source strip, timeline and settings
  drawer with deliberate motion and complete interaction states.

### Excluded

- No server-side conversation history, account synchronisation, sharing,
  billing, user profile, or database schema change.
- No change to source ownership, upload validation, SSRF controls, queue,
  retrieval, citations, cancellation, model service, vLLM, Ray or GPU logic.
- No game assets, game names, copied typography, character art, sound effects
  or visual clone. Persona is a reference for information hierarchy and
  decisive state feedback only.

## Information Architecture

```text
Desktop (>= 980px)

+---------------------------+-----------------------------------------------+
| Mergenetic                | RESEARCH / active session title               |
| RESEARCH ARCHIVE           | source count | model state | session settings |
| [ + 新建会话 ]             +-----------------------------------------------+
| [ search sessions       ] | timeline                                      |
|                            |  user question                                |
| TODAY                      |  assistant answer / research result            |
|  active session            |  citation locator                              |
|  recent session            |                                               |
|                            | selected-source strip                          |
| LAST 7 DAYS                | composer: prompt | source tools | send         |
|  sessions, max 20 total    |                                               |
|                            |                                               |
| [ local only ] [ clear ]  |                                               |
+---------------------------+-----------------------------------------------+
```

### Left Session Rail

- Fixed desktop width: 272px. It is never silently removed on desktop.
- It contains one primary command, `新建会话`, then local search and sessions
  grouped as `今天`, `最近 7 天`, and `更早`.
- Each row shows a concise title, last update time and a small source-count
  indicator. The selected row uses a red cut-marker and text label, not colour
  alone.
- Row actions are keyboard reachable: rename and delete. Deletion requires a
  small confirmation dialog naming the local session; it never calls a server
  delete endpoint.
- `清除本机记录` is a destructive, separately confirmed action that removes
  all local session records but leaves server-owned sources to their existing
  TTL.
- At narrow widths, the rail becomes a modal drawer opened by a labelled
  `会话` button. The active session title remains in the top bar.

### Main Research Stage

- The top bar presents an identity lockup, the active session title and compact
  context status. It does not contain a competing administration link.
- The timeline is the sole task narrative: questions, answers, citations,
  preparation, queue, error and cancellation states appear in chronological
  order.
- The source strip remains directly above the composer. It exposes selection
  and status, while the session settings drawer retains full model and source
  management.
- The composer remains the single entry point. Its right-side source icon
  opens upload, public-link and current-material actions without moving the
  user out of the conversation.

## Local Session Persistence

### Storage Choice

Use browser-native `IndexedDB`, database name
`mergeneticResearchArchiveV1`, with no new dependency. `localStorage` holds
only the non-secret `activeSessionId` pointer. IndexedDB is asynchronous,
handles bounded text history safely and avoids the small synchronous payload
limit of `localStorage`.

### Session Record

```text
ResearchSessionRecord
  id: UUID generated in browser
  title: user title or first question truncated to 48 characters
  createdAt / updatedAt: epoch milliseconds
  messages: [{ id, role, content, status, route, model, citations, locators }]
  sources: [{ id, name, sourceKind, status, selected }]
  selectedModel: model identifier only
```

The record deliberately excludes API Keys, authorization headers, raw file
bytes, parsed text, chunks, embeddings, full research payloads, raw HTTP
responses and hidden task metadata.

### Capacity And Expiry

- Display and store at most 20 sessions. This is a privacy and storage bound,
  not a server quota.
- Creating a 21st session must not silently erase a user's work. The UI asks
  the user to delete one or more older local sessions, or cancel creation.
- Source IDs are retained only as local display metadata. Before a restored
  session can send a research request, the existing ownership/status lookup
  runs. Missing or expired sources are displayed as `资料已过期`, excluded from
  the new request and never silently substituted with uncited chat.
- A migration imports the existing bounded `sessionStorage` conversation into
  one IndexedDB session on first upgrade, then removes the old snapshot only
  after the new record is successfully committed.

## Visual Direction: Research Archive Signal System

### Design Principle

The workspace should feel like a precise research console, not a generic chat
page. The visual signature is a **diagonal signal cut**: an asymmetric red
marker that identifies the active session, active context and send action.
Every other surface remains disciplined so long technical answers are easy to
read.

### Token Direction

| Role | Proposed value | Use |
| --- | --- | --- |
| Archive ink | `#17191D` | rail, strong text, high-contrast surface |
| Paper signal | `#F6F7F4` | main reading background |
| Signal red | `#D9362B` | active session, primary command, important state |
| Graphite | `#34383F` | secondary structure and dividers |
| Steel | `#9AA2AC` | muted metadata |
| Verified green | `#14805E` | ready/success status only |

The primary red is reserved for active choice and command emphasis. Green is
semantic status, not a competing brand accent. Gradients, glow fields, bokeh,
ambient particles and decorative illustrations are prohibited.

### Typography And Shape

- Keep MiSans for body readability. Add a narrow utility treatment through
  weight, uppercase Latin labels and stable tabular numerals rather than adding
  an unverified font dependency.
- Use a sharp, documented radius system: 4px for controls, 8px for content
  surfaces, no pills except compact count/status tags.
- Use diagonal pseudo-element markers and hard divider lines to encode state;
  never use arbitrary slanted containers that compromise readable text width.
- The Mergenetic mark becomes a layered `M` archive stamp used in the rail and
  top bar. It remains CSS/HTML, not a copied logo or external artwork.

## Motion And Interaction

GSAP is loaded already and will be used only when available and when reduced
motion is not requested. CSS retains all essential hover, focus and press
feedback as the functional fallback.

| Interaction | Treatment | Budget |
| --- | --- | --- |
| Initial stage | Rail stamp, title and first visible timeline items enter as one sequence | 260ms, `opacity` + `transform` |
| Session switch | Previous timeline fades/moves 12px out; new timeline enters after state swap | 220ms, cancellable timeline |
| Rail selection | Red cut-marker sweeps into the selected row; label changes immediately | 160ms, `transform` only |
| Composer tools/drawer | Surface fades and translates from its anchor; focus moves after open | 180-220ms, `opacity` + `transform` |
| New message/result | One short upward entry from the composer direction | 180ms, stagger only for new visible items |
| Buttons | Hover elevates 1px; press scales to 0.98; focus is always visible | CSS transition |

- No looped animation, scroll-jacking, layout-property tweening, panel width
  animation, timer-driven visual effect or animation for a background list.
- GSAP timelines are killed before a new session switch begins. They only
  animate visible nodes and never run after the drawer or page is closed.
- `prefers-reduced-motion: reduce` uses immediate state changes and disables
  transform motion while retaining selection, focus and status feedback.

## State And Error Design

- Empty archive: show one directive, `新建会话开始研究`, rather than a marketing
  screen.
- Empty active session: show three compact suggested actions: `直接提问`,
  `添加资料`, `导入链接`.
- A source-backed running turn shows its source count and a readable task
  state in the timeline; cancellation remains where it is today.
- Offline/missing/expired sources keep their visible history but cannot be
  selected for a new request until the user adds a valid replacement.
- Search has a clear empty result state and does not delete or hide unmatched
  sessions permanently.
- Local-storage access failure falls back to the existing bounded
  `sessionStorage` session and shows a concise notice. The user can still use
  the research portal without history persistence.

## Implementation Boundaries

### Template

Add stable, purpose-named regions only:

- `research-session-rail`
- `research-session-search`
- `research-session-list`
- `research-new-session`
- `research-session-drawer-toggle`
- `research-active-session-title`

The existing composer, source APIs, settings drawer IDs and request routes
remain intact to avoid breaking the accepted contracts.

### JavaScript

Split the current portal script into small local responsibilities without a
framework:

- `archiveStore`: IndexedDB open, migration, bounded CRUD and sanitation.
- `sessionController`: active-session selection, title generation, source
  rehydration and retention limit interaction.
- `motionController`: optional GSAP entry/switch/drawer sequences and reduced
  motion guard.
- Existing request functions remain the only code that reaches Gateway APIs.

This is a behavioural refactor inside one static file only if a separate file
would complicate cache/versioning. No frontend state library is justified.

### CSS

Use CSS Grid for desktop rail/main-stage layout and a drawer breakpoint below
980px. Set stable rail width, `min-width: 0`, wrapping rules and mobile touch
targets before adding visual treatment. Motion classes must not change layout
measurements.

## Acceptance Gates

### Automated

- Unit/static contracts cover all new DOM regions, IndexedDB record whitelist,
  migration success/failure, 20-session non-destructive limit, title edit,
  delete/clear confirmation, local search and source-expiry exclusion.
- Existing direct-chat, citation-required research, idempotency, cancellation,
  API-Key clearing and source security tests remain green.
- `node --check`, `git diff --check`, `docker compose config --quiet` and the
  full container `unittest` suite pass.

### Browser

- Desktop: rail remains visible, session switch does not overlap timeline or
  composer, search/title/delete/new controls are keyboard usable.
- Mobile: session drawer traps focus while open, restores focus when closed,
  and does not create horizontal overflow at the actual Firefox narrow width.
- Persistence: refresh and browser restart restore a local session without
  restoring a Key; an expired source cannot submit; a storage failure falls
  back safely.
- Motion: capture normal and reduced-motion runs. Verify animation uses only
  opacity/transforms, cancels on rapid session changes and leaves no active
  GSAP tween after navigation.

### Runtime Safety

Default acceptance uses synthetic local records and stubbed client routes. It
does not upload a file, fetch a URL, create a job, start a model or allocate a
GPU. A real user research task remains a separately approved runtime gate with
an administrator-started model and an idle non-GPU-2 device.

## Rollback

- Export only this UX batch's template/CSS/JS/tests/docs patch before edits.
- IndexedDB migration retains the old `sessionStorage` snapshot until a new
  record transaction succeeds. On migration failure, keep existing behaviour.
- A rollback removes only the `mergeneticResearchArchiveV1` browser database
  and `activeSessionId` pointer if a user explicitly chooses to clear local
  archive data. It never changes server sources, jobs, queues, databases,
  models or GPUs.
- Failed acceptance freezes the batch. Only the affected UI files and local
  browser test state may be repaired or restored.

## Design Self-Review

- The session rail restores discoverable navigation without reintroducing
  separate execution modes or a permanent research/source sidebar.
- Long-lived local history has a clear privacy boundary and no silent data
  loss at the 20-session limit.
- Persona-like energy is translated into original hierarchy, state markers and
  timing rather than copied visual property.
- The strongest visual device is one red signal-cut system; content surfaces
  remain calm enough for industrial and scientific reading.
- All behaviour reuses accepted Gateway contracts and remains testable without
  GPU or model work.
