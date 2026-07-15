# Research Portal Three-Mode UX

## Purpose

Make `/research` approachable for invited users who prefer conversational
industrial, technical and scientific work, without removing the evidence-first
research workflow.

## Confirmed Decisions

The user workspace has three local presentation modes. The desktop sidebar is
expanded by default. The selected mode is stored only in the browser on the
current device; it is not persisted in the Gateway database or associated with
an API Key.

| Mode | Default use | Visible workspace |
| --- | --- | --- |
| `chat` | Everyday question-and-answer | Expanded sidebar, conversation timeline, fixed composer; no source is required. |
| `research` | Multi-source research | Source processing states, research task state, answer and citations. |
| `focus` | Reading a long answer | Answer, citations and compact composer; sidebar and secondary controls are hidden. |

`chat` is the default mode for a first visit. The last explicitly selected mode
is restored on the same trusted device.

## Interaction Model

- A compact, labelled three-item mode control is placed in the workspace header.
  Icon, text, active state and keyboard focus identify each mode; colour alone
  never conveys the selected state.
- Source imports and current research jobs are shared across all modes. A layout
  switch never cancels, recreates or hides the actual job state.
- In chat mode, a source-free prompt calls the existing API-Key-authenticated
  `/v1/chat/completions` endpoint and remains only in the current page memory.
  Completed research answers also appear as citation-bearing timeline entries.
  Research mode retains the existing task/result emphasis. Focus mode keeps
  source locators reachable from the reading surface.
- The current cancellation action remains available only for a cancelable job.

## Motion And Feedback

The visual language takes only general interaction principles from high-density
game interfaces: explicit state hierarchy, decisive selection feedback and
short transitions. It does not copy game assets, typography, palette or layout.

| Interaction | Motion | Limit |
| --- | --- | --- |
| Mode switch | Previous surface fades and translates out; new surface enters after the layout class changes. | 220-280 ms, `opacity` and `transform` only. |
| Sidebar change | Sidebar visually translates/fades while the layout changes once. | No continuous resize animation. |
| New chat/result entry | One short upward entrance from the composer direction. | 180 ms, once per entry. |
| Buttons, source rows, citations | Hover/focus changes border, background and a 1-2 px transform; press scales to `0.98`. | CSS transition only; no timer-driven animation. |
| Reduced motion | No positional transitions or entry animation. | Immediate layout/state update, visible focus preserved. |

GSAP is limited to the mode and message entry sequences. It must animate only
`transform` and `opacity`, create no background loop, and be bypassed when
`prefers-reduced-motion: reduce` is enabled. CSS handles hover, focus and press
feedback so controls remain usable if the GSAP CDN is unavailable.

## Data And Privacy Boundary

- The local preference key contains only `chat`, `research` or `focus`.
- API Key handling remains unchanged: session storage by default, explicit local
  persistence only on a trusted device.
- No research payload, source text or answer history is newly persisted by this
  UX work. The existing short-lived server-side job/result TTL remains intact.

## Acceptance

1. First visit opens chat mode with desktop sidebar expanded.
2. Each mode is reachable by mouse, keyboard and visible focus; reload restores
   the selected mode on the same browser.
3. File status, queued job status, completion, failure and cancellation remain
   correct after any mode switch.
4. Every interactive control has hover, active and `:focus-visible` feedback.
5. Reduced-motion mode has no transform/opacity animation but retains all
   controls and state changes.
6. `node --check`, targeted portal tests, browser smoke at desktop/mobile and
   the existing container test suite pass. No model, fusion or GPU test is
   required for this UI batch.

## Rollback

The work is isolated to the research template, CSS, JavaScript and portal tests.
Rollback restores those files and removes the local preference key
`mergeneticResearchWorkspace`; no database, queue, model service or document
data migration is involved.
