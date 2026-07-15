# Research Command Desk Visual Design

> Status: proposed for implementation review
> Date: 2026-07-14
> Scope: visual and interaction redesign of the user-facing `/research` portal

## Design Read

Reading this as: an internal industrial and scientific research workspace for
technical users, with a restrained tactical-console language that uses a
bright reading surface, dark structural framing, and deliberate motion.

The direction learns from the *information design* of tactical interfaces,
including Arknights: cold material contrast, limited warning colour, technical
labels, cut markers, and dense-but-legible state hierarchy. It does not copy
game assets, character art, logos, fonts, text, screen layouts, or sound.

## Goal

Make `/research` feel like a distinctive Mergenetic research workspace without
making it harder to use as a normal multi-turn chat. The portal must keep its
existing local-only conversation archive, sources, citations, upload flow,
research queue and model API contract unchanged.

## Product Constraints

- Do not change gateway APIs, research route selection, source ownership,
  local IndexedDB records, server retention, worker queues, Redis, PostgreSQL,
  model service or GPU use.
- Do not create a user-to-admin navigation path.
- Do not add an external visual dependency or copy third-party product assets.
- Keep the existing Remix Icon family and MiSans load path.
- Never animate layout size, list scrolling, background decoration, or a
  non-visible DOM node. Respect `prefers-reduced-motion`.
- Keep source and citation information plain, inspectable and reachable by
  keyboard. Visual hierarchy cannot hide factual provenance.

## Visual System

### Palette

The main workspace is cool and bright for long technical reading. Its dark
frame creates the operational character; amber is limited to choices requiring
attention, not used as an all-purpose decorative colour.

| Token | Value | Role |
| --- | --- | --- |
| `--rcd-void` | `#171A1F` | session rail, top telemetry strip, strong text |
| `--rcd-graphite` | `#2B3037` | raised dark controls and inactive rail states |
| `--rcd-alloy` | `#68727D` | technical labels and subdued controls |
| `--rcd-fog` | `#EEF1F2` | cool page field and recessed panels |
| `--rcd-paper` | `#FAFBFB` | reading canvas and composer surface |
| `--rcd-line` | `#CBD2D6` | panel joins, rules and grid marks |
| `--rcd-amber` | `#E6B422` | active session, send command, pending attention |
| `--rcd-ready` | `#18765E` | ready/verified state only |
| `--rcd-danger` | `#B5473C` | destructive action and error state only |

Amber must not appear on passive metadata or every button. Green and red
remain semantic state colours; they never compete with amber for branding.

### Materiality

The material effect comes from structure, not blur or gradients:

- `paper`: near-white content surfaces with a 1px cool-grey edge and a
  background-relative shadow of no more than 10% opacity;
- `alloy`: dark rails with two tonal planes, fine inset divider lines and
  modest contrast between active and inactive rows;
- `recess`: quiet fog-grey fields separated from paper by a single cool-grey
  rule. Material distinction comes from the tonal plane and its edge, not a
  patterned background;
- `signal`: a 3px diagonal amber cut used only for selected session, active
  composer and a cited evidence locator.

No glass blur, bokeh, blobs, background video, generic gradient, decorative
particle field, or fake telemetry is permitted.

### Typography

- **Reading:** MiSans at normal weight, 16px-equivalent body size and 1.72
  line height for assistant results.
- **Display:** MiSans at 700/800, compact scale for session title and turn
  headings. Headings remain readable Chinese, never use artificial condensed
  text or negative letter spacing.
- **Utility:** system monospace for short English status labels, IDs, source
  locators and timestamp numerals. It is not used for Chinese paragraphs.
- **Label rule:** uppercase English is reserved for real stable categories
  such as `RESEARCH`, `SOURCES`, and `LOCAL ARCHIVE`; it is not filler.

## Layout

```text
Desktop >= 1080px

  + rail / graphite +================ command strip / void ===============+
  | Mergenetic stamp | RESEARCH / current title       sources / settings   |
  | [ new session ]  +-----------------------------------------------------+
  | [ session search ]| timeline: question -> answer -> evidence index     |
  |                  |                                                      |
  | archive groups    |                                                       |
  | active row / amber| [ source rail ]                                    |
  |                  | [ composition deck: question + tools + send ]       |
  +------------------+------------------------------------------------------+

Tablet 720-1079px: compact rail becomes a controlled overlay drawer.
Mobile < 720px: command strip keeps the title, source count and session
control; the composer remains after the current conversation, with no overlap.
```

The rail is a permanent archive tool on desktop, not an ornamental sidebar.
The stage is intentionally asymmetrical: left-side persistent history,
right-side focused research narrative. The empty state uses a small action
module aligned with the composer, not a centered marketing hero.

## Signature Components

### 1. Archive Signal Rail

Each local session remains an ordinary accessible button. The active session
gets an amber diagonal cut and a short source-count indicator. Rename and
delete controls only appear on row focus or hover, but are never inaccessible
to keyboard users. The search and clear-local controls retain their current
behaviour.

### 2. Command Strip

Replace the plain header with a shallow dark telemetry strip. It contains only
true state: active title, local archive state, selected source count and the
session settings command. It does not invent model health, task duration,
confidence scores or synthetic charts.

### 3. Evidence Index

Research responses keep their current citation data but render it as a compact
numbered index beneath the answer. Each entry has a locator chip (page or
slide), source name and a short expansion affordance. This makes provenance
look deliberate without making citations less exact.

### 4. Composition Deck

The composer becomes a low-profile paper deck with a recessed tool channel:
source tools, selected source chips and send control form one clear visual
assembly. The primary send control is amber; file, link and material controls
are icon-first with tooltips and visible focus rings.

### 5. Empty Research Prompt

With no turns, show a compact `开始研究` module with three direct actions:
`直接提问`, `添加资料`, and `导入链接`. These are real shortcuts to the existing
composer or its tool menu, not a tutorial or a new state machine.

## Interaction and Motion

Use the already loaded GSAP instance only when it exists and reduced motion is
not selected. CSS still provides all hover, focus and pressed feedback.

| Event | Visual outcome | Budget |
| --- | --- | --- |
| Initial entry | rail stamp, command strip and visible timeline settle in sequence | 260ms |
| Session switch | old timeline fades 10px out, next timeline enters after content swap | 220ms |
| Active row | amber cut sweeps into the current session; status text changes immediately | 160ms |
| Composer tools | anchored tool tray fades and moves 8px from the add control | 180ms |
| Evidence expand | index body fades and moves down 6px | 160ms |
| Button feedback | hover lifts 1px, press scales to `.98`, focus stays high contrast | CSS only |

All GSAP motion is limited to `opacity`, `x`, and `y`. New session-switch
timelines cancel the previous timeline. Under reduced motion, state changes
are immediate and no transform animation occurs.

## Accessibility and Responsive Acceptance

- Text and controls meet contrast expectations against both paper and alloy
  surfaces. Amber is never the sole state indication.
- Every icon-only control has `aria-label` and a tooltip. Existing labels
  remain concise.
- Focus never lands behind a rail, dialog, settings drawer or tool tray.
- Long Chinese titles, URLs, source names and citation locators must wrap or
  truncate without increasing the fixed rail width or overlapping controls.
- At 390px wide: the archive rail is a modal drawer; the command strip,
  message list, evidence index and composer remain independently usable.

## Deliberate Non-Goals

- No game-like HUD, score, rank, character, battle, item or progression
  metaphor.
- No presentation-only widgets that have no actual product data behind them.
- No visual change to the administrator portal in this task.
- No long-running animation or GPU/model test needed to validate the redesign.

## Review Checklist

- The chosen palette is a new cold-grey/amber system, not the old red/black
  scheme.
- Materiality is readable at normal zoom and derives from borders, planes and
  restrained shadows, not prohibited visual effects.
- The archive rail, source workflow, citations and composer remain visible
  product tools rather than decorative cards.
- Every proposed component maps to existing front-end state or an explicit
  minimal template marker; no backend feature is implied.
- The design stays specific to a research workspace rather than becoming a
  generic game-themed dashboard.
