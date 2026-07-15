# Admin Portal Visual UX

## Scope

This batch improves visual hierarchy, concise guidance and interaction feedback
for the existing Gateway administrator portal at `/model-gateway`. It also
removes the administrator portal link from the user research workspace at
`/research`.

The service publication form, API Key management, direct-call validation,
request cancellation and OpenAI-compatible developer integration remain in the
same portal. No Gateway API, schema, queue, model-runtime or GPU behavior
changes in this work.

## User Boundary

`/research` is a user workspace and must not display an administrator portal
link. Its header brand returns to `/research`, and its supporting copy becomes
one short instruction: direct question-answering or source-backed research.

## Administrator Console

The administrator page remains an operational control console rather than a
marketing page.

| Area | Visual treatment | Information retained |
| --- | --- | --- |
| Header and status | Compact service-control title, live state rail and three count indicators. | Running-service, Key and available-model counts. |
| Publication | Form and service list remain side by side with icon-led panel heading and short field labels. | All existing vLLM controls and manual start/stop actions. |
| API Keys | Compact operational card with clear one-time-secret emphasis. | Owner, model allowlist, Key lifecycle actions. |
| Direct validation | Keep request form, response and request cancellation together. | User API Key, model, generation controls, request lookup/cancellation. |
| OpenAI compatibility | Keep a dedicated, readable integration area instead of hero marketing copy. | cURL example, Base URL, authentication, usage, request-status and cancellation routes. |
| Publication flow | Keep a short visible flow near publication controls. | Select model, configure service, manually start, validate via API Key. |

Explanations are retained where they answer an operational question. Long
repetition is removed from headings and panel introductions; labels, one-line
descriptions, tooltips and risk notices remain visible.

## Visual Language

- Use the existing Mergenetic teal, neutral paper/panel and warning/error
  colors; no new brand palette or decorative graphics.
- Each panel heading includes a familiar Remix icon and concise state/action
  label. Existing text buttons remain for destructive or irreversible commands.
- Running, stopped and failed state indicators always pair colour with visible
  text. High-risk controls retain a warning treatment and readable label.
- Buttons, service rows, Key rows, request controls and copy controls receive
  hover lift, press scale and teal keyboard focus. Motion is CSS-first.
- GSAP is reserved for one page-entry sequence and state-surface updates. It
  animates only `transform` and `opacity`, has no repeating effect, and is
  bypassed for `prefers-reduced-motion: reduce`.

## Acceptance And Rollback

1. `/research` has no administrator portal navigation or link.
2. `/model-gateway` retains every existing form ID, list ID, action ID and API
   example endpoint referenced by portal tests and `console.js`.
3. OpenAI-compatible instructions and the publication flow are visible without
   relying on hidden documentation.
4. All interactive controls have hover, active and keyboard-focus feedback;
   reduced motion preserves every operation without transform animation.
5. Portal tests, Node syntax checks, full container tests and HTTP smoke pass;
   no model or fusion run is required.
6. Rollback restores only `research.html`, `research.css`, `console.html`,
   `console.css`, `console.js`, portal tests and this visual documentation. No
   persisted service, API Key, request or model data changes.
