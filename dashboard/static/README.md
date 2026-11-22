# Durance AI dashboard: browser UI

Static files served by the dashboard backend from `/` (FastAPI mounts this folder).
Plain ES modules, no build step, no CDN: everything the page loads comes from this folder.
The HTTP contract is `../docs/API.md`.

## Design system: "Calm dark"

- Tokens on `:root` in `css/app.css` (bg `#0d1117`, surface `#161b22`, raised `#1c2330`, border `#2a3441`,
  text `#e6edf3`, muted `#8b98a5`, one accent: teal `#2dd4bf`; success / warning / danger only where needed).
  `[data-theme="light"]` on `<html>` swaps the tokens (toggle in the user menu).
- System font stack, 14 px / 1.5; page title 22 px / 600 in the top bar, section titles 15 px / 600,
  labels 12 px muted; numbers use tabular figures; mono only for logs, md5 and paths.
- Spacing 4/8/12/16/24/32, cards radius 12 / padding 20, controls radius 8 / height 36.
- Status = 8 px dot + text (`ui.status()`); capsules (`ui.chip()`) only for exceptions
  (Locked, REAL ROBOT). Icons: `js/icons.js` (inline SVG built with createElementNS, Lucide-like geometry).
- Buttons: primary teal (one per view), secondary, ghost, danger = red outline; red fill only for E-STOP and
  the drive STOP.

## Files

```
index.html                 shell markup: sidebar, top bar, E-STOP bar, banners, page host, task drawer
favicon.svg
css/app.css                the whole design system + page layouts (dark default, light theme, responsive)
js/main.js                 entry: boot, auth gate (login / first-run setup), hash router, Escape -> E-STOP
js/shell.js                sidebar (232 px / 64 px rail / drawer < 1024 px), top bar (target switch,
                           System status popover, lock + REAL ROBOT chips, Tasks, user menu, E-STOP),
                           E-STOP bar, connection banner, slim alert banner, shared 3 s /overview poll
js/system.js               system health rows + summary (top bar and Home), trainingState() ("not set up" is
                           neutral), modelLabel() (readable board model label; file name only in Board > Details)
js/api.js                  fetch wrapper: CSRF header, 401 -> login, 403 -> refresh token + retry once,
                           409 {confirm} -> modal + resend with "confirm": true, ApiError, net-down probe
js/events.js               event bus + the single EventSource on /api/events (backoff reconnect, auth re-check)
js/scope.js                per-page lifetime: timers, polls, subscriptions, listeners, AbortSignal
js/tasks.js                task drawer: SSE `task` + 1 s polling fallback; onTaskUpdate() for inline logs
js/ui.js                   UI kit: el() (never innerHTML), btn/iconBtn, status, chip, switch, segmented,
                           createTabs, collapsible, menuButton ("..."), drawer, popover, tooltip, tile,
                           emptyState, dataTable, toasts, confirm dialog, act() (busy buttons)
js/icons.js                inline SVG icon set
js/format.js               number / time formatting (%, ms, s, SI, bytes)
js/auth.js                 login / setup forms, password strength meter
js/components/
  cameraView.js            MJPEG <img> with placeholder + 2 s retry
  rateBars.js              five output-rate bars with trigger markers
  logView.js               event log (newest first, kind filter, max 300)
  settingsForm.js          gesture-control settings from the schema, staged changes + sticky Apply bar
  virtualPerson.js         simulator virtual person segmented control
  clipPlayer.js            base64 event clip -> pixelated canvas, transport, seek(), onFrame
  spikePanel.js            network output spike raster on the player's playhead + details charts
  machines.js              training machines: list (label, type, status only), Add / Edit drawer with SSH key
                           select / generate + copyable public key, "Server ID" trust step, Test checklist, Prepare
js/pages/                  one module per page; each exports mount(root, scope, app)
  home.js live.js simulation.js drive.js record.js dataset.js training.js board.js security.js
vendor/novnc-1.5.0/        noVNC client (RViz from the container over /api/rviz/vnc); do not edit
vendor/three-0.160.0/      no longer used by the UI (kept on disk)
```

Routes: `#/home`, `#/live`, `#/simulation`, `#/drive`, `#/record`, `#/dataset`, `#/training`, `#/board`,
`#/security` (old `#/overview`, `#/operation`, `#/manual`, `#/recording` redirect). Pages are loaded lazily
(`import()`), so noVNC is only fetched on the Simulation page. A page registers everything it starts on its
`scope`; the router disposes it when the user navigates away (this also disconnects the RViz VNC session).

## Content-Security-Policy rules

The server sends
`default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'`.
Keep the UI compatible with it:

- No inline `<script>` blocks, no `on*=` attributes, no `javascript:` URLs, no import maps, no `eval` / `new Function`.
  Scripts are ES modules loaded with `<script type="module" src="/js/main.js">`; imports are relative files.
- No `style="..."` attributes in HTML or generated markup. Setting `el.style.prop = ...` / `setProperty('--x')`
  from JS (CSSOM) is fine; `ui.el()` accepts `{style: {...}}` and applies it that way.
- No external fonts, stylesheets or CDNs (system font stack only; the robot network has no internet).
- Images only from the same origin, `data:` or `blob:` (the recording preview uses blob URLs).
- WebSocket only to our own origin (`connect-src 'self'`): the RViz viewer connects to `/api/rviz/vnc`
  and refuses any other path the server might send.

## Security rules for UI code

- Never put server-provided text (API answers, logs, file names, user names, gesture labels) into `innerHTML`.
  Build DOM with `el()` / `textContent`; `el()` has no innerHTML path. Canvases only draw ticks and lines.
- The CSRF token lives only in memory (`api.js`). `localStorage` holds UI preferences only
  (`durance.*`: theme, last page, sidebar state, remembered tabs / collapsibles, speed limit, dataset source / view,
  training host, simulation source / panel, record person).
- Risky actions are confirmed by the server's 409 `confirm` question; the UI never sends `"confirm": true` on its own.

## Safety behaviour

- E-STOP (top bar, every page; `Escape` when no modal dialog is open) posts `/api/estop` immediately, retries twice on
  network failure, and shows a persistent "E-STOP active" bar with Clear. The Escape listener runs on `window` in the
  capture phase, so it also works while the RViz canvas has keyboard focus. Escape also closes open menus / popovers.
- The shell never scrolls and drawers open below the top bar: the E-STOP is always visible and clickable.
- Drive is a dead-man control: `/api/robot/move` every 100 ms only while a key / the pad / a turn button is held;
  `/api/robot/stop` on release, window blur, hidden tab, pointer cancel, E-STOP and page leave.

## Checking changes

```
for f in $(find js -name '*.js'); do node --experimental-default-type=module --check "$f"; done
```
