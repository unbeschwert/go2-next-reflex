three.js 0.160.0 (MIT, see LICENSE), from the npm package `three@0.160.0`.
- three.module.min.js: package/build/three.module.min.js, unchanged
  (sha256 3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6)
- OrbitControls.js: package/examples/jsm/controls/OrbitControls.js with the import `from 'three'`
  changed to `from './three.module.min.js'` (no import map needed, so the CSP can stay script-src 'self').
Served from the dashboard itself: no CDN at runtime (the robot network has no internet).
