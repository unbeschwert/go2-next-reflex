// Event bus + the single Server-Sent-Events connection to /api/events.
//
// SSE event names (API.md): live, log, task, record, robot, pose, alert.
// Local events published by the UI itself on the same bus:
//   overview      - latest GET /api/overview answer (shell polls every 3 s)
//   sse-state     - {connected, wanted, failed}
//   net-state     - {down}   (fetch-level reachability, from api.js)
//   estop         - E-STOP pressed in this browser
//   theme         - theme changed ('dark' | 'light')

const SSE_NAMES = ['live', 'log', 'task', 'record', 'robot', 'pose', 'alert'];
const LOG_MAX = 300;
const BACKOFF_MIN = 1000;
const BACKOFF_MAX = 15000;

const listeners = new Map();
const lastValues = new Map();
const logRing = [];

let source = null;
let wanted = false;
let connected = false;
let failed = false;
let backoff = BACKOFF_MIN;
let reconnectTimer = null;
let authCheck = null;

/** Subscribe; returns an unsubscribe function. */
export function on(name, fn) {
  let set = listeners.get(name);
  if (!set) {
    set = new Set();
    listeners.set(name, set);
  }
  set.add(fn);
  return () => set.delete(fn);
}

export function emit(name, data) {
  lastValues.set(name, data);
  const set = listeners.get(name);
  if (!set) return;
  for (const fn of Array.from(set)) {
    try {
      fn(data);
    } catch (err) {
      console.error(`[events] handler for "${name}" failed`, err);
    }
  }
}

/** Last payload seen for an event name (or undefined). */
export function last(name) {
  return lastValues.get(name);
}

/** Recent `log` events, oldest first (max 300). */
export function logHistory() {
  return logRing.slice();
}

export function sseConnected() {
  return connected;
}

function publishState() {
  emit('sse-state', { connected, wanted, failed });
}

function setConnected(value) {
  if (value) failed = false;
  if (connected === value) return;
  connected = value;
  publishState();
}

function handleMessage(name, ev) {
  let data;
  try {
    data = JSON.parse(ev.data);
  } catch {
    return; // malformed payload: ignore rather than break the stream
  }
  if (data === null || typeof data !== 'object') return;
  if (!connected) setConnected(true);
  if (name === 'log') {
    logRing.push(data);
    if (logRing.length > LOG_MAX) logRing.splice(0, logRing.length - LOG_MAX);
  }
  emit(name, data);
}

function open() {
  if (!wanted || source) return;
  let es;
  try {
    es = new EventSource('/api/events');
  } catch {
    failed = true;
    setConnected(false);
    publishState();
    scheduleReconnect();
    return;
  }
  source = es;
  es.addEventListener('open', () => {
    if (source !== es) return;
    backoff = BACKOFF_MIN;
    setConnected(true);
  });
  es.addEventListener('error', () => {
    if (source !== es) return;
    // Take over reconnection ourselves so we can back off and re-check auth.
    es.close();
    source = null;
    failed = true;
    connected = false;
    publishState();
    scheduleReconnect();
  });
  for (const name of SSE_NAMES) {
    es.addEventListener(name, (ev) => {
      if (source === es) handleMessage(name, ev);
    });
  }
}

function scheduleReconnect() {
  if (!wanted || reconnectTimer) return;
  const delay = backoff + Math.random() * 400;
  backoff = Math.min(backoff * 2, BACKOFF_MAX);
  reconnectTimer = setTimeout(async () => {
    reconnectTimer = null;
    if (!wanted) return;
    if (authCheck) {
      let ok;
      try {
        ok = await authCheck();
      } catch {
        ok = null; // server unreachable
      }
      if (!wanted) return;
      if (ok === false) return; // logged out: the auth handler shows the login screen
      if (ok === null) {
        scheduleReconnect();
        return;
      }
    }
    open();
  }, delay);
}

/**
 * Open the stream (once). checkAuth: async () => true (logged in) / false (not);
 * throws when the server is unreachable. Called before each reconnect.
 */
export function startEvents(checkAuth) {
  authCheck = checkAuth || null;
  wanted = true;
  failed = false;
  backoff = BACKOFF_MIN;
  publishState();
  open();
}

export function stopEvents() {
  wanted = false;
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (source) {
    source.close();
    source = null;
  }
  connected = false;
  failed = false;
  publishState();
}

/** Forget cached payloads (on logout). Listeners stay registered. */
export function resetEventState() {
  lastValues.clear();
  logRing.length = 0;
}
