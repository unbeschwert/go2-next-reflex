// Task drawer: slow server actions answer {task_id}; progress arrives as SSE
// `task` events. When no SSE update came for a task in the last 1.5 s (or SSE
// is down) we poll GET /api/tasks/{id} every second instead.

import { api } from './api.js';
import { on, sseConnected } from './events.js';
import { el, setText, toast, setLogText, status, setStatus, iconBtn, setDrawerWidth } from './ui.js';
import * as fmt from './format.js';

const MAX_TASKS = 12;
const POLL_MS = 1000;
const SSE_STALE_MS = 1500;
const AUTO_HIDE_MS = 10000;

const entries = new Map(); // id -> entry
const watchers = new Map(); // id -> Set(fn)
let drawerEl = null;
let list = null;
let countEl = null;
let toggleBtn = null;
let bubble = null;
let unsubscribe = null;
let ticker = null;
let autoHideTimer = null;
let pinned = false; // opened by the user: never auto-hide
let wired = false;

const isFinal = (state) => state === 'done' || state === 'failed';

/** toggle: the top-bar Tasks button; bubbleEl: its count bubble. */
export function initTasks(toggle, bubbleEl) {
  drawerEl = document.getElementById('task-drawer');
  list = document.getElementById('task-list');
  countEl = document.getElementById('task-count');
  toggleBtn = toggle;
  bubble = bubbleEl;
  if (!wired) {
    wired = true;
    const closeBtn = iconBtn('x', 'Close tasks');
    closeBtn.addEventListener('click', () => setOpen(false));
    document.getElementById('task-close-slot').replaceWith(closeBtn);
    document.getElementById('task-clear').addEventListener('click', clearFinished);
    drawerEl.addEventListener('mouseenter', cancelAutoHide);
    drawerEl.addEventListener('focusin', cancelAutoHide);
    drawerEl.addEventListener('mouseleave', scheduleAutoHide);
  }
  toggleBtn.addEventListener('click', () => {
    const open = drawerEl.hidden;
    setOpen(open);
    pinned = open;
  });
  unsubscribe = on('task', (t) => {
    if (t && t.id != null) update(t, true);
  });
  ticker = setInterval(tick, 1000);
  updateCounts();
}

export function teardownTasks() {
  if (unsubscribe) unsubscribe();
  unsubscribe = null;
  clearInterval(ticker);
  ticker = null;
  cancelAutoHide();
  for (const entry of entries.values()) finishEntry(entry, true);
  entries.clear();
  watchers.clear();
  if (list) list.replaceChildren();
  setOpen(false);
  updateCounts();
}

function setOpen(open) {
  if (!drawerEl) return;
  drawerEl.hidden = !open;
  drawerEl.style.width = 'min(440px, 100vw)';
  setDrawerWidth(null, open ? 440 : null);
  if (toggleBtn) toggleBtn.setAttribute('aria-expanded', String(open));
  if (!open) {
    pinned = false;
    cancelAutoHide();
  }
}

function cancelAutoHide() {
  clearTimeout(autoHideTimer);
  autoHideTimer = null;
}

/** Hide the drawer 10 s after the last running task finished (unless pinned / in use). */
function scheduleAutoHide() {
  cancelAutoHide();
  if (!drawerEl || drawerEl.hidden || pinned) return;
  for (const e of entries.values()) if (!isFinal(e.task.state)) return;
  autoHideTimer = setTimeout(() => {
    autoHideTimer = null;
    if (drawerEl.matches(':hover') || drawerEl.contains(document.activeElement)) return;
    setOpen(false);
  }, AUTO_HIDE_MS);
}

/**
 * Subscribe to every update of one task (SSE or poll). Returns an unsubscribe function.
 * The current state is delivered immediately when known.
 */
export function onTaskUpdate(id, fn) {
  const key = String(id);
  let set = watchers.get(key);
  if (!set) {
    set = new Set();
    watchers.set(key, set);
  }
  set.add(fn);
  const entry = entries.get(key);
  if (entry) fn(entry.task);
  return () => set.delete(fn);
}

/**
 * Follow a task started by this browser and show its log.
 * Accepts the {task_id} answer or an id. Resolves with the final task
 * ({state: 'done' | 'failed', ...}) or null when there was no task id.
 * opts.open = false keeps the drawer closed (the page shows the log itself).
 */
export function trackTask(idOrAnswer, label, { open = true } = {}) {
  const id = idOrAnswer && typeof idOrAnswer === 'object' ? idOrAnswer.task_id : idOrAnswer;
  if (id == null || id === '') return Promise.resolve(null);
  let entry = entries.get(String(id));
  if (!entry) entry = createEntry({ id: String(id), label, state: 'running', log: [] });
  if (!entry.tracked && isFinal(entry.task.state)) {
    const name = label || entry.task.label || `Task ${id}`;
    toast(`${name}: ${entry.task.state}`, entry.task.state === 'done' ? 'success' : 'error');
  }
  entry.tracked = true;
  cancelAutoHide();
  if (label && !entry.task.label) {
    entry.task.label = label;
    render(entry);
  }
  if (open) setOpen(true);
  return new Promise((resolve) => {
    if (isFinal(entry.task.state)) {
      scheduleAutoHide();
      resolve(entry.task);
      return;
    }
    entry.waiters.push(resolve);
    startPolling(entry);
  });
}

function createEntry(task) {
  const stateEl = status('busy', 'running');
  const labelEl = el('span', { class: 'task-label' });
  const timeEl = el('span', { class: 'task-time' });
  const dismiss = iconBtn('x', 'Remove from the list', { size: 'sm' });
  const logEl = el('pre', { class: 'logbox task-log', tabindex: '0', 'aria-label': 'Task log' });
  const resultEl = el('div', { class: 'task-result', hidden: true });
  const node = el('article', { class: 'task-item' },
    el('div', { class: 'task-head' }, labelEl, timeEl, dismiss),
    el('div', { class: 'small' }, stateEl), logEl, resultEl);
  const entry = {
    task: { ...task }, node, stateEl, labelEl, timeEl, logEl, resultEl,
    waiters: [], pollTimer: null, polling: false, lastSse: 0, firstSeen: Date.now(), finishedAt: null, tracked: false,
  };
  dismiss.addEventListener('click', () => {
    finishEntry(entry, true);
    entries.delete(entry.task.id);
    node.remove();
    updateCounts();
  });
  entries.set(task.id, entry);
  list.prepend(node);
  trimEntries();
  render(entry);
  return entry;
}

function trimEntries() {
  if (entries.size <= MAX_TASKS) return;
  for (const [id, entry] of entries) {
    if (entries.size <= MAX_TASKS) break;
    if (isFinal(entry.task.state)) {
      entry.node.remove();
      entries.delete(id);
    }
  }
}

function update(task, fromSse) {
  const id = String(task.id);
  let entry = entries.get(id);
  if (!entry) entry = createEntry({ ...task, id });
  const wasFinal = isFinal(entry.task.state);
  entry.task = { ...entry.task, ...task, id, label: task.label || entry.task.label };
  if (fromSse) entry.lastSse = Date.now();
  render(entry);
  const set = watchers.get(id);
  if (set) {
    for (const fn of Array.from(set)) {
      try {
        fn(entry.task);
      } catch (err) {
        console.error('[tasks] watcher failed', err);
      }
    }
  }
  if (!wasFinal && isFinal(entry.task.state)) {
    entry.finishedAt = Date.now();
    if (entry.tracked) {
      const name = entry.task.label || entry.task.kind || `Task ${id}`;
      if (entry.task.state === 'done') toast(`${name}: done`, 'success');
      else toast(`${name}: failed`, 'error');
    }
    finishEntry(entry, false);
    scheduleAutoHide();
  }
  updateCounts();
}

function finishEntry(entry, abandon) {
  clearInterval(entry.pollTimer);
  entry.pollTimer = null;
  const waiters = entry.waiters.splice(0);
  const result = abandon && !isFinal(entry.task.state) ? { ...entry.task, state: 'unknown' } : entry.task;
  for (const w of waiters) w(result);
}

function startPolling(entry) {
  if (entry.pollTimer) return;
  const poll = async (force) => {
    if (entry.polling || isFinal(entry.task.state)) return;
    if (!force && sseConnected() && Date.now() - entry.lastSse < SSE_STALE_MS) return;
    entry.polling = true;
    try {
      const t = await api.get(`/tasks/${encodeURIComponent(entry.task.id)}`, null, { timeout: 8000 });
      if (t && typeof t === 'object') update({ ...t, id: entry.task.id }, false);
    } catch (err) {
      if (err.status === 404) {
        update({ id: entry.task.id, state: 'failed', log: [...(entry.task.log || []), 'Task not found on the server.'] }, false);
      }
      // network errors: keep polling; the connection banner is already visible
    } finally {
      entry.polling = false;
    }
  };
  entry.pollTimer = setInterval(() => poll(false), POLL_MS);
  poll(true);
}

const STATE = { running: ['busy', 'Running'], done: ['ok', 'Done'], failed: ['bad', 'Failed'] };

function render(entry) {
  const t = entry.task;
  const [st, word] = STATE[t.state] || ['off', t.state || 'unknown'];
  setStatus(entry.stateEl, st, word);
  setText(entry.labelEl, t.label || fmt.humanize(t.kind) || `Task ${t.id}`);
  renderTime(entry);
  const log = Array.isArray(t.log) ? t.log.slice(-500) : [];
  setLogText(entry.logEl, log.length ? log : ['(no output yet)']);
  if (t.result != null && isFinal(t.state)) {
    entry.resultEl.hidden = false;
    setText(entry.resultEl, `Result: ${fmt.describe(t.result, 400)}`);
  } else {
    entry.resultEl.hidden = true;
  }
}

function renderTime(entry) {
  const t = entry.task;
  if (isFinal(t.state)) {
    const fin = t.finished != null ? fmt.timeOfDay(t.finished) : fmt.timeOfDay(entry.finishedAt || Date.now());
    setText(entry.timeEl, fin);
  } else {
    setText(entry.timeEl, fmt.duration((Date.now() - entry.firstSeen) / 1000));
  }
}

function tick() {
  for (const entry of entries.values()) {
    if (!isFinal(entry.task.state)) renderTime(entry);
  }
}

function clearFinished() {
  for (const [id, entry] of entries) {
    if (isFinal(entry.task.state)) {
      entry.node.remove();
      entries.delete(id);
    }
  }
  updateCounts();
}

function updateCounts() {
  if (!toggleBtn) return;
  let running = 0;
  for (const e of entries.values()) if (!isFinal(e.task.state)) running += 1;
  const total = entries.size;
  setText(countEl, total ? `${running} running · ${total - running} finished` : 'No tasks yet. Slow actions (deploy, cut, fetch, ...) show their progress here.');
  if (bubble) {
    bubble.hidden = running === 0;
    setText(bubble, running ? String(running) : '');
  }
  toggleBtn.setAttribute('aria-label', running ? `Tasks, ${running} running` : 'Tasks');
}
