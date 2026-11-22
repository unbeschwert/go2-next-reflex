// Persistent shell: sidebar navigation (232 px, 64 px icon rail, drawer below
// 1024 px), top bar (page title, target switch, System status popover, lock /
// real-robot chips, Tasks, user menu, E-STOP), E-STOP bar, connection banner,
// slim alert banner, and the shared 3 s GET /api/overview poll (bus event `overview`).

import { api } from './api.js';
import { on, emit, last } from './events.js';
import {
  el, setText, toast, act, prefs, segmented, icon, btn, iconBtn, menuButton, bindPopover, status,
  attachTip, chip, closeTransient,
} from './ui.js';
import { systemRows, summarize } from './system.js';
import { initTasks, teardownTasks } from './tasks.js';

const OVERVIEW_MS = 3000;
const LIVE_STALE_MS = 3000;
const ESTOP_GRACE_MS = 4000;
const MAX_SSE_ALERTS = 8;

let ctx = null;
let wired = false;
let unsubs = [];
let running = false;

let ovTimer = null;
let ovInflight = null;
let ovFailed = false;

let lastLiveAt = 0;
let lastRobot = null;
let staleTimer = null;
let estopPressedAt = 0;
let estopShown = false;
let estopFlashTimer = null;

let netDown = false;
let sseState = { connected: false, wanted: false, failed: false };

const refs = {};
const alerts = new Map(); // key -> {level, text, source, t, timer}
const dismissedOverview = new Set();
let alertsExpanded = false;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- lifecycle

/** context: {pages: [{id, label, icon}], user, logout()} */
export function initShell(context) {
  ctx = context;
  running = true;
  if (!wired) wireOnce();
  refs.userBtnLabel.textContent = ctx.user || 'User';
  initTasks(refs.tasksBtn, refs.tasksBubble);
  unsubs = [
    on('overview', applyOverview),
    on('live', () => {
      const fresh = Date.now() - lastLiveAt > LIVE_STALE_MS;
      lastLiveAt = Date.now();
      if (fresh) renderSystem();
    }),
    on('robot', (r) => {
      lastRobot = r;
      renderSystem();
      renderChips();
    }),
    on('alert', (a) => addAlert(a, 'sse')),
    on('net-state', (s) => {
      netDown = !!s.down;
      renderConnBanner();
      renderSystem();
    }),
    on('sse-state', (s) => {
      sseState = s;
      renderConnBanner();
      renderSystem();
    }),
  ];
  staleTimer = setInterval(renderSystem, 1500);
  document.addEventListener('visibilitychange', onVisibility);
  renderAll();
  refreshOverview();
}

export function teardownShell() {
  running = false;
  for (const u of unsubs) u();
  unsubs = [];
  clearInterval(staleTimer);
  staleTimer = null;
  clearTimeout(ovTimer);
  ovTimer = null;
  document.removeEventListener('visibilitychange', onVisibility);
  teardownTasks();
  for (const a of alerts.values()) clearTimeout(a.timer);
  alerts.clear();
  dismissedOverview.clear();
  renderAlerts();
  hideEstopBar();
  netDown = false;
  sseState = { connected: false, wanted: false, failed: false };
  renderConnBanner();
  lastRobot = null;
  lastLiveAt = 0;
  setNavOpen(false);
  closeTransient();
}

function onVisibility() {
  if (!document.hidden && running) refreshOverview();
}

// ---------------------------------------------------------------- overview poll

async function fetchOverview() {
  if (ovInflight) return ovInflight;
  ovInflight = (async () => {
    try {
      const ov = await api.get('/overview', null, { timeout: 8000 });
      ovFailed = false;
      if (running) emit('overview', ov);
      return ov;
    } catch {
      ovFailed = true;
      if (running) renderSystem();
      return null;
    } finally {
      ovInflight = null;
    }
  })();
  return ovInflight;
}

function scheduleOverview() {
  clearTimeout(ovTimer);
  if (!running) return;
  ovTimer = setTimeout(async () => {
    if (!document.hidden) await fetchOverview();
    scheduleOverview();
  }, OVERVIEW_MS);
}

/** Fetch /overview now (shared with the 3 s poll). Resolves with it or null. */
export async function refreshOverview() {
  const ov = await fetchOverview();
  scheduleOverview();
  return ov;
}

// ---------------------------------------------------------------- one-time DOM wiring

function wireOnce() {
  wired = true;
  $('logo-mark').append(icon('activity', { size: 18 }));

  // sidebar collapse + mobile drawer
  const collapseBtn = $('sidebar-collapse');
  collapseBtn.append(icon('panelClose', { cls: 'collapse-icon' }), el('span', { class: 'collapse-label', text: 'Collapse' }));
  collapseBtn.addEventListener('click', () => setCollapsed(!document.getElementById('app').classList.contains('is-collapsed')));
  const closeNav = iconBtn('x', 'Close navigation');
  closeNav.classList.add('nav-close');
  closeNav.addEventListener('click', () => setNavOpen(false));
  $('sidebar-close-slot').append(closeNav);
  const openNav = iconBtn('menu', 'Open navigation', { cls: 'nav-open' });
  openNav.addEventListener('click', () => setNavOpen(true));
  $('nav-open-slot').replaceWith(openNav);
  $('scrim').addEventListener('click', () => setNavOpen(false));
  window.matchMedia('(max-width: 1023px)').addEventListener('change', () => setNavOpen(false));

  // top bar (right side)
  const right = $('topbar-right');
  refs.realChip = chip('REAL ROBOT', { kind: 'danger', icon: 'alert' });
  refs.realChip.hidden = true;
  attachTip(refs.realChip, 'Robot link is the real Go2: gestures and manual commands move the physical dog.');
  refs.lockChip = el('button', { type: 'button', class: 'chip chip-warn', hidden: true });
  refs.lockChip.addEventListener('click', () => {
    location.hash = '#/board';
  });

  refs.target = segmented([
    { value: 'board', label: 'Board', icon: 'cpu', tip: 'Gesture control runs on the Kria board (SSH)' },
    { value: 'sim', label: 'Simulator', icon: 'box', tip: 'Gesture control runs in the local simulator' },
  ], { label: 'Target', onChange: switchTarget, cls: 'target-switch' });

  refs.systemBtn = btn('Checking…', { kind: 'ghost', cls: 'system-btn' });
  refs.systemBtn.replaceChildren(el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { class: 'btn-label', text: 'Checking…' }), icon('chevronDown', { size: 16 }));
  refs.systemBtn.setAttribute('aria-haspopup', 'dialog');
  bindPopover(refs.systemBtn, () => buildSystemPopover(), { cls: 'popover sys-pop' });

  refs.tasksBtn = btn('Tasks', { kind: 'ghost', icon: 'list', iconOnly: true });
  refs.tasksBubble = el('span', { class: 'count-bubble', hidden: true });
  const tasksWrap = el('span', { class: 'count-wrap' }, refs.tasksBtn, refs.tasksBubble);

  const userMenu = menuButton(() => [
    { label: currentTheme() === 'dark' ? 'Light theme' : 'Dark theme', icon: currentTheme() === 'dark' ? 'sun' : 'moon', onSelect: () => applyTheme(currentTheme() === 'dark' ? 'light' : 'dark') },
    { label: 'Change password', icon: 'key', onSelect: () => openAccount() },
    { label: 'Audit log', icon: 'history', onSelect: () => { prefs.set('tab.security', 'audit'); location.hash = '#/security'; } },
    'sep',
    { label: 'Log out', icon: 'logout', onSelect: () => ctx && ctx.logout() },
  ], { label: 'User menu', text: 'User', icon: 'user' });
  userMenu.classList.add('user-btn');
  refs.userBtnLabel = userMenu.querySelector('.btn-label');

  const estop = el('button', { type: 'button', id: 'estop-btn', class: 'estop', 'aria-keyshortcuts': 'Escape' },
    icon('octagon', { size: 18 }), el('span', { text: 'E-STOP' }), el('kbd', { text: 'Esc' }));
  attachTip(estop, 'Emergency stop: stops the robot and pauses gestures (Escape key)');
  estop.addEventListener('click', () => triggerEstop());
  refs.estop = estop;

  right.append(refs.realChip, refs.lockChip, refs.target.el, refs.systemBtn,
    el('span', { class: 'topbar-sep', 'aria-hidden': 'true' }), tasksWrap, userMenu, estop);

  // bars
  $('estop-bar-icon').replaceWith(icon('octagon', { size: 22 }));
  $('estop-clear').addEventListener('click', clearEstop);
  $('conn-icon').replaceWith(icon('alert', { size: 18 }));

  // nav
  buildNav();
  setCollapsed(!!prefs.get('sidebar.collapsed', false), false);
}

function buildNav() {
  const list = $('nav-list');
  list.replaceChildren(...ctx.pages.map((p) => {
    const a = el('a', { href: `#/${p.id}`, class: 'nav-item', dataset: { page: p.id } },
      icon(p.icon, { size: 18 }), el('span', { class: 'nav-label', text: p.label }));
    attachTip(a, '');
    a.addEventListener('click', () => setNavOpen(false));
    return el('li', null, a);
  }));
}

function setCollapsed(collapsed, remember = true) {
  const app = $('app');
  app.classList.toggle('is-collapsed', collapsed);
  const b = $('sidebar-collapse');
  b.setAttribute('aria-expanded', String(!collapsed));
  b.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
  const ic = b.querySelector('.collapse-icon');
  if (ic) ic.replaceWith(icon(collapsed ? 'panelOpen' : 'panelClose', { cls: 'collapse-icon' }));
  for (const a of document.querySelectorAll('.nav-item')) {
    attachTip(a, collapsed ? a.textContent : '');
  }
  attachTip(b, collapsed ? 'Expand sidebar' : '');
  if (remember) prefs.set('sidebar.collapsed', collapsed);
}

function setNavOpen(open) {
  const app = $('app');
  if (!app) return;
  app.classList.toggle('nav-open', open);
  $('scrim').hidden = !open;
}

/** Highlight the current page in the nav and set the top-bar title. */
export function setPage(id, title, subtitle) {
  for (const a of document.querySelectorAll('.nav-item')) {
    if (a.dataset.page === id) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  setPageMeta(title, subtitle);
}

export function setPageMeta(title, subtitle) {
  if (title != null) setText($('page-title'), title);
  if (subtitle !== undefined) setText($('page-subtitle'), subtitle || '');
}

function openAccount() {
  prefs.set('tab.security', 'account');
  if (location.hash === '#/security') emit('open-account', null);
  else location.hash = '#/security';
}

// ---------------------------------------------------------------- theme

export function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

export function applyTheme(theme) {
  const t = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  prefs.set('theme', t);
  emit('theme', t);
}

// ---------------------------------------------------------------- target

async function switchTarget(mode, b) {
  const ov = last('overview');
  if (ov && ov.target && ov.target.mode === mode) return;
  await act(b, async () => {
    refs.target.setDisabled(true);
    try {
      const answer = await api.post('/target', { mode }, { risk: 'warn' });
      if (answer && answer.target) emit('overview', answer);
      else await refreshOverview();
      toast(mode === 'sim' ? 'Target: local simulator' : 'Target: board via SSH', 'success');
    } finally {
      refs.target.setDisabled(false);
    }
  });
}

// ---------------------------------------------------------------- status rendering

function freshLive() {
  return Date.now() - lastLiveAt < LIVE_STALE_MS ? last('live') : null;
}

function currentRows() {
  return systemRows(last('overview'), freshLive(), lastRobot);
}

function renderSystem() {
  if (!refs.systemBtn) return;
  let sum;
  if (netDown) sum = { state: 'bad', text: 'Server unreachable' };
  else if (sseState.wanted && sseState.failed && !sseState.connected) sum = { state: 'warn', text: 'Live events reconnecting' };
  else if (ovFailed && !last('overview')) sum = { state: 'off', text: 'Status unknown' };
  else sum = summarize(currentRows());
  refs.systemBtn.className = `btn btn-ghost system-btn status-${sum.state}`;
  setText(refs.systemBtn.querySelector('.btn-label'), sum.text);
  refs.systemBtn.setAttribute('aria-label', `System status: ${sum.text}. Show details`);
  if (refs.sysPop && refs.sysPop.isConnected) fillSystemPopover(refs.sysPop);
}

function buildSystemPopover() {
  const box = el('div', { role: 'dialog', 'aria-label': 'System status' });
  refs.sysPop = box;
  fillSystemPopover(box);
  return box;
}

function fillSystemPopover(box) {
  const ov = last('overview');
  box.replaceChildren(
    el('div', { class: 'menu-label', text: ov && ov.target ? `Target: ${ov.target.label || ov.target.mode}` : 'System status' }),
    ...currentRows().map((r) => el('div', { class: 'sys-row' },
      el('span', { class: 'sys-name', text: r.name }),
      status(r.state, r.value),
      el('a', { href: r.href, 'aria-label': `Open ${r.name}` }, icon('chevronRight', { size: 16 })))));
}

function renderChips() {
  const ov = last('overview');
  const link = (lastRobot && lastRobot.link) || (ov && ov.robot && ov.robot.link);
  refs.realChip.hidden = link !== 'go2';
  const lock = ov && ov.board ? ov.board.lock : null;
  if (lock && lock.owner) {
    refs.lockChip.hidden = false;
    const mine = !!lock.mine;
    refs.lockChip.className = `chip ${mine ? 'chip-neutral' : 'chip-warn'}`;
    refs.lockChip.replaceChildren(icon('lock', { size: 14 }), el('span', { text: mine ? 'Your lock' : `Locked by ${lock.owner}` }));
    attachTip(refs.lockChip, `${mine ? 'You hold' : `${lock.owner} holds`} the board lock${lock.purpose ? ` (${lock.purpose})` : ''}${lock.since ? `, since ${lock.since}` : ''}. Click for the Board page.`);
  } else {
    refs.lockChip.hidden = true;
  }
}

function renderAll() {
  const ov = last('overview');
  if (ov && ov.target) refs.target.set(ov.target.mode);
  renderSystem();
  renderChips();
}

function applyOverview(ov) {
  if (!ov || typeof ov !== 'object') return;
  renderAll();
  syncOverviewAlerts(Array.isArray(ov.alerts) ? ov.alerts : []);
  const estop = ov.safety && ov.safety.estop;
  if (estop === true && !estopShown) {
    showEstopBar('E-STOP is active on the server. Clear it when it is safe.');
  } else if (estop === false && estopShown && Date.now() - estopPressedAt > ESTOP_GRACE_MS) {
    hideEstopBar();
  }
}

// ---------------------------------------------------------------- connection banner

function renderConnBanner() {
  const banner = $('conn-banner');
  const sseLost = sseState.wanted && sseState.failed && !sseState.connected;
  const down = netDown || sseLost;
  banner.hidden = !down;
  if (!down) return;
  setText($('conn-detail'), netDown
    ? 'The dashboard server is unreachable.'
    : 'The live event stream is disconnected; live data is not updating.');
}

// ---------------------------------------------------------------- alerts (one slim banner)

const LEVEL_RANK = { danger: 3, warn: 2, info: 1 };
const LEVEL_ICON = { danger: 'octagon', warn: 'alert', info: 'info' };
const LEVEL_WORD = { danger: 'Danger', warn: 'Warning', info: 'Info' };

function alertKey(level, text) {
  return `${level}|${text}`;
}

function normLevel(l) {
  return LEVEL_RANK[l] ? l : 'info';
}

function addAlert(a, source) {
  if (!a || typeof a.text !== 'string' || !a.text) return;
  const level = normLevel(a.level);
  const key = alertKey(level, a.text);
  const existing = alerts.get(key);
  if (existing) {
    existing.t = Date.now();
    renderAlerts();
    return;
  }
  const entry = { level, text: a.text, source, t: Date.now(), timer: null };
  if (source === 'sse' && level === 'info') entry.timer = setTimeout(() => removeAlert(key), 20000);
  alerts.set(key, entry);
  const sse = Array.from(alerts.entries()).filter(([, e]) => e.source === 'sse').sort((x, y) => x[1].t - y[1].t);
  while (sse.length > MAX_SSE_ALERTS) removeAlert(sse.shift()[0], false);
  renderAlerts();
}

function removeAlert(key, render = true) {
  const e = alerts.get(key);
  if (!e) return;
  clearTimeout(e.timer);
  alerts.delete(key);
  if (render) renderAlerts();
}

function dismiss(key) {
  const e = alerts.get(key);
  if (e && e.source === 'overview') dismissedOverview.add(key);
  removeAlert(key);
}

function syncOverviewAlerts(list) {
  const current = new Set();
  for (const a of list) {
    if (a && a.text) current.add(alertKey(normLevel(a.level), a.text));
  }
  for (const [key, e] of Array.from(alerts.entries())) {
    if (e.source === 'overview' && !current.has(key)) removeAlert(key, false);
  }
  for (const key of Array.from(dismissedOverview)) {
    if (!current.has(key)) dismissedOverview.delete(key);
  }
  for (const a of list) {
    if (!a || !a.text) continue;
    const key = alertKey(normLevel(a.level), a.text);
    if (!dismissedOverview.has(key) && !alerts.has(key)) {
      alerts.set(key, { level: normLevel(a.level), text: a.text, source: 'overview', t: Date.now(), timer: null });
    }
  }
  renderAlerts();
  emit('alerts', sortedAlerts());
}

function sortedAlerts() {
  return Array.from(alerts.entries())
    .map(([key, e]) => ({ key, ...e }))
    .sort((x, y) => (LEVEL_RANK[y.level] - LEVEL_RANK[x.level]) || (y.t - x.t));
}

/** Current alerts, most severe first (used by the Home page). */
export function currentAlerts() {
  return sortedAlerts();
}

function alertLine(a, { withMore = 0 } = {}) {
  const close = iconBtn('x', 'Dismiss alert', { size: 'sm' });
  close.addEventListener('click', () => dismiss(a.key));
  const line = el('div', { class: ['alert-line', `lvl-${a.level}`], role: a.level === 'danger' ? 'alert' : null },
    icon(LEVEL_ICON[a.level], { size: 16, cls: 'lvl-icon' }),
    el('span', { class: 'sr-only', text: `${LEVEL_WORD[a.level]}: ` }),
    el('span', { class: 'alert-text', text: a.text }));
  if (withMore) {
    const more = btn(alertsExpanded ? 'Show less' : `+${withMore} more`, { kind: 'ghost', size: 'sm', cls: 'alert-more' });
    more.setAttribute('aria-expanded', String(alertsExpanded));
    more.addEventListener('click', () => {
      alertsExpanded = !alertsExpanded;
      renderAlerts();
    });
    line.append(more);
  }
  line.append(close);
  return line;
}

function renderAlerts() {
  const banner = $('alert-banner');
  if (!banner) return;
  const list = sortedAlerts();
  if (!list.length) {
    banner.hidden = true;
    banner.replaceChildren();
    alertsExpanded = false;
    return;
  }
  banner.hidden = false;
  banner.className = `alert-banner lvl-${list[0].level}`;
  const top = alertLine(list[0], { withMore: list.length - 1 });
  const nodes = [top];
  if (alertsExpanded && list.length > 1) {
    nodes.push(el('div', { class: 'alert-list' }, ...list.slice(1).map((a) => alertLine(a))));
  }
  banner.replaceChildren(...nodes);
  emit('alerts', list);
}

// ---------------------------------------------------------------- E-STOP

/** Send E-STOP immediately (no confirm). Retries twice on network failure. */
export async function triggerEstop() {
  const b = refs.estop;
  if (!b) return;
  b.classList.remove('flash');
  void b.offsetWidth; // restart the animation
  b.classList.add('flash');
  clearTimeout(estopFlashTimer);
  estopFlashTimer = setTimeout(() => b.classList.remove('flash'), 1400);
  estopPressedAt = Date.now();
  emit('estop', { t: estopPressedAt }); // pages stop local loops (drive joystick)
  showEstopBar('Sending stop to the robot…');
  let lastErr = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await api.post('/estop', {}, { noConfirm: true, timeout: 4000 });
      const actions = Array.isArray(res.actions) ? res.actions.map(String) : [];
      showEstopBar(actions.length ? actions.join(' · ') : 'Acknowledged by the server.');
      refreshOverview();
      return;
    } catch (err) {
      lastErr = err;
      if (err.unauthorized) return;
      if (!(err.network || err.timeout)) break;
    }
  }
  const msg = `E-STOP request FAILED (${lastErr ? lastErr.message : 'unknown error'}). Use the robot's own stop / remote now!`;
  showEstopBar(msg, true);
  toast(msg, 'error', { timeout: 0 });
}

function showEstopBar(text, failed = false) {
  const bar = $('estop-bar');
  estopShown = true;
  bar.hidden = false;
  bar.classList.toggle('is-failed', failed);
  setText($('estop-bar-text'), text || 'Robot motion stopped. Gestures are paused.');
}

function hideEstopBar() {
  estopShown = false;
  const bar = $('estop-bar');
  if (bar) bar.hidden = true;
}

async function clearEstop(ev) {
  const b = ev.currentTarget;
  await act(b, async () => {
    await api.post('/estop/clear', {});
    estopPressedAt = 0;
    hideEstopBar();
    toast('E-STOP cleared. Gestures stay paused until you resume them on the Live page.', 'info', { timeout: 7000 });
    refreshOverview();
  });
}

