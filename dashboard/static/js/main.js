// Entry point: boot, auth gate, shell and hash router (#/live ...).

import { api, getAuthState, setCsrf, onUnauthorized } from './api.js';
import { startEvents, stopEvents, resetEventState, last } from './events.js';
import { initShell, teardownShell, refreshOverview, triggerEstop, applyTheme, setPage, setPageMeta } from './shell.js';
import { renderAuth } from './auth.js';
import { createScope } from './scope.js';
import { el, emptyState, prefs, anyDialogOpen, setText, closeTransient, btn } from './ui.js';

export const PAGES = [
  { id: 'home', label: 'Home', icon: 'home', title: 'Home', subtitle: 'Is the system ready?', load: () => import('./pages/home.js') },
  { id: 'live', label: 'Live', icon: 'activity', title: 'Live', subtitle: 'Camera, detected gestures and gesture-control settings', load: () => import('./pages/live.js') },
  { id: 'simulation', label: 'Simulation', icon: 'box', title: 'Simulation', subtitle: 'RViz of the simulated robot or the board test mode', load: () => import('./pages/simulation.js') },
  { id: 'drive', label: 'Drive', icon: 'gamepad', title: 'Drive', subtitle: 'Manual control of the robot', load: () => import('./pages/drive.js') },
  { id: 'record', label: 'Record', icon: 'record', title: 'Record', subtitle: 'Guided recording of labelled gesture takes', load: () => import('./pages/record.js') },
  { id: 'dataset', label: 'Dataset', icon: 'database', title: 'Dataset', subtitle: 'Browse, relabel and clean recorded clips', load: () => import('./pages/dataset.js') },
  { id: 'training', label: 'Training', icon: 'trending', title: 'Training', subtitle: 'Training machines, runs and jobs', load: () => import('./pages/training.js') },
  { id: 'board', label: 'Board', icon: 'cpu', title: 'Board', subtitle: 'Kria KV260 over Ethernet / SSH', load: () => import('./pages/board.js') },
  { id: 'security', label: 'Security', icon: 'shield', title: 'Security', subtitle: 'Audit log and account', load: () => import('./pages/security.js') },
];

// Routes of the previous UI version (remembered last page, bookmarks).
const ALIASES = { overview: 'home', operation: 'live', manual: 'drive', recording: 'record' };

const $ = (id) => document.getElementById(id);

let appRunning = false;
let currentUser = null;
let current = null; // {id, scope}
let navToken = 0;
let bootRetry = null;

const appCtx = {
  /** Last GET /api/overview answer (or undefined). */
  overview: () => last('overview'),
  refreshOverview: () => refreshOverview(),
  user: () => currentUser,
  logout: () => logout(),
  /** Pages may refine the top-bar subtitle (e.g. with the target label). */
  setSubtitle: (text) => setPageMeta(null, text),
};

// ---------------------------------------------------------------- boot

async function boot() {
  applyTheme(prefs.get('theme', 'dark'));
  // Capture phase on window: runs before any page / noVNC handler can stop the event.
  window.addEventListener('keydown', onGlobalKey, true);
  window.addEventListener('hashchange', () => {
    if (appRunning) route(true);
  });
  onUnauthorized(() => showAuth());
  document.querySelector('.skip-link').addEventListener('click', (ev) => {
    ev.preventDefault();
    $('page').focus();
  });
  await loadAuthState();
}

async function loadAuthState() {
  clearTimeout(bootRetry);
  let state;
  try {
    state = await getAuthState();
  } catch (err) {
    showBootMessage(`Cannot reach the dashboard server (${err.message}). Retrying…`, true);
    bootRetry = setTimeout(loadAuthState, 3000);
    return;
  }
  $('boot').hidden = true;
  if (state.logged_in) startApp(state);
  else showAuth(state);
}

function showBootMessage(text, isError) {
  const b = $('boot');
  b.hidden = false;
  b.classList.toggle('is-error', !!isError);
  setText($('boot-text'), text);
}

// ---------------------------------------------------------------- auth gate

async function showAuth(state) {
  stopApp();
  let st = state;
  if (!st) {
    try {
      st = await getAuthState();
    } catch {
      st = { setup_needed: false, logged_in: false };
    }
    if (st.logged_in) {
      startApp(st);
      return;
    }
  }
  $('boot').hidden = true;
  const screen = $('auth-screen');
  screen.hidden = false;
  renderAuth(screen, st, (newState) => onLoggedIn(newState));
}

async function onLoggedIn(state) {
  let st = state;
  if (!st || !st.csrf) {
    try {
      st = await getAuthState();
    } catch {
      /* keep what the login answered */
    }
  }
  if (st && st.logged_in) startApp(st);
  else showAuth(st);
}

function startApp(state) {
  if (appRunning) return;
  appRunning = true;
  setCsrf(state.csrf);
  currentUser = state.user || null;
  $('auth-screen').hidden = true;
  $('auth-screen').replaceChildren();
  $('boot').hidden = true;
  $('app').hidden = false;
  initShell({ pages: PAGES, user: currentUser, logout });
  startEvents(async () => {
    const st = await getAuthState();
    if (!st.logged_in) {
      showAuth(st);
      return false;
    }
    return true;
  });
  route(false);
}

function stopApp() {
  if (!appRunning) return;
  appRunning = false;
  disposeCurrent();
  stopEvents();
  teardownShell();
  resetEventState();
  for (const d of document.querySelectorAll('dialog[open]')) d.close();
  for (const d of document.querySelectorAll('.drawer:not(#task-drawer)')) d.remove();
  $('page').replaceChildren();
  $('app').hidden = true;
  currentUser = null;
}

async function logout() {
  try {
    await api.post('/auth/logout', {}, { timeout: 5000 });
  } catch {
    /* even if the server call fails, leave the UI */
  }
  setCsrf(null);
  showAuth();
}

// ---------------------------------------------------------------- router

function parseHash() {
  const m = /^#\/([a-z-]+)/.exec(location.hash || '');
  if (!m) return null;
  return ALIASES[m[1]] || m[1];
}

function disposeCurrent() {
  closeTransient();
  if (current) {
    current.scope.dispose();
    current = null;
  }
}

async function route(fromNavigation) {
  let wanted = parseHash();
  if (!wanted) {
    const saved = prefs.get('lastPage', 'home');
    wanted = ALIASES[saved] || saved;
  }
  const page = PAGES.find((p) => p.id === wanted) || PAGES[0];
  if ((location.hash || '') !== `#/${page.id}`) history.replaceState(null, '', `#/${page.id}`);
  if (current && current.id === page.id) return;
  disposeCurrent();
  const token = ++navToken;
  prefs.set('lastPage', page.id);
  $('app').dataset.page = page.id;
  setPage(page.id, page.title, page.subtitle);
  document.title = `${page.title} · Durance AI`;
  const host = $('page');
  host.replaceChildren(el('div', { class: 'page' }, emptyState({ kind: 'loading', title: `Loading ${page.title}…` })));

  let mod;
  try {
    mod = await page.load();
  } catch (err) {
    if (token !== navToken) return;
    host.replaceChildren(el('div', { class: 'page' }, emptyState({
      kind: 'error',
      title: `Could not load the ${page.title} page`,
      text: err.message,
      actions: btn('Try again', { kind: 'secondary', icon: 'refresh', onClick: () => { current = null; route(false); } }),
    })));
    return;
  }
  if (token !== navToken || !appRunning) return;

  const scope = createScope();
  current = { id: page.id, scope };
  // Remove drawers a page opened (they are disposed with the page scope too).
  const root = el('div', { class: ['page', `page-${page.id}`] });
  host.replaceChildren(root);
  host.scrollTop = 0;
  try {
    await mod.mount(root, scope, appCtx);
  } catch (err) {
    if (!scope.alive) return;
    console.error(`[router] ${page.id} failed to mount`, err);
    root.replaceChildren(emptyState({ kind: 'error', title: `The ${page.title} page failed to start`, text: err.message }));
  }
  if (fromNavigation && scope.alive) $('page-title').focus({ preventScroll: true });
}

// ---------------------------------------------------------------- keyboard

function onGlobalKey(ev) {
  if (ev.key !== 'Escape' || ev.repeat || !appRunning) return;
  // With a modal dialog open, Escape closes the dialog (native behaviour) instead.
  if (anyDialogOpen()) return;
  closeTransient();
  triggerEstop();
}

boot();
