// Home: "Is the system ready?" One hero card with a status sentence and the next
// sensible action, four stat tiles, and the alert list (only when there are alerts).

import { api } from '../api.js';
import { last } from '../events.js';
import {
  el, setText, card, act, btn, menuButton, tile, icon, toast, prefs, emptyState,
} from '../ui.js';
import { trackTask } from '../tasks.js';
import { currentAlerts } from '../shell.js';
import { MODE_TEXT, modelLabel, trainingState } from '../system.js';
import * as fmt from '../format.js';

const LIVE_STALE_MS = 3000;
const STATUS_TEXT = {
  listening: 'Listening', candidate: 'Gesture candidate', 'waiting-for-calm': 'Waiting to re-arm', busy: 'Robot busy', paused: 'Paused', 'no-board': 'No board',
};
const LEVEL_ICON = { danger: 'octagon', warn: 'alert', info: 'info' };

export async function mount(root, scope, app) {
  let sim = null;
  let simPoll = null;
  let lastLiveAt = 0;

  // ---- hero
  const heroDot = el('span', { class: 'hero-dot', 'aria-hidden': 'true' });
  const heroTitle = el('div', { class: 'hero-title', role: 'status', 'aria-live': 'polite' });
  const heroText = el('div', { class: 'hero-text' });
  const heroActions = el('div', { class: 'hero-actions' });
  const hero = el('section', { class: 'card hero status-off', 'aria-label': 'System readiness' },
    el('div', { class: 'hero-main' }, heroDot, el('div', null, heroTitle, heroText)),
    heroActions);

  // ---- tiles
  const tBoard = tile({ label: 'Board mode', icon: 'cpu', href: '#/board' });
  const tModel = tile({ label: 'Model', icon: 'sparkles', href: '#/board' });
  tModel.el.addEventListener('click', () => prefs.set('tab.board', 'models'));
  const tLive = tile({ label: 'Live', icon: 'activity', href: '#/live' });
  const tTrain = tile({ label: 'Training', icon: 'trending', href: '#/training' });
  const tiles = el('div', { class: 'tiles' }, tBoard.el, tModel.el, tLive.el, tTrain.el);

  // ---- alerts
  const alertRows = el('div', { class: 'alert-rows' });
  const alertsCard = card({ title: 'Alerts', icon: 'bell' }, alertRows);
  alertsCard.hidden = true;

  root.append(hero, tiles, alertsCard);

  // ---------------------------------------------------------------- actions

  async function inference(action, b, label) {
    const res = await act(b, () => api.post('/board/inference', { action }, { risk: action === 'robot' ? 'danger' : 'warn', signal: scope.signal }));
    if (!res) return;
    await trackTask(res, label);
    if (scope.alive) app.refreshOverview();
  }

  async function simAction(path, b, success) {
    const st = await act(b, () => api.post(path, {}, { signal: scope.signal }), { success });
    if (st && scope.alive) {
      sim = st;
      render();
    }
  }

  async function resumeGestures(b) {
    const res = await act(b, () => api.put('/settings', { values: { paused: false } }, { signal: scope.signal }), { success: 'Gestures resumed' });
    if (res && scope.alive) app.refreshOverview();
  }

  async function clearEstop(b) {
    const res = await act(b, () => api.post('/estop/clear', {}, { signal: scope.signal }));
    if (res && scope.alive) {
      toast('E-STOP cleared. Gestures stay paused until you resume them.', 'info');
      app.refreshOverview();
    }
  }

  const A = {
    startTest: () => ({ label: 'Start test mode', icon: 'play', run: (b) => inference('start', b, 'Start test mode') }),
    startRobot: () => ({ label: 'Start robot mode', icon: 'zap', run: (b) => inference('robot', b, 'Start robot mode') }),
    stop: () => ({ label: 'Stop inference', icon: 'square', run: (b) => inference('stop', b, 'Stop inference') }),
    resume: () => ({ label: 'Resume gestures', icon: 'play', run: resumeGestures }),
    clearEstop: () => ({ label: 'Clear E-STOP', icon: 'check', run: clearEstop }),
    openBoard: () => ({ label: 'Open Board', icon: 'cpu', href: '#/board' }),
    openLive: () => ({ label: 'Open Live', icon: 'activity', href: '#/live' }),
    startSim: () => ({ label: 'Start simulator', icon: 'play', run: (b) => simAction('/sim/start', b, 'Simulator started') }),
    stopSim: () => ({ label: 'Stop simulator', icon: 'square', run: (b) => simAction('/sim/stop', b, 'Simulator stopped') }),
  };

  function actionButton(a, kind) {
    if (a.href) {
      const link = el('a', { class: `btn btn-${kind}`, href: a.href }, icon(a.icon), el('span', { class: 'btn-label', text: a.label }));
      return link;
    }
    const b = btn(a.label, { kind, icon: a.icon });
    b.addEventListener('click', () => a.run(b));
    return b;
  }

  // ---------------------------------------------------------------- state -> hero

  function liveNow() {
    return Date.now() - lastLiveAt < LIVE_STALE_MS ? last('live') : null;
  }

  function heroModel(ov) {
    if (!ov) return { state: 'off', title: 'Checking the system…', text: 'Waiting for the first status update.', actions: [] };
    const safety = ov.safety || {};
    if (safety.estop) {
      return {
        state: 'bad', title: 'E-STOP active', text: 'Robot motion is stopped and gestures are paused. Clear the E-STOP when it is safe.',
        actions: [A.clearEstop()], menu: [],
      };
    }
    const pausedNote = safety.paused ? ', gestures paused' : ', gestures active';
    const extra = safety.paused ? [A.resume()] : [];
    if (ov.target && ov.target.mode === 'sim') {
      if (!sim) return { state: 'off', title: 'Simulator target', text: 'Checking the simulator…', actions: [] };
      if (!sim.running) {
        return { state: 'warn', title: 'Simulator is stopped', text: 'Start it to try gesture control without the board or the dog.', actions: [A.startSim()] };
      }
      return {
        state: safety.paused ? 'warn' : 'ok',
        title: `Ready: simulator running${pausedNote}`,
        text: `The virtual person performs: ${sim.person || 'nothing'}. Change it on the Live or Simulation page.`,
        actions: [...extra, A.openLive()],
        menu: [A.stopSim()],
      };
    }
    const b = ov.board || {};
    if (!b.reachable) {
      return { state: 'bad', title: 'Board offline', text: b.error || 'Check the Ethernet cable and the network profile for the board.', actions: [A.openBoard()] };
    }
    if (b.ssh === false) {
      return { state: 'warn', title: 'Board reachable, but SSH is not available', text: b.error || 'The dashboard cannot control the board right now.', actions: [A.openBoard()] };
    }
    const lv = liveNow();
    const streamDown = lv ? !lv.connected : (ov.live ? !ov.live.connected : false);
    switch (b.mode) {
      case 'stopped':
        return {
          state: 'off', title: 'Not running: inference is stopped',
          text: 'Start test mode to see gestures in RViz, or robot mode to let the real dog react.',
          actions: [A.startTest()], menu: [A.startRobot(), { label: 'Robot dry-run', icon: 'eye', run: (bb) => inference('robot-dry', bb, 'Robot dry-run') }, { label: 'Test mode without viewer', icon: 'play', run: (bb) => inference('start-headless', bb, 'Start test mode (headless)') }],
        };
      case 'test':
        return {
          state: streamDown ? 'warn' : (safety.paused ? 'warn' : 'ok'),
          title: streamDown ? 'Board in test mode, but no live stream' : `Ready: board in test mode${pausedNote}`,
          text: 'Gestures drive the RViz animator on this laptop. Switch to robot mode to drive the real Go2.',
          actions: [...extra, A.startRobot()], menu: [A.stop(), { label: 'Robot dry-run', icon: 'eye', run: (bb) => inference('robot-dry', bb, 'Robot dry-run') }],
        };
      case 'robot-dry':
        return {
          state: streamDown ? 'warn' : 'ok',
          title: `Robot dry-run${pausedNote}`,
          text: 'The bridge decides like in robot mode but never commands the dog.',
          actions: [...extra, A.startRobot()], menu: [A.stop(), A.startTest()],
        };
      case 'robot':
        return {
          state: streamDown ? 'warn' : 'ok',
          title: streamDown ? 'Robot mode, but no live stream' : `Robot mode: the real Go2 reacts to gestures${safety.paused ? ' (paused)' : ''}`,
          text: 'Keep the area clear. E-STOP is top right (or press Escape).',
          actions: [...extra, { ...A.stop(), danger: true }], menu: [A.startTest()],
        };
      default:
        return { state: 'warn', title: 'Board mode unknown', text: 'The board did not report its inference mode.', actions: [A.openBoard()], menu: [A.stop()] };
    }
  }

  let heroSig = '';
  function render() {
    const ov = app.overview();
    const m = heroModel(ov);
    hero.className = `card hero status-${m.state}`;
    setText(heroTitle, m.title);
    setText(heroText, m.text);
    // Rebuild the buttons only when the set of actions changes (keeps focus / busy state).
    const sig = JSON.stringify([...(m.actions || []).map((a) => a.label), '|', ...(m.menu || []).map((a) => a.label)]);
    if (sig !== heroSig) {
      heroSig = sig;
      const nodes = (m.actions || []).map((a, i) => actionButton(a, a.danger ? 'danger' : (i === (m.actions.length - 1) ? 'primary' : 'secondary')));
      if (m.menu && m.menu.length) {
        nodes.push(menuButton(m.menu.map((a) => ({ label: a.label, icon: a.icon, onSelect: (b) => (a.href ? (location.hash = a.href) : a.run(b)) })), { label: 'More actions' }));
      }
      heroActions.replaceChildren(...nodes);
    }
    renderTiles(ov);
    if (ov && ov.target) app.setSubtitle(ov.target.label || (ov.target.mode === 'sim' ? 'Local simulator' : 'Board'));
  }

  function renderTiles(ov) {
    const b = ov && ov.board;
    if (!b) tBoard.set('–', 'Checking…', 'off');
    else if (!b.reachable) tBoard.set('Offline', 'Board not reachable', 'bad');
    else tBoard.set(MODE_TEXT[b.mode] || 'Unknown', fmt.isNum(b.ping_ms) ? `Reachable · ${fmt.ms(b.ping_ms)}` : 'Reachable', b.ssh === false ? 'warn' : 'ok');

    const ml = modelLabel(b && b.model);
    const model = b && b.model;
    if (ml) {
      const sub = ml.detail || `${model.classes || '?'} gestures${fmt.isNum(model.input) ? ` \u00b7 ${model.input}\u00d7${model.input}` : ''}`;
      tModel.set(ml.title, sub, 'ok', ml.full);
    } else {
      tModel.set('\u2013', b && b.reachable ? 'Unknown model' : 'No board', 'off');
    }

    const lv = liveNow();
    if (lv) {
      if (lv.connected) tLive.set(`${fmt.int(lv.fps)} fps`, STATUS_TEXT[lv.status] || lv.status || 'Live', 'ok');
      else tLive.set('No stream', 'Board output not connected', b && b.mode === 'stopped' ? 'off' : 'bad');
    } else if (ov && ov.live) {
      tLive.set(ov.live.connected ? `${fmt.int(ov.live.fps)} fps` : 'No stream', STATUS_TEXT[ov.live.status] || (ov.live.connected ? 'Live' : 'Not connected'), ov.live.connected ? 'ok' : 'off');
    } else {
      tLive.set('–', 'No data', 'off');
    }

    const t = ov && ov.training;
    const ts = trainingState(t);
    if (!t) tTrain.set('\u2013', 'Checking\u2026', 'off');
    else if (ts.configured === false) tTrain.set('Not set up', 'Optional: add a machine', 'off');
    else if (t.ok === true) tTrain.set(`${fmt.int(t.running_jobs ?? 0)} running`, `${t.host || 'Training machine'} \u00b7 ${t.detail || 'reachable'}`, 'ok');
    else tTrain.set(t.host || 'Training', t.ok === false ? (t.detail || 'unreachable') : 'checking\u2026', t.ok === false ? 'warn' : 'off');
  }

  function renderAlerts(list) {
    const items = Array.isArray(list) ? list : [];
    alertsCard.hidden = !items.length;
    alertRows.replaceChildren(...items.map((a) => el('div', { class: ['alert-row', `lvl-${a.level}`] },
      icon(LEVEL_ICON[a.level] || 'info', { size: 18, cls: 'lvl-icon' }),
      el('span', { class: 'sr-only', text: `${a.level}: ` }),
      el('span', { text: a.text }))));
  }

  // ---------------------------------------------------------------- data

  const loadSim = async () => {
    try {
      const st = await api.get('/sim/state', null, { signal: scope.signal, timeout: 6000 });
      if (scope.alive) {
        sim = st;
        render();
      }
    } catch {
      /* hero keeps "checking"; the sim page shows details */
    }
  };

  function onOverview(ov) {
    const isSim = ov && ov.target && ov.target.mode === 'sim';
    if (isSim && !simPoll) simPoll = scope.poll(loadSim, 3000);
    else if (!isSim && simPoll) {
      simPoll.stop();
      simPoll = null;
      sim = null;
    }
    render();
  }

  scope.sub('overview', onOverview);
  scope.sub('live', () => {
    const wasStale = Date.now() - lastLiveAt > LIVE_STALE_MS;
    lastLiveAt = Date.now();
    if (wasStale) render();
  });
  scope.sub('alerts', renderAlerts);
  scope.interval(() => {
    if (Date.now() - lastLiveAt > LIVE_STALE_MS) render();
  }, 2000);

  renderAlerts(currentAlerts());
  const cached = app.overview();
  if (cached) onOverview(cached);
  else render();
  const fresh = await app.refreshOverview();
  if (!scope.alive) return;
  if (!fresh && !app.overview()) {
    hero.after(emptyState({ kind: 'error', title: 'Could not load the system status', text: 'The dashboard server did not answer. It retries automatically.' }));
  }
}
