// Simulation: real RViz 2 from the durance-rviz container, shown with the
// vendored noVNC client over the dashboard's authenticated WebSocket
// (/api/rviz/vnc). Side panel: simulator, virtual person, live outputs, details.

import RFB from '../../vendor/novnc-1.5.0/core/rfb.js';
import { initLogging } from '../../vendor/novnc-1.5.0/core/util/logging.js';
import { api } from '../api.js';
import { last } from '../events.js';
import {
  el, setText, card, act, btn, iconBtn, setBtnLabel, segmented, status, setStatus, collapsible, kv, emptyState,
  menuButton, prefs, icon,
} from '../ui.js';
import { trackTask } from '../tasks.js';
import { createRateBars } from '../components/rateBars.js';
import { createVirtualPerson } from '../components/virtualPerson.js';
import * as fmt from '../format.js';

// Our UI shows connection problems itself; keep noVNC's console logging quiet.
initLogging('none');

const COMPONENT_LABELS = {
  fake_board: 'Fake board', bridge: 'Gesture bridge', animator: 'Go2 animator', rviz: 'Desktop RViz', tap: 'Event tap',
};
const SOURCE_TEXT = { sim: 'Simulator', 'board-test': 'Board test mode' };
const RETRY_MS = 3000;

function componentState(v) {
  if (v === true) return ['ok', 'running'];
  if (v === false || v == null) return ['off', 'stopped'];
  if (typeof v === 'string') {
    const s = v.toLowerCase();
    if (['running', 'ok', 'up', 'active', 'started'].includes(s)) return ['ok', v];
    if (['error', 'failed', 'crashed', 'dead'].includes(s)) return ['bad', v];
    if (['starting', 'stopping'].includes(s)) return ['busy', v];
    return ['off', v];
  }
  if (typeof v === 'object') {
    const ok = v.running ?? v.ok ?? v.alive;
    return [ok ? 'ok' : 'off', v.detail || v.status || (ok ? 'running' : 'stopped')];
  }
  return ['off', String(v)];
}

/** Only connect to our own WebSocket endpoint. */
function safeVncPath(p) {
  return typeof p === 'string' && /^\/api\/rviz\/[A-Za-z0-9/_-]*$/.test(p) ? p : null;
}

export async function mount(root, scope) {
  let rviz = null; // GET /rviz/state
  let rvizError = null;
  let sim = null;
  let source = prefs.get('sim.source', 'sim');
  let rfb = null;
  let rfbToken = 0;
  let rfbState = 'idle'; // idle | connecting | connected | disconnected
  let rfbDetail = '';
  let connectedKey = '';
  let userDisconnected = false;
  let retryTimer = null;
  let viewOnly = !!prefs.get('sim.viewOnly', false);

  // ---------------------------------------------------------------- RViz card
  const sourceSeg = segmented([
    { value: 'sim', label: 'Simulator', icon: 'box', tip: 'Show the local simulator (ROS domain 88)' },
    { value: 'board-test', label: 'Board test', icon: 'cpu', tip: 'Board test mode: show the animator commanded by the board (domain 42)' },
  ], { label: 'RViz source', value: source, onChange: (v, b) => chooseSource(v, b) });
  const startStopBtn = btn('Start RViz', { kind: 'primary', icon: 'play' });
  const fullBtn = iconBtn('maximize', 'Full screen');
  const panelBtn = iconBtn('panelRight', 'Show or hide the side panel');
  const rvizStatus = status('off', 'Checking…');
  const moreBtn = menuButton(() => [
    { label: 'Reconnect', icon: 'refresh', disabled: !(rviz && rviz.container && rviz.container.running), onSelect: reconnect },
    { label: viewOnly ? 'Allow mouse and keyboard in RViz' : 'View only (no input to RViz)', icon: viewOnly ? 'hand' : 'eye', onSelect: toggleViewOnly },
    { label: 'Rebuild the RViz image', icon: 'hammer', onSelect: (b) => buildImage(b) },
  ], { label: 'More RViz actions' });
  const screen = el('div', { class: 'rviz-screen', 'aria-label': 'RViz view' });
  const overlay = el('div', { class: 'rviz-overlay' });
  const stage = el('div', { class: 'rviz-stage' }, screen, overlay);
  const rvizToolbar = el('div', { class: 'rviz-toolbar' }, sourceSeg.el, startStopBtn, rvizStatus, el('span', { class: 'spacer' }), fullBtn, moreBtn, panelBtn);
  const rvizCard = el('section', { class: 'card rviz-card', 'aria-label': 'RViz' }, rvizToolbar, stage);

  // ---------------------------------------------------------------- side panel
  const simStatus = status('off', 'Checking…');
  const simBtn = btn('Start', { kind: 'secondary', size: 'sm', icon: 'play' });
  const simMenu = menuButton(() => [
    { label: 'Reset robot pose', icon: 'reset', disabled: !(sim && sim.running), onSelect: (b) => simAction(b, '/sim/reset', {}, 'Robot back at the origin') },
    { label: 'Open RViz window on this laptop', icon: 'monitor', onSelect: (b) => simAction(b, '/sim/rviz', { action: 'start', source }, 'Desktop RViz launched') },
    { label: 'Close the laptop RViz window', icon: 'x', onSelect: (b) => simAction(b, '/sim/rviz', { action: 'stop', source }, 'Desktop RViz closed') },
  ], { label: 'More simulator actions' });
  const simCard = card({ title: 'Simulator', actions: [simBtn, simMenu] }, simStatus);

  const person = createVirtualPerson({ compact: true, onState: (st) => applySim(st) });
  const personText = el('div', { class: 'small muted', text: '' });
  const personCard = card({ title: 'Virtual person' }, person.el, personText);

  const rates = createRateBars({ compact: true });
  const liveStatus = status('off', 'No live data');
  const ratesCard = card({ title: 'Live outputs', actions: [liveStatus] }, rates.el);

  const detailsBody = el('div', { class: 'stack-sm' });
  const details = collapsible({ title: 'Details', key: 'sim.details', meta: '' }, detailsBody);
  const detailsCard = el('section', { class: 'card' }, details);

  const side = el('aside', { class: 'side-panel', 'aria-label': 'Simulator panel' }, simCard, personCard, ratesCard, detailsCard);
  const layout = el('div', { class: 'sim-layout' }, rvizCard, side);
  root.append(layout);

  function setPanel(open) {
    side.hidden = !open;
    layout.classList.toggle('side-collapsed', !open);
    panelBtn.setAttribute('aria-pressed', String(open));
    prefs.set('sim.panel', open);
  }
  setPanel(prefs.get('sim.panel', true) !== false);
  panelBtn.addEventListener('click', () => setPanel(side.hidden));

  // ---------------------------------------------------------------- RFB (noVNC)

  function disconnectRfb() {
    clearTimeout(retryTimer);
    retryTimer = null;
    rfbToken += 1;
    if (rfb) {
      const r = rfb;
      rfb = null;
      try {
        r.disconnect();
      } catch {
        /* already closed */
      }
    }
    screen.replaceChildren();
  }

  function connectRfb(vnc) {
    disconnectRfb();
    const path = safeVncPath(vnc && vnc.path);
    if (!path) {
      rfbState = 'disconnected';
      rfbDetail = 'The server sent an unexpected VNC path.';
      render();
      return;
    }
    const token = rfbToken;
    const url = `${location.protocol === 'https:' ? 'wss://' : 'ws://'}${location.host}${path}`;
    rfbState = 'connecting';
    rfbDetail = '';
    connectedKey = `${path}|${vnc.password || ''}`;
    try {
      rfb = new RFB(screen, url, { credentials: { password: vnc.password || '' } });
    } catch (err) {
      rfb = null;
      rfbState = 'disconnected';
      rfbDetail = err.message;
      render();
      return;
    }
    rfb.scaleViewport = true;
    rfb.background = 'transparent';
    rfb.focusOnClick = true;
    rfb.viewOnly = viewOnly;
    rfb.addEventListener('connect', () => {
      if (token !== rfbToken) return;
      rfbState = 'connected';
      render();
    });
    rfb.addEventListener('disconnect', (e) => {
      if (token !== rfbToken) return;
      rfb = null;
      rfbState = 'disconnected';
      rfbDetail = e.detail && e.detail.clean ? 'The RViz session ended.' : 'The connection to RViz was lost.';
      render();
      if (!userDisconnected && rviz && rviz.container && rviz.container.running && !retryTimer) {
        retryTimer = scope.timeout(() => {
          retryTimer = null;
          if (rfbState === 'disconnected' && rviz && rviz.vnc) connectRfb(rviz.vnc);
        }, RETRY_MS);
      }
    });
    rfb.addEventListener('credentialsrequired', () => {
      if (token === rfbToken && rfb) rfb.sendCredentials({ password: (vnc && vnc.password) || '' });
    });
    rfb.addEventListener('securityfailure', (e) => {
      if (token !== rfbToken) return;
      rfbDetail = `Authentication failed${e.detail && e.detail.reason ? `: ${e.detail.reason}` : ''}`;
    });
    render();
  }
  scope.add(disconnectRfb); // disconnect on page leave

  function syncRfb() {
    const running = rviz && rviz.container && rviz.container.running && rviz.vnc;
    if (!running) {
      if (rfb || rfbState !== 'idle') {
        disconnectRfb();
        rfbState = 'idle';
      }
      return;
    }
    const key = `${safeVncPath(rviz.vnc.path)}|${rviz.vnc.password || ''}`;
    if (userDisconnected) return;
    if (rfbState === 'idle' || (key !== connectedKey && rfbState !== 'connecting')) connectRfb(rviz.vnc);
  }

  // ---------------------------------------------------------------- RViz state -> view

  function overlayContent() {
    if (rvizError && !rviz) {
      return emptyState({
        kind: 'error', title: 'RViz service unavailable', text: rvizError,
        actions: btn('Try again', { kind: 'secondary', icon: 'refresh', onClick: () => statePoll.now() }),
      });
    }
    if (!rviz) return emptyState({ kind: 'loading', title: 'Checking RViz…' });
    const docker = rviz.docker || {};
    const image = rviz.image || {};
    const ctr = rviz.container || {};
    if (docker.ok === false) {
      return emptyState({
        icon: 'alert', title: 'Docker is not usable',
        text: `${docker.detail || 'The dashboard cannot use Docker.'} RViz runs in a container; fix Docker access (for example add your user to the docker group and log in again), then reload.`,
        actions: btn('Check again', { kind: 'secondary', icon: 'refresh', onClick: () => statePoll.now() }),
      });
    }
    if (image.present === false) {
      if (image.build_task) {
        return emptyState({
          kind: 'loading', title: 'Building the RViz image…', text: 'This takes several minutes the first time.',
          actions: btn('Show progress', { kind: 'secondary', icon: 'list', onClick: () => trackTask(image.build_task, 'Build RViz image') }),
        });
      }
      const b = btn('Build RViz image', { kind: 'primary', icon: 'hammer' });
      b.addEventListener('click', () => buildImage(b));
      return emptyState({
        icon: 'box', title: 'The RViz image is not built yet',
        text: `The ${image.name || 'durance-rviz'} Docker image (RViz 2 with the Go2 model) is needed once. Building takes several minutes.`,
        actions: b,
      });
    }
    if (!ctr.running || !rviz.vnc) {
      // Not running, or running from an earlier dashboard session (no VNC password): Start (re)connects it.
      const b = btn('Start RViz', { kind: 'primary', icon: 'play' });
      b.addEventListener('click', () => startRviz(b, source));
      return emptyState({
        icon: 'monitor', title: ctr.running ? 'RViz needs a restart' : 'RViz is not running',
        text: ctr.error ? `Last error: ${ctr.error}` : `Start RViz to see the ${SOURCE_TEXT[source].toLowerCase()} in 3D.`,
        actions: b,
      });
    }
    if (rfbState === 'connecting' || rfbState === 'idle') return emptyState({ kind: 'loading', title: 'Connecting to RViz…' });
    if (rfbState === 'disconnected') {
      const b = btn('Reconnect', { kind: 'primary', icon: 'refresh' });
      b.addEventListener('click', reconnect);
      return emptyState({ icon: 'wifi', title: 'Disconnected from RViz', text: rfbDetail || 'The connection was closed.', actions: b });
    }
    return null;
  }

  let overlaySig = '';
  function render() {
    const ctr = (rviz && rviz.container) || {};
    const running = !!ctr.running;
    // toolbar
    if (running) {
      setBtnLabel(startStopBtn, 'Stop RViz');
      startStopBtn.className = 'btn btn-secondary';
      startStopBtn.querySelector('.icon').replaceWith(icon('square'));
    } else {
      setBtnLabel(startStopBtn, 'Start RViz');
      startStopBtn.className = 'btn btn-primary';
      startStopBtn.querySelector('.icon').replaceWith(icon('play'));
    }
    const usable = rviz && (rviz.docker || {}).ok !== false && (rviz.image || {}).present !== false;
    startStopBtn.hidden = !usable; // the empty state offers the right action instead
    sourceSeg.setDisabled(!usable);
    if (running && ctr.source) {
      source = ctr.source;
      sourceSeg.set(source);
    }
    if (!rviz) setStatus(rvizStatus, rvizError ? 'bad' : 'off', rvizError ? 'Unavailable' : 'Checking…');
    else if ((rviz.docker || {}).ok === false) setStatus(rvizStatus, 'bad', 'Docker unavailable');
    else if ((rviz.image || {}).present === false) setStatus(rvizStatus, 'warn', (rviz.image || {}).build_task ? 'Building image' : 'Image not built');
    else if (!running) setStatus(rvizStatus, 'off', 'Stopped');
    else if (rfbState === 'connected') setStatus(rvizStatus, 'ok', `Connected · ${SOURCE_TEXT[ctr.source] || ctr.source || ''}`);
    else if (rfbState === 'disconnected') setStatus(rvizStatus, 'warn', 'Disconnected');
    else setStatus(rvizStatus, 'busy', 'Connecting');
    // overlay
    const sig = JSON.stringify([rviz && rviz.docker, rviz && rviz.image, ctr.running, ctr.error, rfbState, rfbDetail, rvizError, source]);
    if (sig !== overlaySig) {
      overlaySig = sig;
      const content = overlayContent();
      overlay.hidden = !content;
      overlay.classList.toggle('is-dim', rfbState === 'disconnected' && running);
      overlay.replaceChildren(...(content ? [content] : []));
    }
  }

  // ---------------------------------------------------------------- actions

  async function startRviz(b, src) {
    userDisconnected = false;
    const st = await act(b, () => api.post('/rviz/start', { source: src }, { signal: scope.signal, timeout: 60000 }), { success: `RViz started: ${SOURCE_TEXT[src]}` });
    if (st && scope.alive) applyRviz(st);
  }

  async function chooseSource(v, b) {
    source = v;
    prefs.set('sim.source', v);
    sourceSeg.set(v);
    const ctr = (rviz && rviz.container) || {};
    if (ctr.running && ctr.source === v) return;
    if (!rviz || (rviz.docker || {}).ok === false || (rviz.image || {}).present === false) {
      render();
      return;
    }
    await startRviz(b, v);
  }

  startStopBtn.addEventListener('click', async () => {
    const running = rviz && rviz.container && rviz.container.running;
    if (running) {
      userDisconnected = true;
      disconnectRfb();
      rfbState = 'idle';
      const st = await act(startStopBtn, () => api.post('/rviz/stop', {}, { signal: scope.signal, timeout: 30000 }), { success: 'RViz stopped' });
      if (st && scope.alive) applyRviz(st);
    } else {
      await startRviz(startStopBtn, source);
    }
  });

  function reconnect() {
    userDisconnected = false;
    if (rviz && rviz.vnc) connectRfb(rviz.vnc);
    else statePoll.now();
  }

  async function buildImage(b) {
    const res = await act(b, () => api.post('/rviz/build', {}, { signal: scope.signal }));
    if (!res) return;
    statePoll.now();
    const t = await trackTask(res, 'Build RViz image');
    if (scope.alive && t) statePoll.now();
  }

  function toggleViewOnly() {
    viewOnly = !viewOnly;
    prefs.set('sim.viewOnly', viewOnly);
    if (rfb) rfb.viewOnly = viewOnly;
  }

  // full screen (Fullscreen API on the card)
  fullBtn.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await rvizCard.requestFullscreen();
    } catch {
      /* not allowed (iframe / browser setting) */
    }
  });
  // ---- stage size: the largest rectangle with the VNC screen's aspect ratio that fits
  // (no letterbox bands); the card then ends right under it.
  let aspect = 16 / 9;
  let lastFit = '';
  const wide = window.matchMedia('(min-width: 1024px)');
  function fitStage() {
    const full = document.fullscreenElement === rvizCard;
    const tb = rvizToolbar.offsetHeight;
    const w = full ? window.innerWidth : rvizCard.clientWidth;
    let availH;
    if (full) availH = window.innerHeight - tb;
    else if (wide.matches) availH = layout.clientHeight - tb - 2;
    else availH = window.innerHeight * 0.7;
    let sw = w;
    let sh = w / aspect;
    if (sh > availH) {
      sh = Math.max(220, availH);
      sw = Math.min(w, sh * aspect);
    }
    const key = `${Math.floor(sw)}x${Math.floor(sh)}`;
    if (key === lastFit) return;
    lastFit = key;
    stage.style.width = `${Math.floor(sw)}px`;
    stage.style.height = `${Math.floor(sh)}px`;
  }
  const fitObserver = new ResizeObserver(() => fitStage());
  scope.add(() => fitObserver.disconnect());
  scope.listen(window, 'resize', fitStage);

  scope.listen(document, 'fullscreenchange', () => {
    lastFit = '';
    fitStage();
    const on = document.fullscreenElement === rvizCard;
    fullBtn.replaceChildren(icon(on ? 'minimize' : 'maximize'));
    fullBtn.setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
    fullBtn.dataset.tip = on ? 'Exit full screen' : 'Full screen';
  });
  scope.add(() => {
    if (document.fullscreenElement === rvizCard) document.exitFullscreen().catch(() => {});
  });

  function applyRviz(st) {
    if (!st || typeof st !== 'object') return;
    rviz = st;
    rvizError = null;
    const vw = Number(st.vnc && st.vnc.width);
    const vh = Number(st.vnc && st.vnc.height);
    if (vw > 0 && vh > 0 && Math.abs(vw / vh - aspect) > 1e-3) {
      aspect = vw / vh;
      lastFit = '';
      fitStage();
    }
    syncRfb();
    render();
  }

  const statePoll = scope.poll(async () => {
    try {
      const st = await api.get('/rviz/state', null, { signal: scope.signal, timeout: 8000 });
      if (scope.alive) applyRviz(st);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      rvizError = err.message;
      if (!rviz) render();
    }
  }, 3000);

  // ---------------------------------------------------------------- simulator side panel

  function applySim(st) {
    if (!st || typeof st !== 'object') return;
    sim = st;
    const running = !!st.running;
    setStatus(simStatus, running ? 'ok' : 'off', running ? `Running${st.domain != null ? ` · ROS domain ${st.domain}` : ''}` : 'Stopped');
    setBtnLabel(simBtn, running ? 'Stop' : 'Start');
    simBtn.querySelector('.icon').replaceWith(icon(running ? 'square' : 'play'));
    person.setState(st);
    setText(personText, running ? `Now: ${st.person || 'nothing'}` : 'Start the simulator to use the virtual person.');
    const comps = st.components && typeof st.components === 'object' ? st.components : {};
    const names = Object.keys(comps);
    const okCount = names.filter((n) => componentState(comps[n])[0] === 'ok').length;
    setText(details.metaEl, names.length ? `${okCount}/${names.length} running` : '');
    detailsBody.replaceChildren(
      ...names.map((n) => {
        const [s, text] = componentState(comps[n]);
        return el('div', { class: 'row-between' }, el('span', { class: 'muted', text: COMPONENT_LABELS[n] || n }), status(s, text));
      }),
      kv([['ROS domain', st.domain != null ? String(st.domain) : null], ['Followed by RViz', SOURCE_TEXT[st.view_source] || st.view_source || null]]));
  }

  async function simAction(b, path, body, success) {
    const st = await act(b, () => api.post(path, body, { signal: scope.signal, timeout: 30000 }), { success });
    if (st && scope.alive) applySim(st);
  }

  simBtn.addEventListener('click', () => {
    const running = sim && sim.running;
    simAction(simBtn, running ? '/sim/stop' : '/sim/start', {}, running ? 'Simulator stopped' : 'Simulator started');
  });

  scope.poll(async () => {
    try {
      const st = await api.get('/sim/state', null, { signal: scope.signal, timeout: 6000 });
      if (scope.alive) applySim(st);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      setStatus(simStatus, 'bad', 'Simulator state unavailable');
    }
  }, 3000);

  // live outputs
  let lastLiveAt = 0;
  scope.sub('live', (lv) => {
    lastLiveAt = Date.now();
    rates.update(lv);
    setStatus(liveStatus, lv.connected === false ? 'bad' : 'ok', lv.connected === false ? 'No stream' : `${fmt.int(lv.fps)} fps`);
  });
  scope.interval(() => {
    if (Date.now() - lastLiveAt > 3000) {
      rates.reset();
      setStatus(liveStatus, 'off', 'No live data');
    }
  }, 1000);
  const lv = last('live');
  if (lv) rates.update(lv);

  fitObserver.observe(layout);
  fitObserver.observe(rvizCard);
  requestAnimationFrame(fitStage);
  render();
}
