// Drive (manual control): dead-man joystick (pointer + W/S A/D Q/E keys),
// command tiles in tabs, big STOP. While an input is held the UI sends
// POST /robot/move every 100 ms; on release / blur / hidden tab / pointercancel
// / E-STOP / page leave it sends /robot/stop immediately. The server also stops
// the robot by itself when moves stop arriving for deadman_s.

import { api } from '../api.js';
import { last } from '../events.js';
import {
  el, setText, card, act, btn, icon, status, setStatus, switchControl, prefs, toast, isTypingTarget, anyDialogOpen,
  emptyState, kv, setEnabled, uid, createTabs, collapsible, attachTip, kbd,
} from '../ui.js';
import { LINK_TEXT } from '../system.js';
import * as fmt from '../format.js';

const MOVE_MS = 100;
const DEADZONE = 0.08;
const DEFAULT_LIMITS = { vx: 0.5, vy: 0.3, wz: 1.0 };
const KEYS = { w: 'w', s: 's', a: 'a', d: 'd', q: 'q', e: 'e' };
const CMD_ICON = {
  stop: 'square', standup: 'arrowUp', balance: 'scale', sit: 'arrowDown', risesit: 'arrowUp', standdown: 'arrowDownLine',
  recovery: 'rotateCcw', hello: 'hand', stretch: 'move', content: 'smile', heart: 'heart', dance1: 'music', dance2: 'music',
  scrape: 'sparkles', damp: 'power',
};

export async function mount(root, scope) {
  let state = null;
  let speed = fmt.clamp(Number(prefs.get('manual.speed', 0.3)) || 0.3, 0.05, 1);

  // ---- input state
  const keys = new Set();
  const pad = { active: false, x: 0, y: 0, pointer: null };
  let turnHold = 0;
  let turnPointer = null;
  let timer = null;
  let inflight = false;
  let stopAfterInflight = false;
  let driveError = null;

  // ---------------------------------------------------------------- layout
  const realStrip = el('div', { class: 'real-strip', role: 'alert', hidden: true },
    icon('alert', { size: 20 }), el('strong', { text: 'REAL ROBOT' }),
    el('span', { text: 'Commands on this page move the physical Go2. Keep the area clear and a hand on E-STOP.' }));

  const modeSwitch = switchControl('Manual control', {
    onChange: async (next, b) => {
      const st = await act(b, () => api.post('/robot/manual', { enabled: next }, { risk: 'danger', signal: scope.signal }), {
        success: next ? 'Manual control on: gesture control paused' : 'Manual control off: gestures in control',
      });
      if (st && scope.alive) applyState(st);
    },
  });
  modeSwitch.button.classList.add('switch-lg');
  modeSwitch.el.querySelector('.switch-label').classList.add('mode-title');
  const modeText = el('div', { class: 'mode-text', text: 'Gesture control is paused while manual control is on.' });
  const linkStatus = status('off', 'Link: loading…');
  const modeCard = el('section', { class: 'card mode-card', 'aria-label': 'Manual control mode' },
    el('div', { class: 'mode-main' }, modeSwitch.el, modeText), linkStatus);

  // joystick
  const knob = el('div', { class: 'joy-knob', 'aria-hidden': 'true' });
  const padEl = el('div', {
    class: 'joystick', role: 'application', 'aria-roledescription': 'joystick',
    'aria-label': 'Drive pad: press and hold, then drag. Up is forward, left and right strafe. Keyboard: W and S forward and back, A and D strafe, Q and E turn.',
  }, knob,
  el('span', { class: 'joy-hint joy-hint-top', 'aria-hidden': 'true', text: 'forward' }),
  el('span', { class: 'joy-hint joy-hint-bottom', 'aria-hidden': 'true', text: 'back' }));
  const turnLeft = btn('Turn left', { kind: 'secondary', icon: 'rotateCcw', tip: 'Hold to turn left (Q)', cls: 'turn-btn' });
  const turnRight = btn('Turn right', { kind: 'secondary', icon: 'rotateCw', tip: 'Hold to turn right (E)', cls: 'turn-btn' });
  const kbdHints = el('div', { class: 'kbd-hints', 'aria-hidden': 'true' },
    kbd('W'), kbd('A'), kbd('S'), kbd('D'), el('span', { text: 'move' }), el('span', { text: '\u00b7' }), kbd('Q'), kbd('E'), el('span', { text: 'turn' }));
  const joyCol = el('div', { class: 'joy-col' }, padEl, el('div', { class: 'turn-row' }, turnLeft, turnRight), kbdHints);

  // controls
  const speedId = uid('speed');
  const speedInput = el('input', { type: 'range', id: speedId, min: '0.05', max: '1', step: '0.05', value: String(speed) });
  const speedText = el('output', { class: 'speed-value', for: speedId });
  const speedMax = el('div', { class: 'small muted tabular' });
  const rVx = el('div', { class: 'readout-value', text: '0.00' });
  const rVy = el('div', { class: 'readout-value', text: '0.00' });
  const rWz = el('div', { class: 'readout-value', text: '0.00' });
  const cell = (label, v, unit) => el('div', { class: 'readout-cell' }, el('div', { class: 'readout-label', text: label }), v, el('div', { class: 'readout-unit', text: unit }));
  const driveStatus = el('div', { class: 'drive-status', role: 'status' });
  const stopBtn = btn('STOP', { kind: 'danger-solid', icon: 'square', cls: 'stop-btn' });
  const controls = el('div', { class: 'drive-controls' },
    el('div', { class: 'field' },
      el('div', { class: 'field-label-row' }, el('label', { class: 'field-label', for: speedId, text: 'Speed limit' })),
      el('div', { class: 'speed-row' }, speedInput, speedText), speedMax),
    el('div', { class: 'readout', 'aria-label': 'Command being sent' }, cell('Forward', rVx, 'vx \u00b7 m/s'), cell('Sideways', rVy, 'vy \u00b7 m/s'), cell('Turn', rWz, 'wz \u00b7 rad/s')),
    driveStatus,
    stopBtn);
  const driveCard = card({ title: 'Drive', sub: 'Hold to move, release to stop' }, el('div', { class: 'drive-card-body' }, joyCol, controls));

  // commands
  const cmdHost = el('div', null, emptyState({ kind: 'loading', title: 'Loading commands…' }));
  const commandsCard = card({ title: 'Commands' }, cmdHost);
  const commandButtons = [];
  const stateBody = el('div');
  const stateCard = el('section', { class: 'card' }, collapsible({ title: 'Robot state', key: 'drive.state' }, stateBody));

  root.append(realStrip, modeCard, el('div', { class: 'drive-grid' }, driveCard, el('div', { class: 'stack' }, commandsCard, stateCard)));

  // ---------------------------------------------------------------- state

  const enabled = () => !!(state && state.manual);
  const limits = () => {
    const l = (state && state.limits) || {};
    return {
      vx: fmt.isNum(l.vx) ? l.vx : DEFAULT_LIMITS.vx,
      vy: fmt.isNum(l.vy) ? l.vy : DEFAULT_LIMITS.vy,
      wz: fmt.isNum(l.wz) ? l.wz : DEFAULT_LIMITS.wz,
    };
  };

  function renderSpeed() {
    const l = limits();
    setText(speedText, fmt.pct(speed));
    setText(speedMax, `Max ${fmt.fixed(l.vx * speed, 2)} m/s forward · ${fmt.fixed(l.wz * speed, 2)} rad/s turn`);
  }

  function applyState(st) {
    if (!st || typeof st !== 'object') return;
    state = { ...(state || {}), ...st };
    const on = enabled();
    const link = state.link || 'none';
    setStatus(linkStatus, link === 'go2' ? 'real' : (link === 'none' ? 'off' : 'ok'), `Link · ${LINK_TEXT[link] || link}`);
    realStrip.hidden = link !== 'go2';
    modeSwitch.set(on);
    setText(modeText, on ? 'On: gesture control is paused while you drive.' : 'Off: gestures control the robot. Turn on to drive by hand.');
    padEl.classList.toggle('is-disabled', !on);
    padEl.setAttribute('aria-disabled', String(!on));
    setEnabled(turnLeft, on);
    setEnabled(turnRight, on);
    for (const b of commandButtons) setEnabled(b, on);
    if (!on) {
      releaseAll(false);
      setText(driveStatus, 'Turn on manual control to drive.');
    } else if (!driveError && !timer) {
      setText(driveStatus, 'Ready: hold the pad or a key to move.');
    }
    stateBody.replaceChildren(kv([
      ['Link', LINK_TEXT[link] || link],
      ['Teleop', fmt.describe(state.teleop)],
      ['Last command', fmt.describe(state.last_cmd)],
      ['Dead-man timeout', fmt.isNum(state.deadman_s) ? fmt.secs(state.deadman_s, 1) : null],
      ['Limits', `vx ${fmt.fixed(limits().vx, 2)} m/s · vy ${fmt.fixed(limits().vy, 2)} m/s · wz ${fmt.fixed(limits().wz, 2)} rad/s`],
    ]));
    renderSpeed();
  }

  // ---------------------------------------------------------------- driving

  function currentCmd() {
    let fx = 0;
    let fy = 0;
    let fw = 0;
    if (pad.active) {
      fx += -pad.y; // screen up = forward (+vx)
      fy += -pad.x; // screen right = strafe right (-vy, ROS y points left)
    }
    if (keys.has('w')) fx += 1;
    if (keys.has('s')) fx -= 1;
    if (keys.has('a')) fy += 1;
    if (keys.has('d')) fy -= 1;
    if (keys.has('q')) fw += 1;
    if (keys.has('e')) fw -= 1;
    fw += turnHold;
    const l = limits();
    return {
      vx: fmt.roundTo(fmt.clamp(fx, -1, 1) * l.vx * speed, 3),
      vy: fmt.roundTo(fmt.clamp(fy, -1, 1) * l.vy * speed, 3),
      wz: fmt.roundTo(fmt.clamp(fw, -1, 1) * l.wz * speed, 3),
    };
  }

  const anyHeld = () => pad.active || keys.size > 0 || turnHold !== 0;

  function showCmd(c) {
    setText(rVx, fmt.fixed(c.vx, 2));
    setText(rVy, fmt.fixed(c.vy, 2));
    setText(rWz, fmt.fixed(c.wz, 2));
  }

  function inputChanged() {
    if (!enabled()) {
      releaseAll(false);
      return;
    }
    if (anyHeld()) {
      showCmd(currentCmd());
      if (!timer) {
        driveError = null;
        driveStatus.classList.remove('is-error');
        setText(driveStatus, 'Driving: release to stop.');
        sendMove();
        timer = setInterval(sendMove, MOVE_MS);
      }
    } else {
      stopDriving();
    }
  }

  async function sendMove() {
    if (!anyHeld() || inflight) return; // never queue moves; the server dead-man covers gaps
    const cmd = currentCmd();
    showCmd(cmd);
    inflight = true;
    try {
      await api.post('/robot/move', cmd, { noConfirm: true, timeout: 2000 });
    } catch (err) {
      if (!err.aborted && scope.alive) failDrive(err);
    } finally {
      inflight = false;
      if (stopAfterInflight) {
        stopAfterInflight = false;
        sendStop(); // a move may have overtaken the stop: stop once more
      }
    }
  }

  async function sendStop() {
    try {
      await api.post('/robot/stop', {}, { noConfirm: true, timeout: 3000 });
    } catch (err) {
      if (err.unauthorized) return;
      try {
        await api.post('/robot/stop', {}, { noConfirm: true, timeout: 3000 });
      } catch (err2) {
        toast(`STOP request failed (${err2.message}). The robot stops by itself when moves stop arriving; use E-STOP if needed.`, 'error');
      }
    }
  }

  function stopDriving() {
    const wasDriving = timer !== null;
    clearInterval(timer);
    timer = null;
    showCmd({ vx: 0, vy: 0, wz: 0 });
    if (wasDriving || inflight) {
      sendStop();
      if (inflight) stopAfterInflight = true;
      if (!driveError && enabled()) setText(driveStatus, 'Stopped. Hold the pad or a key to move.');
    }
  }

  function releaseAll(sendIfDriving = true) {
    keys.clear();
    pad.active = false;
    pad.pointer = null;
    pad.x = 0;
    pad.y = 0;
    knob.style.transform = '';
    padEl.classList.remove('is-active');
    turnHold = 0;
    turnPointer = null;
    turnLeft.classList.remove('is-held');
    turnRight.classList.remove('is-held');
    if (sendIfDriving || timer) stopDriving();
  }

  function failDrive(err) {
    if (driveError) return;
    driveError = err;
    releaseAll(true);
    const msg = err.needsConfirm ? err.message : `Move rejected: ${err.message}`;
    setText(driveStatus, msg);
    driveStatus.classList.add('is-error');
    toast(msg, 'error');
    scope.timeout(() => {
      driveError = null;
      driveStatus.classList.remove('is-error');
    }, 3000);
  }

  // ---- joystick pad (pointer events: mouse, touch, pen)
  function updatePad(ev) {
    const r = padEl.getBoundingClientRect();
    const radius = Math.max(10, Math.min(r.width, r.height) / 2 - 32);
    let dx = (ev.clientX - (r.left + r.width / 2)) / radius;
    let dy = (ev.clientY - (r.top + r.height / 2)) / radius;
    const len = Math.hypot(dx, dy);
    if (len > 1) {
      dx /= len;
      dy /= len;
    }
    const dead = Math.hypot(dx, dy) < DEADZONE;
    pad.x = dead ? 0 : dx;
    pad.y = dead ? 0 : dy;
    knob.style.transform = `translate(${(dx * radius).toFixed(1)}px, ${(dy * radius).toFixed(1)}px)`;
  }

  padEl.addEventListener('pointerdown', (ev) => {
    if (!enabled() || pad.active) return;
    ev.preventDefault();
    try {
      padEl.setPointerCapture(ev.pointerId);
    } catch {
      /* capture is best-effort */
    }
    pad.active = true;
    pad.pointer = ev.pointerId;
    padEl.classList.add('is-active');
    updatePad(ev);
    inputChanged();
  });
  padEl.addEventListener('pointermove', (ev) => {
    if (!pad.active || ev.pointerId !== pad.pointer) return;
    updatePad(ev);
    showCmd(currentCmd());
  });
  const padEnd = (ev) => {
    if (!pad.active || ev.pointerId !== pad.pointer) return;
    pad.active = false;
    pad.pointer = null;
    pad.x = 0;
    pad.y = 0;
    knob.style.transform = '';
    padEl.classList.remove('is-active');
    inputChanged();
  };
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) padEl.addEventListener(type, padEnd);

  // ---- hold-to-turn buttons
  function holdButton(b, dir) {
    b.addEventListener('pointerdown', (ev) => {
      if (!enabled() || b.disabled) return;
      ev.preventDefault();
      try {
        b.setPointerCapture(ev.pointerId);
      } catch {
        /* best-effort */
      }
      turnHold = dir;
      turnPointer = ev.pointerId;
      b.classList.add('is-held');
      inputChanged();
    });
    const end = (ev) => {
      if (turnPointer !== ev.pointerId) return;
      turnHold = 0;
      turnPointer = null;
      b.classList.remove('is-held');
      inputChanged();
    };
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(type, end);
    b.addEventListener('contextmenu', (ev) => ev.preventDefault());
  }
  holdButton(turnLeft, 1);
  holdButton(turnRight, -1);

  // ---- keyboard
  scope.listen(document, 'keydown', (ev) => {
    const k = KEYS[(ev.key || '').toLowerCase()];
    if (!k || ev.ctrlKey || ev.metaKey || ev.altKey || isTypingTarget(ev.target) || anyDialogOpen()) return;
    ev.preventDefault();
    if (!enabled()) {
      if (!ev.repeat) setText(driveStatus, 'Turn on manual control to drive.');
      return;
    }
    if (keys.has(k)) return;
    keys.add(k);
    inputChanged();
  });
  scope.listen(document, 'keyup', (ev) => {
    const k = KEYS[(ev.key || '').toLowerCase()];
    if (!k || !keys.has(k)) return;
    keys.delete(k);
    inputChanged();
  });

  // ---- safety releases
  scope.listen(window, 'blur', () => releaseAll(true));
  scope.listen(window, 'pagehide', () => releaseAll(true));
  scope.listen(document, 'visibilitychange', () => {
    if (document.hidden) releaseAll(true);
  });
  scope.sub('estop', () => releaseAll(true));
  scope.add(() => releaseAll(true)); // leaving the page stops driving

  // ---- speed
  speedInput.addEventListener('input', () => {
    speed = fmt.clamp(Number(speedInput.value) || 0.3, 0.05, 1);
    prefs.set('manual.speed', speed);
    renderSpeed();
    if (anyHeld()) showCmd(currentCmd());
  });

  // ---- STOP (always enabled)
  stopBtn.addEventListener('click', async () => {
    releaseAll(false);
    clearInterval(timer);
    timer = null;
    const ok = await act(stopBtn, () => api.post('/robot/stop', {}, { noConfirm: true, timeout: 3000 }));
    if (ok) setText(driveStatus, 'STOP sent.');
  });

  // ---------------------------------------------------------------- commands

  function cmdButton(c) {
    const b = el('button', { type: 'button', class: ['cmd-btn', c.confirm && 'is-risky'], disabled: !enabled() },
      icon(CMD_ICON[c.name] || 'zap', { size: 22 }), el('span', { text: c.label || c.name }));
    const tip = [c.confirm ? 'Asks for confirmation.' : '', c.note ? String(c.note) : ''].filter(Boolean).join(' ');
    if (tip) attachTip(b, tip);
    b.addEventListener('click', async () => {
      const res = await act(b, () => api.post('/robot/command', { name: c.name }, { risk: c.confirm ? 'danger' : 'warn' }));
      if (res) toast(`Sent: ${c.label || c.name}`, 'success');
    });
    commandButtons.push(b);
    return b;
  }

  async function loadCommands() {
    try {
      const res = await api.get('/robot/commands', null, { signal: scope.signal });
      if (!scope.alive) return;
      const cmds = Array.isArray(res.commands) ? res.commands : [];
      if (!cmds.length) {
        cmdHost.replaceChildren(emptyState({ title: 'No robot commands available' }));
        return;
      }
      const groups = new Map();
      for (const c of cmds) {
        const g = c.group || 'Other';
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g).push(c);
      }
      const names = Array.from(groups.keys());
      const tabs = createTabs({
        key: 'tab.drive.commands',
        label: 'Command groups',
        value: names.includes('Posture') ? 'Posture' : names[0],
        items: names.map((g) => ({ value: g, label: g })),
      });
      commandButtons.length = 0;
      for (const [g, list] of groups) {
        tabs.panel(g).append(el('div', { class: 'cmd-grid' }, ...list.map(cmdButton)));
        tabs.panel(g).style.paddingTop = '16px';
      }
      cmdHost.replaceChildren(tabs.el, ...names.map((g) => tabs.panel(g)));
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      cmdHost.replaceChildren(emptyState({
        kind: 'error', title: 'Could not load commands', text: err.message,
        actions: btn('Retry', { kind: 'secondary', icon: 'refresh', onClick: () => loadCommands() }),
      }));
    }
  }

  // ---------------------------------------------------------------- data

  scope.sub('robot', applyState);
  const loadState = async () => {
    try {
      const st = await api.get('/robot/state', null, { signal: scope.signal, timeout: 6000 });
      if (scope.alive) applyState(st);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      if (!state) setStatus(linkStatus, 'bad', `Robot state unavailable: ${err.message}`);
    }
  };
  const cached = last('robot');
  if (cached) {
    applyState(cached);
  } else {
    // Controls start disabled until the real state arrives.
    applyState({ manual: false, link: 'none' });
    state = null;
    setStatus(linkStatus, 'off', 'Link: loading…');
  }
  showCmd({ vx: 0, vy: 0, wz: 0 });
  scope.poll(loadState, 5000);
  await loadCommands();
  if (state) applyState({});
}
