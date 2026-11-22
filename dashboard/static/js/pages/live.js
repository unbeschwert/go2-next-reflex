// Live (operation): camera + detected gesture side by side, then tabs
// [Activity] [Detection settings]. Header: bridge status, sensitivity preset,
// "Pause gestures" switch. Settings changes are staged (sticky footer bar).

import { api } from '../api.js';
import { last } from '../events.js';
import {
  el, setText, act, icon, status, setStatus, switchControl, select, setOptions, toolbar, createTabs, attachTip,
} from '../ui.js';
import * as fmt from '../format.js';
import { createCameraView } from '../components/cameraView.js';
import { createRateBars } from '../components/rateBars.js';
import { createLogView } from '../components/logView.js';
import { createSettingsForm, same } from '../components/settingsForm.js';
import { createVirtualPerson } from '../components/virtualPerson.js';

const LIVE_STALE_MS = 3000;

// Client-side sensitivity presets (applied through the staged settings + Apply).
const PRESETS = {
  balanced: {},
  responsive: { window_s: 0.4, stable_s: 0.1, min_rate: 0.45, margin: 0.25, rearm_max_s: 1.0, cooldown_s: 0.3 },
  strict: { window_s: 0.6, stable_s: 0.35, min_rate: 0.6, margin: 0.35, beat_none: 0.3 },
};
const PRESET_LABELS = { balanced: 'Balanced', responsive: 'Responsive', strict: 'Strict' };
const PRESET_SKIP_GROUPS = new Set(['Robot', 'Safety']);

const GESTURE_ICON = [
  [/heart/i, 'heart'], [/left/i, 'rotateCcw'], [/right/i, 'rotateCw'], [/follow/i, 'footprints'],
  [/hello|wave/i, 'hand'], [/dance/i, 'music'], [/lay|down/i, 'arrowDownLine'], [/sit/i, 'arrowDown'],
];

function gestureIcon(label) {
  if (!label) return 'hand';
  for (const [re, name] of GESTURE_ICON) if (re.test(label)) return name;
  return 'sparkles';
}

function labelAt(labels, idx) {
  if (Number.isInteger(idx) && labels && labels[idx] != null) return String(labels[idx]);
  if (typeof idx === 'string' && idx) return idx;
  return null;
}

/** The preset's full value set over the sensitivity keys (defaults + overrides, clamped). */
function presetValues(name, rows, defaults) {
  const over = PRESETS[name] || {};
  const out = {};
  for (const r of rows) {
    if (r.type === 'bool' || PRESET_SKIP_GROUPS.has(r.group)) continue;
    let v = defaults[r.key];
    if (r.key.startsWith('min_rate_o') && 'min_rate' in over) v = over.min_rate;
    else if (r.key in over) v = over[r.key];
    if (!fmt.isNum(v)) continue;
    if (fmt.isNum(r.min)) v = Math.max(r.min, v);
    if (fmt.isNum(r.max)) v = Math.min(r.max, v);
    out[r.key] = v;
  }
  return out;
}

export async function mount(root, scope, app) {
  let labels = null;
  let paused = null;
  let lastLiveAt = 0;
  let lastCmd = null; // {label, t}
  let flashTimer = null;

  // ---------------------------------------------------------------- toolbar
  const bridgeStatus = status('off', 'No live data');
  const presetSel = select([
    { value: 'balanced', label: 'Balanced' }, { value: 'responsive', label: 'Responsive' }, { value: 'strict', label: 'Strict' },
    { value: 'custom', label: 'Custom', disabled: true, hidden: true },
  ], { value: 'balanced', ariaLabel: 'Sensitivity preset' });
  presetSel.disabled = true;
  attachTip(presetSel, 'Sensitivity preset: stages detection settings; press Apply to use them');
  const presetField = el('label', { class: 'row' }, el('span', { class: 'small muted', text: 'Sensitivity' }), presetSel);
  const pauseSwitch = switchControl('Pause gestures', {
    tip: 'Paused: the bridge keeps watching but sends no robot commands',
    onChange: async (next, b) => {
      const data = await act(b, () => settings.save({ paused: next }), { success: next ? 'Gestures paused' : 'Gestures resumed' });
      if (data && data.values && typeof data.values.paused === 'boolean') setPaused(data.values.paused);
    },
  });
  root.append(toolbar(bridgeStatus, [presetField, pauseSwitch.el]));

  // ---------------------------------------------------------------- camera card
  const camera = createCameraView(scope, { src: '/api/live/camera' });
  const camInfo = el('span', { class: 'tabular' });
  const person = createVirtualPerson();
  const personWrap = el('div', { class: 'row' }, el('span', { class: 'small muted', text: 'Virtual person' }), person.el);
  personWrap.hidden = true;
  const cameraCard = el('section', { class: 'card camera-card', 'aria-label': 'Event camera' },
    camera.el, el('div', { class: 'camera-footer' }, camInfo, personWrap));

  // ---------------------------------------------------------------- detected card
  const gIcon = el('div', { class: 'gesture-icon', 'aria-hidden': 'true' }, icon('hand', { size: 28 }));
  let gIconName = 'hand';
  const gName = el('div', { class: 'gesture-name', 'aria-live': 'polite', 'aria-atomic': 'true', text: 'No live data' });
  const gState = el('div', { class: 'gesture-state', text: 'Waiting for the live stream' });
  const rates = createRateBars();
  const lastCmdText = el('strong', { text: 'none yet' });
  const lastCmdAgo = el('span', { class: 'tabular' });
  const detected = el('section', { class: 'card detected-card', 'aria-label': 'Detected gesture' },
    el('div', { class: 'detected-label', text: 'Detected' }),
    el('div', { class: 'detected-main' }, gIcon, el('div', { class: 'stack-sm' }, gName, gState)),
    rates.el,
    el('div', { class: 'last-cmd' }, icon('send', { size: 16 }), el('span', { text: 'Last command' }), el('span', { 'aria-live': 'polite' }, lastCmdText), el('span', { class: 'spacer' }), lastCmdAgo));

  root.append(el('div', { class: 'live-grid' }, cameraCard, detected));

  // ---------------------------------------------------------------- tabs: activity / settings
  const settings = createSettingsForm(scope, {
    exclude: ['paused'],
    onData: (data) => {
      if (Array.isArray(data.classes) && data.classes.length && !labels) labels = data.classes.map(String);
      if (data.values && typeof data.values.paused === 'boolean' && Date.now() - lastLiveAt > LIVE_STALE_MS) setPaused(data.values.paused);
      presetSel.disabled = false;
      updatePreset();
    },
    onLocalChange: () => updatePreset(),
  });
  const log = createLogView(scope);
  const tabs = createTabs({
    key: 'tab.live',
    label: 'Live sections',
    items: [{ value: 'activity', label: 'Activity', icon: 'list' }, { value: 'settings', label: 'Detection settings', icon: 'sliders' }],
    onChange: (v) => showTools(v),
  });
  const toolsHost = el('div', { class: 'tabbar-actions' });
  tabs.panel('activity').append(log.el);
  tabs.panel('settings').append(settings.el);
  const tabsCard = el('section', { class: 'card live-tabs-card' },
    el('div', { class: 'tabbar' }, tabs.el, toolsHost),
    tabs.panel('activity'), tabs.panel('settings'));
  function showTools(v) {
    toolsHost.replaceChildren(v === 'activity' ? log.tools : settings.tools);
  }
  showTools(tabs.get());
  root.append(tabsCard, settings.footer);

  // ---------------------------------------------------------------- presets
  function updatePreset() {
    if (!settings.loaded) return;
    const rows = settings.schemaRows();
    const defs = settings.defaults();
    const locals = settings.locals();
    let match = 'custom';
    for (const name of Object.keys(PRESETS)) {
      const vals = presetValues(name, rows, defs);
      if (Object.keys(vals).length && Object.entries(vals).every(([k, v]) => same(locals[k], v))) {
        match = name;
        break;
      }
    }
    setOptions(presetSel, [
      ...Object.keys(PRESETS).map((p) => ({ value: p, label: PRESET_LABELS[p] })),
      { value: 'custom', label: 'Custom', disabled: true, hidden: match !== 'custom' },
    ], match);
  }
  presetSel.addEventListener('change', () => {
    const name = presetSel.value;
    if (!PRESETS[name]) return;
    settings.stage(presetValues(name, settings.schemaRows(), settings.defaults()));
    tabs.select('settings');
  });

  // ---------------------------------------------------------------- pause
  function setPaused(v) {
    if (typeof v !== 'boolean') return;
    paused = v;
    pauseSwitch.set(v);
  }

  // ---------------------------------------------------------------- live updates
  function setGesture(iconName, name, stateText, active) {
    if (iconName !== gIconName) {
      gIconName = iconName;
      gIcon.replaceChildren(icon(iconName, { size: 28 }));
    }
    setText(gName, name);
    setText(gState, stateText);
    detected.classList.toggle('is-active', !!active);
  }

  function render(lv) {
    const stale = !lv || Date.now() - lastLiveAt > LIVE_STALE_MS;
    const lbls = (lv && Array.isArray(lv.labels) && lv.labels.length) ? lv.labels : labels;
    if (stale) {
      setStatus(bridgeStatus, 'off', 'No live data');
      setGesture('hand', 'No live data', 'Waiting for the live stream', false);
      setText(camInfo, 'No live data');
      return;
    }
    const fps = fmt.isNum(lv.fps) ? `${fmt.int(lv.fps)} fps` : '';
    setText(camInfo, [lv.connected ? fps : 'Not connected', fmt.isNum(lv.window_frames) ? `window ${lv.window_frames} frames` : ''].filter(Boolean).join(' · '));
    if (lv.connected === false) {
      setStatus(bridgeStatus, 'bad', 'No board output');
      setGesture('ban', 'No board', 'The board output stream is not connected', false);
      return;
    }
    const th = lv.thresholds || {};
    switch (lv.status) {
      case 'candidate': {
        const name = labelAt(lbls, lv.candidate) || 'Unknown gesture';
        setStatus(bridgeStatus, 'info', `Candidate · ${fps}`);
        setGesture(gestureIcon(name), name, `Holding ${fmt.secs(lv.candidate_s, 2)}${fmt.isNum(th.stable_s) ? ` of ${fmt.secs(th.stable_s, 2)}` : ''}`, true);
        break;
      }
      case 'busy': {
        const name = labelAt(lbls, lv.last_trick) || (lastCmd && lastCmd.label) || 'Trick';
        setStatus(bridgeStatus, 'busy', `Busy · ${fps}`);
        setGesture(gestureIcon(name), name, `Busy: ${/turn/i.test(name) ? 'turning' : 'performing'} (${fmt.secs(lv.busy_s)})`, true);
        break;
      }
      case 'waiting-for-calm':
        setStatus(bridgeStatus, 'warn', `Waiting to re-arm · ${fps}`);
        setGesture('clock', lastCmd ? lastCmd.label : 'No gesture', 'Waiting to re-arm: lower your hands', false);
        break;
      case 'paused':
        setStatus(bridgeStatus, 'warn', `Paused · ${fps}`);
        setGesture('pause', 'Paused', 'Gesture control is paused: no robot commands', false);
        break;
      case 'no-board':
        setStatus(bridgeStatus, 'bad', 'No board');
        setGesture('ban', 'No board', 'The board output stream is not connected', false);
        break;
      default:
        setStatus(bridgeStatus, 'ok', `Listening · ${fps}`);
        setGesture('hand', 'No gesture', 'Listening: show a gesture to the camera', false);
    }
  }

  function renderLastCmd() {
    if (!lastCmd) return;
    setText(lastCmdText, lastCmd.label);
    setText(lastCmdAgo, lastCmd.t ? fmt.ago(lastCmd.t) : '');
  }

  scope.sub('live', (lv) => {
    lastLiveAt = Date.now();
    rates.update(lv);
    render(lv);
    if (typeof lv.paused === 'boolean' && lv.paused !== paused) setPaused(lv.paused);
  });

  scope.sub('log', (entry) => {
    if (entry.kind !== 'command') return;
    lastCmd = { label: entry.line == null ? 'command' : String(entry.line), t: entry.t };
    renderLastCmd();
    detected.classList.remove('flash');
    void detected.offsetWidth; // restart the animation
    detected.classList.add('flash');
    if (flashTimer) scope.clearTimeout(flashTimer);
    flashTimer = scope.timeout(() => detected.classList.remove('flash'), 1300);
  });

  scope.interval(() => {
    if (Date.now() - lastLiveAt > LIVE_STALE_MS) {
      render(null);
      rates.reset();
    }
    renderLastCmd();
  }, 1000);

  // ---------------------------------------------------------------- overview: target / virtual person
  let personPoll = null;
  const loadSimState = async () => {
    try {
      const st = await api.get('/sim/state', null, { signal: scope.signal, timeout: 6000 });
      if (scope.alive) person.setState(st);
    } catch {
      if (scope.alive) person.setState(null);
    }
  };
  function applyOverview(ov) {
    if (!ov) return;
    const isSim = ov.target && ov.target.mode === 'sim';
    personWrap.hidden = !isSim;
    if (isSim && !personPoll) personPoll = scope.poll(loadSimState, 3000);
    else if (!isSim && personPoll) {
      personPoll.stop();
      personPoll = null;
    }
    if (!lastCmd && ov.live && ov.live.last_command && ov.live.last_command.label) {
      lastCmd = { label: String(ov.live.last_command.label), t: ov.live.last_command.t };
      renderLastCmd();
    }
    if (paused === null && ov.safety && typeof ov.safety.paused === 'boolean') setPaused(ov.safety.paused);
  }
  scope.sub('overview', applyOverview);
  applyOverview(app.overview());

  const lv = last('live');
  if (lv) rates.update(lv);
  render(null);
}
