// Board: tabs [Status] [Models] [Auto mode] [Logs] [Software].

import { api } from '../api.js';
import {
  el, setText, card, act, btn, iconBtn, status, setStatus, segmented, switchControl, select, field, input,
  collapsible, createTabs, dataTable, strSort, emptyState, kv, tile, menuButton, chip, setEnabled, setLogText, icon,
  setFieldError, toast, attachTip,
} from '../ui.js';
import { trackTask } from '../tasks.js';
import { MODE_TEXT, shortModel, modelLabel } from '../system.js';
import * as fmt from '../format.js';

const STATUS_MS = 5000;
const LOG_MS = 2000;
const LOG_NAMES = ['bridge', 'obstacle', 'camera', 'server'];
const MODE_ACTION = { stopped: 'stop', test: 'start', 'robot-dry': 'robot-dry', robot: 'robot' };
const HELPERS = {
  bridge_settings_py: ['bridge_settings.py', 'Gesture-control settings and the E-STOP pause'],
  go2_cmd_py: ['go2_cmd.py', 'Robot command helper'],
  robot_link: ['robot_link.sh', 'Network link to the Go2 (robot mode)'],
  laptop_link: ['laptop_link.sh', 'Network link to this laptop (test mode)'],
};

function metric(model, key) {
  const v = model.metrics ? model.metrics[key] : null;
  return fmt.isNum(v) ? v : -Infinity;
}

function problemsOf(m) {
  return Array.isArray(m.problems) ? m.problems : [];
}

function modelSub(m) {
  if (m.kind === 'setup') {
    const f = m.formats || {};
    return ['run', f.weights ? `weights ${f.weights}` : null, m.firmware ? m.firmware.label : 'no firmware yet',
      m.metrics && m.metrics.kind ? `${m.metrics.kind.replace('_', '-')} accuracy` : null].filter(Boolean).join(' · ');
  }
  return [m.source, m.md5 ? String(m.md5).slice(0, 8) : null].filter(Boolean).join(' · ');
}

export async function mount(root, scope, app) {
  let st = null; // /board/status
  let autostart = null;
  let autoFormFilled = false;

  const tabs = createTabs({
    key: 'tab.board',
    label: 'Board sections',
    items: [
      { value: 'status', label: 'Status', icon: 'gauge' }, { value: 'models', label: 'Models', icon: 'sparkles' },
      { value: 'auto', label: 'Auto mode', icon: 'hardDrive' }, { value: 'logs', label: 'Logs', icon: 'file' },
      { value: 'software', label: 'Software', icon: 'wrench' },
    ],
  });
  const boardStatus = status('off', 'Checking…');
  root.append(el('div', { class: 'tabbar' }, tabs.el, el('div', { class: 'tabbar-actions' }, boardStatus)));
  for (const v of ['status', 'models', 'auto', 'logs', 'software']) {
    tabs.panel(v).classList.add('stack');
    root.append(tabs.panel(v));
  }

  // ================================================================ Status tab
  const warnHost = el('div', { class: 'stack-sm' });
  const warnCard = card({ title: 'Warnings', icon: 'alert' }, warnHost);
  warnCard.hidden = true;

  let modeChoice = null;
  const modeSeg = segmented([
    { value: 'stopped', label: 'Stopped', icon: 'square' },
    { value: 'test', label: 'Test', icon: 'monitor', tip: 'Camera + FPGA model + bridge; commands go to the RViz animator' },
    { value: 'robot-dry', label: 'Robot dry-run', icon: 'eye', tip: 'Decides like robot mode but never commands the dog' },
    { value: 'robot', label: 'Robot', icon: 'zap', tip: 'The REAL Go2 moves on gestures (LiDAR obstacle check on). Asks for confirmation.' },
  ], { label: 'Inference mode', cls: 'mode-seg', size: 'md', onChange: (v) => { modeChoice = v; modeSeg.set(v); renderMode(); } });
  const applyMode = btn('Apply', { kind: 'primary' });
  const modeNow = el('p', { class: 'small muted' });
  const modeMore = menuButton([
    { label: 'Test mode without viewer (headless)', icon: 'monitor', onSelect: (b) => inference('start-headless', b, 'Start test mode (headless)') },
  ], { label: 'More mode actions' });
  const modeCard = card({ title: 'Mode', actions: [modeMore] },
    el('div', { class: 'stack' }, modeSeg.el, el('div', { class: 'form-actions' }, applyMode, modeNow)));

  const tTemp = tile({ label: 'Temperature', icon: 'gauge' });
  const tMem = tile({ label: 'Memory free', icon: 'server' });
  const tDisk = tile({ label: 'Disk free', icon: 'hardDrive' });
  const tUp = tile({ label: 'Uptime', icon: 'clock' });
  const healthTiles = el('div', { class: 'tiles tiles-sm' }, tTemp.el, tMem.el, tDisk.el, tUp.el);

  const detailsBody = el('div', { class: 'stack' });
  const detailsCard = el('section', { class: 'card' }, collapsible({ title: 'Details', key: 'board.details', meta: 'processes, model, configuration' }, detailsBody));

  // lock
  const lockLine = status('off', 'Checking…');
  const purpose = input({ maxLength: 80, placeholder: 'e.g. demo, 80x80 test' });
  const purposeField = field('Purpose', purpose, { help: 'Shown to other operators while you hold the board.' });
  const acquireBtn = btn('Acquire', { kind: 'primary', icon: 'lock' });
  const releaseBtn = btn('Release', { kind: 'secondary', icon: 'unlock' });
  const takeoverBtn = btn('Take over', { kind: 'warn', icon: 'lock' });
  const lockCard = card({ title: 'Board lock' }, el('div', { class: 'stack' }, lockLine, purposeField, el('div', { class: 'form-actions' }, acquireBtn, releaseBtn, takeoverBtn)));

  tabs.panel('status').append(warnCard,
    el('div', { class: 'board-grid' }, el('div', { class: 'stack' }, modeCard, healthTiles, detailsCard), lockCard));

  function renderMode() {
    const cur = st ? st.mode : null;
    const choice = modeChoice ?? cur;
    modeSeg.set(choice);
    setText(modeNow, cur ? `Now: ${MODE_TEXT[cur] || cur}` : 'Mode unknown');
    setEnabled(applyMode, !!(st && st.reachable && st.ssh !== false && choice && MODE_ACTION[choice] && choice !== cur));
    modeSeg.setDisabled(!(st && st.reachable && st.ssh !== false));
    applyMode.className = `btn ${choice === 'robot' ? 'btn-danger' : 'btn-primary'}`;
  }

  async function inference(action, b, label) {
    const res = await act(b, () => api.post('/board/inference', { action }, { risk: action === 'robot' ? 'danger' : 'warn', signal: scope.signal }));
    if (!res) return;
    await trackTask(res, label);
    if (scope.alive) {
      modeChoice = null;
      statusPoll.now();
      app.refreshOverview();
    }
  }
  applyMode.addEventListener('click', () => {
    const choice = modeChoice ?? (st && st.mode);
    if (!MODE_ACTION[choice]) return;
    inference(MODE_ACTION[choice], applyMode, choice === 'stopped' ? 'Stop inference' : `Start ${MODE_TEXT[choice].toLowerCase()}`);
  });

  async function lockAction(action, b) {
    const res = await act(b, () => api.post('/board/lock', { action, purpose: purpose.value.trim() }, { risk: 'warn', signal: scope.signal }), {
      success: action === 'release' ? 'Lock released' : 'Lock acquired',
    });
    if (!res || !scope.alive) return;
    if (st) {
      st = { ...st, lock: res.lock };
      renderLock();
    }
    statusPoll.now();
    app.refreshOverview();
  }
  acquireBtn.addEventListener('click', () => lockAction('acquire', acquireBtn));
  releaseBtn.addEventListener('click', () => lockAction('release', releaseBtn));
  takeoverBtn.addEventListener('click', () => lockAction('takeover', takeoverBtn));

  function renderLock() {
    const lock = st ? st.lock : null;
    const held = !!(lock && lock.owner);
    const mine = held && lock.mine;
    const extra = held ? `${lock.purpose ? ` (${lock.purpose})` : ''}${lock.since ? ` since ${lock.since}` : ''}` : '';
    if (!held) setStatus(lockLine, 'off', 'Nobody holds the board lock');
    else if (mine) setStatus(lockLine, 'ok', `You hold the lock${extra}`);
    else setStatus(lockLine, 'warn', `Locked by ${lock.owner}${extra}`);
    acquireBtn.hidden = held;
    releaseBtn.hidden = !mine;
    takeoverBtn.hidden = !held || mine;
  }

  function renderStatus() {
    const b = st;
    if (!b.reachable) setStatus(boardStatus, 'bad', 'Board offline');
    else if (b.ssh === false) setStatus(boardStatus, 'warn', 'No SSH');
    else setStatus(boardStatus, 'ok', `${MODE_TEXT[b.mode] || 'Unknown'}${fmt.isNum(b.ping_ms) ? ` · ${fmt.ms(b.ping_ms)}` : ''}`);
    renderMode();
    renderLock();
    const warnings = Array.isArray(b.warnings) ? b.warnings : [];
    warnCard.hidden = !warnings.length && !b.error;
    warnHost.replaceChildren(...[b.error, ...warnings].filter(Boolean).map((w) => el('div', { class: 'row' }, icon('alert', { size: 16, cls: 'lvl-icon' }), el('span', { text: String(w) }))));
    warnHost.classList.add('lvl-warn');

    const h = b.health || {};
    tTemp.set(fmt.isNum(h.temp_c) ? `${fmt.fixed(h.temp_c, 1)} °C` : '–', fmt.isNum(h.temp_c) ? (h.temp_c > 80 ? 'Hot' : 'Normal') : 'No sensor', fmt.isNum(h.temp_c) ? (h.temp_c > 80 ? 'warn' : 'ok') : 'off');
    tMem.set(fmt.isNum(h.mem_free_pct) ? `${fmt.fixed(h.mem_free_pct, 0)} %` : '–', Array.isArray(h.load) && fmt.isNum(h.load[0]) ? `load ${fmt.fixed(h.load[0], 2)}` : '', fmt.isNum(h.mem_free_pct) ? (h.mem_free_pct < 10 ? 'warn' : 'ok') : 'off');
    tDisk.set(fmt.isNum(h.disk_free_mb) ? fmt.bytes(h.disk_free_mb * 1e6) : '–', fmt.isNum(h.disk_free_mb) ? (h.disk_free_mb < 100 ? 'Almost full' : 'OK') : '', fmt.isNum(h.disk_free_mb) ? (h.disk_free_mb < 100 ? 'warn' : 'ok') : 'off');
    tUp.set(fmt.isNum(h.uptime_s) ? fmt.duration(h.uptime_s) : '–', h.time ? `clock ${String(h.time).slice(11, 16)}` : '', fmt.isNum(h.uptime_s) ? 'ok' : 'off');

    const procs = b.processes && typeof b.processes === 'object' ? b.processes : {};
    const model = b.model || {};
    detailsBody.replaceChildren(
      el('div', { class: 'row', style: { gap: '8px 20px' } }, ...(Object.keys(procs).length
        ? Object.entries(procs).map(([n, up]) => status(up ? 'ok' : 'off', `${n}${up ? '' : ' (off)'}`))
        : [el('span', { class: 'muted', text: 'No processes reported' })])),
      kv([
        ['Model', (modelLabel(model) || {}).full || null],
        ['File', model.name ? el('span', { class: 'mono', text: String(model.name) }) : undefined],
        ['Model MD5', model.md5 ? el('code', { text: String(model.md5_full || model.md5) }) : null],
        ['Weights MD5', b.weights_md5 ? el('code', { text: String(b.weights_md5) }) : undefined],
        ['Gestures', model.classes || null],
        ['Firmware', b.firmware && b.firmware.md5 ? `${b.firmware.label || 'unknown'}${b.firmware.weights ? ` · weights ${b.firmware.weights}, potentials ${b.firmware.potentials}` : ''} (md5 ${b.firmware.md5})` : undefined],
        ['Input size', fmt.isNum(b.input_size) ? `${b.input_size}×${b.input_size}` : (fmt.isNum(model.input) ? `${model.input}×${model.input}` : null)],
        ['Scaling', b.scaling != null ? fmt.describe(b.scaling) : undefined],
        ['ROS domain', b.ros_domain != null ? String(b.ros_domain) : undefined],
        ['Board time', b.time != null ? String(b.time) : undefined],
      ]));
    renderSoftware();
  }

  async function loadStatus() {
    try {
      const res = await api.get('/board/status', null, { signal: scope.signal, timeout: 10000 });
      if (!scope.alive) return;
      st = res || {};
      renderStatus();
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      if (!st) {
        setStatus(boardStatus, 'bad', 'Status unavailable');
        detailsBody.replaceChildren(emptyState({ kind: 'error', title: 'Could not load the board status', text: err.message }));
      }
    }
  }
  const statusPoll = scope.poll(loadStatus, STATUS_MS);

  // ================================================================ Models tab
  const restoreMenu = menuButton([
    { label: 'Restore the previous model (weights, or the whole setup)', icon: 'reset', onSelect: (b) => restore(b) },
  ], { label: 'More model actions' });
  const modelsTable = dataTable({
    caption: 'Models available for deployment',
    empty: 'No models on this laptop yet. Fetch runs from the Training page.',
    sortKey: 'test',
    sortDir: 'desc',
    rowClass: (m) => (m.deployed ? 'is-selected' : null),
    columns: [
      {
        key: 'name', label: 'Model', sort: strSort('name'),
        render: (m) => el('div', null,
          el('div', { class: 'row', style: { flexWrap: 'nowrap' } }, el('span', { class: 'cell-main', text: m.name || m.id }), m.deployed ? chip('Deployed', { kind: 'accent', icon: 'check' }) : null),
          el('div', { class: 'cell-sub', text: modelSub(m) }),
          problemsOf(m).length ? el('div', { class: 'cell-sub', style: { color: 'var(--warn, #b26b00)' }, text: problemsOf(m).join(' · ') }) : null),
      },
      { key: 'arch', label: 'Network', sort: strSort('arch') },
      { key: 'input', label: 'Input', numeric: true, render: (m) => (fmt.isNum(m.input) ? `${m.input}×${m.input}` : fmt.DASH) },
      { key: 'classes', label: 'Gestures' },
      { key: 'test', label: 'Test', numeric: true, sort: (a, b) => metric(a, 'test') - metric(b, 'test'), render: (m) => fmt.accuracy(m.metrics && m.metrics.test) },
      { key: 'synops', label: 'SynOps', numeric: true, sort: (a, b) => metric(a, 'synops') - metric(b, 'synops'), render: (m) => fmt.si(m.metrics && m.metrics.synops) },
      {
        key: 'deploy', label: '', cls: 'actions',
        render: (m) => {
          if (m.deployed) return el('span', { class: 'small muted', text: 'On the board' });
          if (m.kind === 'setup') { // a run of our pipeline: installs firmware + configs + weights
            if (!m.firmware) {
              const fb = btn('Build firmware', { kind: 'ghost', size: 'sm', icon: 'cpu', tip: 'riversdk.dev job via scripts/firmware_job.py (6-8 min; an identical job is reused)' });
              fb.addEventListener('click', () => buildFirmware(m, fb));
              return fb;
            }
            const b = btn('Install', { kind: 'ghost', size: 'sm', icon: 'upload', tip: `Firmware ${m.firmware.label}, configs, weights and classes` });
            b.disabled = problemsOf(m).length > 0;
            b.addEventListener('click', () => deploy(m, m.classes, b));
            return b;
          }
          if (m.classes === 'own' || m.classes === 'dataset') {
            const b = btn('Deploy', { kind: 'ghost', size: 'sm', icon: 'upload' });
            b.addEventListener('click', () => deploy(m, m.classes, b));
            return b;
          }
          return menuButton([
            { label: 'Deploy with our gestures', icon: 'upload', onSelect: (b) => deploy(m, 'own', b) },
            { label: 'Deploy with dataset gestures', icon: 'upload', onSelect: (b) => deploy(m, 'dataset', b) },
          ], { label: 'Deploy', text: 'Deploy', icon: 'upload' });
        },
      },
    ],
  });
  tabs.panel('models').append(el('section', { class: 'card card-flush' },
    el('div', { class: 'card-head' }, el('div', { class: 'card-heading' }, el('h2', { class: 'card-title', text: 'Models' }), el('span', { class: 'card-sub', text: 'Runs (runs/) install their firmware, configs and weights; weights files replace only weights.txt. Both back up first and restart inference in test mode.' })), restoreMenu),
    modelsTable.el));

  async function loadModels() {
    try {
      const res = await api.get('/board/models', null, { signal: scope.signal, timeout: 15000 });
      if (scope.alive) modelsTable.setRows(Array.isArray(res.models) ? res.models : []);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      modelsTable.setEmpty(`Could not load models: ${err.message}`);
      modelsTable.setRows([]);
    }
  }
  async function deploy(model, classes, b) {
    const res = await act(b, () => api.post('/board/deploy', { model_id: model.id, classes }, { risk: 'danger', signal: scope.signal }));
    if (!res) return;
    await trackTask(res, `Deploy ${shortModel(model.name) || model.id}`);
    if (scope.alive) {
      loadModels();
      statusPoll.now();
      app.refreshOverview();
    }
  }
  async function buildFirmware(model, b) {
    const res = await act(b, () => api.post('/board/firmware', { model_id: model.id }, { risk: 'warn', signal: scope.signal }));
    if (!res) return;
    await trackTask(res, `Firmware for ${model.name}`);
    if (scope.alive) loadModels();
  }
  async function restore(b) {
    const res = await act(b, () => api.post('/board/restore', {}, { risk: 'warn', signal: scope.signal }));
    if (!res) return;
    await trackTask(res, 'Restore the previous model');
    if (scope.alive) {
      loadModels();
      statusPoll.now();
    }
  }
  tabs.onFirstShow('models', loadModels);

  // ================================================================ Auto mode tab
  const autoState = el('div', { class: 'row', style: { gap: '8px 20px' } }, status('off', 'Checking…'));
  const autoSwitch = switchControl('Start automatically at power-on', {
    onChange: (next, b) => autoAction(next ? 'enable' : 'disable', b),
  });
  const modeSel = select([
    { value: 'robot', label: 'Robot: real Go2 with LiDAR obstacle check' },
    { value: 'robot-dry', label: 'Robot dry-run: decide, never command' },
    { value: 'test', label: 'Test: RViz animator' },
  ], { value: 'robot' });
  const classesSel = select([{ value: 'own', label: 'Our gestures' }, { value: 'dataset', label: 'Dataset gestures' }], { value: 'own' });
  const delayInput = input({ type: 'number', min: '0', max: '600', step: '1', value: '10' });
  const delayField = field('Start delay (s)', delayInput, { help: 'Wait after power-on before starting (0 to 600 s).' });
  const installBtn = btn('Install', { kind: 'primary', icon: 'download' });
  const autoMore = menuButton([
    { label: 'Uninstall auto mode', icon: 'trash', danger: true, onSelect: (b) => autoAction('uninstall', b) },
  ], { label: 'More auto-mode actions' });
  const autoError = el('div', { class: 'form-error', role: 'alert', hidden: true });
  tabs.panel('auto').append(card({ title: 'Auto mode (SD card)', actions: [autoMore] },
    el('div', { class: 'stack' },
      el('p', { class: 'prose', text: 'With auto mode installed and enabled, the board starts the event camera, the FPGA model and the gesture bridge by itself at power-on, without this laptop. Use it for stand-alone demos: power the board, wait for the start delay, and the dog reacts to gestures. Disable it to give the board back to laptop control.' }),
      autoState,
      autoSwitch.el,
      el('div', { class: 'form-grid' }, field('Mode at boot', modeSel), field('Gestures', classesSel), delayField),
      autoError,
      el('div', { class: 'form-actions' }, installBtn))));

  function confFromForm() {
    const d = Number(delayInput.value);
    if (!Number.isFinite(d) || d < 0 || d > 600) {
      setFieldError(delayField, 'Between 0 and 600 s.');
      return null;
    }
    setFieldError(delayField, '');
    return { mode: modeSel.value, classes: classesSel.value, start_delay_s: d };
  }

  async function autoAction(action, b) {
    const body = { action };
    if (action === 'install' || action === 'enable') {
      const conf = confFromForm();
      if (!conf) return;
      body.conf = conf;
    }
    autoError.hidden = true;
    const risk = body.conf && body.conf.mode === 'robot' ? 'danger' : 'warn';
    const res = await act(b, () => api.post('/board/autostart', body, { risk, signal: scope.signal }), {
      onError: (err) => {
        if (err.status !== 422) return false;
        autoError.replaceChildren(icon('alert', { size: 18 }), el('span', { text: err.message }));
        autoError.hidden = false;
        return true;
      },
    });
    if (!res) {
      if (autostart) autoSwitch.set(!!autostart.enabled);
      return;
    }
    await trackTask(res, `Auto mode: ${action}`);
    if (scope.alive) {
      autoFormFilled = false;
      loadAutostart();
      app.refreshOverview();
    }
  }
  installBtn.addEventListener('click', () => autoAction('install', installBtn));

  function renderAutostart() {
    const a = autostart;
    autoState.replaceChildren(
      status(a.installed ? 'ok' : 'off', a.installed ? 'Installed' : 'Not installed'),
      status(a.enabled ? 'info' : 'off', a.enabled ? 'Enabled at power-on' : 'Disabled'),
      status(a.active ? 'ok' : 'off', a.active ? 'Running now' : 'Not running'));
    autoSwitch.set(!!a.enabled);
    autoSwitch.setDisabled(!a.installed);
    const c = a.conf || {};
    if (!autoFormFilled && a.conf) {
      autoFormFilled = true;
      if (c.mode) modeSel.value = String(c.mode);
      if (c.classes) classesSel.value = String(c.classes);
      if (fmt.isNum(c.start_delay_s)) delayInput.value = String(c.start_delay_s);
    }
    installBtn.querySelector('.btn-label').textContent = a.installed ? 'Save and reinstall' : 'Install';
  }

  async function loadAutostart() {
    try {
      const res = await api.get('/board/autostart', null, { signal: scope.signal, timeout: 10000 });
      if (!scope.alive) return;
      autostart = res || {};
      renderAutostart();
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      if (!autostart) autoState.replaceChildren(status('bad', `Could not read the auto-mode state: ${err.message}`));
    }
  }
  tabs.onFirstShow('auto', () => scope.poll(loadAutostart, 15000));

  // ================================================================ Logs tab
  let logName = 'bridge';
  let logLines = 200;
  const logPre = el('pre', { class: 'logbox', tabindex: '0', 'aria-label': 'Board log', text: 'Loading…' });
  logPre.style.maxHeight = 'calc(100dvh - 300px)';
  logPre.style.minHeight = '360px';
  const logSeg = segmented(LOG_NAMES.map((n) => ({ value: n, label: n })), {
    label: 'Log', value: logName,
    onChange: (v) => {
      logName = v;
      logSeg.set(v);
      logPre.textContent = 'Loading…';
      loadLog();
    },
  });
  const linesSel = select([100, 200, 500, 1000].map((n) => ({ value: n, label: `${n} lines` })), {
    value: logLines, ariaLabel: 'Number of log lines', cls: 'input-sm',
    onChange: (v) => { logLines = Number(v) || 200; loadLog(); },
  });
  const follow = switchControl('Follow', { checked: true, tip: 'Refresh every 2 s and stay at the newest line' });
  const logInfo = el('span', { class: 'small faint tabular' });
  const logRefresh = iconBtn('refresh', 'Refresh now');
  logRefresh.addEventListener('click', () => act(logRefresh, loadLog));
  tabs.panel('logs').append(el('section', { class: 'card' },
    el('div', { class: 'card-head' }, logSeg.el, el('div', { class: 'card-actions' }, logInfo, linesSel, follow.el, logRefresh)),
    logPre));

  async function loadLog() {
    const name = logName;
    try {
      const res = await api.get('/board/logs', { name, lines: logLines }, { signal: scope.signal, timeout: 10000 });
      if (!scope.alive || name !== logName) return;
      const lines = Array.isArray(res.lines) ? res.lines : [];
      setLogText(logPre, lines.length ? lines : ['(empty)'], follow.get());
      setText(logInfo, fmt.timeOfDay(Date.now()));
    } catch (err) {
      if (!scope.alive || err.aborted || name !== logName) return;
      setText(logInfo, `refresh failed: ${err.message}`);
      if (logPre.textContent === 'Loading…') logPre.textContent = `Could not load the ${name} log: ${err.message}`;
    }
  }
  tabs.onFirstShow('logs', () => {
    loadLog();
    scope.interval(() => {
      if (tabs.get() === 'logs' && follow.get() && !document.hidden) loadLog();
    }, LOG_MS);
  });

  // ================================================================ Software tab
  const helpersHost = el('div');
  const swBtn = btn('Install board software', { kind: 'primary', icon: 'download' });
  attachTip(swBtn, 'Copies the current gesture bridge and helpers from this laptop to the board (old files are backed up)');
  tabs.panel('software').append(card({ title: 'Board software', sub: 'Helper files the gesture bridge needs on the board' },
    el('div', { class: 'stack' },
      el('p', { class: 'prose', text: 'Installs the current gesture bridge, bridge_settings.py, lidar_follow.py and robot_link.sh from this laptop on the board. The old files are backed up; restart inference afterwards.' }),
      helpersHost,
      el('div', { class: 'form-actions' }, swBtn))));

  function renderSoftware() {
    const h = st && st.helpers && typeof st.helpers === 'object' ? st.helpers : null;
    if (!h) {
      helpersHost.replaceChildren(el('p', { class: 'muted', text: st && st.reachable ? 'The board did not report its helper files.' : 'The board is not reachable.' }));
      return;
    }
    const keys = Array.from(new Set([...Object.keys(HELPERS), ...Object.keys(h)]));
    helpersHost.replaceChildren(...keys.map((k) => {
      const [file, what] = HELPERS[k] || [k, ''];
      return el('div', { class: 'helper-row' },
        el('div', null, el('div', { class: 'mono', text: file }), what ? el('div', { class: 'cell-sub', text: what }) : null),
        status(h[k] ? 'ok' : 'warn', h[k] ? 'Installed' : 'Missing'));
    }));
  }

  swBtn.addEventListener('click', async () => {
    const res = await act(swBtn, () => api.post('/board/software', {}, { risk: 'warn', signal: scope.signal }));
    if (!res) return;
    const t = await trackTask(res, 'Install board software');
    if (scope.alive) {
      statusPoll.now();
      if (t && t.state === 'done') toast('Board software installed. Restart inference to use it.', 'info');
    }
  });
}
