// Record: tabs [Recording] [Sessions]. Recording is a 3-step flow:
// 1 Set up (session, person, minutes, gestures; Advanced collapsed)
// 2 Record (cue panel + live preview, one control bar; Space / R / N keys)
// 3 Cut clips (session select, Cut, inline task log)

import { api } from '../api.js';
import { last } from '../events.js';
import {
  el, setText, card, act, btn, icon, status, setStatus, field, setFieldError, switchControl, select, setOptions,
  collapsible, createTabs, dataTable, strSort, numSort, emptyState, prefs, shortcutAllowed, setEnabled,
  attachTip, setLogText, input, setBtnLabel,
} from '../ui.js';
import { trackTask, onTaskUpdate } from '../tasks.js';
import * as fmt from '../format.js';

const PHASES = ['idle', 'connecting', 'waiting', 'ready', 'go', 'paused', 'done', 'error'];
const PHASE_TEXT = {
  idle: 'Idle', connecting: 'Connecting', waiting: 'Waiting for Go', ready: 'Get ready', go: 'Recording', paused: 'Paused', done: 'Done', error: 'Error',
};
const PREVIEW_MS = 200;
const SSE_STALE_MS = 2500;

function styleNames(v) {
  if (Array.isArray(v)) return v.map((s) => (typeof s === 'string' ? s : (s && (s.name || s.id)) || '')).filter(Boolean);
  if (typeof v === 'string') return v.split(',').map((s) => s.trim()).filter(Boolean);
  return [];
}

export async function mount(root, scope) {
  let options = null;
  let rec = null;
  let sessions = [];
  let lastSseAt = 0;
  let stepNow = 1;

  // ---------------------------------------------------------------- page tabs
  const recStatus = status('off', 'Recorder idle');
  const tabs = createTabs({
    key: 'tab.record',
    label: 'Record sections',
    items: [{ value: 'recording', label: 'Recording', icon: 'record' }, { value: 'sessions', label: 'Sessions', icon: 'list' }],
  });
  root.append(el('div', { class: 'tabbar' }, tabs.el, el('div', { class: 'tabbar-actions' }, recStatus)));
  root.append(tabs.panel('recording'), tabs.panel('sessions'));
  tabs.panel('recording').classList.add('stack');
  tabs.panel('sessions').classList.add('stack');

  // ---------------------------------------------------------------- stepper
  const stepBtns = [
    ['Set up', 1], ['Record', 2], ['Cut clips', 3],
  ].map(([label, n]) => {
    const b = el('button', { type: 'button', class: 'step' }, el('span', { class: 'step-num', text: String(n) }), el('span', { text: label }));
    b.addEventListener('click', () => setStep(n));
    return b;
  });
  const stepper = el('nav', { class: 'stepper', 'aria-label': 'Recording steps' },
    stepBtns[0], el('span', { class: 'step-line', 'aria-hidden': 'true' }), stepBtns[1], el('span', { class: 'step-line', 'aria-hidden': 'true' }), stepBtns[2]);
  const step1 = el('div', { class: 'stack' });
  const step2 = el('div', { class: 'stack' });
  const step3 = el('div', { class: 'stack' });
  tabs.panel('recording').append(stepper, step1, step2, step3);

  function setStep(n) {
    stepNow = n;
    [step1, step2, step3].forEach((s, i) => {
      s.hidden = i !== n - 1;
    });
    stepBtns.forEach((b, i) => {
      if (i === n - 1) b.setAttribute('aria-current', 'step');
      else b.removeAttribute('aria-current');
    });
  }

  // ---------------------------------------------------------------- step 1: set up
  const sessionInput = input({ type: 'number', min: '0', step: '1' });
  const personInput = input({ maxLength: 40, value: prefs.get('record.person', ''), autocomplete: 'off' });
  const minutesInput = input({ type: 'number', min: '0.5', max: '60', step: '0.5' });
  const sessionField = field('Session', sessionInput);
  const personField = field('Person', personInput, { help: 'Who performs the gestures (stored with the takes).' });
  const minutesField = field('Minutes per gesture', minutesInput);
  const gestureChips = el('div', { class: 'toggle-chips', role: 'group', 'aria-label': 'Gestures to record' });
  const styleChips = el('div', { class: 'toggle-chips', role: 'group', 'aria-label': 'Styles' });
  const shuffleSw = switchControl('Shuffle the order', { checked: true });
  const noActionSw = switchControl('Also record "no action" segments', { checked: true });
  const formError = el('div', { class: 'form-error', role: 'alert', hidden: true });
  const timingText = el('p', { class: 'small muted' });
  const startBtn = btn('Start recording', { kind: 'primary', icon: 'record' });
  const setupBody = el('div', { class: 'stack' }, emptyState({ kind: 'loading', title: 'Loading recording options…' }));
  step1.append(card({ title: 'Session' }, setupBody));

  function toggleChip(label, { pressed = true, tip, iconName } = {}) {
    const b = el('button', { type: 'button', class: 'toggle-chip', 'aria-pressed': String(pressed) },
      iconName ? icon(iconName, { size: 16 }) : null, el('span', { text: label }));
    if (tip) attachTip(b, tip);
    b.addEventListener('click', () => b.setAttribute('aria-pressed', String(b.getAttribute('aria-pressed') !== 'true')));
    return b;
  }

  function buildForm(opts) {
    options = opts;
    const defaults = opts.defaults || {};
    sessionInput.value = opts.next_session != null ? String(opts.next_session) : '';
    minutesInput.value = fmt.isNum(defaults.minutes) ? String(defaults.minutes) : '1';
    const classes = Array.isArray(opts.classes) ? opts.classes : [];
    gestureChips.replaceChildren(...(classes.length ? classes.map((c) => {
      const b = toggleChip(String(c.name), { tip: c.how ? `How: ${c.how}` : undefined, iconName: 'check' });
      b.dataset.id = JSON.stringify(c.id);
      return b;
    }) : [el('span', { class: 'muted', text: 'No gesture classes defined.' })]));
    const known = new Set(styleNames(defaults.styles));
    for (const c of classes) for (const s of styleNames(c.styles)) known.add(s);
    const defaultStyles = new Set(styleNames(defaults.styles));
    styleChips.replaceChildren(...Array.from(known).map((s) => {
      const b = toggleChip(s, { pressed: defaultStyles.size ? defaultStyles.has(s) : true });
      b.dataset.style = s;
      return b;
    }));
    if (typeof defaults.no_action === 'boolean') noActionSw.set(defaults.no_action);
    const timing = [];
    if (fmt.isNum(defaults.ready_s)) timing.push(`${defaults.ready_s} s countdown before each take`);
    if (fmt.isNum(defaults.switch_s)) timing.push(`${defaults.switch_s} s between gestures`);
    setText(timingText, `${timing.length ? `${timing.join(' · ')}. ` : ''}Starting stops board inference (the camera streams to one program only).`);
    setupBody.replaceChildren(
      el('div', { class: 'form-grid' }, sessionField, personField, minutesField),
      el('div', { class: 'field' }, el('div', { class: 'field-label', text: 'Gestures' }), gestureChips),
      collapsible({ title: 'Advanced', key: 'record.advanced' },
        el('div', { class: 'stack' },
          el('div', { class: 'field' }, el('div', { class: 'field-label', text: 'Styles' }), known.size ? styleChips : el('span', { class: 'muted small', text: 'Server default styles' })),
          shuffleSw.el, noActionSw.el)),
      formError,
      el('div', { class: 'form-actions' }, startBtn, timingText));
    renderControls();
  }

  async function loadOptions() {
    try {
      const opts = await api.get('/record/options', null, { signal: scope.signal });
      if (scope.alive) buildForm(opts || {});
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      setupBody.replaceChildren(emptyState({
        kind: 'error', title: 'Could not load the recording options', text: err.message,
        actions: btn('Retry', { kind: 'secondary', icon: 'refresh', onClick: () => loadOptions() }),
      }));
    }
  }

  function collectStart() {
    for (const f of [sessionField, personField, minutesField]) setFieldError(f, '');
    formError.hidden = true;
    let ok = true;
    const rawSession = sessionInput.value.trim();
    if (!rawSession) {
      setFieldError(sessionField, 'Enter a session number.');
      ok = false;
    }
    const person = personInput.value.trim();
    if (!person) {
      setFieldError(personField, 'Enter who is recording.');
      ok = false;
    }
    const minutes = Number(minutesInput.value);
    if (!Number.isFinite(minutes) || minutes <= 0) {
      setFieldError(minutesField, 'Enter the minutes per gesture.');
      ok = false;
    }
    const classes = Array.from(gestureChips.querySelectorAll('.toggle-chip[aria-pressed="true"]')).map((b) => JSON.parse(b.dataset.id));
    if (!classes.length) {
      formError.replaceChildren(icon('alert', { size: 18 }), el('span', { text: 'Select at least one gesture.' }));
      formError.hidden = false;
      ok = false;
    }
    if (!ok) return null;
    const styles = Array.from(styleChips.querySelectorAll('.toggle-chip[aria-pressed="true"]')).map((b) => b.dataset.style);
    const session = typeof (options && options.next_session) === 'number' ? Number(rawSession) : rawSession;
    return { session, person, classes, minutes, styles: styles.join(','), no_action: noActionSw.get(), shuffle: shuffleSw.get() };
  }

  startBtn.addEventListener('click', async () => {
    const body = collectStart();
    if (!body) return;
    prefs.set('record.person', body.person);
    const st = await act(startBtn, () => api.post('/record/start', body, { risk: 'warn', signal: scope.signal }), {
      onError: (err) => {
        if (err.status !== 422) return false;
        formError.replaceChildren(icon('alert', { size: 18 }), el('span', { text: err.message }));
        formError.hidden = false;
        return true;
      },
    });
    if (st && scope.alive) {
      applyRecord(st);
      setStep(2);
    }
  });

  // ---------------------------------------------------------------- step 2: record
  const cuePhase = el('span', { class: 'cue-phase', text: 'Idle' });
  const cueProgress = el('span', { class: 'tabular' });
  const cueCam = status('off', 'Camera idle');
  const cueRate = el('span', { class: 'tabular' });
  const cueBig = el('div', { class: 'cue-big', text: 'Not recording' });
  const cueSub = el('div', { class: 'cue-sub' });
  const cueHow = el('div', { class: 'cue-how' });
  const cueAnnounce = el('div', { class: 'sr-only', 'aria-live': 'assertive', 'aria-atomic': 'true' });
  const progressBar = el('div', { class: 'progress-fill' });
  const progress = el('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0', 'aria-label': 'Session progress' }, progressBar);
  const cue = el('section', { class: 'cue phase-idle', 'aria-label': 'Recording cue' },
    el('div', { class: 'cue-top' }, cuePhase, cueProgress, el('span', { class: 'spacer' }), cueCam, cueRate),
    cueBig, cueSub, cueHow, progress, cueAnnounce);
  const previewImg = el('img', { class: 'preview-img', alt: 'Live preview of the recorded events' });
  const previewMsg = el('div', { class: 'preview-msg', text: 'The live preview appears while recording.' });
  const previewBox = el('div', { class: 'preview' }, previewImg, previewMsg);

  const mainBtn = btn('Go', { kind: 'primary', icon: 'play', size: 'lg' });
  const redoBtn = btn('Redo last', { kind: 'secondary', icon: 'reset', size: 'lg', tip: 'Record the last take again (R)' });
  const skipBtn = btn('Skip', { kind: 'secondary', icon: 'skipForward', size: 'lg', tip: 'Skip this take (N)' });
  const stopBtn = btn('Stop session', { kind: 'danger', icon: 'square', size: 'lg' });
  attachTip(mainBtn, 'Space');
  const controls = el('div', { class: 'rec-controls', role: 'group', 'aria-label': 'Take controls' }, mainBtn, redoBtn, skipBtn, el('span', { class: 'spacer' }), stopBtn);

  const takesList = el('div');
  const takes = collapsible({ title: 'Saved takes', key: 'record.takes' }, takesList);
  const planList = el('div');
  const plan = collapsible({ title: 'Plan', key: 'record.plan' }, planList);
  step2.append(el('div', { class: 'rec-stage' }, cue, previewBox), controls, el('section', { class: 'card' }, takes, plan));

  // ---------------------------------------------------------------- step 3: cut clips
  const cutSession = select([], { ariaLabel: 'Session to cut' });
  const recutSw = switchControl('Re-cut sessions that already have clips');
  const cutBtn = btn('Cut clips', { kind: 'primary', icon: 'scissors' });
  const cutStatus = status('off', 'Not started');
  const cutLog = el('pre', { class: 'logbox', tabindex: '0', 'aria-label': 'Cut task log', text: 'The task log appears here.' });
  step3.append(card({ title: 'Cut into clips', sub: 'Cuts the labelled takes of a session into 1 s clips for the dataset' },
    el('div', { class: 'stack' },
      el('div', { class: 'form-actions' }, field('Session', cutSession), el('div', { class: 'field' }, el('div', { class: 'field-label', text: ' ' }), recutSw.el)),
      el('div', { class: 'form-actions' }, cutBtn, cutStatus),
      cutLog)));

  async function cut(session, b) {
    if (session == null || session === '') return;
    const res = await act(b, () => api.post('/record/cut', { session, best: 0, recut: recutSw.get() }, { signal: scope.signal }));
    if (!res || !scope.alive) return;
    setStatus(cutStatus, 'busy', `Cutting session ${session}…`);
    const unsub = onTaskUpdate(res.task_id, (t) => {
      if (!scope.alive) return;
      setLogText(cutLog, Array.isArray(t.log) && t.log.length ? t.log : ['(no output yet)'], true);
    });
    scope.add(unsub);
    const t = await trackTask(res, `Cut session ${session} into clips`, { open: false });
    unsub();
    if (!scope.alive) return;
    if (t && t.state === 'done') setStatus(cutStatus, 'ok', `Session ${session} cut${t.result ? `: ${fmt.describe(t.result, 80)}` : ''}`);
    else setStatus(cutStatus, 'bad', `Cutting failed (${t ? t.state : 'unknown'})`);
    loadSessions();
  }
  cutBtn.addEventListener('click', () => cut(cutSession.value, cutBtn));

  // ---------------------------------------------------------------- sessions tab
  const sessionsTable = dataTable({
    caption: 'Recorded sessions',
    empty: 'No sessions recorded yet.',
    sortKey: 'session',
    columns: [
      { key: 'session', label: 'Session', sort: strSort('session') },
      { key: 'person', label: 'Person', sort: strSort('person') },
      { key: 'raw_takes', label: 'Takes', numeric: true, sort: numSort('raw_takes') },
      { key: 'labeled_takes', label: 'Labelled', numeric: true, sort: numSort('labeled_takes') },
      {
        key: 'clips', label: 'Clips per gesture',
        render: (r) => (r.clips && typeof r.clips === 'object' && Object.keys(r.clips).length
          ? el('span', { class: 'muted', text: Object.entries(r.clips).map(([k, v]) => `${k} ${v}`).join(' · ') })
          : el('span', { class: 'faint', text: 'not cut yet' })),
      },
      { key: 'total_clips', label: 'Clips', numeric: true, sort: numSort('total_clips') },
      {
        key: 'cut', label: '', cls: 'actions',
        render: (r) => btn('Cut', {
          kind: 'ghost', size: 'sm', icon: 'scissors',
          onClick: () => {
            tabs.select('recording');
            setStep(3);
            cutSession.value = String(r.session);
          },
        }),
      },
    ],
  });
  const sessionsCard = el('section', { class: 'card card-flush' }, sessionsTable.el);
  tabs.panel('sessions').append(sessionsCard);

  async function loadSessions() {
    try {
      const res = await api.get('/record/sessions', null, { signal: scope.signal });
      if (!scope.alive) return;
      sessions = Array.isArray(res.sessions) ? res.sessions : [];
      sessionsTable.setRows(sessions);
      const prev = cutSession.value || (rec && rec.session) || '';
      setOptions(cutSession, sessions.map((s) => ({ value: String(s.session), label: `Session ${s.session}${s.person ? ` · ${s.person}` : ''}` })), prev);
      setEnabled(cutBtn, sessions.length > 0);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      sessionsTable.setEmpty(`Could not load sessions: ${err.message}`);
    }
  }

  // ---------------------------------------------------------------- record state rendering

  function gestureName(r) {
    return r.gesture && r.gesture.name ? String(r.gesture.name) : '';
  }

  function renderCue(r) {
    const phase = PHASES.includes(r.phase) ? r.phase : 'idle';
    cue.className = `cue phase-${phase}`;
    setText(cuePhase, PHASE_TEXT[phase]);
    const name = gestureName(r).toUpperCase();
    let big;
    let sub = '';
    switch (phase) {
      case 'connecting': big = 'Connecting…'; sub = 'Opening the event camera.'; break;
      case 'waiting': big = name ? `Next: ${name}` : 'Ready'; sub = 'Press Go (Space) when the person is in position.'; break;
      case 'ready': big = `${name} in ${fmt.isNum(r.seconds_left) ? Math.ceil(r.seconds_left) : ''}`.trim(); break;
      case 'go': big = `GO ${name} ${fmt.clock(r.seconds_left)}`.trim(); break;
      case 'paused': big = 'Paused'; sub = name ? `Next: ${name}. Press Resume (Space).` : 'Press Resume (Space).'; break;
      case 'done': big = 'Done'; sub = `${Array.isArray(r.saved) ? r.saved.length : 0} takes saved. Cut them into clips in step 3.`; break;
      case 'error': big = 'Error'; sub = r.error ? String(r.error) : 'The recorder stopped with an error.'; break;
      default: big = 'Not recording'; sub = 'Set up a session in step 1 and press Start recording.';
    }
    const active = ['waiting', 'ready', 'go', 'paused'].includes(phase);
    if (active && (r.style || r.hint)) sub = [r.style ? `Style: ${r.style}` : null, r.hint, sub].filter(Boolean).join(' · ');
    setText(cueBig, big);
    setText(cueSub, sub);
    setText(cueHow, active && r.gesture && r.gesture.how ? `How: ${r.gesture.how}` : '');
    const announce = `${phase}|${name}`;
    if (cueAnnounce.dataset.last !== announce) {
      cueAnnounce.dataset.last = announce;
      setText(cueAnnounce, `${PHASE_TEXT[phase]}${name ? `: ${gestureName(r)}` : ''}`);
    }
    const steps = fmt.isNum(r.steps) ? r.steps : 0;
    const stepN = fmt.isNum(r.step) ? r.step : 0;
    setText(cueProgress, steps ? `Take ${stepN} of ${steps}` : '');
    const frac = steps ? fmt.clamp(stepN / steps, 0, 1) : 0;
    progressBar.style.width = `${(frac * 100).toFixed(1)}%`;
    progress.setAttribute('aria-valuenow', String(Math.round(frac * 100)));
    setText(cueRate, r.running && fmt.isNum(r.events_per_s) ? `${fmt.si(r.events_per_s)} ev/s` : '');
    if (r.running) setStatus(cueCam, r.connected ? 'ok' : 'bad', r.connected ? 'Camera' : 'No camera');
    else setStatus(cueCam, 'off', 'Camera idle');
    const st = phase === 'error' ? 'bad' : (r.running ? (phase === 'go' ? 'busy' : 'info') : 'off');
    setStatus(recStatus, st, r.running ? `${PHASE_TEXT[phase]}${r.session ? ` · session ${r.session}` : ''}` : (phase === 'done' ? 'Session done' : 'Recorder idle'));
  }

  function mainAction(r) {
    const phase = r ? r.phase : 'idle';
    if (phase === 'paused') return ['resume', 'Resume', 'play'];
    if (phase === 'go' || phase === 'ready') return ['pause', 'Pause', 'pause'];
    return ['go', 'Go', 'play'];
  }

  function renderControls() {
    const r = rec;
    const phase = r ? r.phase : 'idle';
    const running = !!(r && r.running);
    const saved = r && Array.isArray(r.saved) ? r.saved.length : 0;
    const [action, label, ic] = mainAction(r);
    setBtnLabel(mainBtn, label);
    mainBtn.querySelector('.icon').replaceWith(icon(ic, { size: 20 }));
    mainBtn.dataset.action = action;
    setEnabled(mainBtn, running && ['waiting', 'ready', 'go', 'paused'].includes(phase));
    setEnabled(redoBtn, running && saved > 0 && ['waiting', 'ready', 'go', 'paused'].includes(phase));
    setEnabled(skipBtn, running && ['waiting', 'ready', 'go', 'paused'].includes(phase));
    setEnabled(stopBtn, running);
    setEnabled(startBtn, !running && !!options);
    for (const c of setupBody.querySelectorAll('input, .toggle-chip, .switch')) c.disabled = running;
    stepBtns[0].classList.toggle('is-done', running || phase === 'done');
    stepBtns[1].classList.toggle('is-done', phase === 'done');
  }

  function renderLists(r) {
    const saved = r && Array.isArray(r.saved) ? r.saved.slice(-100).reverse() : [];
    setText(takes.metaEl, saved.length ? `${saved.length}` : '');
    takesList.replaceChildren(...(saved.length ? saved.map((t) => el('div', { class: 'take-row' },
      el('span', { class: 'mono break', text: String(t.file || '?') }),
      el('span', { class: 'small muted tabular', text: `${fmt.si(t.events)} events · ${fmt.secs(t.seconds)}` })))
      : [el('p', { class: 'muted small', text: 'No takes saved yet.' })]));
    const pl = r && Array.isArray(r.plan) ? r.plan.slice(0, 300) : [];
    setText(plan.metaEl, pl.length ? `${pl.length} takes` : '');
    const current = r && fmt.isNum(r.step) ? r.step - 1 : -1;
    planList.replaceChildren(...(pl.length ? pl.map((p, i) => el('div', { class: ['take-row', 'plan-row', i === current && 'is-current', i < current && 'is-done'], 'aria-current': i === current ? 'step' : null },
      el('span', { text: `${i + 1}. ${p.name || p.cls}` }),
      el('span', { class: 'small tabular', text: `${p.style || ''} · ${fmt.secs(p.seconds, 0)}` })))
      : [el('p', { class: 'muted small', text: 'The plan appears when a session starts.' })]));
  }

  let listSig = '';
  function applyRecord(r) {
    if (!r || typeof r !== 'object') return;
    const prev = rec;
    rec = r;
    renderCue(r);
    renderControls();
    const sig = `${Array.isArray(r.saved) ? r.saved.length : 0}|${r.step}|${Array.isArray(r.plan) ? r.plan.length : 0}|${r.session}`;
    if (sig !== listSig) {
      listSig = sig;
      renderLists(r);
    }
    if (r.running) startPreview();
    if ((!prev || !prev.running) && r.running && stepNow === 1) setStep(2);
    if (prev && prev.running && !r.running) {
      loadSessions();
      loadOptions();
      if (r.phase === 'done') {
        setStep(3);
        if (r.session != null) cutSession.value = String(r.session);
      }
    }
  }

  // ---------------------------------------------------------------- preview (fetch -> blob URL, no flicker)
  let previewUrl = null;
  let previewRunning = false;
  async function previewTick() {
    if (!scope.alive || !rec || !rec.running || document.hidden) {
      previewRunning = false;
      return;
    }
    const t0 = performance.now();
    let failed = false;
    try {
      const blob = await api.blob('/record/preview', { t: Date.now() }, { signal: scope.signal, timeout: 3000 });
      if (!scope.alive) return;
      const url = URL.createObjectURL(blob);
      previewImg.src = url;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = url;
      previewBox.classList.add('is-live');
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      previewBox.classList.remove('is-live');
      setText(previewMsg, `No preview (${err.message}). Retrying…`);
      failed = true;
    }
    // Next frame 200 ms after the previous request started; back off to 1 s while failing.
    scope.timeout(previewTick, failed ? 1000 : Math.max(0, PREVIEW_MS - (performance.now() - t0)));
  }
  function startPreview() {
    if (previewRunning) return;
    previewRunning = true;
    setText(previewMsg, 'Loading preview…');
    previewTick();
  }
  scope.add(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });
  scope.listen(document, 'visibilitychange', () => {
    if (!document.hidden && rec && rec.running) startPreview();
  });

  // ---------------------------------------------------------------- controls
  async function control(action, b) {
    const st = await act(b, () => api.post('/record/control', { action }, { signal: scope.signal }));
    if (st && scope.alive) applyRecord(st);
  }
  mainBtn.addEventListener('click', () => control(mainBtn.dataset.action || 'go', mainBtn));
  redoBtn.addEventListener('click', () => control('redo', redoBtn));
  skipBtn.addEventListener('click', () => control('skip', skipBtn));
  stopBtn.addEventListener('click', () => control('stop', stopBtn));

  // Space = go / pause / resume, R = redo, N = skip (only while a session runs).
  scope.listen(document, 'keydown', (ev) => {
    if (!rec || !rec.running || !shortcutAllowed(ev)) return;
    const key = (ev.key || '').toLowerCase();
    if (ev.key === ' ' || ev.code === 'Space') {
      ev.preventDefault(); // no page scroll / no activation of the focused button
      if (!ev.repeat && !mainBtn.disabled) control(mainBtn.dataset.action || 'go', mainBtn);
    } else if (key === 'r' && !ev.repeat) {
      ev.preventDefault();
      if (!redoBtn.disabled) control('redo', redoBtn);
    } else if (key === 'n' && !ev.repeat) {
      ev.preventDefault();
      if (!skipBtn.disabled) control('skip', skipBtn);
    }
  });
  scope.listen(document, 'keyup', (ev) => {
    if (rec && rec.running && (ev.key === ' ' || ev.code === 'Space') && shortcutAllowed(ev)) ev.preventDefault();
  });

  // ---------------------------------------------------------------- data flow
  scope.sub('record', (r) => {
    lastSseAt = Date.now();
    applyRecord(r);
  });
  const loadState = async () => {
    if (rec && Date.now() - lastSseAt < SSE_STALE_MS) return; // SSE carries it 4x per second
    try {
      const st = await api.get('/record/state', null, { signal: scope.signal, timeout: 5000 });
      if (scope.alive) applyRecord(st);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      if (!rec) {
        setText(cueBig, 'Recorder unavailable');
        setText(cueSub, err.message);
      }
    }
  };
  const cached = last('record');
  if (cached) applyRecord(cached);
  else {
    renderCue({ phase: 'idle' });
    renderControls();
    renderLists(null);
  }
  setStep(rec && rec.running ? 2 : 1);
  scope.poll(loadState, 1000);
  loadOptions();
  loadSessions();
}
