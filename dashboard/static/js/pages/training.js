// Training: tabs [Runs] [Jobs] [New run] [Machines]; machine status in the header.
// Runs: sortable compact table + Fetch. Jobs: state dots, log drawer, cancel in "...".
// New run: machine / preset / dataset / architecture / name; Advanced flags collapsed.
// Machines: GPU machines / Slurm clusters added in the dashboard (components/machines.js).

import { api, detailKeys } from '../api.js';
import {
  el, setText, card, act, btn, status, setStatus, field, setFieldError, switchControl, select, setOptions, collapsible,
  createTabs, dataTable, numSort, strSort, emptyState, prefs, toast, menuButton, drawer, setLogText, input, infoTip,
  setBtnLabel, icon,
} from '../ui.js';
import { trackTask } from '../tasks.js';
import { createMachinesPanel } from '../components/machines.js';
import * as fmt from '../format.js';

const JOBS_MS = 10000;
const HOSTS_MS = 30000;
const LOG_MS = 5000;
const FLAG_GROUPS = [
  ['Training', (k) => ['epochs', 'batch-size', 'lr', 'weight-decay', 'ce-weight', 'seed'].includes(k) || k.startsWith('qat-')],
  ['Neurons', (k) => ['learnable-leak', 'initial-tau', 'spike-reg'].includes(k)],
  ['Augmentation', () => true],
];

function jobState(state) {
  const s = String(state || '').toUpperCase();
  if (s.startsWith('RUN')) return 'busy';
  if (s.startsWith('PEND') || s.startsWith('CONF')) return 'warn';
  if (s.startsWith('COMP') || s === 'DONE') return 'ok';
  if (/^(FAIL|CANC|TIME|OUT_OF|NODE_FAIL)/.test(s)) return 'bad';
  return 'off';
}

function optionOf(x) {
  if (x && typeof x === 'object') return { value: String(x.id ?? x.value ?? x.name), label: String(x.label ?? x.name ?? x.id) };
  return { value: String(x), label: String(x) };
}

export async function mount(root, scope) {
  let hosts = null; // null = not loaded yet
  let runningJobs = 0;
  let host = prefs.get('train.host', null);
  let options = null;
  const flagFields = new Map(); // key -> {flag, kind, field, get, set}

  // ---------------------------------------------------------------- header
  const tabs = createTabs({
    key: 'tab.training',
    label: 'Training sections',
    items: [
      { value: 'runs', label: 'Runs', icon: 'trending' }, { value: 'jobs', label: 'Jobs', icon: 'list' },
      { value: 'new', label: 'New run', icon: 'plus' }, { value: 'machines', label: 'Machines', icon: 'server' },
    ],
    onChange: () => renderHeader(),
  });
  const hostStatus = status('off', 'Checking machines…');
  const hostSel = select([], { ariaLabel: 'Training machine', cls: 'input-sm' });
  hostSel.hidden = true;
  root.append(el('div', { class: 'tabbar' }, tabs.el, el('div', { class: 'tabbar-actions' }, hostStatus, hostSel)));

  // Each work tab has its content and a "no machine yet" state.
  const work = {};
  for (const v of ['runs', 'jobs', 'new']) {
    const content = el('div', { class: 'stack' });
    const none = emptyState({
      icon: 'server', title: 'No training machine yet',
      text: 'Add a cloud GPU or a cluster that you can reach with SSH.',
      actions: btn('Add machine', { kind: 'primary', icon: 'plus', onClick: () => { tabs.select('machines'); machines.openAdd(); } }),
    });
    const noneCard = el('section', { class: 'card' }, none);
    noneCard.hidden = true;
    tabs.panel(v).append(content, noneCard);
    work[v] = { content, noneCard };
  }
  root.append(tabs.panel('runs'), tabs.panel('jobs'), tabs.panel('new'), tabs.panel('machines'));

  const machines = createMachinesPanel(scope, { onChanged: () => loadHosts(true) });
  tabs.panel('machines').append(machines.el);

  // ---------------------------------------------------------------- runs
  const runsTable = dataTable({
    caption: 'Finished training runs',
    empty: 'No finished runs on this machine yet.',
    sortKey: 'test',
    sortDir: 'desc',
    columns: [
      {
        key: 'name', label: 'Name', sort: strSort('name'),
        render: (r) => el('div', null, el('div', { class: 'cell-main', text: r.name || '?' }),
          el('div', { class: 'cell-sub', text: [r.dataset, r.job ? `job ${r.job}` : null, fmt.isNum(r.val) ? `val ${fmt.accuracy(r.val)}` : null].filter(Boolean).join(' · ') })),
      },
      { key: 'arch', label: 'Arch', sort: strSort('arch') },
      { key: 'test', label: 'Test', numeric: true, sort: numSort('test'), render: (r) => fmt.accuracy(r.test) },
      { key: 'synops', label: 'SynOps', numeric: true, sort: numSort('synops'), render: (r) => fmt.si(r.synops) },
      { key: 'finished', label: 'Finished', sort: strSort('finished'), render: (r) => el('span', { class: 'muted', text: fmt.dateTime(r.finished) }) },
      {
        key: 'fetch', label: '', cls: 'actions',
        render: (r) => {
          const b = btn('Fetch', { kind: 'ghost', size: 'sm', icon: 'download', tip: 'Copy the weights and metrics to this laptop (then deployable from Board > Models)' });
          b.addEventListener('click', () => fetchRun(r, b));
          return b;
        },
      },
    ],
  });
  work.runs.content.append(el('section', { class: 'card card-flush' }, runsTable.el));

  async function loadRuns() {
    if (!host) return;
    try {
      const res = await api.get('/train/runs', { host }, { signal: scope.signal, timeout: 20000 });
      if (scope.alive) runsTable.setRows(Array.isArray(res.runs) ? res.runs : []);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      runsTable.setEmpty(`Could not load runs: ${err.message}`);
      runsTable.setRows([]);
    }
  }

  async function fetchRun(run, b) {
    const res = await act(b, () => api.post('/train/fetch', { host, run: run.name }, { signal: scope.signal }));
    if (!res) return;
    const t = await trackTask(res, `Fetch ${run.name}`);
    if (t && t.state === 'done' && scope.alive) toast(`${run.name} is now listed under Board › Models`, 'info');
  }

  // ---------------------------------------------------------------- jobs
  let logDrawer = null;
  const jobsTable = dataTable({
    caption: 'Training jobs',
    empty: 'No jobs on this machine.',
    columns: [
      { key: 'id', label: 'Job', render: (j) => el('span', { class: 'mono', text: String(j.id) }) },
      {
        key: 'name', label: 'Name',
        render: (j) => el('div', null, el('div', { class: 'cell-main', text: j.name || '?' }), j.reason ? el('div', { class: 'cell-sub', text: String(j.reason) }) : null),
      },
      { key: 'state', label: 'State', render: (j) => status(jobState(j.state), fmt.humanize(String(j.state || 'unknown').toLowerCase())) },
      { key: 'time', label: 'Time', render: (j) => el('span', { class: 'tabular', text: j.time || fmt.DASH }) },
      { key: 'node', label: 'Node', render: (j) => el('span', { class: 'muted', text: j.node || fmt.DASH }) },
      {
        key: 'actions', label: '', cls: 'actions',
        render: (j) => {
          const logBtn = btn('Log', { kind: 'ghost', size: 'sm', icon: 'file' });
          logBtn.addEventListener('click', () => openLog(j));
          const finished = ['ok', 'bad'].includes(jobState(j.state));
          const more = menuButton([
            { label: 'Cancel job', icon: 'x', danger: true, disabled: finished, onSelect: (b) => cancelJob(j, b) },
          ], { label: `More actions for job ${j.id}` });
          return el('div', { class: 'row', style: { justifyContent: 'flex-end', flexWrap: 'nowrap' } }, logBtn, more);
        },
      },
    ],
  });
  const jobsInfo = el('span', { class: 'small muted' });
  work.jobs.content.append(el('section', { class: 'card card-flush' },
    el('div', { class: 'card-head' }, el('div', { class: 'card-heading' }, el('h2', { class: 'card-title', text: 'Jobs' }), jobsInfo),
      btn('Refresh', { kind: 'ghost', size: 'sm', icon: 'refresh', onClick: (ev) => act(ev.currentTarget, loadJobs) })),
    jobsTable.el));

  async function loadJobs() {
    if (!host) return;
    try {
      const res = await api.get('/train/jobs', { host }, { signal: scope.signal, timeout: 15000 });
      if (!scope.alive) return;
      const jobs = Array.isArray(res.jobs) ? res.jobs : [];
      jobsTable.setRows(jobs);
      runningJobs = jobs.filter((j) => jobState(j.state) === 'busy').length;
      tabs.setCount('jobs', runningJobs);
      setText(jobsInfo, `updated ${fmt.timeOfDay(Date.now())}`);
      renderHeader();
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      setText(jobsInfo, `refresh failed: ${err.message}`);
    }
  }
  const jobsPoll = scope.poll(loadJobs, JOBS_MS, { immediate: false });

  async function cancelJob(job, b) {
    const res = await act(b, () => api.post('/train/cancel', { host, job_id: job.id }, { risk: 'warn', signal: scope.signal }), {
      success: `Job ${job.id} cancelled`,
    });
    if (res) jobsPoll.now();
  }

  function openLog(job) {
    if (logDrawer) logDrawer.close();
    const pre = el('pre', { class: 'logbox', tabindex: '0', 'aria-label': `Log of job ${job.id}`, text: 'Loading…' });
    const follow = switchControl('Follow', { checked: true });
    let timer = null;
    const d = drawer({
      title: `Job ${job.id}${job.name ? ` · ${job.name}` : ''}`,
      width: 640,
      onClose: () => {
        logDrawer = null;
        clearInterval(timer);
      },
    });
    d.actions.append(follow.el);
    d.body.append(pre);
    logDrawer = d;
    const jobHost = host;
    const load = async () => {
      try {
        const res = await api.get('/train/log', { host: jobHost, job: job.id, lines: 200 }, { signal: scope.signal, timeout: 15000 });
        if (!scope.alive || logDrawer !== d) return;
        const lines = Array.isArray(res.lines) ? res.lines : [];
        setLogText(pre, lines.length ? lines : ['(log is empty)'], follow.get());
      } catch (err) {
        if (!scope.alive || err.aborted) return;
        setLogText(pre, [`Could not load the log: ${err.message}`]);
      }
    };
    timer = setInterval(() => {
      if (!document.hidden && follow.get()) load();
    }, LOG_MS);
    load();
  }
  scope.add(() => {
    if (logDrawer) logDrawer.close();
  });

  // ---------------------------------------------------------------- new run
  const machineSel = select([], { ariaLabel: 'Machine' });
  const machineField = field('Machine', machineSel);
  const presetSel = select([], {});
  const datasetSel = select([], {});
  const archSel = select([], {});
  const nameInput = input({ maxLength: 80, placeholder: 'e.g. own-nb40-q8h-v2', attrs: { spellcheck: 'false' } });
  const nameField = field('Name', nameInput);
  const datasetField = field('Dataset', datasetSel);
  const archField = field('Architecture', archSel);
  const dryRun = switchControl('Dry run: check the machine and show the command, start nothing', { checked: true });
  const submitBtn = btn('Dry run', { kind: 'primary', icon: 'terminal' });
  const formError = el('div', { class: 'form-error', role: 'alert', hidden: true });
  const resultStatus = status('off', '');
  resultStatus.hidden = true;
  const resultPre = el('pre', { class: 'logbox cmd-out', hidden: true, tabindex: '0', 'aria-label': 'Training command' });
  const flagsHost = el('div', { class: 'stack' });
  const advanced = collapsible({ title: 'Advanced flags', key: 'training.advanced' }, flagsHost);
  const formBody = el('div', { class: 'stack' }, emptyState({ kind: 'loading', title: 'Loading training options…' }));
  work.new.content.append(card({ title: 'New training run' }, formBody));

  machineSel.addEventListener('change', () => setHost(machineSel.value));

  function flagControl(f) {
    const type = String(f.type || 'str');
    const k = f.key;
    let control;
    let get;
    let set;
    let kind = type;
    if (type === 'choice' && Array.isArray(f.choices)) {
      control = select(f.choices.map((c) => ({ value: String(c), label: String(c) })), { value: f.default != null ? String(f.default) : undefined });
      get = () => control.value;
      set = (v) => { if (v != null) control.value = String(v); };
    } else if (type === 'bool' || (type === 'int' && f.min === 0 && f.max === 1)) {
      kind = type === 'bool' ? 'bool' : 'int01';
      const sw = switchControl(k, { checked: !!Number(f.default ?? 0), hideLabel: true });
      get = () => (kind === 'bool' ? sw.get() : (sw.get() ? 1 : 0));
      set = (v) => sw.set(!!Number(v === true ? 1 : v));
      const wrap = el('div', { class: 'field' },
        el('div', { class: 'field-label-row' }, el('span', { class: 'field-label mono', text: k }), f.help ? infoTip(String(f.help), k) : null),
        sw.el, el('div', { class: 'field-error', role: 'alert', hidden: true }));
      return { flag: f, kind, field: wrap, get, set };
    } else if (type === 'int' || type === 'float') {
      control = input({ type: 'number', step: type === 'int' ? '1' : 'any', min: fmt.isNum(f.min) ? String(f.min) : null, max: fmt.isNum(f.max) ? String(f.max) : null, value: f.default != null ? String(f.default) : '' });
      control.classList.add('tabular');
      get = () => control.value.trim();
      set = (v) => { control.value = v == null ? '' : String(v); };
    } else {
      control = input({ value: f.default != null ? String(f.default) : '', attrs: { spellcheck: 'false' } });
      get = () => control.value.trim();
      set = (v) => { control.value = v == null ? '' : String(v); };
    }
    const range = fmt.isNum(f.min) || fmt.isNum(f.max) ? `Range ${fmt.isNum(f.min) ? f.min : '…'} to ${fmt.isNum(f.max) ? f.max : '…'}.` : '';
    const help = [f.help ? String(f.help) : '', range].filter(Boolean).join(' ');
    const wrap = field(k, control, { help: help || undefined });
    wrap.querySelector('.field-label').classList.add('mono');
    return { flag: f, kind, field: wrap, get, set };
  }

  function applyPreset(id) {
    const preset = options && Array.isArray(options.presets) ? options.presets.find((p) => String(p.id) === String(id)) : null;
    const args = preset && preset.args && typeof preset.args === 'object' ? preset.args : {};
    for (const [key, entry] of flagFields) {
      entry.set(key in args ? args[key] : entry.flag.default);
      setFieldError(entry.field, '');
    }
    if (args.dataset != null && !flagFields.has('dataset')) datasetSel.value = String(args.dataset);
    if (args.arch != null && !flagFields.has('arch')) archSel.value = String(args.arch);
  }

  function buildForm(opts) {
    options = opts;
    const presets = Array.isArray(opts.presets) ? opts.presets : [];
    setOptions(presetSel, [{ value: '', label: 'No preset (defaults)' }, ...presets.map((p) => ({ value: String(p.id), label: p.label || String(p.id) }))]);
    setOptions(datasetSel, (Array.isArray(opts.datasets) ? opts.datasets : []).map(optionOf));
    setOptions(archSel, (Array.isArray(opts.archs) ? opts.archs : []).map(optionOf));
    flagFields.clear();
    const flags = (Array.isArray(opts.flags) ? opts.flags : []).filter((f) => f && f.key);
    const grouped = new Map(FLAG_GROUPS.map(([g]) => [g, []]));
    for (const f of flags) {
      const c = flagControl(f);
      flagFields.set(f.key, c);
      grouped.get(FLAG_GROUPS.find(([, test]) => test(f.key))[0]).push(c.field);
    }
    flagsHost.replaceChildren(...Array.from(grouped).filter(([, list]) => list.length).map(([g, list]) => el('div', null,
      el('div', { class: 'flag-group-title', text: g }), el('div', { class: 'form-grid' }, ...list))));
    setText(advanced.metaEl, `${flags.length} flags`);
    formBody.replaceChildren(
      el('div', { class: 'form-grid' }, machineField, field('Preset', presetSel, { help: 'Fills the flags with a known recipe.' }), datasetField, archField, nameField),
      flags.length ? advanced : null,
      formError,
      el('div', { class: 'form-actions' }, submitBtn, dryRun.el),
      resultStatus,
      resultPre);
    if (presets.length) {
      presetSel.value = String(presets[0].id);
      applyPreset(presets[0].id);
    }
  }

  async function loadOptions() {
    try {
      const res = await api.get('/train/options', null, { signal: scope.signal });
      if (scope.alive) buildForm(res || {});
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      formBody.replaceChildren(emptyState({
        kind: 'error', title: 'Could not load the training options', text: err.message,
        actions: btn('Retry', { kind: 'secondary', icon: 'refresh', onClick: () => loadOptions() }),
      }));
    }
  }

  presetSel.addEventListener('change', () => applyPreset(presetSel.value));
  dryRun.button.addEventListener('click', () => {
    const dry = dryRun.get();
    setBtnLabel(submitBtn, dry ? 'Dry run' : 'Start training');
    submitBtn.querySelector('.icon').replaceWith(icon(dry ? 'terminal' : 'send'));
  });

  function collectArgs() {
    const args = {};
    let ok = true;
    for (const [key, entry] of flagFields) {
      setFieldError(entry.field, '');
      const raw = entry.get();
      if (entry.kind === 'bool' || entry.kind === 'int01') {
        args[key] = raw;
        continue;
      }
      if (raw === '' || raw == null) continue;
      if (entry.kind === 'int' || entry.kind === 'float') {
        const v = Number(raw);
        const f = entry.flag;
        let msg = '';
        if (!Number.isFinite(v)) msg = 'Enter a number.';
        else if (entry.kind === 'int' && !Number.isInteger(v)) msg = 'Enter a whole number.';
        else if (fmt.isNum(f.min) && v < f.min) msg = `Minimum is ${f.min}.`;
        else if (fmt.isNum(f.max) && v > f.max) msg = `Maximum is ${f.max}.`;
        if (msg) {
          setFieldError(entry.field, msg);
          ok = false;
          continue;
        }
        args[key] = v;
      } else {
        args[key] = raw;
      }
    }
    return ok ? args : null;
  }

  function showError(text) {
    formError.replaceChildren(icon('alert', { size: 18 }), el('span', { text }));
    formError.hidden = false;
  }

  submitBtn.addEventListener('click', async () => {
    formError.hidden = true;
    for (const f of [nameField, datasetField, archField, machineField]) setFieldError(f, '');
    const args = collectArgs();
    if (!args) {
      advanced.open = true;
      showError('Fix the marked flags first.');
      return;
    }
    if (!host) {
      setFieldError(machineField, 'Add a training machine first (Machines tab).');
      return;
    }
    const dry = dryRun.get();
    const body = {
      host, name: nameInput.value.trim(), dataset: datasetSel.value, arch: archSel.value, preset: presetSel.value || null, args, dry_run: dry,
    };
    const res = await act(submitBtn, () => api.post('/train/submit', body, { risk: 'warn', signal: scope.signal, timeout: 120000 }), {
      onError: (err) => {
        if (err.status !== 422) return false;
        let placed = false;
        for (const k of detailKeys(err.detail)) {
          const target = (flagFields.get(k) || {}).field || { name: nameField, dataset: datasetField, arch: archField, host: machineField }[k];
          if (target) {
            setFieldError(target, err.message);
            placed = true;
            if (flagFields.has(k)) advanced.open = true;
          }
        }
        if (!placed) {
          const m = /^\s*([a-z_-]+)\s*:/.exec(err.message || '');
          const target = m && ({ name: nameField, dataset: datasetField, arch: archField, host: machineField }[m[1]] || (flagFields.get(m[1]) || {}).field);
          if (target) {
            setFieldError(target, err.message);
            placed = true;
          }
        }
        showError(placed ? 'The server rejected a value (see the marked field).' : err.message);
        return true;
      },
    });
    if (!res || !scope.alive) return;
    resultStatus.hidden = false;
    if (res.dry_run) setStatus(resultStatus, 'info', 'Dry run: the machine was checked and nothing was started. This command would run:');
    else {
      setStatus(resultStatus, 'ok', `Started${res.job_id != null ? ` as job ${res.job_id}` : ''}.`);
      jobsPoll.now();
    }
    resultPre.hidden = !res.command;
    setLogText(resultPre, res.command ? String(res.command) : '');
  });

  // ---------------------------------------------------------------- machines (hosts)

  function current() {
    return (hosts || []).find((x) => String(x.id) === String(host)) || null;
  }

  function renderHeader() {
    if (hosts == null) return;
    const h = current();
    const onMachinesTab = tabs.get() === 'machines';
    hostSel.hidden = hosts.length < 2 || onMachinesTab;
    if (!hosts.length) {
      setStatus(hostStatus, 'off', 'No training machine set up');
      return;
    }
    if (!h) {
      setStatus(hostStatus, 'off', 'No machine selected');
      return;
    }
    const name = hostSel.hidden ? `${h.label || h.id} · ` : ''; // with a visible select the name is there
    setStatus(hostStatus, h.ok ? 'ok' : 'bad', `${name}${h.ok ? `${runningJobs} running` : (h.detail || 'unreachable')}`, h.detail || undefined);
  }

  function setHost(id) {
    if (String(id) === String(host)) return;
    host = id;
    prefs.set('train.host', host);
    hostSel.value = host;
    machineSel.value = host;
    if (logDrawer) logDrawer.close();
    runningJobs = 0;
    runsTable.setRows([]);
    jobsTable.setRows([]);
    renderHeader();
    jobsPoll.now();
    loadRuns();
  }
  hostSel.addEventListener('change', () => setHost(hostSel.value));

  let firstLoad = true;
  async function loadHosts(refreshWork = false) {
    try {
      const res = await api.get('/train/hosts', null, { signal: scope.signal, timeout: 20000 });
      if (!scope.alive) return;
      const prevHost = host;
      hosts = Array.isArray(res.hosts) ? res.hosts : [];
      if (!hosts.some((h) => String(h.id) === String(host))) host = hosts.length ? String((hosts.find((h) => h.ok) || hosts[0]).id) : null;
      const opts = hosts.map((h) => ({ value: String(h.id), label: `${h.label || h.id}${h.ok ? '' : ' (unreachable)'}` }));
      setOptions(hostSel, opts, host ?? undefined);
      setOptions(machineSel, opts.length ? opts : [{ value: '', label: 'No machine yet' }], host ?? '');
      const configured = hosts.length > 0;
      for (const w of Object.values(work)) {
        w.content.hidden = !configured;
        w.noneCard.hidden = configured;
      }
      tabs.setCount('machines', 0);
      machines.setHosts(hosts);
      if (firstLoad && !configured) tabs.select('machines');
      firstLoad = false;
      renderHeader();
      if (configured && (refreshWork || host !== prevHost)) {
        jobsPoll.now();
        loadRuns();
      }
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      setStatus(hostStatus, 'bad', `Machines unavailable: ${err.message}`);
      if (hosts == null) machines.setHosts([]);
    }
  }

  // ---------------------------------------------------------------- start
  tabs.onFirstShow('new', loadOptions);
  await loadHosts(true);
  if (!scope.alive) return;
  scope.poll(() => loadHosts(false), HOSTS_MS, { immediate: false });
}
