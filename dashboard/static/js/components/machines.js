// Training machines added in the dashboard (GPU servers reached over SSH):
// list (label, status; never host / user / paths), Add / Edit drawer with
// SSH key selection or generation, the "Server ID" step (host key trust), Test
// checklist and Prepare (datasets upload task with inline progress).

import { api, detailKeys } from '../api.js';
import {
  el, setText, act, btn, iconBtn, icon, chip, status, emptyState, placeholder, menuButton, drawer, field, input,
  select, setOptions, collapsible, setFieldError, confirmDialog, checkbox, toast, setLogText, attachTip,
} from '../ui.js';
import { trackTask, onTaskUpdate } from '../tasks.js';

const PREPARE_DATASETS = [
  { id: 'organisers', label: 'Durance AI team sessions 01-14 (sit, lay down, hello, dance)' },
  { id: 'own', label: 'Our sessions 20-30 (heart, turn left, turn right)' },
];
const FIELDS = ['id', 'label', 'host', 'user', 'port', 'identity_file', 'ssh_host', 'workdir', 'python', 'uv', 'max_submits_per_hour'];
const KEY_PREF = { 'ssh-ed25519': 0, 'ecdsa-sha2-nistp256': 1, 'ecdsa-sha2-nistp384': 1, 'ecdsa-sha2-nistp521': 1, 'ssh-rsa': 2 };

/** "My GPU box #2" -> "my-gpu-box-2" (backend: ^[a-z0-9][a-z0-9-]{1,30}$) */
export function slugify(name) {
  let s = String(name || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  s = s.slice(0, 31).replace(/-+$/, '');
  if (s.length < 2) s = s ? `${s}-gpu` : 'gpu';
  return s;
}

/** The field a 422 detail is about ("host: invalid value" -> "host"). */
function fieldOf(err) {
  const keys = detailKeys(err.detail);
  for (const k of keys) if (FIELDS.includes(k)) return k;
  const m = /^\s*([a-z_]+)\s*:/.exec(err.message || '');
  if (m && FIELDS.includes(m[1])) return m[1];
  if (/^key name/i.test(err.message || '')) return 'identity_file';
  return null;
}

function checklist(steps) {
  const list = Array.isArray(steps) ? steps : [];
  if (!list.length) return el('p', { class: 'muted small', text: 'No checks were reported.' });
  return el('ul', { class: 'checklist' }, ...list.map((s) => el('li', { class: ['check-row', s.ok ? 'is-ok' : 'is-bad'] },
    icon(s.ok ? 'check' : 'x', { size: 16, cls: 'check-icon' }),
    el('span', { class: 'sr-only', text: s.ok ? 'passed: ' : 'failed: ' }),
    el('span', { class: 'check-name', text: String(s.name || '?') }),
    el('span', { class: 'check-detail', text: s.detail == null ? '' : String(s.detail) }))));
}

/** Copyable read-only public key box. */
function publicKeyBox(pub) {
  const code = el('code', { class: 'pubkey-text', text: pub });
  const copyBtn = btn('Copy', { kind: 'secondary', size: 'sm', icon: 'file' });
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pub);
      toast('Public key copied', 'success');
    } catch {
      const r = document.createRange();
      r.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      toast('Selected: press Ctrl+C to copy', 'info');
    }
  });
  return el('div', { class: 'pubkey' },
    el('div', { class: 'row-between' }, el('span', { class: 'field-label', text: 'Public key' }), copyBtn),
    code,
    el('p', { class: 'field-hint', text: "Add this public key to your cloud provider's SSH keys (or ~/.ssh/authorized_keys on the machine)." }));
}

/**
 * onChanged(): machines were added / edited / removed (the page reloads /train/hosts).
 */
export function createMachinesPanel(scope, { onChanged } = {}) {
  let hosts = null;
  let openDrawer = null;
  const extras = new Map(); // id -> element under the row (test / prepare output)

  const addBtn = btn('Add machine', { kind: 'primary', icon: 'plus' });
  addBtn.addEventListener('click', () => openForm(null));
  const list = el('div', { class: 'machines' }, placeholder('loading', 'Loading machines…'));
  const root = el('section', { class: 'card card-flush', 'aria-label': 'Training machines' },
    el('div', { class: 'card-head' },
      el('div', { class: 'card-heading' }, el('h2', { class: 'card-title', text: 'Training machines' }),
        el('span', { class: 'card-sub', text: 'Cloud GPUs or clusters reachable with an SSH key' })),
      addBtn),
    list);

  scope.add(() => {
    if (openDrawer) openDrawer.close();
  });

  // ---------------------------------------------------------------- list

  function render() {
    if (hosts == null) return;
    addBtn.hidden = !hosts.length;
    if (!hosts.length) {
      const b = btn('Add machine', { kind: 'primary', icon: 'plus', onClick: () => openForm(null) });
      list.replaceChildren(emptyState({
        icon: 'server', title: 'No training machine yet',
        text: 'Add a cloud GPU or a cluster that you can reach with SSH.',
        actions: b,
      }));
      return;
    }
    list.replaceChildren(...hosts.map((h) => {
      const testBtn = btn('Test', { kind: 'ghost', size: 'sm', icon: 'check' });
      testBtn.addEventListener('click', () => test(h, testBtn));
      const prepBtn = btn('Prepare', { kind: 'ghost', size: 'sm', icon: 'upload' });
      attachTip(prepBtn, 'Copy the code and riverops, install the training environment, upload dataset sessions');
      prepBtn.addEventListener('click', () => prepare(h, prepBtn));
      const editBtn = btn('Edit', { kind: 'ghost', size: 'sm', icon: 'sliders' });
      editBtn.addEventListener('click', () => openForm(h.id));
      const more = menuButton([
        { label: 'Server ID', icon: 'shield', onSelect: () => openServerId(h.id, h.label) },
        'sep',
        { label: 'Remove from the dashboard', icon: 'trash', danger: true, onSelect: (b) => remove(h, b) },
      ], { label: `More actions for ${h.label}` });
      let extra = extras.get(h.id);
      if (!extra) {
        extra = el('div', { class: 'machine-extra', hidden: true });
        extras.set(h.id, extra);
      }
      return el('div', { class: 'machine' },
        el('div', { class: 'machine-main' },
          el('div', { class: 'machine-name' }, el('span', { class: 'cell-main', text: h.label || h.id }),
            chip('GPU server', { kind: 'neutral', icon: 'cpu' })),
          status(h.ok ? 'ok' : (h.ok === false ? 'bad' : 'off'), h.detail || (h.ok ? 'reachable' : 'unknown')),
          el('div', { class: 'machine-actions' }, testBtn, prepBtn, editBtn, more)),
        extra);
    }));
  }

  function showExtra(id, node) {
    const extra = extras.get(id);
    if (!extra) return null;
    const close = iconBtn('x', 'Hide', { size: 'sm' });
    close.addEventListener('click', () => {
      extra.hidden = true;
      extra.replaceChildren();
    });
    extra.replaceChildren(el('div', { class: 'machine-extra-head' }, node.title, close), node.body);
    extra.hidden = false;
    return extra;
  }

  // ---------------------------------------------------------------- test / prepare / remove

  async function runTest(id, b) {
    return act(b, () => api.post(`/train/machines/${encodeURIComponent(id)}/test`, {}, { signal: scope.signal, timeout: 90000 }));
  }

  async function test(h, b) {
    showExtra(h.id, { title: el('strong', { text: 'Test' }), body: placeholder('loading', 'Checking the machine…') });
    const res = await runTest(h.id, b);
    if (!scope.alive) return;
    if (!res) {
      showExtra(h.id, { title: el('strong', { text: 'Test' }), body: placeholder('error', 'The test could not run.') });
      return;
    }
    showExtra(h.id, {
      title: status(res.ok ? 'ok' : 'bad', res.ok ? 'Ready for training' : 'Not ready yet'),
      body: checklist(res.steps),
    });
    if (onChanged) onChanged();
  }

  async function prepare(h, b) {
    const boxes = PREPARE_DATASETS.map((d) => ({ ...d, box: checkbox(d.label, { checked: d.id === 'organisers' }) }));
    const yes = await confirmDialog({
      title: `Prepare ${h.label}`,
      message: 'Copies the repository code and the riverops wheel, installs the training environment (uv sync --group train) and uploads the dataset sessions you choose (data/motion_buddy_dataset). Only changed files move; the first time can take a while.',
      confirmLabel: 'Prepare',
      risk: 'normal',
      body: el('div', { class: 'stack-sm' }, el('div', { class: 'field-label', text: 'Datasets' }), ...boxes.map((x) => x.box.el)),
    });
    if (!yes || !scope.alive) return;
    const datasets = boxes.filter((x) => x.box.input.checked).map((x) => x.id);
    const res = await act(b, () => api.post(`/train/machines/${encodeURIComponent(h.id)}/prepare`, { datasets }, { risk: 'warn', signal: scope.signal }));
    if (!res || !scope.alive) return;
    const st = status('busy', 'Preparing…');
    const log = el('pre', { class: 'logbox task-log', tabindex: '0', 'aria-label': 'Prepare log', text: '(starting)' });
    showExtra(h.id, { title: st, body: log });
    const unsub = onTaskUpdate(res.task_id, (t) => {
      if (scope.alive) setLogText(log, Array.isArray(t.log) && t.log.length ? t.log : ['(no output yet)'], true);
    });
    scope.add(unsub);
    const t = await trackTask(res, `Prepare ${h.label}`, { open: false });
    unsub();
    if (!scope.alive) return;
    st.replaceWith(t && t.state === 'done' ? status('ok', 'Prepared') : status('bad', `Prepare failed (${t ? t.state : 'unknown'})`));
  }

  async function remove(h, b) {
    const res = await act(b, () => api.post(`/train/machines/${encodeURIComponent(h.id)}/delete`, {}, { risk: 'warn', signal: scope.signal }), {
      success: `${h.label} removed from the dashboard`,
    });
    if (res && onChanged) onChanged();
  }

  // ---------------------------------------------------------------- add / edit drawer

  async function openForm(id) {
    if (openDrawer) openDrawer.close();
    const editing = !!id;
    const d = drawer({ title: editing ? 'Edit machine' : 'Add training machine', width: 520, onClose: () => { if (openDrawer === d) openDrawer = null; } });
    openDrawer = d;
    d.body.append(placeholder('loading', 'Loading…'));
    let current = null;
    let keys = [];
    try {
      const [k, m] = await Promise.all([
        api.get('/train/keys', null, { signal: scope.signal }),
        editing ? api.get(`/train/machines/${encodeURIComponent(id)}`, null, { signal: scope.signal }) : Promise.resolve(null),
      ]);
      keys = Array.isArray(k.keys) ? k.keys : [];
      current = m;
    } catch (err) {
      if (!scope.alive || err.aborted || openDrawer !== d) return;
      d.body.replaceChildren(emptyState({ kind: 'error', title: 'Could not open the form', text: err.message }));
      return;
    }
    if (!scope.alive || openDrawer !== d) return;
    buildForm(d, current, keys);
  }

  function buildForm(d, current, keysIn) {
    let keys = keysIn;
    const editing = !!current;
    const v = current || {};
    const taken = new Set((hosts || []).map((h) => h.id));

    const nameIn = input({ value: v.label || '', maxLength: 40, placeholder: 'e.g. Cloud A100' });
    const idText = el('span', { class: 'mono' });
    const nameF = field('Name', nameIn);
    nameF.append(el('div', { class: 'field-hint' }, 'ID: ', idText, editing ? ' (fixed)' : ''));
    const hostIn = input({ value: v.host || '', placeholder: 'e.g. 203.0.113.7 or gpu.example.org', attrs: { spellcheck: 'false', autocapitalize: 'off' } });
    const userIn = input({ value: v.user || '', placeholder: 'e.g. ubuntu', attrs: { spellcheck: 'false', autocapitalize: 'off' } });
    const portIn = input({ type: 'number', min: '1', max: '65535', value: String(v.port || 22) });
    const hostF = field('Host', hostIn);
    const userF = field('User', userIn);
    const portF = field('Port', portIn);
    const connRow = el('div', { class: 'form-grid conn-grid' }, hostF, userF, portF);

    const keySel = select([], { ariaLabel: 'SSH key' });
    const keyF = field('SSH key', keySel, { help: 'The private key stays on this laptop; only its public half goes to the machine.' });
    const genBtn = btn('Generate new key', { kind: 'secondary', size: 'sm', icon: 'key' });
    const pubHost = el('div');
    keyF.append(el('div', { class: 'form-actions' }, genBtn));
    keyF.append(pubHost);

    const workIn = input({ value: v.workdir || '~/durance-training', attrs: { spellcheck: 'false' } });
    const pyIn = input({ value: v.python || 'python3', attrs: { spellcheck: 'false' } });
    const uvIn = input({ value: v.uv || '', placeholder: 'uv on the PATH or ~/.local/bin/uv', attrs: { spellcheck: 'false' } });
    const workF = field('Work folder', workIn, { help: 'Folder on the machine for a copy of the repository: code, data/ and runs/.' });
    const pyF = field('Python', pyIn, { help: 'Plain Python 3 for small status scripts, e.g. python3.' });
    const uvF = field('uv', uvIn, { help: 'Runs the training environment. Empty: found on the PATH, or installed to ~/.local/bin by Prepare.' });

    const aliasIn = input({ value: v.ssh_host || '', placeholder: 'e.g. my-gpu', attrs: { spellcheck: 'false' } });
    const aliasF = field('Use an ~/.ssh/config alias instead', aliasIn, { help: 'When set, host, user and port come from your ~/.ssh/config entry.' });
    const maxIn = input({ type: 'number', min: '1', max: '1000', value: String(v.max_submits_per_hour ?? 20) });
    const maxF = field('Max submits per hour', maxIn);
    const advanced = collapsible({ title: 'Advanced', open: !!v.ssh_host }, el('div', { class: 'stack' }, aliasF, maxF));

    const formError = el('div', { class: 'form-error', role: 'alert', hidden: true });
    const saveBtn = btn(editing ? 'Save' : 'Save and continue', { kind: 'primary', type: 'submit' });
    const cancelBtn = btn('Cancel', { kind: 'ghost', onClick: () => d.close() });
    const serverIdBtn = editing ? btn('Server ID', { kind: 'ghost', icon: 'shield', onClick: () => showServerId(d, current.id, current.label) }) : null;
    const form = el('form', { class: 'stack', novalidate: true },
      nameF, connRow, keyF, workF, pyF, uvF, advanced, formError,
      el('div', { class: 'form-actions' }, saveBtn, cancelBtn, el('span', { class: 'spacer' }), serverIdBtn));
    d.body.replaceChildren(form);
    d.setTitle(editing ? `Edit ${current.label}` : 'Add training machine');

    const fieldMap = {
      label: nameF, id: nameF, host: hostF, user: userF, port: portF, identity_file: keyF,
      workdir: workF, python: pyF, uv: uvF, ssh_host: aliasF, max_submits_per_hour: maxF,
    };

    function newId() {
      if (editing) return current.id;
      let base = slugify(nameIn.value);
      let candidate = base;
      let n = 2;
      while (taken.has(candidate)) {
        candidate = `${base.slice(0, 28)}-${n}`;
        n += 1;
      }
      return candidate;
    }

    function renderKeys(selected) {
      setOptions(keySel, [
        { value: '', label: keys.length ? 'Choose a key…' : 'No keys yet: generate one' },
        ...keys.map((k) => ({ value: k.ref, label: `${k.name}${k.source === 'dashboard' ? ' (made here)' : ' (~/.ssh)'}` })),
      ], selected ?? keySel.value);
      const k = keys.find((x) => x.ref === keySel.value);
      pubHost.replaceChildren(...(k && k.public ? [publicKeyBox(k.public)] : []));
    }

    function sync() {
      setText(idText, newId());
      const alias = aliasIn.value.trim() !== '';
      connRow.hidden = alias;
    }

    nameIn.addEventListener('input', sync);
    aliasIn.addEventListener('input', sync);
    keySel.addEventListener('change', () => renderKeys());

    genBtn.addEventListener('click', async () => {
      const base = (newId() || 'gpu').replace(/[^A-Za-z0-9_.-]/g, '-');
      let name = `durance-${base}`;
      const names = new Set(keys.map((k) => k.name));
      let n = 2;
      while (names.has(name)) {
        name = `durance-${base}-${n}`;
        n += 1;
      }
      setFieldError(keyF, '');
      const k = await act(genBtn, () => api.post('/train/keys', { name }, { signal: scope.signal }), {
        onError: (err) => {
          setFieldError(keyF, err.message);
          return true;
        },
      });
      if (!k || !scope.alive) return;
      keys = [...keys.filter((x) => x.ref !== k.ref), k];
      renderKeys(k.ref); // the public key box below is the feedback
    });

    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      for (const f of Object.values(fieldMap)) setFieldError(f, '');
      formError.hidden = true;
      if (!nameIn.value.trim()) {
        setFieldError(nameF, 'Give the machine a name.');
        return;
      }
      const alias = aliasIn.value.trim();
      const port = Number(portIn.value);
      const body = {
        id: newId(),
        label: nameIn.value.trim(),
        host: alias ? '' : hostIn.value.trim(),
        user: alias ? '' : userIn.value.trim(),
        port: Number.isInteger(port) ? port : 22,
        identity_file: keySel.value,
        ssh_host: alias,
        workdir: workIn.value.trim(),
        python: pyIn.value.trim() || 'python3',
        uv: uvIn.value.trim(),
        max_submits_per_hour: Number(maxIn.value) || 20,
      };
      const saved = await act(saveBtn, () => api.post('/train/machines', body, { signal: scope.signal }), {
        onError: (err) => {
          if (err.status !== 422) return false;
          const f = fieldOf(err);
          if (f && fieldMap[f]) {
            if ((f === 'ssh_host' || f === 'max_submits_per_hour') && !advanced.open) advanced.open = true;
            const msg = err.message.replace(/^\s*[a-z_ ]+:\s*/i, '') || err.message;
            setFieldError(fieldMap[f], /^invalid value$/i.test(msg) ? 'Not valid: check for typos or spaces.' : msg);
          } else {
            formError.replaceChildren(icon('alert', { size: 18 }), el('span', { text: err.message }));
            formError.hidden = false;
          }
          return true;
        },
      });
      if (!saved || !scope.alive) return;
      toast(editing ? 'Machine saved' : `${saved.label || body.label} added`, 'success');
      if (onChanged) onChanged();
      if (!editing) showServerId(d, saved.id || body.id, saved.label || body.label);
      else d.close();
    });

    renderKeys(v.identity_file || '');
    sync();
    nameIn.focus();
  }

  // ---------------------------------------------------------------- Server ID step

  function openServerId(id, label) {
    if (openDrawer) openDrawer.close();
    const d = drawer({ title: `Server ID · ${label}`, width: 520, onClose: () => { if (openDrawer === d) openDrawer = null; } });
    openDrawer = d;
    showServerId(d, id, label);
  }

  async function showServerId(d, id, label) {
    d.setTitle(`Server ID · ${label}`);
    const body = el('div', { class: 'stack' });
    d.body.replaceChildren(body);
    const testHost = el('div');
    const nextRow = el('div', { class: 'form-actions' });

    async function load() {
      body.replaceChildren(placeholder('loading', 'Asking the server for its ID…'));
      let res;
      try {
        res = await api.get(`/train/machines/${encodeURIComponent(id)}/server-id`, null, { signal: scope.signal, timeout: 30000 });
      } catch (err) {
        if (!scope.alive || err.aborted) return;
        body.replaceChildren(
          el('p', { class: 'prose', text: 'This code identifies the server. Compare it with the one your provider shows, or trust it if you just created the machine.' }),
          el('div', { class: 'form-error', role: 'alert' }, icon('alert', { size: 18 }), el('span', { text: err.message })),
          el('div', { class: 'form-actions' }, btn('Try again', { kind: 'secondary', icon: 'refresh', onClick: load }), btn('Close', { kind: 'ghost', onClick: () => d.close() })));
        return;
      }
      if (!scope.alive) return;
      const ids = Array.isArray(res.keys) ? res.keys : [];
      if (!ids.length) {
        body.replaceChildren(
          el('p', { class: 'prose', text: res.note ? String(res.note) : 'The server did not present an ID.' }),
          nextRow, testHost);
        renderNext(true);
        return;
      }
      const best = ids.filter((k) => !k.trusted).sort((a, b) => (KEY_PREF[a.type] ?? 9) - (KEY_PREF[b.type] ?? 9))[0];
      const anyTrusted = ids.some((k) => k.trusted);
      const trustBtn = btn('Trust this server', { kind: 'primary', icon: 'shield' });
      trustBtn.hidden = anyTrusted || !best;
      trustBtn.addEventListener('click', async () => {
        const r = await act(trustBtn, () => api.post(`/train/machines/${encodeURIComponent(id)}/server-id`, { fingerprint: best.fingerprint }, { signal: scope.signal, timeout: 30000 }));
        if (r && scope.alive) {
          if (onChanged) onChanged();
          load();
        }
      });
      body.replaceChildren(
        el('p', { class: 'prose', text: 'This code identifies the server. Compare it with the one your provider shows, or trust it if you just created the machine.' }),
        el('ul', { class: 'server-ids' }, ...ids.map((k) => el('li', { class: ['server-id', k === best && !anyTrusted && 'is-best'] },
          el('span', { class: 'server-id-type', text: String(k.type || '?') }),
          el('code', { class: 'server-id-fp', text: String(k.fingerprint || '?') }),
          k.trusted ? el('span', { class: 'server-id-ok' }, icon('check', { size: 14 }), 'trusted') : null))),
        anyTrusted ? status('ok', 'This server is trusted.') : el('div', { class: 'form-actions' }, trustBtn),
        nextRow, testHost);
      renderNext(anyTrusted);
    }

    function renderNext(ready) {
      nextRow.replaceChildren();
      if (!ready) return;
      const testBtn = btn('Test the connection', { kind: 'secondary', icon: 'check' });
      testBtn.addEventListener('click', async () => {
        testHost.replaceChildren(placeholder('loading', 'Checking the machine…'));
        const res = await runTest(id, testBtn);
        if (!scope.alive) return;
        testHost.replaceChildren(...(res ? [status(res.ok ? 'ok' : 'bad', res.ok ? 'Ready for training' : 'Not ready yet'), checklist(res.steps)] : [placeholder('error', 'The test could not run.')]));
        if (onChanged) onChanged();
      });
      nextRow.append(testBtn, btn('Done', { kind: 'ghost', onClick: () => d.close() }));
    }

    load();
  }

  return {
    el: root,
    setHosts(next) {
      hosts = Array.isArray(next) ? next : [];
      render();
    },
    openAdd: () => openForm(null),
  };
}
