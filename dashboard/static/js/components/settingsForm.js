// Gesture-control settings generated from GET /api/settings `schema`.
// - groups as collapsible sections (Detection open), label + info tooltip,
//   slider + value + unit, a reset icon only when the value differs from its default
// - changes are staged locally; a sticky footer bar ("N unsaved changes · Discard · Apply")
//   appears only when something is pending; Apply sends only the changed keys (PUT)
// - "Reset all" (header tool) asks first, then POST /settings/reset {}
// - 422 errors are shown next to the offending setting; `applied` is watched after a save

import { api, detailKeys } from '../api.js';
import {
  el, setText, act, confirmDialog, placeholder, toast, uid, setEnabled, reportError, collapsible, infoTip, btn,
  iconBtn, icon, emptyState,
} from '../ui.js';
import * as fmt from '../format.js';

const REFRESH_MS = 10000;
const APPLIED_POLL_MS = 1500;
const APPLIED_WAIT_MS = 20000;

export function same(a, b) {
  if (typeof a === 'number' && typeof b === 'number') {
    return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
  }
  return a === b;
}

/**
 * exclude: schema keys rendered elsewhere (`paused` is the page's switch).
 * onData(answer): after every GET/PUT/reset answer. onLocalChange(): after any staged edit.
 */
export function createSettingsForm(scope, { exclude = ['paused'], onData, onLocalChange } = {}) {
  let schema = [];
  let defaults = {};
  let server = {};
  let local = {};
  let applied = null;
  let loaded = false;
  let saving = false;
  let lastError = null;
  let waitingApplied = false;
  let appliedTimedOut = false;
  let appliedPoll = null;
  let appliedDeadline = 0;
  let schemaKey = '';
  const rows = new Map();
  const invalid = new Map(); // key -> message

  // ---- header tools (go into the tab bar)
  const stateText = el('span', { class: 'settings-state', role: 'status', 'aria-live': 'polite' });
  const resetAllBtn = btn('Reset all', { kind: 'ghost', size: 'sm', icon: 'reset' });
  const tools = el('div', { class: 'row' }, stateText, resetAllBtn);

  // ---- body
  const body = el('div', { class: 'settings-groups' }, placeholder('loading', 'Loading settings…'));

  // ---- sticky footer (placed by the page at the bottom of the page)
  const footerText = el('span', { class: 'unsaved-text' });
  const footerErr = el('div', { class: 'unsaved-error', hidden: true });
  const discardBtn = btn('Discard', { kind: 'ghost' });
  const applyBtn = btn('Apply', { kind: 'primary' });
  const footer = el('div', { class: 'unsaved-bar', role: 'region', 'aria-label': 'Unsaved settings', hidden: true },
    icon('alert', { size: 18 }), el('div', { class: 'stack-sm unsaved-main' }, footerText, footerErr), discardBtn, applyBtn);

  applyBtn.addEventListener('click', () => apply());
  discardBtn.addEventListener('click', discard);
  resetAllBtn.addEventListener('click', resetAll);

  const visibleSchema = () => schema.filter((s) => s && s.key && !exclude.includes(s.key));

  function dirtyKeys() {
    return visibleSchema().map((s) => s.key).filter((k) => k in local && !same(local[k], server[k]));
  }

  function ingest(data, { force = null } = {}) {
    if (!data || typeof data !== 'object') return;
    const prevServer = server;
    schema = Array.isArray(data.schema) ? data.schema : schema;
    defaults = data.defaults && typeof data.defaults === 'object' ? data.defaults : defaults;
    server = data.values && typeof data.values === 'object' ? { ...data.values } : server;
    applied = typeof data.applied === 'boolean' ? data.applied : applied;
    loaded = true;
    for (const s of schema) {
      if (!s || !s.key) continue;
      const k = s.key;
      const wasDirty = k in local && !same(local[k], prevServer[k]);
      if (force === 'all' || (force && force.has(k)) || !wasDirty) {
        local[k] = server[k];
        invalid.delete(k);
      }
    }
    const key = visibleSchema().map((s) => `${s.key}:${s.type}:${s.min}:${s.max}:${s.step}:${s.group}`).join('|');
    if (key !== schemaKey) {
      schemaKey = key;
      build();
    } else {
      for (const k of rows.keys()) syncControls(k);
    }
    if (applied === true) {
      waitingApplied = false;
      appliedTimedOut = false;
      stopAppliedPoll();
    }
    renderState();
    if (onData) onData(data);
  }

  async function load() {
    try {
      const data = await api.get('/settings', null, { signal: scope.signal });
      if (scope.alive) ingest(data);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      if (!loaded) {
        body.replaceChildren(emptyState({
          kind: 'error', title: 'Could not load the settings', text: err.message,
          actions: btn('Retry', { kind: 'secondary', icon: 'refresh', onClick: () => load() }),
        }));
      }
    }
  }

  // ---- rendering

  function build() {
    rows.clear();
    const groups = new Map();
    for (const s of visibleSchema()) {
      const g = s.group || 'General';
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(s);
    }
    if (!groups.size) {
      body.replaceChildren(emptyState({ title: 'No adjustable settings', text: 'The bridge reports no gesture-control settings.' }));
      return;
    }
    let first = true;
    body.replaceChildren(...Array.from(groups, ([name, entries]) => {
      const sec = collapsible({ title: name, open: first || name === 'Detection', key: `settings.group.${name}`, meta: `${entries.length} setting${entries.length > 1 ? 's' : ''}` },
        ...entries.map(buildRow));
      first = false;
      return sec;
    }));
    for (const k of rows.keys()) syncControls(k);
  }

  function buildRow(s) {
    const k = s.key;
    const isBool = s.type === 'bool';
    const isInt = s.type === 'int';
    const step = fmt.isNum(s.step) && s.step > 0 ? s.step : (isInt ? 1 : 0.01);
    const decimals = isInt ? 0 : fmt.decimalsFor(step);
    const hasRange = fmt.isNum(s.min) && fmt.isNum(s.max) && s.max > s.min;
    const labelId = uid('set');
    const errId = uid('err');
    const labelText = s.label || k;
    const r = { entry: s, isBool, isInt, step, decimals };

    const name = el('div', { class: 'setting-name' },
      el('span', { id: labelId, text: labelText }),
      s.help ? infoTip(String(s.help), labelText) : null);

    let slider = el('span');
    let value;
    if (isBool) {
      r.check = el('button', { type: 'button', role: 'switch', class: 'switch', 'aria-labelledby': labelId, 'aria-describedby': errId, 'aria-checked': 'false' },
        el('span', { class: 'switch-thumb', 'aria-hidden': 'true' }));
      r.check.addEventListener('click', () => edit(k, r.check.getAttribute('aria-checked') !== 'true'));
      value = el('div', { class: 'setting-value' }, r.check);
    } else {
      if (hasRange) {
        r.range = el('input', {
          type: 'range', min: String(s.min), max: String(s.max), step: String(step), 'aria-labelledby': labelId, 'aria-describedby': errId,
        });
        r.range.addEventListener('input', () => edit(k, Number(r.range.value), false));
        slider = r.range;
      }
      r.number = el('input', {
        type: 'number', class: 'input input-sm input-num', step: String(step), inputMode: isInt ? 'numeric' : 'decimal',
        min: fmt.isNum(s.min) ? String(s.min) : null, max: fmt.isNum(s.max) ? String(s.max) : null,
        'aria-labelledby': labelId, 'aria-describedby': errId,
      });
      r.number.addEventListener('input', () => {
        const v = parseValue(r, r.number.value);
        if (v.error) {
          invalid.set(k, v.error);
          refreshRow(k);
          renderState();
        } else {
          edit(k, v.value, true);
        }
      });
      value = el('div', { class: 'setting-value' }, r.number, el('span', { class: 'setting-unit', text: s.unit ? String(s.unit) : '' }));
    }
    r.resetBtn = iconBtn('reset', `Reset ${labelText} to its default`, { size: 'sm', cls: 'setting-reset' });
    r.resetBtn.addEventListener('click', () => {
      if (defaults[k] !== undefined) edit(k, defaults[k], false);
    });
    r.errEl = el('div', { class: 'field-error setting-error', id: errId, role: 'alert', hidden: true });
    r.row = el('div', { class: 'setting-row', dataset: { key: k } }, name, slider, value, r.resetBtn, r.errEl);
    rows.set(k, r);
    return r.row;
  }

  function parseValue(r, raw) {
    const s = r.entry;
    if (raw === '' || raw == null) return { error: 'Enter a value.' };
    const v = Number(raw);
    if (!Number.isFinite(v)) return { error: 'Enter a number.' };
    if (r.isInt && !Number.isInteger(v)) return { error: 'Enter a whole number.' };
    if (fmt.isNum(s.min) && v < s.min) return { error: `Minimum is ${fmt.withUnit(s.min, s.unit)}.` };
    if (fmt.isNum(s.max) && v > s.max) return { error: `Maximum is ${fmt.withUnit(s.max, s.unit)}.` };
    return { value: v };
  }

  function display(r, v) {
    return fmt.isNum(v) ? String(fmt.roundTo(v, r.decimals)) : '';
  }

  function syncControls(k) {
    const r = rows.get(k);
    if (!r) return;
    const v = local[k];
    if (r.isBool) {
      r.check.setAttribute('aria-checked', String(!!v));
    } else {
      if (r.range && fmt.isNum(v)) r.range.value = String(v);
      if (document.activeElement !== r.number) r.number.value = display(r, v);
    }
    refreshRow(k);
  }

  function refreshRow(k) {
    const r = rows.get(k);
    if (!r) return;
    const def = defaults[k];
    const modified = def !== undefined && !same(local[k], def);
    const dirty = !same(local[k], server[k]);
    r.row.classList.toggle('is-modified', modified);
    r.row.classList.toggle('is-dirty', dirty);
    r.resetBtn.dataset.tip = def === undefined ? '' : `Reset to the default (${r.isBool ? (def ? 'on' : 'off') : fmt.withUnit(def, r.entry.unit, r.decimals)})`;
    const msg = invalid.get(k) || '';
    setText(r.errEl, msg);
    r.errEl.hidden = !msg;
    for (const c of [r.number, r.range, r.check]) {
      if (!c) continue;
      if (msg) c.setAttribute('aria-invalid', 'true');
      else c.removeAttribute('aria-invalid');
    }
  }

  function edit(k, value, fromNumber = false) {
    local[k] = value;
    invalid.delete(k);
    lastError = null;
    const r = rows.get(k);
    if (r && !r.isBool) {
      if (r.range) r.range.value = String(value);
      if (!fromNumber) r.number.value = display(r, value);
    } else if (r) {
      r.check.setAttribute('aria-checked', String(!!value));
    }
    refreshRow(k);
    renderState();
    if (onLocalChange) onLocalChange();
  }

  function renderState() {
    const dirty = dirtyKeys();
    const bad = Array.from(invalid.keys()).filter((k) => rows.has(k));
    footer.hidden = !(dirty.length || bad.length || (lastError && saving));
    setText(footerText, saving ? 'Applying…' : `${dirty.length} unsaved change${dirty.length === 1 ? '' : 's'}${bad.length ? ` · ${bad.length} invalid` : ''}`);
    footerErr.hidden = !lastError;
    setText(footerErr, lastError || '');
    if (lastError && dirty.length) footer.hidden = false;
    setEnabled(applyBtn, loaded && dirty.length > 0 && !saving && bad.length === 0);
    setEnabled(discardBtn, !saving);
    setEnabled(resetAllBtn, loaded && !saving);
    let text = '';
    if (!loaded) text = 'Loading…';
    else if (waitingApplied) text = 'Saved, waiting for the bridge…';
    else if (appliedTimedOut) text = 'Saved; the bridge has not confirmed yet';
    else if (applied === false) text = 'Not yet picked up by the bridge';
    else text = 'Applied by the bridge';
    setText(stateText, text);
  }

  // ---- actions

  function handleSaveError(err, sentKeys) {
    if (err.cancelled || err.aborted || err.unauthorized) return;
    if (err.status === 422) {
      const msg = err.message;
      let placed = false;
      for (const k of detailKeys(err.detail)) {
        if (rows.has(k)) {
          invalid.set(k, msg);
          placed = true;
        }
      }
      if (!placed) {
        for (const k of sentKeys) {
          const r = rows.get(k);
          if (msg.includes(k) || (r && r.entry.label && msg.includes(r.entry.label))) {
            invalid.set(k, msg);
            placed = true;
          }
        }
      }
      if (!placed && sentKeys.length === 1 && rows.has(sentKeys[0])) {
        invalid.set(sentKeys[0], msg);
        placed = true;
      }
      for (const k of rows.keys()) refreshRow(k);
      lastError = placed ? 'The server rejected a value (see the marked setting).' : msg;
    } else {
      lastError = err.message;
      reportError(err, 'Settings not applied');
    }
  }

  function watchApplied() {
    if (applied !== false) {
      waitingApplied = false;
      return;
    }
    waitingApplied = true;
    appliedTimedOut = false;
    appliedDeadline = Date.now() + APPLIED_WAIT_MS;
    if (appliedPoll) return;
    appliedPoll = scope.interval(async () => {
      if (Date.now() > appliedDeadline) {
        stopAppliedPoll();
        waitingApplied = false;
        appliedTimedOut = true;
        renderState();
        return;
      }
      try {
        const data = await api.get('/settings', null, { signal: scope.signal, timeout: 5000 });
        if (scope.alive) ingest(data);
      } catch {
        /* keep waiting until the deadline */
      }
    }, APPLIED_POLL_MS);
  }

  function stopAppliedPoll() {
    if (appliedPoll) clearInterval(appliedPoll);
    appliedPoll = null;
  }

  /** PUT values immediately (used by Apply and by the page's pause switch). Throws on error. */
  async function save(values) {
    const keys = Object.keys(values);
    const data = await api.put('/settings', { values }, { signal: scope.signal });
    if (!scope.alive) return data;
    ingest(data, { force: new Set(keys) });
    watchApplied();
    renderState();
    return data;
  }

  async function apply() {
    if (saving || !loaded) return;
    const keys = dirtyKeys();
    if (!keys.length || keys.some((k) => invalid.has(k))) return;
    const values = {};
    for (const k of keys) values[k] = local[k];
    saving = true;
    lastError = null;
    renderState();
    await act(applyBtn, async () => {
      try {
        await save(values);
        toast(`${keys.length} setting${keys.length > 1 ? 's' : ''} applied`, 'success');
      } catch (err) {
        handleSaveError(err, keys);
      }
    });
    saving = false;
    if (scope.alive) {
      renderState();
      if (onLocalChange) onLocalChange();
    }
  }

  function discard() {
    for (const s of visibleSchema()) {
      local[s.key] = server[s.key];
      invalid.delete(s.key);
    }
    lastError = null;
    for (const k of rows.keys()) syncControls(k);
    renderState();
    if (onLocalChange) onLocalChange();
  }

  async function resetAll() {
    const yes = await confirmDialog({
      title: 'Reset all settings?',
      message: 'Every gesture-control setting goes back to its tuned default and is applied now. Unsaved changes are lost.',
      confirmLabel: 'Reset all',
      risk: 'warn',
    });
    if (!yes || !scope.alive) return;
    const data = await act(resetAllBtn, () => api.post('/settings/reset', {}, { signal: scope.signal }));
    if (!data || !scope.alive) return;
    invalid.clear();
    lastError = null;
    ingest(data, { force: 'all' });
    watchApplied();
    renderState();
    if (onLocalChange) onLocalChange();
    toast('All settings reset to defaults', 'success');
  }

  scope.poll(async () => {
    if (saving) return;
    await load();
  }, REFRESH_MS);

  return {
    el: body,
    tools,
    footer,
    save,
    reload: load,
    get loaded() {
      return loaded;
    },
    /** Current staged values (local), defaults and schema rows (for presets). */
    locals: () => ({ ...local }),
    defaults: () => ({ ...defaults }),
    schemaRows: () => visibleSchema(),
    /** Stage several values at once (shown as unsaved until Apply). */
    stage(values) {
      for (const [k, v] of Object.entries(values)) {
        if (!rows.has(k)) continue;
        local[k] = v;
        invalid.delete(k);
        syncControls(k);
      }
      lastError = null;
      renderState();
      if (onLocalChange) onLocalChange();
    },
  };
}
