// Security: tabs [Audit log] [Account].

import { api } from '../api.js';
import {
  el, setText, card, act, btn, field, setFieldError, dataTable, select, toast, status, createTabs, input, kv, icon, iconBtn,
} from '../ui.js';
import { strengthMeter } from '../auth.js';
import * as fmt from '../format.js';

const AUDIT_MS = 15000;
const MIN_PASSWORD = 10;

function paramsText(p) {
  if (p == null || p === '') return '';
  if (typeof p === 'string') return p;
  try {
    return JSON.stringify(p);
  } catch {
    return String(p);
  }
}

export async function mount(root, scope, app) {
  const tabs = createTabs({
    key: 'tab.security',
    label: 'Security sections',
    items: [{ value: 'audit', label: 'Audit log', icon: 'history' }, { value: 'account', label: 'Account', icon: 'user' }],
  });
  root.append(el('div', { class: 'tabbar' }, tabs.el), tabs.panel('audit'), tabs.panel('account'));
  scope.sub('open-account', () => tabs.select('account'));

  // ---------------------------------------------------------------- audit
  let entries = [];
  let limit = 200;
  const filterInput = input({ type: 'search', placeholder: 'Filter by user, action or text', 'aria-label': 'Filter the audit log' });
  filterInput.classList.add('input-sm');
  const limitSel = select([50, 200, 500].map((n) => ({ value: n, label: `Last ${n}` })), {
    value: limit, ariaLabel: 'Number of audit entries', cls: 'input-sm',
    onChange: (v) => { limit = Number(v) || 200; auditPoll.now(); },
  });
  const auditInfo = el('span', { class: 'small muted tabular' });
  const refresh = iconBtn('refresh', 'Refresh');
  const auditTable = dataTable({
    caption: 'Audit log, newest first',
    empty: 'No audit entries yet.',
    columns: [
      { key: 't', label: 'Time', render: (e) => el('span', { class: 'tabular muted', text: fmt.dateTime(e.t) }) },
      { key: 'user', label: 'User' },
      { key: 'action', label: 'Action', render: (e) => el('span', { class: 'mono', text: e.action || '?' }) },
      {
        key: 'params', label: 'Details',
        render: (e) => {
          const s = paramsText(e.params);
          if (!s || s === '{}') return el('span', { class: 'faint', text: '–' });
          if (s.length <= 90) return el('code', { class: 'break', text: s });
          return el('details', null, el('summary', { class: 'mono small', text: `${s.slice(0, 90)}…` }), el('code', { class: 'break', text: s }));
        },
      },
      { key: 'result', label: 'Result', render: (e) => el('span', { class: 'muted', text: fmt.describe(e.result, 120) }) },
    ],
  });
  tabs.panel('audit').append(el('section', { class: 'card card-flush' },
    el('div', { class: 'card-head' }, el('div', { class: 'card-heading' }, el('h2', { class: 'card-title', text: 'Audit log' }), auditInfo),
      el('div', { class: 'card-actions' }, filterInput, limitSel, refresh)),
    auditTable.el));

  function renderAudit() {
    const q = filterInput.value.trim().toLowerCase();
    const rows = q ? entries.filter((e) => [e.user, e.action, paramsText(e.params), fmt.describe(e.result, 400)]
      .some((v) => String(v ?? '').toLowerCase().includes(q))) : entries;
    auditTable.setRows(rows);
    setText(auditInfo, q ? `${rows.length} of ${entries.length}` : `${entries.length} entries`);
  }

  async function loadAudit() {
    try {
      const res = await api.get('/audit', { limit }, { signal: scope.signal, timeout: 10000 });
      if (!scope.alive) return;
      entries = Array.isArray(res.entries) ? res.entries : [];
      renderAudit();
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      if (!entries.length) {
        auditTable.setEmpty(`Could not load the audit log: ${err.message}`);
        auditTable.setRows([]);
      } else {
        setText(auditInfo, `refresh failed: ${err.message}`);
      }
    }
  }
  filterInput.addEventListener('input', renderAudit);
  refresh.addEventListener('click', () => act(refresh, loadAudit));
  const auditPoll = scope.poll(loadAudit, AUDIT_MS);

  // ---------------------------------------------------------------- account
  const sessionBody = el('div');
  const logoutBtn = btn('Log out', { kind: 'secondary', icon: 'logout' });
  logoutBtn.addEventListener('click', () => act(logoutBtn, () => app.logout()));
  function renderSession(health) {
    sessionBody.replaceChildren(kv([
      ['Logged in as', app.user() || fmt.DASH],
      ['Server', health ? status(health.ok ? 'ok' : 'bad', health.ok ? 'Healthy' : 'Not healthy') : fmt.DASH],
      ['Version', health && health.version != null ? String(health.version) : fmt.DASH],
      ['Uptime', health && fmt.isNum(health.uptime_s) ? fmt.duration(health.uptime_s) : fmt.DASH],
    ]));
  }
  renderSession(null);
  api.get('/health', null, { signal: scope.signal, timeout: 5000 }).then((h) => scope.alive && renderSession(h)).catch(() => {});

  const oldPw = input({ type: 'password', autocomplete: 'current-password' });
  const newPw = input({ type: 'password', autocomplete: 'new-password', minLength: MIN_PASSWORD });
  const repeatPw = input({ type: 'password', autocomplete: 'new-password', minLength: MIN_PASSWORD });
  const oldField = field('Current password', oldPw);
  const newField = field('New password', newPw, { hint: `At least ${MIN_PASSWORD} characters.` });
  const repeatField = field('Repeat the new password', repeatPw);
  const meter = strengthMeter(newPw);
  newField.insertBefore(meter.el, newField.querySelector('.field-hint'));
  const pwError = el('div', { class: 'form-error', role: 'alert', hidden: true });
  const pwBtn = btn('Change password', { kind: 'primary', type: 'submit' });
  const pwForm = el('form', { class: 'stack', novalidate: true }, oldField, newField, repeatField, pwError, el('div', { class: 'form-actions' }, pwBtn));

  tabs.panel('account').append(el('div', { class: 'security-grid' },
    card({ title: 'Session' }, el('div', { class: 'stack' }, sessionBody,
      el('p', { class: 'small muted', text: 'The session cookie is HttpOnly and SameSite=Strict; every change request carries a CSRF token kept in memory only.' }),
      el('div', { class: 'form-actions' }, logoutBtn))),
    card({ title: 'Change password' }, pwForm)));

  pwForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    pwError.hidden = true;
    let ok = true;
    setFieldError(oldField, oldPw.value ? '' : 'Enter your current password.');
    if (!oldPw.value) ok = false;
    if (newPw.value.length < MIN_PASSWORD) {
      setFieldError(newField, `The new password needs at least ${MIN_PASSWORD} characters.`);
      ok = false;
    } else if (newPw.value === oldPw.value) {
      setFieldError(newField, 'The new password must differ from the current one.');
      ok = false;
    } else {
      setFieldError(newField, '');
    }
    if (repeatPw.value !== newPw.value) {
      setFieldError(repeatField, 'The passwords do not match.');
      ok = false;
    } else {
      setFieldError(repeatField, '');
    }
    if (!ok) return;
    const res = await act(pwBtn, () => api.post('/auth/password', { old: oldPw.value, new: newPw.value }, { signal: scope.signal }), {
      onError: (err) => {
        if (err.aborted || err.cancelled) return true;
        pwError.replaceChildren(icon('alert', { size: 18 }), el('span', { text: `Not changed: ${err.message}` }));
        pwError.hidden = false;
        return true;
      },
    });
    if (!res || !scope.alive) return;
    oldPw.value = '';
    newPw.value = '';
    repeatPw.value = '';
    meter.update();
    toast('Password changed', 'success');
  });
}
