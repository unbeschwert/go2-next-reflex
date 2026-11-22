// Login and first-run setup screens (centered forms).

import { api } from './api.js';
import { el, field, setFieldError, setText, icon, btn, setBtnLabel } from './ui.js';

const MIN_PASSWORD = 10;

/** Rough strength estimate: returns {score 0..4, label}. Shared with the password-change form. */
export function passwordStrength(pw) {
  if (!pw) return { score: 0, label: 'Enter a password' };
  if (pw.length < MIN_PASSWORD) return { score: 0, label: `Too short: ${pw.length}/${MIN_PASSWORD} characters` };
  let score = 1;
  if (pw.length >= 14) score += 1;
  if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) score += 1;
  if (/\d/.test(pw)) score += 0.5;
  if (/[^A-Za-z0-9]/.test(pw)) score += 0.5;
  if (/^(.)\1+$/.test(pw) || /^(?:1234567890|password|qwerty)/i.test(pw)) score = 1;
  const s = Math.min(4, Math.floor(score));
  const labels = ['Too short', 'Weak: use more characters or mix letters, digits, symbols', 'Fair', 'Good', 'Strong'];
  return { score: s, label: labels[s] };
}

/** Password input + strength meter. Returns {wrap, input, update()}. */
export function strengthMeter(input) {
  const meter = el('meter', { class: 'strength', min: 0, max: 4, low: 2, high: 3, optimum: 4, value: 0, 'aria-label': 'Password strength' });
  const text = el('span', { class: 'small muted', 'aria-live': 'polite' });
  const update = () => {
    const s = passwordStrength(input.value);
    meter.value = s.score;
    setText(text, s.label);
  };
  input.addEventListener('input', update);
  update();
  return { el: el('div', { class: 'strength-row' }, meter, text), update };
}

function brand(subtitle) {
  return el('div', { class: 'auth-brand' },
    el('span', { class: 'logo-mark logo-mark-lg', 'aria-hidden': 'true' }, icon('activity', { size: 26 })),
    el('h1', { class: 'auth-title', text: 'Durance AI' }),
    el('p', { class: 'auth-subtitle', text: subtitle }));
}

function errorBox() {
  const text = el('span');
  const box = el('div', { class: 'form-error', role: 'alert', hidden: true }, icon('alert', { size: 18 }), text);
  box.textEl = text;
  return box;
}

/**
 * Render login (or setup when state.setup_needed) into container.
 * onDone(authState) is called after a successful login/setup.
 */
export function renderAuth(container, state, onDone) {
  container.replaceChildren(state && state.setup_needed ? setupForm(container, onDone) : loginForm(onDone));
  const first = container.querySelector('input');
  if (first) first.focus();
}

function loginForm(onDone) {
  const user = el('input', { class: 'input', type: 'text', name: 'username', autocomplete: 'username', required: true, attrs: { autocapitalize: 'off', spellcheck: 'false' } });
  const pass = el('input', { class: 'input', type: 'password', name: 'password', autocomplete: 'current-password', required: true });
  const error = errorBox();
  const submit = btn('Log in', { kind: 'primary', type: 'submit', size: 'lg' });
  const form = el('form', { class: 'auth-card', novalidate: true },
    brand('Gesture-controlled Go2 dashboard'),
    field('Username', user),
    field('Password', pass),
    error,
    submit);

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (submit.disabled) return;
    error.hidden = true;
    if (!user.value.trim() || !pass.value) {
      setText(error.textEl, 'Enter your username and password.');
      error.hidden = false;
      return;
    }
    submit.disabled = true;
    setBtnLabel(submit, 'Logging in…');
    try {
      const st = await api.post('/auth/login', { username: user.value.trim(), password: pass.value });
      pass.value = '';
      onDone(st);
    } catch (err) {
      let msg = err.message;
      if (err.status === 429) msg = `Too many attempts. ${err.message}`;
      else if (err.status === 401 || err.status === 403) msg = err.message && !/HTTP/.test(err.message) ? err.message : 'Wrong username or password.';
      setText(error.textEl, msg);
      error.hidden = false;
      pass.select();
    } finally {
      submit.disabled = false;
      setBtnLabel(submit, 'Log in');
    }
  });
  return form;
}

function setupForm(container, onDone) {
  const user = el('input', { class: 'input', type: 'text', name: 'username', autocomplete: 'username', required: true, attrs: { autocapitalize: 'off', spellcheck: 'false' } });
  const pass = el('input', { class: 'input', type: 'password', name: 'new-password', autocomplete: 'new-password', required: true, minLength: MIN_PASSWORD });
  const repeat = el('input', { class: 'input', type: 'password', name: 'repeat-password', autocomplete: 'new-password', required: true, minLength: MIN_PASSWORD });
  const meter = strengthMeter(pass);
  const userField = field('Username', user);
  const passField = field('Password', pass, { hint: `At least ${MIN_PASSWORD} characters. A passphrase of several words works well.` });
  const repeatField = field('Repeat password', repeat);
  passField.insertBefore(meter.el, passField.querySelector('.field-hint'));
  const error = errorBox();
  const submit = btn('Create account', { kind: 'primary', type: 'submit', size: 'lg' });
  const form = el('form', { class: 'auth-card', novalidate: true },
    brand('First run: create the operator account. It is needed for every later login.'),
    userField, passField, repeatField, error, submit);

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (submit.disabled) return;
    error.hidden = true;
    let ok = true;
    setFieldError(userField, user.value.trim() ? '' : 'Choose a username.');
    if (!user.value.trim()) ok = false;
    if (pass.value.length < MIN_PASSWORD) {
      setFieldError(passField, `The password needs at least ${MIN_PASSWORD} characters.`);
      ok = false;
    } else {
      setFieldError(passField, '');
    }
    if (repeat.value !== pass.value) {
      setFieldError(repeatField, 'The passwords do not match.');
      ok = false;
    } else {
      setFieldError(repeatField, '');
    }
    if (!ok) return;
    submit.disabled = true;
    setBtnLabel(submit, 'Creating…');
    try {
      const st = await api.post('/auth/setup', { username: user.value.trim(), password: pass.value });
      pass.value = '';
      repeat.value = '';
      if (st && st.logged_in) onDone(st);
      else renderAuth(container, { setup_needed: false }, onDone);
    } catch (err) {
      setText(error.textEl, err.message);
      error.hidden = false;
    } finally {
      submit.disabled = false;
      setBtnLabel(submit, 'Create account');
    }
  });
  return form;
}
