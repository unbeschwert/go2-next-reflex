// UI kit: DOM builder, buttons, status dots, switches, tabs, collapsibles, menus,
// drawers, popovers, tooltips, toasts and dialogs.
//
// Rule for the whole UI: text that comes from the server (API answers, logs,
// file names, user names) only ever reaches the DOM through textContent.
// el() has no innerHTML path and refuses inline style / on* attributes (the
// Content-Security-Policy would block them anyway); styles are set via CSSOM.

import { icon } from './icons.js';

let uidCounter = 0;

/** Unique id for label/aria wiring. */
export function uid(prefix = 'id') {
  uidCounter += 1;
  return `${prefix}-${uidCounter}`;
}

const ATTR_KEYS = new Set(['role', 'for', 'tabindex', 'colspan', 'rowspan', 'scope', 'list', 'form', 'inputmode', 'autocomplete', 'enterkeyhint']);

function applyProps(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (key.startsWith('aria-')) {
      if (value != null) node.setAttribute(key, String(value));
      continue;
    }
    if (value == null || value === false) continue;
    if (key === 'innerHTML' || key === 'outerHTML') throw new Error('el(): innerHTML is not allowed');
    if (key === 'class' || key === 'className') {
      node.className = Array.isArray(value) ? value.filter(Boolean).join(' ') : String(value);
    } else if (key === 'text') {
      node.textContent = String(value);
    } else if (key === 'style') {
      for (const [prop, v] of Object.entries(value)) {
        if (prop.startsWith('--')) node.style.setProperty(prop, String(v));
        else node.style[prop] = v;
      }
    } else if (key === 'dataset') {
      Object.assign(node.dataset, value);
    } else if (key === 'on') {
      for (const [type, fn] of Object.entries(value)) node.addEventListener(type, fn);
    } else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'attrs') {
      for (const [a, v] of Object.entries(value)) {
        if (a === 'style' || a.toLowerCase().startsWith('on')) throw new Error(`el(): attribute ${a} is not allowed`);
        if (v != null && v !== false) node.setAttribute(a, v === true ? '' : String(v));
      }
    } else if (ATTR_KEYS.has(key) || key.startsWith('data-')) {
      node.setAttribute(key, value === true ? '' : String(value));
    } else if (key in node) {
      node[key] = value;
    } else {
      node.setAttribute(key, value === true ? '' : String(value));
    }
  }
}

/** Append children: strings become text nodes, arrays are flattened, null/false skipped. */
export function append(node, children) {
  for (const child of children) {
    if (child == null || child === false) continue;
    if (Array.isArray(child)) append(node, child);
    else if (child instanceof Node) node.append(child);
    else node.append(document.createTextNode(String(child)));
  }
  return node;
}

/** el('button', {class: 'btn', onclick: fn, 'aria-label': 'x'}, 'Text', otherNode) */
export function el(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) applyProps(node, props);
  append(node, children);
  return node;
}

/** Set textContent only when it changes (cheap for 10 Hz updates). */
export function setText(node, text) {
  const s = text == null ? '' : String(text);
  if (node.textContent !== s) node.textContent = s;
}

export { icon };

// ---------------------------------------------------------------- prefs & misc

const PREF_PREFIX = 'durance.';

/** localStorage for UI preferences only (theme, last page, speeds). Never secrets. */
export const prefs = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(PREF_PREFIX + key);
      return raw == null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(PREF_PREFIX + key, JSON.stringify(value));
    } catch {
      /* storage blocked: preference is simply not remembered */
    }
  },
};

export function debounce(fn, wait) {
  let timer = null;
  const d = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, wait);
  };
  d.cancel = () => {
    clearTimeout(timer);
    timer = null;
  };
  return d;
}

/** True when the keyboard event target is a text-entry control. */
export function isTypingTarget(target) {
  if (!(target instanceof Element)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag === 'INPUT') {
    const type = (target.getAttribute('type') || 'text').toLowerCase();
    return !['checkbox', 'radio', 'range', 'button', 'submit', 'reset', 'color', 'file'].includes(type);
  }
  return false;
}

export function anyDialogOpen() {
  return document.querySelector('dialog[open]') !== null;
}

/** Shared filter for page keyboard shortcuts. */
export function shortcutAllowed(ev) {
  return !ev.ctrlKey && !ev.metaKey && !ev.altKey && !isTypingTarget(ev.target) && !anyDialogOpen()
    && !(ev.target instanceof Element && ev.target.closest('.rviz-screen'));
}

/**
 * Enable/disable a button without fighting act(): while a request runs, the
 * wanted state is remembered and applied when the request finishes.
 */
export function setEnabled(button, enabled) {
  if (!button) return;
  if (button.dataset.busy === '1') {
    button.dataset.wasDisabled = enabled ? '' : '1';
    return;
  }
  button.disabled = !enabled;
}

// ---------------------------------------------------------------- transient UI (menus, popovers, tooltip)

const transient = new Set();

/** Close every open menu / popover / tooltip (route change, Escape). */
export function closeTransient() {
  for (const close of Array.from(transient)) close();
  hideTip();
}

// ---------------------------------------------------------------- tooltip

let tipEl = null;
let tipTarget = null;
let tipRaf = 0;

function ensureTip() {
  if (!tipEl) {
    tipEl = el('div', { class: 'tooltip', id: 'ui-tooltip', role: 'tooltip', hidden: true });
    document.body.append(tipEl);
    window.addEventListener('scroll', hideTip, true);
  }
  return tipEl;
}

function hideTip() {
  if (!tipEl || tipEl.hidden) return;
  tipEl.hidden = true;
  cancelAnimationFrame(tipRaf);
  if (tipTarget) {
    if (tipTarget.dataset.prevDescribedby) tipTarget.setAttribute('aria-describedby', tipTarget.dataset.prevDescribedby);
    else tipTarget.removeAttribute('aria-describedby');
    delete tipTarget.dataset.prevDescribedby;
  }
  tipTarget = null;
}

function showTip(target) {
  const text = target.dataset.tip;
  if (!text) return;
  const t = ensureTip();
  if (tipTarget && tipTarget !== target) hideTip();
  setText(t, text);
  t.hidden = false;
  tipTarget = target;
  const prev = target.getAttribute('aria-describedby');
  if (prev && prev !== 'ui-tooltip') target.dataset.prevDescribedby = prev;
  target.setAttribute('aria-describedby', 'ui-tooltip');
  const r = target.getBoundingClientRect();
  const tw = t.offsetWidth;
  const th = t.offsetHeight;
  let x = r.left + r.width / 2 - tw / 2;
  x = Math.max(8, Math.min(window.innerWidth - tw - 8, x));
  let y = r.top - th - 8;
  if (y < 8) y = r.bottom + 8;
  t.style.left = `${Math.round(x)}px`;
  t.style.top = `${Math.round(y)}px`;
  const watch = () => {
    if (!tipTarget) return;
    if (!tipTarget.isConnected) {
      hideTip();
      return;
    }
    tipRaf = requestAnimationFrame(watch);
  };
  tipRaf = requestAnimationFrame(watch);
}

/** Attach a hover/focus tooltip (text only). Call again to change the text. */
export function attachTip(target, text) {
  if (!target) return target;
  target.dataset.tip = text || '';
  if (target.dataset.tipBound) return target;
  target.dataset.tipBound = '1';
  target.addEventListener('mouseenter', () => showTip(target));
  target.addEventListener('mouseleave', hideTip);
  target.addEventListener('focus', () => {
    if (target.matches(':focus-visible')) showTip(target);
  });
  target.addEventListener('blur', hideTip);
  target.addEventListener('pointerdown', hideTip);
  return target;
}

/** Small info icon that shows help text on hover / keyboard focus. */
export function infoTip(text, label = 'More information') {
  const b = el('button', { type: 'button', class: 'info-tip', 'aria-label': `${label}: ${text}` }, icon('info', { size: 15 }));
  attachTip(b, text);
  return b;
}

// ---------------------------------------------------------------- buttons

/**
 * btn('Save', {kind: 'primary'|'secondary'|'ghost'|'danger'|'danger-solid', icon, size: 'sm'|'lg',
 *              tip, ariaLabel, onClick, iconOnly, pressed, cls, type, disabled})
 */
export function btn(label, opts = {}) {
  const {
    kind = 'secondary', icon: ic, size, tip, ariaLabel, onClick, iconOnly = false, pressed, cls, type = 'button', disabled,
  } = opts;
  const b = el('button', {
    type,
    class: ['btn', `btn-${kind}`, size && `btn-${size}`, iconOnly && 'btn-icon', cls],
    disabled: !!disabled,
    'aria-label': iconOnly ? (ariaLabel || label) : ariaLabel,
    'aria-pressed': pressed == null ? null : String(!!pressed),
  }, ic ? icon(ic, { size: size === 'lg' ? 20 : 18 }) : null, iconOnly ? null : el('span', { class: 'btn-label', text: label }));
  if (onClick) b.addEventListener('click', onClick);
  if (tip || iconOnly) attachTip(b, tip || label);
  return b;
}

/** Ghost icon-only button with a tooltip. */
export function iconBtn(ic, label, opts = {}) {
  return btn(label, { kind: 'ghost', ...opts, icon: ic, iconOnly: true });
}

/** Replace the visible label of a btn(). */
export function setBtnLabel(b, text) {
  const l = b.querySelector('.btn-label');
  if (l) setText(l, text);
  else setText(b, text);
}

export function kbd(text) {
  return el('kbd', { text });
}

// ---------------------------------------------------------------- status

const STATE_WORD = { ok: 'OK', warn: 'Warning', bad: 'Problem', off: 'Inactive', info: 'Info', busy: 'Busy', real: 'Real robot' };

/** Status line: 8px dot + text (colour is never the only signal: the text says it). */
export function status(state, text, { tip } = {}) {
  const node = el('span', { class: 'status' }, el('span', { class: 'dot', 'aria-hidden': 'true' }), el('span', { class: 'status-text' }));
  setStatus(node, state, text, tip);
  return node;
}

export function setStatus(node, state, text, tip) {
  const s = STATE_WORD[state] ? state : 'off';
  const cls = `status status-${s}`;
  if (node.className !== cls) node.className = cls;
  setText(node.lastChild, text);
  if (tip !== undefined) attachTip(node, tip);
}

/** Boolean -> status state. */
export function boolState(v) {
  if (v === true) return 'ok';
  if (v === false) return 'bad';
  return 'off';
}

/** Capsule for exceptions only (Locked, REAL ROBOT, E-STOP active). */
export function chip(text, { kind = 'neutral', icon: ic } = {}) {
  return el('span', { class: ['chip', `chip-${kind}`] }, ic ? icon(ic, { size: 14 }) : null, el('span', { text }));
}

// ---------------------------------------------------------------- toasts

const TOAST_ICON = { info: 'info', success: 'check', warn: 'alert', error: 'alert', danger: 'octagon' };
const TOAST_MAX = 4;

export function toast(message, level = 'info', { timeout } = {}) {
  const host = document.getElementById('toasts');
  if (!host) return;
  const lvl = TOAST_ICON[level] ? level : 'info';
  const urgent = lvl === 'error' || lvl === 'danger';
  const close = el('button', { type: 'button', class: 'toast-close', 'aria-label': 'Dismiss notification' }, icon('x', { size: 16 }));
  const node = el('div', { class: ['toast', `toast-${lvl}`], role: urgent ? 'alert' : 'status' },
    icon(TOAST_ICON[lvl], { size: 18, cls: 'toast-icon' }),
    el('span', { class: 'toast-text', text: message }),
    close);
  let timer = null;
  const remove = () => {
    clearTimeout(timer);
    node.remove();
  };
  close.addEventListener('click', remove);
  host.prepend(node);
  while (host.children.length > TOAST_MAX) host.lastElementChild.remove();
  const ms = timeout ?? (urgent ? 9000 : 4000);
  if (ms > 0) timer = setTimeout(remove, ms);
}

/** Toast an error unless it is a cancelled confirm, an abort or a 401 (login screen shows). */
export function reportError(err, prefix) {
  if (!err) return;
  if (err.aborted || err.cancelled || err.unauthorized) return;
  const msg = err.message || String(err);
  toast(prefix ? `${prefix}: ${msg}` : msg, 'error');
}

/**
 * Run an async action tied to one or more buttons: the buttons are disabled
 * (no double submit) while it runs, errors are reported as a toast.
 * Returns the action's result, or undefined when it failed / was cancelled.
 */
export async function act(target, fn, { success, errorPrefix, onError } = {}) {
  const buttons = (Array.isArray(target) ? target : [target]).filter(Boolean);
  if (buttons.some((b) => b.dataset.busy === '1')) return undefined;
  for (const b of buttons) {
    b.dataset.wasDisabled = b.disabled ? '1' : '';
    b.dataset.busy = '1';
    b.disabled = true;
    b.setAttribute('aria-busy', 'true');
  }
  try {
    const result = await fn();
    if (success) toast(typeof success === 'function' ? success(result) : success, 'success');
    return result;
  } catch (err) {
    if (onError && onError(err) === true) return undefined;
    reportError(err, errorPrefix);
    return undefined;
  } finally {
    for (const b of buttons) {
      const was = b.dataset.wasDisabled === '1';
      delete b.dataset.busy;
      delete b.dataset.wasDisabled;
      b.disabled = was;
      b.removeAttribute('aria-busy');
    }
  }
}

// ---------------------------------------------------------------- dialogs (modal)

function riskKind(risk) {
  if (risk === 'danger') return 'danger'; // red outline: red fill is reserved for E-STOP and the drive STOP
  if (risk === 'warn') return 'warn';
  return 'primary';
}

/**
 * Modal confirm. Resolves true only when the user presses Confirm.
 * Escape / Cancel / closing resolve false. Focus starts on Cancel.
 */
export function confirmDialog({
  title = 'Please confirm', message, detail, confirmLabel = 'Confirm', cancelLabel = 'Cancel', risk = 'warn', body,
} = {}) {
  return new Promise((resolve) => {
    const titleId = uid('dlg-title');
    const msgId = uid('dlg-msg');
    const cancelBtn = btn(cancelLabel, { kind: 'secondary' });
    const confirmBtn = btn(confirmLabel, { kind: riskKind(risk) });
    const dlg = el('dialog', { class: ['dialog', `dialog-${risk}`], 'aria-labelledby': titleId, 'aria-describedby': msgId },
      el('div', { class: 'dialog-head' },
        risk === 'danger' || risk === 'warn' ? icon(risk === 'danger' ? 'octagon' : 'alert', { size: 22, cls: 'dialog-icon' }) : null,
        el('h2', { id: titleId, class: 'dialog-title', text: title })),
      el('p', { id: msgId, class: 'dialog-message', text: message || '' }),
      detail && detail !== message ? el('p', { class: 'dialog-detail', text: detail }) : null,
      body ? el('div', { class: 'dialog-body' }, body) : null,
      el('div', { class: 'dialog-actions' }, cancelBtn, confirmBtn));
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      if (dlg.open) dlg.close();
      dlg.remove();
      resolve(value);
    };
    cancelBtn.addEventListener('click', () => finish(false));
    confirmBtn.addEventListener('click', () => finish(true));
    dlg.addEventListener('cancel', () => finish(false));
    dlg.addEventListener('close', () => finish(false));
    document.body.append(dlg);
    closeTransient();
    dlg.showModal();
    cancelBtn.focus();
  });
}

// ---------------------------------------------------------------- menus & popovers (non-modal)

function placeBelow(anchor, panel, align = 'end') {
  const r = anchor.getBoundingClientRect();
  const pw = panel.offsetWidth;
  const ph = panel.offsetHeight;
  let x = align === 'end' ? r.right - pw : r.left;
  x = Math.max(8, Math.min(window.innerWidth - pw - 8, x));
  let y = r.bottom + 6;
  if (y + ph > window.innerHeight - 8 && r.top - ph - 6 > 8) y = r.top - ph - 6;
  panel.style.left = `${Math.round(x)}px`;
  panel.style.top = `${Math.round(y)}px`;
}

/**
 * Toggle a floating panel from a button (non-modal; closes on outside click,
 * scroll, route change and Escape). panel: element or () => element.
 * Returns {open, close, isOpen}.
 */
export function bindPopover(anchor, makePanel, { align = 'end', onOpen, onClose, cls = 'popover' } = {}) {
  let panel = null;
  const outside = (e) => {
    if (panel && !panel.contains(e.target) && !anchor.contains(e.target)) close();
  };
  const onScroll = (e) => {
    if (panel && !panel.contains(e.target)) close();
  };
  function close() {
    if (!panel) return;
    panel.remove();
    panel = null;
    anchor.setAttribute('aria-expanded', 'false');
    transient.delete(close);
    document.removeEventListener('pointerdown', outside, true);
    window.removeEventListener('scroll', onScroll, true);
    window.removeEventListener('resize', close);
    if (onClose) onClose();
  }
  function open() {
    if (panel) return;
    closeTransient();
    const content = typeof makePanel === 'function' ? makePanel(close) : makePanel;
    panel = el('div', { class: cls }, content);
    document.body.append(panel);
    placeBelow(anchor, panel, align);
    anchor.setAttribute('aria-expanded', 'true');
    transient.add(close);
    document.addEventListener('pointerdown', outside, true);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', close);
    if (onOpen) onOpen(panel);
  }
  anchor.setAttribute('aria-expanded', 'false');
  anchor.addEventListener('click', () => (panel ? close() : open()));
  return { open, close, isOpen: () => !!panel, reposition: () => panel && placeBelow(anchor, panel, align) };
}

/**
 * "More" menu button. items: [{label, icon, danger, disabled, onSelect(button)} | 'sep'] or a function returning them.
 */
export function menuButton(items, { label = 'More actions', icon: ic = 'more', kind = 'ghost', text } = {}) {
  const b = text
    ? btn(text, { kind, icon: ic })
    : btn(label, { kind, icon: ic, iconOnly: true });
  b.setAttribute('aria-haspopup', 'menu');
  const pop = bindPopover(b, (close) => {
    const list = (typeof items === 'function' ? items() : items).filter(Boolean);
    const menu = el('div', { role: 'menu', class: 'menu', 'aria-label': label });
    for (const it of list) {
      if (it === 'sep') {
        menu.append(el('div', { class: 'menu-sep', role: 'separator' }));
        continue;
      }
      const mi = el('button', {
        type: 'button', role: 'menuitem', class: ['menu-item', it.danger && 'is-danger'], disabled: !!it.disabled,
      }, it.icon ? icon(it.icon, { size: 16 }) : el('span', { class: 'menu-icon-gap' }), el('span', { text: it.label }));
      mi.addEventListener('click', () => {
        close();
        it.onSelect(b);
      });
      menu.append(mi);
    }
    menu.addEventListener('keydown', (ev) => {
      const itemsEls = Array.from(menu.querySelectorAll('.menu-item:not(:disabled)'));
      const i = itemsEls.indexOf(document.activeElement);
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
        ev.preventDefault();
        const n = itemsEls.length;
        if (!n) return;
        itemsEls[(i + (ev.key === 'ArrowDown' ? 1 : n - 1) + n) % n].focus();
      } else if (ev.key === 'Tab') {
        close();
      }
    });
    return menu;
  }, {
    cls: 'popover popover-menu',
    onOpen: (panel) => {
      const first = panel.querySelector('.menu-item:not(:disabled)');
      if (first) first.focus();
    },
  });
  b.closeMenu = pop.close;
  return b;
}

// ---------------------------------------------------------------- drawer (non-modal side panel)

/** Toasts centre in the page area left of an open drawer (CSS reads --drawer-w). */
export function setDrawerWidth(host, width) {
  const app = host && host.id === 'app' ? host : document.getElementById('app');
  if (!app) return;
  const open = Array.from(app.querySelectorAll(':scope > .drawer:not([hidden])'));
  const w = width != null ? Math.min(width, window.innerWidth) : (open.length ? open[open.length - 1].getBoundingClientRect().width : 0);
  if (w > 0) app.style.setProperty('--drawer-w', `${Math.round(w)}px`);
  else app.style.removeProperty('--drawer-w');
}

/** Right-side panel over the page. Returns {el, body, actions, setTitle, close}. */
export function drawer({ title, onClose, width = 480 } = {}) {
  const host = document.getElementById('app') || document.body;
  const titleEl = el('h2', { class: 'drawer-title', text: title || '' });
  const actions = el('div', { class: 'drawer-actions' });
  const closeBtn = iconBtn('x', 'Close panel');
  const body = el('div', { class: 'drawer-body' });
  const node = el('aside', { class: 'drawer', 'aria-label': title || 'Panel' },
    el('div', { class: 'drawer-head' }, titleEl, actions, closeBtn), body);
  node.style.width = `min(${width}px, 100vw)`;
  host.append(node);
  setDrawerWidth(host, width);
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    node.remove();
    setDrawerWidth(host, null);
    if (onClose) onClose();
  };
  closeBtn.addEventListener('click', close);
  return { el: node, body, actions, close, setTitle: (t) => setText(titleEl, t) };
}

// ---------------------------------------------------------------- layout pieces

/** Card. opts: {title, sub, actions, cls, icon} */
export function card({ title, sub, actions, cls, icon: ic, id } = {}, ...children) {
  const head = title || actions
    ? el('div', { class: 'card-head' },
      el('div', { class: 'card-heading' },
        ic ? icon(ic, { cls: 'card-icon' }) : null,
        title ? el('h2', { class: 'card-title', text: title }) : null,
        sub ? el('span', { class: 'card-sub', text: sub }) : null),
      actions ? el('div', { class: 'card-actions' }, actions) : null)
    : null;
  return el('section', { class: ['card', cls], id }, head, ...children);
}

/** Collapsible section (native <details>). opts: {title, open, key, meta, cls} */
export function collapsible({ title, open = false, key, meta, cls, icon: ic } = {}, ...children) {
  const isOpen = key ? !!prefs.get(key, open) : open;
  const metaEl = el('span', { class: 'collapse-meta' }, meta ?? null);
  const d = el('details', { class: ['collapse', cls], open: isOpen },
    el('summary', { class: 'collapse-head' },
      icon('chevronRight', { size: 16, cls: 'collapse-chevron' }),
      ic ? icon(ic, { size: 16 }) : null,
      el('span', { class: 'collapse-title', text: title }),
      metaEl),
    el('div', { class: 'collapse-body' }, ...children));
  if (key) d.addEventListener('toggle', () => prefs.set(key, d.open));
  d.metaEl = metaEl;
  d.bodyEl = d.lastChild;
  return d;
}

/**
 * Tabs with panels. items: [{value, label, icon}]. key: localStorage key for the selected tab.
 * Returns {el, panel(v), select(v), get(), onFirstShow(v, fn), setCount(v, n)}.
 */
export function createTabs({ items, value, key, onChange, label = 'Sections' } = {}) {
  const saved = key ? prefs.get(key, null) : null;
  let current = items.some((i) => i.value === saved) ? saved : (value ?? items[0].value);
  const list = el('div', { class: 'tabs', role: 'tablist', 'aria-label': label });
  const tabsMap = new Map();
  const panels = new Map();
  const counts = new Map();
  const shown = new Set();
  const lazy = new Map();
  for (const it of items) {
    const tid = uid('tab');
    const pid = uid('panel');
    const count = el('span', { class: 'tab-count', hidden: true });
    const t = el('button', { type: 'button', role: 'tab', class: 'tab', id: tid, 'aria-controls': pid },
      it.icon ? icon(it.icon, { size: 16 }) : null, el('span', { text: it.label }), count);
    const p = el('div', { role: 'tabpanel', class: 'tab-panel', id: pid, 'aria-labelledby': tid });
    t.addEventListener('click', () => selectTab(it.value));
    t.addEventListener('keydown', (ev) => {
      if (ev.key !== 'ArrowRight' && ev.key !== 'ArrowLeft') return;
      ev.preventDefault();
      const idx = items.findIndex((x) => x.value === current);
      const next = items[(idx + (ev.key === 'ArrowRight' ? 1 : items.length - 1)) % items.length];
      selectTab(next.value, true);
    });
    tabsMap.set(it.value, t);
    panels.set(it.value, p);
    counts.set(it.value, count);
    list.append(t);
  }
  function paint(focus) {
    for (const [v, t] of tabsMap) {
      const on = v === current;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      panels.get(v).hidden = !on;
      if (on && focus) t.focus();
    }
  }
  function selectTab(v, focus = false) {
    if (!tabsMap.has(v)) return;
    const changed = v !== current;
    current = v;
    paint(focus);
    if (key) prefs.set(key, v);
    if (!shown.has(v) && lazy.has(v)) {
      shown.add(v);
      lazy.get(v)();
    }
    if (changed && onChange) onChange(v);
  }
  paint(false);
  return {
    el: list,
    panel: (v) => panels.get(v),
    select: selectTab,
    get: () => current,
    onFirstShow(v, fn) {
      if (v === current && !shown.has(v)) {
        shown.add(v);
        fn();
      } else if (!shown.has(v)) {
        lazy.set(v, fn);
      }
    },
    setCount(v, n) {
      const c = counts.get(v);
      if (!c) return;
      c.hidden = !n;
      setText(c, n ? String(n) : '');
    },
  };
}

/** Page toolbar row: left (tabs / status) and right (actions). */
export function toolbar(left, right) {
  return el('div', { class: 'page-toolbar' },
    el('div', { class: 'toolbar-left' }, left || null),
    el('div', { class: 'toolbar-right' }, right || null));
}

/** Stat tile: {label, icon, href}. Returns {el, set(value, sub, state)} */
export function tile({ label, icon: ic, href } = {}) {
  const value = el('div', { class: 'tile-value', text: '–' });
  const sub = status('off', '');
  const node = el(href ? 'a' : 'div', { class: 'tile', href },
    el('div', { class: 'tile-label' }, ic ? icon(ic, { size: 16 }) : null, el('span', { text: label })),
    value, sub);
  return {
    el: node,
    set(v, subText, state = 'off', tip) {
      setText(value, v);
      setStatus(sub, state, subText || '');
      if (tip !== undefined) value.title = tip || '';
    },
  };
}

/** Empty / loading / error state. kind: 'empty' | 'loading' | 'error' */
export function emptyState({ kind = 'empty', icon: ic, title, text, actions } = {}) {
  const iconName = ic || (kind === 'error' ? 'alert' : kind === 'loading' ? null : 'info');
  return el('div', { class: ['empty', `empty-${kind}`], role: kind === 'error' ? 'alert' : 'status' },
    kind === 'loading' ? el('span', { class: 'spinner spinner-lg', 'aria-hidden': 'true' }) : icon(iconName, { size: 28, cls: 'empty-icon' }),
    title ? el('div', { class: 'empty-title', text: title }) : null,
    text ? el('div', { class: 'empty-text', text }) : null,
    actions ? el('div', { class: 'empty-actions' }, actions) : null);
}

/** Compact loading / empty / error line (inside cards and tables). */
export function placeholder(kind, text, retry) {
  return el('div', { class: ['placeholder', `is-${kind}`], role: kind === 'error' ? 'alert' : 'status' },
    kind === 'loading' ? el('span', { class: 'spinner', 'aria-hidden': 'true' }) : null,
    kind === 'error' ? icon('alert', { size: 16 }) : null,
    el('span', { text }),
    retry ? btn('Retry', { kind: 'ghost', size: 'sm', icon: 'refresh', onClick: retry }) : null);
}

/** Key/value list. pairs: [[label, valueNodeOrText], ...] (undefined values are skipped). */
export function kv(pairs) {
  const dl = el('dl', { class: 'kv' });
  for (const [k, v] of pairs) {
    if (v === undefined) continue;
    dl.append(el('dt', { text: k }), el('dd', null, v == null || v === '' ? '–' : v));
  }
  return dl;
}

// ---------------------------------------------------------------- form controls

/** Labelled form field. opts: {help (info tooltip), hint (visible small text), cls}. Error slot .field-error */
export function field(label, control, { help, hint, cls } = {}) {
  if (!control.id) control.id = uid('f');
  const hintEl = hint ? el('div', { class: 'field-hint', id: uid('hint'), text: hint }) : null;
  const errEl = el('div', { class: 'field-error', id: uid('err'), role: 'alert', hidden: true });
  control.setAttribute('aria-describedby', [hintEl && hintEl.id, errEl.id].filter(Boolean).join(' '));
  return el('div', { class: ['field', cls] },
    el('div', { class: 'field-label-row' },
      el('label', { class: 'field-label', for: control.id, text: label }),
      help ? infoTip(help, label) : null),
    control, hintEl, errEl);
}

export function setFieldError(fieldNode, message) {
  if (!fieldNode) return;
  const errEl = fieldNode.querySelector('.field-error');
  const control = fieldNode.querySelector('input, select, textarea, button[role=switch]');
  if (errEl) {
    setText(errEl, message || '');
    errEl.hidden = !message;
  }
  if (control) {
    if (message) control.setAttribute('aria-invalid', 'true');
    else control.removeAttribute('aria-invalid');
  }
}

export function input(props = {}) {
  return el('input', { class: 'input', type: 'text', ...props });
}

/**
 * Switch (role="switch"). onChange(next, button) decides; call api.set(v) to show the new state
 * (when no onChange is given the switch flips itself).
 */
export function switchControl(label, { checked = false, onChange, disabled, hideLabel = false, tip } = {}) {
  const id = uid('sw');
  const sw = el('button', { type: 'button', role: 'switch', class: 'switch', id, 'aria-checked': String(!!checked), disabled: !!disabled },
    el('span', { class: 'switch-thumb', 'aria-hidden': 'true' }));
  const lab = el('label', { class: hideLabel ? 'sr-only' : 'switch-label', for: id, text: label });
  const wrap = el('span', { class: 'switch-field' }, sw, lab);
  const api = {
    el: wrap,
    button: sw,
    get: () => sw.getAttribute('aria-checked') === 'true',
    set(v) {
      sw.setAttribute('aria-checked', String(!!v));
    },
    setDisabled(d) {
      setEnabled(sw, !d);
    },
  };
  sw.addEventListener('click', () => {
    const next = !api.get();
    if (onChange) onChange(next, sw);
    else api.set(next);
  });
  if (tip) attachTip(sw, tip);
  return api;
}

/** Checkbox with a label. */
export function checkbox(label, { checked = false, onChange, disabled } = {}) {
  const inp = el('input', { type: 'checkbox', checked, disabled: !!disabled });
  if (onChange) inp.addEventListener('change', () => onChange(inp.checked));
  const wrap = el('label', { class: 'check' }, inp, el('span', { text: label }));
  return { el: wrap, input: inp };
}

export function select(options, { value, onChange, id, ariaLabel, cls } = {}) {
  const s = el('select', { class: ['input', cls], id, 'aria-label': ariaLabel });
  setOptions(s, options, value);
  if (onChange) s.addEventListener('change', () => onChange(s.value));
  return s;
}

/** options: [{value, label, disabled}] or strings */
export function setOptions(selectEl, options, value) {
  const prev = value !== undefined ? value : selectEl.value;
  selectEl.replaceChildren(...options.map((o) => {
    const opt = typeof o === 'object' && o !== null ? o : { value: o, label: o };
    return el('option', { value: String(opt.value), text: opt.label ?? String(opt.value), disabled: !!opt.disabled, hidden: !!opt.hidden });
  }));
  if (prev != null && options.some((o) => String(typeof o === 'object' && o !== null ? o.value : o) === String(prev))) {
    selectEl.value = String(prev);
  }
}

/** Segmented control. options: [{value, label, icon, tip}] */
export function segmented(options, { label, value, onChange, size = 'sm', cls } = {}) {
  const group = el('div', { class: ['segmented', `segmented-${size}`, cls], role: 'group', 'aria-label': label });
  const buttons = new Map();
  for (const o of options) {
    const b = el('button', { type: 'button', class: 'seg', 'aria-pressed': 'false' },
      o.icon ? icon(o.icon, { size: 16 }) : null, o.label ? el('span', { text: o.label }) : null);
    if (!o.label) b.setAttribute('aria-label', o.tip || String(o.value));
    if (o.tip) attachTip(b, o.tip);
    b.addEventListener('click', () => {
      if (onChange) onChange(o.value, b);
    });
    buttons.set(o.value, b);
    group.append(b);
  }
  const api = {
    el: group,
    buttons,
    value,
    set(v) {
      api.value = v;
      for (const [val, b] of buttons) b.setAttribute('aria-pressed', String(val === v));
    },
    setDisabled(disabled) {
      for (const b of buttons.values()) setEnabled(b, !disabled);
    },
  };
  api.set(value);
  return api;
}

// ---------------------------------------------------------------- tables

/**
 * Data table.
 * columns: [{key, label, render?(row) -> Node|string, cls?, sort?(a,b) -> number, numeric?, width?}]
 * Returns {el, setRows(rows), setEmpty(text), table, tbody}.
 */
export function dataTable({ columns, empty = 'Nothing to show.', caption, rowClass, onRowClick, sortKey, sortDir = 'desc' } = {}) {
  const table = el('table', { class: 'table' });
  if (caption) table.append(el('caption', { class: 'sr-only', text: caption }));
  const headRow = el('tr');
  const tbody = el('tbody');
  table.append(el('thead', null, headRow), tbody);
  let rows = [];
  let emptyText = empty;
  let sort = { key: sortKey, dir: sortDir };
  const ths = new Map();

  for (const col of columns) {
    const th = el('th', { scope: 'col', class: [col.cls, col.numeric && 'num'] });
    if (col.width) th.style.width = col.width;
    if (col.sort) {
      const b = el('button', { type: 'button', class: 'th-sort' }, el('span', { text: col.label }), icon('sort', { size: 14, cls: 'sort-icon' }));
      b.addEventListener('click', () => {
        sort = { key: col.key, dir: sort.key === col.key && sort.dir === 'desc' ? 'asc' : 'desc' };
        render();
      });
      th.append(b);
    } else {
      th.textContent = col.label;
    }
    ths.set(col.key, th);
    headRow.append(th);
  }

  function cellValue(col, row) {
    if (col.render) return col.render(row);
    const v = row[col.key];
    return v == null || v === '' ? '–' : String(v);
  }

  function render() {
    for (const col of columns) {
      if (!col.sort) continue;
      ths.get(col.key).setAttribute('aria-sort', sort.key === col.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none');
    }
    let list = rows;
    const col = columns.find((c) => c.key === sort.key && c.sort);
    if (col) list = rows.slice().sort((a, b) => (sort.dir === 'asc' ? 1 : -1) * col.sort(a, b));
    if (!list.length) {
      tbody.replaceChildren(el('tr', null, el('td', { class: 'table-empty', colspan: String(columns.length), text: emptyText })));
      return;
    }
    tbody.replaceChildren(...list.map((row) => {
      const tr = el('tr', { class: rowClass ? rowClass(row) : null });
      for (const c of columns) tr.append(el('td', { class: [c.cls, c.numeric && 'num'] }, cellValue(c, row)));
      if (onRowClick) {
        tr.classList.add('is-clickable');
        tr.tabIndex = 0;
        tr.addEventListener('click', (ev) => {
          if (ev.target instanceof Element && ev.target.closest('button, a, input, select, label')) return;
          onRowClick(row, tr);
        });
        tr.addEventListener('keydown', (ev) => {
          if (ev.key === 'Enter' && ev.target === tr) onRowClick(row, tr);
        });
      }
      return tr;
    }));
  }

  render();
  return {
    el: el('div', { class: 'table-wrap' }, table),
    table,
    tbody,
    setRows(next) {
      rows = Array.isArray(next) ? next : [];
      render();
    },
    setEmpty(text) {
      emptyText = text;
      if (!rows.length) render();
    },
  };
}

/** Numeric comparator that sorts null/undefined last. */
export function numSort(key) {
  return (a, b) => {
    const x = typeof a[key] === 'number' ? a[key] : -Infinity;
    const y = typeof b[key] === 'number' ? b[key] : -Infinity;
    return x - y;
  };
}

export function strSort(key) {
  return (a, b) => String(a[key] ?? '').localeCompare(String(b[key] ?? ''), undefined, { numeric: true });
}

/** <pre> log box that keeps the scroll at the bottom when the user was at the bottom (or follow=true). */
export function setLogText(pre, lines, follow = false) {
  const text = Array.isArray(lines) ? lines.join('\n') : String(lines ?? '');
  if (pre.textContent === text) return;
  const atBottom = follow || pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
  pre.textContent = text;
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}
