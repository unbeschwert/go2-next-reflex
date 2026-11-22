// Event log fed by SSE `log` events: newest first, kind shown as dot + word,
// capped at 300 lines, filter by kind (small select). Renders in animation
// frames so bursts of log lines do not thrash the DOM.

import { logHistory } from '../events.js';
import { el, setText, select, iconBtn } from '../ui.js';
import * as fmt from '../format.js';

export const LOG_KINDS = ['command', 'won', 'ready', 'held', 'turn', 'info', 'error'];

const FILTERS = [
  { value: 'all', label: 'All events' },
  { value: 'command', label: 'Commands' },
  { value: 'decisions', label: 'Decisions (won, ready, held, turn)' },
  { value: 'error', label: 'Errors' },
  { value: 'info', label: 'Info' },
];

function kindOf(entry) {
  return LOG_KINDS.includes(entry.kind) ? entry.kind : 'info';
}

function matches(filter, entry) {
  const k = kindOf(entry);
  if (filter === 'all') return true;
  if (filter === 'decisions') return ['won', 'ready', 'held', 'turn'].includes(k);
  return k === filter;
}

/** Returns {el, tools}: tools (filter + count + clear) go into a header / tab bar. */
export function createLogView(scope, { max = 300, compact = false } = {}) {
  let filter = 'all';
  const entries = []; // newest first
  let pending = [];
  let raf = 0;

  const list = el('ol', { class: ['logview-list', compact && 'logview-compact'], 'aria-label': 'Event log, newest first' });
  const count = el('span', { class: 'small muted tabular' });
  const filterSel = select(FILTERS, {
    value: filter,
    ariaLabel: 'Show events',
    cls: 'input-sm',
    onChange: (v) => {
      filter = v;
      rebuild();
    },
  });
  const clearBtn = iconBtn('trash', 'Clear the list', { size: 'sm' });
  clearBtn.addEventListener('click', () => {
    entries.length = 0;
    rebuild();
  });
  const tools = el('div', { class: 'row' }, count, filterSel, clearBtn);

  function item(entry) {
    const kind = kindOf(entry);
    return el('li', { class: ['log-item', `log-kind-${kind}`] },
      el('time', { class: 'log-time', text: fmt.timeOfDay(entry.t) }),
      compact ? null : el('span', { class: 'log-kind', text: entry.kind ? String(entry.kind) : 'info' }),
      el('span', { class: 'log-line', text: entry.line == null ? '' : String(entry.line) }));
  }

  function updateCount() {
    const shown = list.querySelectorAll('.log-item').length;
    setText(count, entries.length === shown ? `${entries.length} events` : `${shown} of ${entries.length}`);
  }

  function rebuild() {
    const nodes = [];
    for (const e of entries) {
      if (matches(filter, e)) nodes.push(item(e));
      if (nodes.length >= max) break;
    }
    if (!nodes.length) {
      list.replaceChildren(el('li', { class: 'log-empty', text: entries.length ? 'No events match the filter.' : 'No events yet. Gesture decisions and commands appear here.' }));
    } else {
      list.replaceChildren(...nodes);
    }
    updateCount();
  }

  function flush() {
    raf = 0;
    const batch = pending;
    pending = [];
    const emptyNode = list.querySelector('.log-empty');
    if (emptyNode && batch.some((e) => matches(filter, e))) emptyNode.remove();
    for (const e of batch) {
      entries.unshift(e);
      if (matches(filter, e)) list.prepend(item(e));
    }
    if (entries.length > max) entries.length = max;
    while (list.childElementCount > max) list.lastElementChild.remove();
    updateCount();
  }

  for (const e of logHistory()) entries.unshift(e);
  if (entries.length > max) entries.length = max;
  rebuild();

  scope.sub('log', (e) => {
    pending.push(e);
    if (!raf) raf = requestAnimationFrame(flush);
  });
  scope.add(() => cancelAnimationFrame(raf));

  return { el: list, tools };
}
