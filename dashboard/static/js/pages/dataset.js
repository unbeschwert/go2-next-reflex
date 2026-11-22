// Dataset: master/detail. Left: paged clip list; right: "Clip + spikes" (player
// and the network's output spikes on one playhead). Toolbar: source, session and
// class filters, Overview toggle (sessions x classes heat map). Relabel / Reject
// live in the "..." menu (writable sources). Keys: J previous, K next clip.

import { api } from '../api.js';
import {
  el, setText, card, act, btn, iconBtn, select, setOptions, prefs, chip, emptyState, placeholder, menuButton, toast,
  shortcutAllowed, setEnabled, toolbar, attachTip,
} from '../ui.js';
import * as fmt from '../format.js';
import { createClipPlayer } from '../components/clipPlayer.js';
import { createSpikePanel } from '../components/spikePanel.js';

const PAGE_SIZE = 50;
const CLIP_SIZE = 80;

function sortedIds(classes) {
  return Object.keys(classes || {}).sort((a, b) => {
    const x = Number(a);
    const y = Number(b);
    return Number.isFinite(x) && Number.isFinite(y) ? x - y : a.localeCompare(b);
  });
}

function baseName(path) {
  const s = String(path || '');
  const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
  return i >= 0 ? s.slice(i + 1) : s;
}

function clsValue(key) {
  return /^\d+$/.test(String(key)) ? Number(key) : key;
}

export async function mount(root, scope) {
  let sources = [];
  let source = null;
  let summary = null;
  let filter = { session: '', cls: '' };
  let clips = [];
  let total = 0;
  let offset = 0;
  let selected = -1;
  let clipCtrl = null;
  let listToken = 0;
  let view = prefs.get('dataset.view', 'list');

  // ---------------------------------------------------------------- toolbar
  const sourceSel = select([], { ariaLabel: 'Dataset source' });
  const roChip = chip('Read-only', { kind: 'neutral', icon: 'lock' });
  roChip.hidden = true;
  attachTip(roChip, 'Clips of this source cannot be relabelled or rejected');
  const sessionSel = select([{ value: '', label: 'All sessions' }], { ariaLabel: 'Session filter' });
  const classSel = select([{ value: '', label: 'All gestures' }], { ariaLabel: 'Gesture filter' });
  const overviewBtn = btn('Overview', { kind: 'secondary', icon: 'grid', pressed: view === 'overview' });
  attachTip(overviewBtn, 'Sessions x gestures table');
  const refreshBtn = iconBtn('refresh', 'Refresh');
  root.append(toolbar(
    [el('label', { class: 'row' }, el('span', { class: 'small muted', text: 'Source' }), sourceSel), roChip, sessionSel, classSel],
    [overviewBtn, refreshBtn]));

  // ---------------------------------------------------------------- list view
  const listTitle = el('span', { class: 'small muted tabular' });
  const list = el('ul', { class: 'ds-list', 'aria-label': 'Clips' });
  const pageText = el('span', { class: 'tabular' });
  const prevBtn = iconBtn('chevronLeft', 'Previous page', { size: 'sm' });
  const nextBtn = iconBtn('chevronRight', 'Next page', { size: 'sm' });
  const listCard = el('section', { class: 'card ds-list-card', 'aria-label': 'Clip list' },
    el('div', { class: 'card-head' }, el('div', { class: 'card-heading' }, el('h2', { class: 'card-title', text: 'Clips' }), listTitle),
      el('span', { class: 'row small faint', title: 'Previous / next clip' }, el('kbd', { text: 'J' }), el('kbd', { text: 'K' }))),
    list,
    el('div', { class: 'ds-pager' }, pageText, el('div', { class: 'row' }, prevBtn, nextBtn)));

  // detail: clip + spikes on one playhead
  const spikes = createSpikePanel(scope, { onSeek: (f) => player.seek(f) });
  const player = createClipPlayer(scope, {
    onPrev: () => step(-1),
    onNext: () => step(1),
    onFrame: (f) => spikes.setFrame(f),
  });
  const moreMenu = menuButton(() => {
    const clip = clips[selected];
    const cls = classes();
    const ids = sortedIds(cls);
    if (!clip || !writable()) return [{ label: 'This source is read-only', icon: 'lock', disabled: true, onSelect: () => {} }];
    return [
      ...ids.filter((id) => String(id) !== String(clip.cls)).map((id) => ({
        label: `Relabel as ${cls[id]}`, icon: 'tag', onSelect: (b) => relabel(clip, id, b),
      })),
      'sep',
      { label: 'Reject clip', icon: 'trash', danger: true, onSelect: (b) => reject(clip, b) },
    ];
  }, { label: 'Clip actions' });
  const detailCard = el('section', { class: 'card ds-detail', 'aria-label': 'Clip and spikes' },
    el('div', { class: 'ds-detail-head' }, player.meta, moreMenu),
    player.stage, player.transport, spikes.el);
  const listView = el('div', { class: 'ds-layout' }, listCard, detailCard);

  // overview view
  const heatHost = el('div', { class: 'table-wrap' }, placeholder('loading', 'Loading…'));
  const overviewCard = card({ title: 'Sessions × gestures', sub: 'Click a cell to list those clips' }, heatHost);

  root.append(listView, overviewCard);

  function setView(v) {
    view = v;
    prefs.set('dataset.view', v);
    listView.hidden = v !== 'list';
    overviewCard.hidden = v !== 'overview';
    overviewBtn.setAttribute('aria-pressed', String(v === 'overview'));
    if (v === 'overview') renderHeat();
  }
  overviewBtn.addEventListener('click', () => setView(view === 'overview' ? 'list' : 'overview'));

  // ---------------------------------------------------------------- helpers

  const classes = () => (summary && summary.classes) || (source && source.classes) || {};
  const writable = () => !!(source && source.writable);

  function renderFilters() {
    const cls = classes();
    const sessions = summary && Array.isArray(summary.sessions)
      ? summary.sessions.map((s) => String(s.session))
      : (source && Array.isArray(source.sessions) ? source.sessions.map(String) : []);
    setOptions(sessionSel, [{ value: '', label: 'All sessions' }, ...sessions.map((s) => ({ value: s, label: `Session ${s}` }))], filter.session);
    setOptions(classSel, [{ value: '', label: 'All gestures' }, ...sortedIds(cls).map((id) => ({ value: id, label: String(cls[id]) }))], filter.cls);
  }

  function setFilter(next) {
    filter = { ...filter, ...next };
    offset = 0;
    renderFilters();
    loadClips({ select: 0 });
  }

  sessionSel.addEventListener('change', () => setFilter({ session: sessionSel.value }));
  classSel.addEventListener('change', () => setFilter({ cls: classSel.value }));

  // ---------------------------------------------------------------- overview heat map

  function renderHeat() {
    if (!summary) return;
    const cls = classes();
    const ids = sortedIds(cls);
    const sessions = Array.isArray(summary.sessions) ? summary.sessions : [];
    if (!sessions.length) {
      heatHost.replaceChildren(emptyState({ title: 'No sessions', text: 'This source has no recorded sessions yet.' }));
      return;
    }
    let max = 0;
    for (const s of sessions) for (const id of ids) max = Math.max(max, Number((s.counts || {})[id]) || 0);
    const cellBtn = (n, session, id) => {
      const b = el('button', { type: 'button', class: ['heat-btn', !n && 'heat-zero'], text: fmt.int(n), 'aria-label': `Session ${session}, ${cls[id]}: ${n} clips` });
      if (n) b.style.setProperty('--heat', String(0.12 + 0.68 * (n / (max || 1))));
      b.addEventListener('click', () => {
        setView('list');
        setFilter({ session: String(session), cls: String(id) });
      });
      return b;
    };
    const totals = Object.fromEntries(ids.map((id) => [id, 0]));
    let grand = 0;
    const rows = sessions.map((s) => {
      const counts = s.counts || {};
      const rowTotal = fmt.isNum(s.total) ? s.total : ids.reduce((a, id) => a + (Number(counts[id]) || 0), 0);
      grand += rowTotal;
      return el('tr', null,
        el('th', { scope: 'row', text: `Session ${s.session}` }),
        ...ids.map((id) => {
          const n = Number(counts[id]) || 0;
          totals[id] += n;
          return el('td', { class: 'heat-cell' }, cellBtn(n, s.session, id));
        }),
        el('td', { class: 'num', text: fmt.int(rowTotal) }));
    });
    heatHost.replaceChildren(el('table', { class: 'table heat-table' },
      el('caption', { class: 'sr-only', text: 'Clip counts per session and gesture' }),
      el('thead', null, el('tr', null, el('th', { text: 'Session' }), ...ids.map((id) => el('th', { class: 'num', text: String(cls[id]) })), el('th', { class: 'num', text: 'Total' }))),
      el('tbody', null, ...rows),
      el('tfoot', null, el('tr', null, el('th', { text: 'All sessions' }), ...ids.map((id) => el('td', { class: 'num', text: fmt.int(totals[id]) })), el('td', { class: 'num', text: fmt.int(grand) })))));
  }

  async function loadSummary() {
    if (!source) return;
    try {
      summary = await api.get('/dataset/summary', { source: source.id }, { signal: scope.signal });
      if (!scope.alive) return;
      renderFilters();
      if (view === 'overview') renderHeat();
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      heatHost.replaceChildren(placeholder('error', `Could not load the summary: ${err.message}`, () => loadSummary()));
    }
  }

  // ---------------------------------------------------------------- clip list

  const rowNodes = [];

  function renderList() {
    rowNodes.length = 0;
    if (!clips.length) {
      list.replaceChildren(el('li', null, emptyState({ title: 'No clips', text: filter.session || filter.cls ? 'No clips match the filters.' : 'This source has no clips.' })));
    } else {
      list.replaceChildren(...clips.map((c, i) => {
        const b = el('button', { type: 'button', class: 'clip-row', 'aria-current': i === selected ? 'true' : null, title: String(c.path || '') },
          el('span', { class: 'clip-label', text: c.label ?? String(c.cls ?? '?') }),
          el('span', { class: 'clip-meta', text: [c.session != null ? `session ${c.session}` : null, c.style, c.take != null ? `take ${c.take}` : null].filter(Boolean).join(' · ') || baseName(c.path) }));
        b.addEventListener('click', () => selectClip(i));
        rowNodes.push(b);
        return el('li', null, b);
      }));
    }
    setText(listTitle, `${fmt.int(total)} clip${total === 1 ? '' : 's'}`);
    const from = total ? offset + 1 : 0;
    const to = Math.min(offset + clips.length, total);
    setText(pageText, `${fmt.int(from)}–${fmt.int(to)} of ${fmt.int(total)}`);
    setEnabled(prevBtn, offset > 0);
    setEnabled(nextBtn, offset + clips.length < total);
  }

  function markSelected() {
    rowNodes.forEach((b, i) => {
      if (i === selected) b.setAttribute('aria-current', 'true');
      else b.removeAttribute('aria-current');
    });
  }

  async function loadClips({ select: want = null } = {}) {
    if (!source) return;
    const token = ++listToken;
    list.replaceChildren(el('li', null, placeholder('loading', 'Loading clips…')));
    try {
      const res = await api.get('/dataset/clips', {
        source: source.id, session: filter.session || null, cls: filter.cls || null, offset, limit: PAGE_SIZE,
      }, { signal: scope.signal });
      if (!scope.alive || token !== listToken) return;
      clips = Array.isArray(res.clips) ? res.clips : [];
      total = fmt.isNum(res.total) ? res.total : clips.length;
      selected = -1;
      renderList();
      if (clips.length && want != null) selectClip(want === 'last' ? clips.length - 1 : Math.min(want, clips.length - 1));
      else if (!clips.length) {
        player.clear('No clip selected.');
        spikes.clear();
      }
    } catch (err) {
      if (!scope.alive || err.aborted || token !== listToken) return;
      list.replaceChildren(el('li', null, placeholder('error', `Could not load clips: ${err.message}`, () => loadClips({ select: want }))));
    }
  }

  async function selectClip(i) {
    if (i < 0 || i >= clips.length) return;
    selected = i;
    markSelected();
    const row = rowNodes[i];
    if (row) row.scrollIntoView({ block: 'nearest' });
    const clip = clips[i];
    if (clipCtrl) clipCtrl.abort();
    const c = scope.controller();
    clipCtrl = c;
    player.clear('Loading clip…');
    spikes.load(source.id, clip.path); // independent request; does not block the player
    try {
      const res = await api.get('/dataset/clip', { source: source.id, path: clip.path, size: CLIP_SIZE }, { signal: c.signal, timeout: 30000 });
      if (!scope.alive || c.signal.aborted) return;
      player.load(res);
    } catch (err) {
      if (!scope.alive || err.aborted || c.signal.aborted) return;
      player.clear(`Could not load the clip: ${err.message}`);
    }
  }

  async function step(dir) {
    if (!clips.length) return;
    const next = selected < 0 ? 0 : selected + dir;
    if (next >= 0 && next < clips.length) {
      selectClip(next);
    } else if (next >= clips.length && offset + clips.length < total) {
      offset += PAGE_SIZE;
      await loadClips({ select: 0 });
    } else if (next < 0 && offset > 0) {
      offset = Math.max(0, offset - PAGE_SIZE);
      await loadClips({ select: 'last' });
    }
  }

  prevBtn.addEventListener('click', () => {
    offset = Math.max(0, offset - PAGE_SIZE);
    loadClips({ select: null });
  });
  nextBtn.addEventListener('click', () => {
    offset += PAGE_SIZE;
    loadClips({ select: null });
  });

  // ---------------------------------------------------------------- relabel / reject

  async function relabel(clip, id, b) {
    const cls = classes();
    const res = await act(b, () => api.post('/dataset/relabel', { source: source.id, path: clip.path, cls: clsValue(id) }, { risk: 'warn', signal: scope.signal }), {
      success: `Relabelled as ${cls[id] || id}`,
    });
    if (!res || !scope.alive) return;
    clip.path = res.path || clip.path;
    clip.cls = clsValue(id);
    clip.label = cls[id] || clip.label;
    const idx = selected;
    renderList();
    selected = idx;
    markSelected();
    loadSummary();
  }

  async function reject(clip, b) {
    const res = await act(b, () => api.post('/dataset/reject', { source: source.id, path: clip.path }, { risk: 'danger', signal: scope.signal }));
    if (!res || !scope.alive) return;
    toast(res.moved_to ? `Clip rejected (moved to ${res.moved_to})` : 'Clip rejected', 'success');
    const idx = selected;
    clips.splice(idx, 1);
    total = Math.max(0, total - 1);
    selected = -1;
    renderList();
    if (clips.length) selectClip(Math.min(idx, clips.length - 1));
    else {
      player.clear('No clip selected.');
      spikes.clear();
    }
    loadSummary();
  }

  // ---------------------------------------------------------------- sources

  function chooseSource(id) {
    source = sources.find((s) => String(s.id) === String(id)) || sources[0] || null;
    if (!source) return;
    prefs.set('dataset.source', source.id);
    sourceSel.value = String(source.id);
    roChip.hidden = writable();
    summary = null;
    filter = { session: '', cls: '' };
    offset = 0;
    clips = [];
    selected = -1;
    player.clear();
    spikes.clear();
    renderFilters();
    loadSummary();
    loadClips({ select: 0 });
  }

  async function loadSources() {
    try {
      const res = await api.get('/dataset/sources', null, { signal: scope.signal });
      if (!scope.alive) return;
      sources = Array.isArray(res.sources) ? res.sources : [];
      if (!sources.length) {
        listView.replaceChildren(emptyState({ title: 'No dataset sources', text: 'No recording folder or dataset zip is configured.' }));
        return;
      }
      setOptions(sourceSel, sources.map((s) => ({ value: s.id, label: s.name || s.id })));
      const remembered = prefs.get('dataset.source', null);
      chooseSource(sources.some((s) => String(s.id) === String(remembered)) ? remembered : sources[0].id);
    } catch (err) {
      if (!scope.alive || err.aborted) return;
      listView.replaceChildren(emptyState({
        kind: 'error', title: 'Could not load the dataset sources', text: err.message,
        actions: btn('Retry', { kind: 'secondary', icon: 'refresh', onClick: () => loadSources() }),
      }));
    }
  }

  sourceSel.addEventListener('change', () => chooseSource(sourceSel.value));
  refreshBtn.addEventListener('click', () => act(refreshBtn, async () => {
    await loadSummary();
    await loadClips({ select: selected >= 0 ? selected : 0 });
  }));

  scope.listen(document, 'keydown', (ev) => {
    if (!shortcutAllowed(ev) || view !== 'list') return;
    const k = (ev.key || '').toLowerCase();
    if (k === 'j') {
      ev.preventDefault();
      step(-1);
    } else if (k === 'k') {
      ev.preventDefault();
      step(1);
    }
  });

  setView(view);
  player.clear();
  spikes.clear();
  await loadSources();
}
