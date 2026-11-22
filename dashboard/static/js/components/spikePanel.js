// Output spikes of a clip replayed through the FPGA integer model
// (GET /api/dataset/spikes), drawn under the clip player and synced to its playhead:
// - verdict line (predicted vs label), model select (GET /api/dataset/spike-models)
// - raster: 5 rows x frames, a tick per spike, playhead, totals / rate per row
// - Details (collapsed): camera events per frame, layer activity (% of neurons)
// Click / drag on the raster seeks the player. Labels and numbers are DOM text;
// the canvases only draw ticks and lines.

import { api } from '../api.js';
import { el, setText, icon, btn, collapsible, attachTip } from '../ui.js';
import * as fmt from '../format.js';

const ROW_H = 22;
const CLASS_NAMES = { own: 'Own gestures', dataset: 'Dataset gestures' };

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function colors() {
  return {
    accent: cssVar('--accent', '#2dd4bf'),
    text: cssVar('--text', '#e6edf3'),
    muted: cssVar('--muted', '#8b98a5'),
    faint: cssVar('--faint', '#6e7b88'),
    border: cssVar('--border', '#2a3441'),
    accentWeak: cssVar('--accent-weak', 'rgba(45,212,191,.12)'),
    layers: [cssVar('--accent', '#2dd4bf'), cssVar('--blue', '#58a6ff'), cssVar('--warning', '#d29922'), '#a371f7', cssVar('--muted', '#8b98a5')],
  };
}

/** Canvas sized to its CSS box at device pixel ratio. Returns {ctx, w, h} in CSS pixels. */
function prepare(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, canvas.clientWidth);
  const h = Math.max(1, canvas.clientHeight);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

export function createSpikePanel(scope, { onSeek } = {}) {
  let data = null;
  let frame = 0;
  let clipRef = null; // {source, path}
  let ctrl = null;
  let modelId = '';
  let models = null;
  let modelsLoading = false;
  let pal = colors();

  // ---- header: verdict + model
  const verdict = el('div', { class: 'verdict', role: 'status' });
  const modelSel = el('select', { class: 'input input-sm spike-model', 'aria-label': 'Network model for the spikes' },
    el('option', { value: '', text: 'Board model (auto)' }));
  attachTip(modelSel, 'Which trained network replays the clip (FPGA integer simulator)');
  const modelHint = el('span', { class: 'small faint ellipsis spike-model-name' });
  const head = el('div', { class: 'spike-head' }, verdict, el('div', { class: 'row spike-model-row' }, modelHint, modelSel));

  // ---- raster
  const labelsCol = el('div', { class: 'raster-labels' });
  const totalsCol = el('div', { class: 'raster-totals' });
  const canvas = el('canvas', { class: 'raster-canvas', role: 'img', 'aria-label': 'Output spike raster' });
  const overlay = el('div', { class: 'raster-overlay' });
  const plot = el('div', { class: 'raster-plot' }, canvas, overlay);
  const raster = el('div', { class: 'raster' }, labelsCol, plot, totalsCol);

  // ---- details
  const evCanvas = el('canvas', { class: 'chart-canvas chart-events', role: 'img', 'aria-label': 'Camera events per frame' });
  const evMax = el('span', { class: 'small faint tabular' });
  const layerCanvas = el('canvas', { class: 'chart-canvas chart-layers', role: 'img', 'aria-label': 'Layer activity per frame' });
  const layerLegend = el('div', { class: 'legend layer-legend' });
  const details = collapsible({ title: 'Details', key: 'dataset.spikeDetails', open: false, cls: 'spike-details' },
    el('div', { class: 'chart-head' }, el('span', { class: 'small muted', text: 'Camera events per frame' }), evMax), evCanvas,
    el('div', { class: 'chart-head' }, el('span', { class: 'small muted', text: 'Layer activity (% of neurons spiking per frame)' }), layerLegend), layerCanvas);
  details.addEventListener('toggle', () => drawDetails());

  const root = el('section', { class: 'spike-panel', 'aria-label': 'Network output spikes' }, head, raster, details);

  // ---------------------------------------------------------------- drawing

  function rowsCount() {
    return data && Array.isArray(data.labels) ? data.labels.length : 5;
  }

  function drawRaster() {
    canvas.style.height = `${rowsCount() * ROW_H}px`;
    const { ctx, w, h } = prepare(canvas);
    const rows = rowsCount();
    ctx.strokeStyle = pal.border;
    ctx.lineWidth = 1;
    for (let r = 1; r < rows; r += 1) {
      ctx.beginPath();
      ctx.moveTo(0, r * ROW_H + 0.5);
      ctx.lineTo(w, r * ROW_H + 0.5);
      ctx.stroke();
    }
    if (!data || !Array.isArray(data.outputs)) return;
    const frames = data.outputs.length || 1;
    const fw = w / frames;
    const tick = Math.max(1, Math.min(fw - 0.5, 4));
    for (let r = 0; r < rows; r += 1) {
      const pred = r === data.predicted;
      ctx.fillStyle = pred ? pal.accent : (r === data.true ? pal.text : pal.faint);
      ctx.globalAlpha = pred || r === data.true ? 1 : 0.8;
      for (let t = 0; t < frames; t += 1) {
        const row = data.outputs[t];
        if (row && row[r]) ctx.fillRect(t * fw + (fw - tick) / 2, r * ROW_H + 5, tick, ROW_H - 10);
      }
    }
    ctx.globalAlpha = 1;
    const x = Math.round((fmt.clamp(frame, 0, frames - 1) + 0.5) * fw) + 0.5;
    ctx.strokeStyle = pal.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }

  function drawSeries(cv, series, { color, fill, max }) {
    const { ctx, w, h } = prepare(cv);
    if (!series.length) return;
    const n = series.length;
    const top = max > 0 ? max : 1;
    const xs = (i) => (n > 1 ? (i / (n - 1)) * w : w / 2);
    const ys = (v) => h - 2 - (fmt.clamp(v / top, 0, 1) * (h - 4));
    ctx.beginPath();
    series.forEach((v, i) => (i ? ctx.lineTo(xs(i), ys(v)) : ctx.moveTo(xs(i), ys(v))));
    if (fill) {
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.beginPath();
      series.forEach((v, i) => (i ? ctx.lineTo(xs(i), ys(v)) : ctx.moveTo(xs(i), ys(v))));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }

  function playheadOn(cv, n) {
    const ctx = cv.getContext('2d');
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    const x = Math.round(n > 1 ? (fmt.clamp(frame, 0, n - 1) / (n - 1)) * w : 0) + 0.5;
    ctx.strokeStyle = pal.text;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawDetails() {
    if (!details.open) return;
    const ev = data && Array.isArray(data.events_per_frame) ? data.events_per_frame.map(Number).map((v) => (Number.isFinite(v) ? v : 0)) : [];
    const evTop = ev.length ? Math.max(...ev) : 0;
    setText(evMax, ev.length ? `max ${fmt.int(evTop)}` : '');
    drawSeries(evCanvas, ev, { color: pal.muted, fill: pal.accentWeak, max: evTop });
    if (ev.length) playheadOn(evCanvas, ev.length);

    const ls = data && Array.isArray(data.layer_spikes) ? data.layer_spikes : [];
    const neurons = data && Array.isArray(data.neurons) ? data.neurons : [];
    const layers = neurons.length || (ls[0] ? ls[0].length : 0);
    const series = [];
    let top = 0;
    for (let l = 0; l < layers; l += 1) {
      const nrn = Number(neurons[l]) || 1;
      const s = ls.map((row) => (row && Number.isFinite(Number(row[l])) ? (100 * Number(row[l])) / nrn : 0));
      series.push(s);
      for (const v of s) if (v > top) top = v;
    }
    // All layers share one canvas and one y scale (max % over all layers).
    const { ctx: c, w, h } = prepare(layerCanvas);
    series.forEach((s, l) => {
      const n = s.length;
      c.beginPath();
      s.forEach((v, i) => {
        const x = n > 1 ? (i / (n - 1)) * w : w / 2;
        const y = h - 2 - (fmt.clamp(v / (top || 1), 0, 1) * (h - 4));
        if (i) c.lineTo(x, y);
        else c.moveTo(x, y);
      });
      c.strokeStyle = pal.layers[l % pal.layers.length];
      c.lineWidth = 1.25;
      c.stroke();
    });
    if (ls.length) playheadOn(layerCanvas, ls.length);
    layerLegend.replaceChildren(...series.map((s, l) => {
      const sw = el('span', { class: 'legend-swatch', 'aria-hidden': 'true' });
      sw.style.background = pal.layers[l % pal.layers.length];
      return el('span', { class: 'legend-item' }, sw, `L${l + 1}`);
    }), series.length ? el('span', { class: 'faint tabular', text: `max ${fmt.fixed(top, 1)} %` }) : null);
  }

  function renderText() {
    labelsCol.replaceChildren();
    totalsCol.replaceChildren();
    if (!data) return;
    const labels = Array.isArray(data.labels) ? data.labels.map(String) : [];
    labels.forEach((label, i) => {
      const isPred = i === data.predicted;
      const isTrue = i === data.true;
      labelsCol.append(el('div', { class: ['raster-label', isPred && 'is-pred', !isPred && !isTrue && 'is-muted'] },
        el('span', { class: 'ellipsis', text: label, title: label }),
        isTrue ? el('span', { class: 'true-tag', text: 'label', title: 'The clip is labelled with this gesture' }) : null));
      const total = Array.isArray(data.totals) ? data.totals[i] : null;
      const rate = Array.isArray(data.rates) ? data.rates[i] : null;
      totalsCol.append(el('div', { class: ['raster-total', isPred && 'is-pred'], text: `${fmt.isNum(total) ? total : '–'} · ${fmt.pct(rate)}` }));
    });
    const pred = labels[data.predicted];
    const truth = labels[data.true];
    const match = Number.isInteger(data.predicted) && data.predicted === data.true;
    const rateP = Array.isArray(data.rates) ? data.rates[data.predicted] : null;
    verdict.replaceChildren(
      el('span', { class: 'muted', text: 'Network says:' }),
      el('strong', { text: pred || 'nothing' }),
      fmt.isNum(rateP) ? el('span', { class: 'muted tabular', text: `(${fmt.pct(rateP)} of frames)` }) : null,
      el('span', { class: 'faint', text: '·' }),
      el('span', { class: 'muted', text: `label: ${truth || 'not in this model'}` }),
      truth ? el('span', { class: ['verdict-mark', match ? 'is-ok' : 'is-bad'] }, icon(match ? 'check' : 'x', { size: 16 }), el('span', { text: match ? 'match' : 'mismatch' })) : null);
    canvas.setAttribute('aria-label', `Output spikes per gesture: ${labels.map((l, i) => `${l} ${Array.isArray(data.totals) ? data.totals[i] : '?'}`).join(', ')}. Predicted ${pred || 'nothing'}.`);
    setText(modelHint, data.model && data.model.name ? data.model.name : '');
    modelHint.title = data.model ? `${data.model.name || ''} · ${data.model.arch || ''} · ${data.model.bits || ''}` : '';
  }

  function redraw() {
    drawRaster();
    drawDetails();
  }

  const ro = new ResizeObserver(() => redraw());
  ro.observe(plot);
  scope.add(() => ro.disconnect());
  scope.sub('theme', () => requestAnimationFrame(() => {
    pal = colors();
    redraw();
  }));

  // ---------------------------------------------------------------- seeking

  let dragging = false;
  function seekFrom(ev) {
    if (!data || !Array.isArray(data.outputs) || !onSeek) return;
    const r = canvas.getBoundingClientRect();
    const f = Math.floor(((ev.clientX - r.left) / Math.max(1, r.width)) * data.outputs.length);
    onSeek(fmt.clamp(f, 0, data.outputs.length - 1));
  }
  canvas.addEventListener('pointerdown', (ev) => {
    if (!data) return;
    dragging = true;
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      /* best-effort */
    }
    seekFrom(ev);
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (dragging) seekFrom(ev);
  });
  for (const t of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(t, () => {
    dragging = false;
  });

  // ---------------------------------------------------------------- loading

  function showOverlay(node) {
    overlay.hidden = !node;
    overlay.replaceChildren(...(node ? [node] : []));
  }

  async function ensureModels() {
    if (models || modelsLoading) return;
    modelsLoading = true;
    try {
      const res = await api.get('/dataset/spike-models', null, { signal: scope.signal });
      if (!scope.alive) return;
      models = Array.isArray(res.models) ? res.models : [];
      const groups = new Map();
      for (const m of models) {
        const k = m.classes || 'other';
        if (!groups.has(k)) groups.set(k, []);
        groups.get(k).push(m);
      }
      modelSel.replaceChildren(el('option', { value: '', text: 'Board model (auto)' }),
        ...Array.from(groups, ([k, list]) => el('optgroup', { label: CLASS_NAMES[k] || k },
          ...list.map((m) => el('option', { value: String(m.id), text: `${m.name}${m.arch ? ` (${m.arch}${m.bits ? `, ${m.bits}` : ''})` : ''}` })))));
      modelSel.value = modelId;
    } catch {
      /* the select keeps the automatic choice only */
    } finally {
      modelsLoading = false;
    }
  }

  async function fetchSpikes() {
    if (!clipRef) return;
    if (ctrl) ctrl.abort();
    const c = scope.controller();
    ctrl = c;
    data = null;
    renderText();
    setText(verdict, '');
    verdict.append(el('span', { class: 'muted', text: 'Running the clip through the network…' }));
    drawRaster();
    showOverlay(el('div', { class: 'raster-loading' }, el('span', { class: 'spinner', 'aria-hidden': 'true' }), el('span', { text: 'Simulating the FPGA network…' })));
    try {
      const res = await api.get('/dataset/spikes', { source: clipRef.source, path: clipRef.path, model: modelId || null }, { signal: c.signal, timeout: 130000 });
      if (!scope.alive || c.signal.aborted) return;
      data = res && typeof res === 'object' ? res : null;
      showOverlay(null);
      renderText();
      redraw();
    } catch (err) {
      if (!scope.alive || c.signal.aborted || err.aborted) return;
      setText(verdict, '');
      const retry = btn('Retry', { kind: 'ghost', size: 'sm', icon: 'refresh', onClick: () => fetchSpikes() });
      showOverlay(el('div', { class: 'raster-error', role: 'alert' }, icon('alert', { size: 16 }), el('span', { text: `No spikes: ${err.message}` }), retry));
    }
  }

  modelSel.addEventListener('change', () => {
    modelId = modelSel.value;
    fetchSpikes();
  });

  return {
    el: root,
    /** New clip selected: cancel the previous request and fetch its spikes. */
    load(source, path) {
      clipRef = { source, path };
      frame = 0;
      ensureModels();
      fetchSpikes();
    },
    clear(message) {
      if (ctrl) ctrl.abort();
      ctrl = null;
      clipRef = null;
      data = null;
      renderText();
      setText(verdict, '');
      drawRaster();
      showOverlay(message ? el('div', { class: 'raster-loading muted', text: message }) : null);
    },
    /** Player moved: move the playhead. */
    setFrame(f) {
      if (f === frame) return;
      frame = f;
      drawRaster();
      if (details.open) drawDetails();
    },
  };
}
