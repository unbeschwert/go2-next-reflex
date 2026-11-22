// Five horizontal bars for the network's output rates (SSE `live.rates`), each
// with a thin marker at its trigger level (thresholds.min_rate[i]). The
// candidate is highlighted; bars above their trigger level are tinted.

import { el, setText, attachTip } from '../ui.js';
import * as fmt from '../format.js';

const DEFAULT_LABELS = ['Heart', 'Turn left 360', 'Turn right 360', 'Follow me', 'none'];

function frac(x) {
  return fmt.isNum(x) ? fmt.clamp(x, 0, 1) : 0;
}

export function createRateBars({ compact = false } = {}) {
  const root = el('div', { class: ['rates', compact && 'rates-compact'], role: 'group', 'aria-label': 'Gesture output rates' });
  let rows = [];
  let labelsKey = '';

  function buildRows(labels) {
    const key = labels.join('\u0000');
    if (key === labelsKey) return;
    labelsKey = key;
    rows = labels.map((label) => {
      const isNone = String(label).toLowerCase() === 'none';
      const fill = el('div', { class: 'rate-fill' });
      const marker = el('div', { class: 'rate-marker', hidden: true });
      const track = el('div', {
        class: 'rate-track', role: 'meter', 'aria-label': `${label} rate`, 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0',
      }, fill, marker);
      const value = el('span', { class: 'rate-value', text: fmt.DASH });
      const row = el('div', { class: ['rate-row', isNone && 'rate-none'] },
        el('span', { class: 'rate-label', text: isNone ? 'No gesture' : label, title: label }), track, value);
      return { row, fill, marker, value, track, isNone, min: null };
    });
    root.replaceChildren(...rows.map((r) => r.row));
  }

  function update(live) {
    const rates = live && Array.isArray(live.rates) ? live.rates : [];
    let labels = live && Array.isArray(live.labels) && live.labels.length ? live.labels : null;
    if (!labels) labels = rates.length ? DEFAULT_LABELS.slice(0, rates.length) : DEFAULT_LABELS;
    buildRows(labels.map(String));
    const th = (live && live.thresholds) || {};
    const minRate = Array.isArray(th.min_rate) ? th.min_rate : [];
    const candidate = live && Number.isInteger(live.candidate) ? live.candidate : null;

    rows.forEach((r, i) => {
      const rate = rates[i];
      const v = frac(rate);
      r.fill.style.width = `${(v * 100).toFixed(1)}%`;
      r.track.setAttribute('aria-valuenow', String(Math.round(v * 100)));
      setText(r.value, fmt.isNum(rate) ? fmt.pct(rate) : fmt.DASH);
      const min = !r.isNone && fmt.isNum(minRate[i]) ? minRate[i] : null;
      if (min !== r.min) {
        r.min = min;
        r.marker.hidden = min == null;
        if (min != null) {
          r.marker.style.left = `${(frac(min) * 100).toFixed(1)}%`;
          attachTip(r.track, `Trigger level ${fmt.pct(min)}`);
        }
      }
      r.row.classList.toggle('is-above', min != null && fmt.isNum(rate) && rate >= min);
      r.row.classList.toggle('is-candidate', candidate === i && !r.isNone);
    });
  }

  function reset() {
    for (const r of rows) {
      r.fill.style.width = '0%';
      setText(r.value, fmt.DASH);
      r.row.classList.remove('is-above', 'is-candidate');
    }
  }

  update(null);
  return { el: root, update, reset };
}
