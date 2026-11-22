// Number and time formatting. Every function returns a plain string and
// tolerates missing / non-numeric input (renders an en dash).

export const DASH = '–';

export function isNum(x) {
  return typeof x === 'number' && Number.isFinite(x);
}

export function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

/** 0.62 -> "62%" */
export function pct(x, digits = 0) {
  return isNum(x) ? `${(x * 100).toFixed(digits)}%` : DASH;
}

export function fixed(x, digits = 2) {
  return isNum(x) ? x.toFixed(digits) : DASH;
}

/** 1.234 -> "1.2 ms", 42.4 -> "42 ms" */
export function ms(x) {
  if (!isNum(x)) return DASH;
  return `${x < 10 ? x.toFixed(1) : Math.round(x)} ms`;
}

/** 0.5 -> "0.5 s" */
export function secs(x, digits = 1) {
  return isNum(x) ? `${x.toFixed(digits)} s` : DASH;
}

/** Countdown style: 42.3 -> "0:43" */
export function clock(s) {
  if (!isNum(s)) return DASH;
  const t = Math.max(0, Math.ceil(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}

/** Human duration: 12 -> "12 s", 192 -> "3 min 12 s", 7300 -> "2 h 1 min" */
export function duration(s) {
  if (!isNum(s)) return DASH;
  const t = Math.max(0, Math.round(s));
  if (t < 60) return `${t} s`;
  if (t < 3600) return `${Math.floor(t / 60)} min ${t % 60} s`;
  const h = Math.floor(t / 3600);
  return `${h} h ${Math.floor((t % 3600) / 60)} min`;
}

/** Integer with thousands separators. */
export function int(n) {
  return isNum(n) ? Math.round(n).toLocaleString('en-US') : DASH;
}

/** SI prefix: 152000 -> "152.0k" */
export function si(n, digits = 1) {
  if (!isNum(n)) return DASH;
  const a = Math.abs(n);
  const units = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
  for (const [v, u] of units) {
    if (a >= v) return `${(n / v).toFixed(digits)}${u}`;
  }
  return String(Math.round(n * 100) / 100);
}

export function bytes(n) {
  if (!isNum(n)) return DASH;
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n;
  let i = -1;
  do {
    v /= 1024;
    i += 1;
  } while (v >= 1024 && i < units.length - 1);
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

/** Accepts epoch seconds, epoch milliseconds or an ISO string. */
export function toDate(t) {
  if (isNum(t)) return new Date(t > 1e12 ? t : t * 1000);
  if (typeof t === 'string' && t) {
    const d = new Date(t);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  return null;
}

/** "14:03:22" (or "14:03:22.4" with tenths). Unparseable strings are returned as-is. */
export function timeOfDay(t, tenths = false) {
  const d = toDate(t);
  if (!d) return typeof t === 'string' && t ? t : DASH;
  const base = d.toLocaleTimeString('en-GB', { hour12: false });
  return tenths ? `${base}.${Math.floor(d.getMilliseconds() / 100)}` : base;
}

export function dateTime(t) {
  const d = toDate(t);
  if (!d) return typeof t === 'string' && t ? t : DASH;
  return d.toLocaleString('en-GB', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
}

/** "3 s ago", "5 min ago" relative to the browser clock. */
export function ago(t) {
  const d = toDate(t);
  if (!d) return DASH;
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 0) return 'just now';
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

/** Accuracy given either as a fraction (0.93) or a percent (93.1). */
export function accuracy(x) {
  if (!isNum(x)) return DASH;
  return x <= 1 ? pct(x, 1) : `${x.toFixed(1)}%`;
}

/** Number of decimals implied by a step (0.05 -> 2). */
export function decimalsFor(step) {
  if (!isNum(step) || step <= 0) return 2;
  const s = String(step);
  if (s.includes('e-')) return parseInt(s.split('e-')[1], 10) || 2;
  const i = s.indexOf('.');
  return i < 0 ? 0 : s.length - i - 1;
}

export function roundTo(x, decimals) {
  const f = 10 ** decimals;
  return Math.round(x * f) / f;
}

/** Value with optional unit: (0.5, "s", 2) -> "0.50 s" */
export function withUnit(v, unit, decimals) {
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (!isNum(v)) return v == null ? DASH : String(v);
  const txt = decimals == null ? String(v) : v.toFixed(decimals);
  return unit ? `${txt} ${unit}` : txt;
}

/**
 * Short readable description of an arbitrary JSON value (used for fields
 * whose exact shape the API leaves open, e.g. last_cmd, teleop, result).
 */
export function describe(v, max = 200) {
  if (v == null) return DASH;
  if (typeof v === 'string') return v.length > max ? `${v.slice(0, max)}…` : v;
  if (typeof v === 'number') return isNum(v) ? String(Math.round(v * 1000) / 1000) : DASH;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  let s;
  if (Array.isArray(v)) {
    s = v.map((x) => describe(x, max)).join(', ');
  } else {
    s = Object.entries(v)
      .map(([k, x]) => `${k}: ${typeof x === 'object' && x !== null ? JSON.stringify(x) : describe(x, max)}`)
      .join(', ');
  }
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

/** Title-case a machine word: "robot-dry" -> "Robot dry" */
export function humanize(s) {
  if (typeof s !== 'string' || !s) return DASH;
  const t = s.replace(/[_-]+/g, ' ');
  return t.charAt(0).toUpperCase() + t.slice(1);
}
