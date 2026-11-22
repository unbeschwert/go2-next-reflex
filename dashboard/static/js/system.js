// System health summary shared by the top-bar "System" button and the Home page:
// four rows (Board, Robot link, Live stream, Training) and the worst state.

import * as fmt from './format.js';

export const MODE_TEXT = {
  stopped: 'Stopped', test: 'Test mode', robot: 'Robot mode', 'robot-dry': 'Robot dry-run', unknown: 'Unknown',
};

export const LINK_TEXT = {
  go2: 'Go2 (real robot)', rviz: 'RViz animator', sim: 'Simulator', none: 'None',
};

const STATUS_TEXT = {
  listening: 'listening', candidate: 'candidate', 'waiting-for-calm': 'waiting to re-arm', busy: 'busy', paused: 'paused', 'no-board': 'no board',
};

const RANK = { bad: 3, warn: 2, real: 1, ok: 0, info: 0, busy: 0, off: 0 };

/**
 * ov: GET /overview answer (or undefined); live: last SSE `live` payload when fresh (else null);
 * robot: last SSE `robot` payload (optional).
 */
export function systemRows(ov, live, robot) {
  const rows = [];
  const target = ov && ov.target ? ov.target.mode : null;
  const b = ov && ov.board;

  // Board
  let board;
  if (!b) board = ['off', 'Checking…'];
  else if (target === 'sim') board = ['off', b.reachable ? 'Not used (simulator target)' : 'Offline (not needed)'];
  else if (!b.reachable) board = ['bad', 'Offline'];
  else if (b.ssh === false) board = ['warn', 'Reachable, no SSH'];
  else board = ['ok', `${MODE_TEXT[b.mode] || 'Unknown mode'}${fmt.isNum(b.ping_ms) ? ` · ${fmt.ms(b.ping_ms)}` : ''}`];
  rows.push({ key: 'board', name: 'Board', state: board[0], value: board[1], href: '#/board', icon: 'cpu' });

  // Robot link
  const link = (robot && robot.link) || (ov && ov.robot && ov.robot.link);
  const linkState = link === 'go2' ? 'real' : (link === 'none' || !link ? 'off' : 'ok');
  rows.push({ key: 'link', name: 'Robot link', state: linkState, value: link ? (LINK_TEXT[link] || link) : 'Unknown', href: '#/drive', icon: 'wifi' });

  // Live stream
  const idleExpected = target !== 'sim' && b && (b.mode === 'stopped' || b.mode === 'unknown');
  let liveRow;
  if (live) {
    if (live.connected) liveRow = ['ok', `${fmt.int(live.fps)} fps · ${STATUS_TEXT[live.status] || live.status || 'live'}`];
    else liveRow = idleExpected ? ['off', 'Idle (inference stopped)'] : ['bad', 'No stream'];
  } else if (ov && ov.live) {
    if (ov.live.connected) liveRow = ['ok', `${fmt.int(ov.live.fps)} fps`];
    else liveRow = idleExpected ? ['off', 'Idle (inference stopped)'] : ['bad', 'No stream'];
  } else {
    liveRow = ['off', 'No data'];
  }
  rows.push({ key: 'live', name: 'Live stream', state: liveRow[0], value: liveRow[1], href: '#/live', icon: 'activity' });

  // Training machine (optional: "not set up" is neutral, not a problem)
  const train = trainingState(ov && ov.training);
  rows.push({ key: 'training', name: 'Training', state: train.state, value: train.text, href: '#/training', icon: 'trending' });

  return rows;
}

const SHORT = {
  board: { bad: 'Board offline', warn: 'Board: no SSH' },
  live: { bad: 'Live stream down' },
  training: { warn: 'Training machine unreachable' },
};

/**
 * /overview `training` -> {state, text, configured}. configured false -> neutral "Not set up".
 */
export function trainingState(t) {
  if (!t) return { state: 'off', text: 'Checking\u2026', configured: null };
  if (t.configured === false) return { state: 'off', text: 'Not set up', configured: false };
  const label = t.host || 'Training machine';
  if (t.ok === true) {
    return { state: 'ok', text: `${label} \u00b7 ${fmt.isNum(t.running_jobs) ? `${fmt.int(t.running_jobs)} running` : (t.detail || 'reachable')}`, configured: true };
  }
  if (t.ok === false) return { state: 'warn', text: `${label}: ${t.detail || 'unreachable'}`, configured: true };
  return { state: 'off', text: `${label} \u00b7 checking\u2026`, configured: true };
}

/**
 * Board model -> {title, detail, full}: prefers the backend's readable `label`
 * ("Own gestures \u00b7 best80 (80\u00d780, 16-bit)" -> title "Own gestures \u00b7 best80", detail "80\u00d780, 16-bit").
 * The file name is only a fallback (and shown in Board > Details).
 */
export function modelLabel(model) {
  if (!model) return null;
  const label = typeof model.label === 'string' && model.label.trim() ? model.label.trim() : null;
  if (label) {
    // "Own gestures · best80 (80×80, 16-bit)" -> title "best80", detail "Own gestures · 80×80, 16-bit"
    const m = /^(.*?)\s*\(([^()]*)\)\s*$/.exec(label);
    const main = m ? m[1] : label;
    const paren = m ? m[2] : null;
    const parts = main.split(/\s+\u00b7\s+/);
    if (parts.length > 1) {
      const title = parts.pop();
      return { title, detail: [parts.join(' \u00b7 '), paren].filter(Boolean).join(' \u00b7 '), full: label };
    }
    return { title: main, detail: paren, full: label };
  }
  if (model.name) return { title: shortModel(model.name), detail: null, full: model.name };
  return null;
}

/** Worst state over the rows + one short summary text. */
export function summarize(rows) {
  let worst = null;
  for (const r of rows) {
    if (!worst || RANK[r.state] > RANK[worst.state]) worst = r;
  }
  if (!worst || RANK[worst.state] < 2) return { state: 'ok', text: 'All systems OK' };
  const issues = rows.filter((r) => RANK[r.state] >= 2).length;
  const text = (SHORT[worst.key] && SHORT[worst.key][worst.state]) || `${worst.name}: ${worst.value}`;
  return { state: worst.state, text: issues > 1 ? `${text} +${issues - 1}` : text };
}

/** Short model name: "own_gestures_nb40_q8h" -> "nb40_q8h". */
export function shortModel(name) {
  if (!name || typeof name !== 'string') return null;
  const parts = name.split('_');
  return parts.length > 2 ? parts.slice(-2).join('_') : name;
}
