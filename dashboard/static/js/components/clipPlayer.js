// Event-clip player. GET /api/dataset/clip returns base64 of
// uint8[frames][2][size][size] (event counts per 10 ms frame, channel 0 = OFF,
// 1 = ON). Frames are drawn into a size x size canvas that CSS scales up with
// nearest-neighbour (image-rendering: pixelated).

import { el, setText, uid, iconBtn, select, icon, attachTip } from '../ui.js';
import * as fmt from '../format.js';

const SPEEDS = [0.25, 0.5, 1];
const EXPOSURES = [1, 3, 5]; // frames summed per displayed image (10 ms each)

/** Decode the clip payload. Throws with a readable message on malformed data. */
export function decodeClip(answer) {
  const size = Number(answer && answer.size);
  const frames = Number(answer && answer.frames);
  if (!Number.isInteger(size) || size <= 0 || !Number.isInteger(frames) || frames <= 0) {
    throw new Error('The clip has no frames.');
  }
  if (typeof answer.data !== 'string') throw new Error('The clip has no event data.');
  let bin;
  try {
    bin = atob(answer.data);
  } catch {
    throw new Error('The clip data is not valid base64.');
  }
  const expected = frames * 2 * size * size;
  if (bin.length < expected) throw new Error(`The clip data is truncated (${bin.length} of ${expected} bytes).`);
  const buf = new Uint8Array(expected);
  for (let i = 0; i < expected; i += 1) buf[i] = bin.charCodeAt(i);
  const fps = fmt.isNum(answer.fps) && answer.fps > 0 ? answer.fps : 100;
  return { size, frames, fps, buf, meta: answer.meta || {} };
}

/**
 * onPrev / onNext: previous / next clip in the list (optional).
 * onFrame(frame, frames): called whenever the shown frame changes (shared playhead).
 */
export function createClipPlayer(scope, { onPrev, onNext, onFrame } = {}) {
  let clip = null;
  let frame = 0;
  let playing = false;
  let speed = 1;
  let exposure = 3;
  let loop = true;
  let raf = 0;
  let lastTs = 0;
  let accum = 0;
  let image = null;

  const canvas = el('canvas', { class: 'clip-canvas', width: 80, height: 80, role: 'img', 'aria-label': 'Event clip frame' });
  const ctx = canvas.getContext('2d');
  const empty = el('div', { class: 'clip-empty', text: 'Select a clip to play it.' });
  const stage = el('div', { class: 'clip-stage' }, canvas, empty);

  const prevBtn = iconBtn('skipBack', 'Previous clip (J)');
  const nextBtn = iconBtn('skipForward', 'Next clip (K)');
  const playBtn = iconBtn('play', 'Play');
  playBtn.classList.replace('btn-ghost', 'btn-secondary');
  const loopBtn = iconBtn('repeat', 'Loop', { pressed: true });
  const speedSel = select(SPEEDS.map((s) => ({ value: s, label: `${s}x` })), { value: 1, ariaLabel: 'Playback speed', cls: 'input-sm' });
  const expSel = select(EXPOSURES.map((n) => ({ value: n, label: `${n * 10} ms` })), { value: exposure, ariaLabel: 'Exposure: frames summed per image', cls: 'input-sm' });
  attachTip(expSel, 'Exposure: how many 10 ms frames are summed per image');
  const scrubId = uid('scrub');
  const scrub = el('input', { type: 'range', id: scrubId, class: 'clip-scrub', min: '0', max: '0', step: '1', value: '0', disabled: true, 'aria-label': 'Frame' });
  const frameText = el('span', { class: 'frame-text' });

  const titleEl = el('div', { class: 'player-title', text: 'No clip selected' });
  const metaEl = el('div', { class: 'small muted' });
  const legend = el('div', { class: 'legend' },
    el('span', { class: 'legend-item' }, el('span', { class: 'legend-swatch swatch-off', 'aria-hidden': 'true' }), 'OFF'),
    el('span', { class: 'legend-item' }, el('span', { class: 'legend-swatch swatch-on', 'aria-hidden': 'true' }), 'ON'),
    el('span', { class: 'legend-item' }, el('span', { class: 'legend-swatch swatch-both', 'aria-hidden': 'true' }), 'both'));

  const transport = el('div', { class: 'transport' }, prevBtn, playBtn, nextBtn, scrub, frameText, speedSel, expSel, loopBtn);
  const meta = el('div', { class: 'player-meta' }, el('div', null, titleEl, metaEl), legend);

  if (onPrev) prevBtn.addEventListener('click', onPrev);
  else prevBtn.hidden = true;
  if (onNext) nextBtn.addEventListener('click', onNext);
  else nextBtn.hidden = true;

  function draw() {
    if (!clip) return;
    const { size, buf } = clip;
    const n = size * size;
    if (!image || image.width !== size) image = ctx.createImageData(size, size);
    const px = image.data;
    const first = Math.max(0, frame - exposure + 1);
    for (let i = 0; i < n; i += 1) {
      let off = 0;
      let on = 0;
      for (let f = first; f <= frame; f += 1) {
        const o = f * 2 * n;
        off += buf[o + i];
        on += buf[o + n + i];
      }
      let r = 0;
      let g = 0;
      let b = 0;
      if (off || on) {
        const k = Math.min(1, 0.45 + 0.18 * (off + on));
        if (off && on) {
          r = 235 * k; g = 235 * k; b = 225 * k; // both: white-ish
        } else if (on) {
          r = 255 * k; g = 140 * k; b = 30 * k; // ON: orange-ish
        } else {
          r = 60 * k; g = 150 * k; b = 255 * k; // OFF: blue-ish
        }
      }
      const j = i * 4;
      px[j] = r;
      px[j + 1] = g;
      px[j + 2] = b;
      px[j + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    scrub.value = String(frame);
    if (onFrame) onFrame(frame, clip.frames);
    setText(frameText, `${fmt.secs(frame / clip.fps, 2)} / ${fmt.secs(clip.frames / clip.fps, 2)}`);
  }

  function setPlaying(p) {
    playing = p && !!clip;
    const ic = icon(playing ? 'pause' : 'play');
    playBtn.replaceChildren(ic);
    playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    playBtn.dataset.tip = playing ? 'Pause' : 'Play';
    cancelAnimationFrame(raf);
    raf = 0;
    if (playing) {
      if (frame >= clip.frames - 1) frame = 0;
      lastTs = 0;
      accum = 0;
      raf = requestAnimationFrame(step);
    }
  }

  function step(ts) {
    raf = 0;
    if (!playing || !clip) return;
    if (lastTs) accum += ((ts - lastTs) / 1000) * clip.fps * speed;
    lastTs = ts;
    if (accum >= 1) {
      const adv = Math.floor(accum);
      accum -= adv;
      let next = frame + adv;
      if (next >= clip.frames) {
        if (loop) {
          next %= clip.frames;
        } else {
          frame = clip.frames - 1;
          draw();
          setPlaying(false);
          return;
        }
      }
      frame = next;
      draw();
    }
    raf = requestAnimationFrame(step);
  }

  playBtn.addEventListener('click', () => setPlaying(!playing));
  speedSel.addEventListener('change', () => {
    speed = Number(speedSel.value) || 1;
  });
  expSel.addEventListener('change', () => {
    exposure = Number(expSel.value) || 1;
    draw();
  });
  loopBtn.addEventListener('click', () => {
    loop = !loop;
    loopBtn.setAttribute('aria-pressed', String(loop));
  });
  scrub.addEventListener('input', () => {
    if (!clip) return;
    if (playing) setPlaying(false);
    frame = fmt.clamp(Number(scrub.value) || 0, 0, clip.frames - 1);
    draw();
  });

  scope.add(() => cancelAnimationFrame(raf));

  function setEnabledAll(on) {
    scrub.disabled = !on;
    playBtn.disabled = !on;
  }

  return {
    stage,
    transport,
    meta,
    /** Load a /dataset/clip answer. Throws on malformed data. */
    load(answer) {
      setPlaying(false);
      const decoded = decodeClip(answer);
      clip = decoded;
      canvas.width = decoded.size;
      canvas.height = decoded.size;
      image = null;
      frame = 0;
      scrub.max = String(decoded.frames - 1);
      setEnabledAll(true);
      empty.hidden = true;
      const m = decoded.meta;
      setText(titleEl, `${m.label ?? m.cls ?? 'Clip'}${m.session != null ? ` · session ${m.session}` : ''}`);
      const parts = [];
      if (fmt.isNum(m.events)) parts.push(`${fmt.int(m.events)} events`);
      parts.push(fmt.secs(fmt.isNum(m.duration_s) ? m.duration_s : decoded.frames / decoded.fps, 2));
      parts.push(`${decoded.frames} frames at ${decoded.fps} fps`);
      setText(metaEl, parts.join(' · '));
      metaEl.title = m.path ? String(m.path) : '';
      draw();
      setPlaying(true);
    },
    clear(message) {
      setPlaying(false);
      clip = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setEnabledAll(false);
      empty.hidden = false;
      setText(empty, message || 'Select a clip to play it.');
      setText(frameText, '');
      setText(titleEl, 'No clip selected');
      setText(metaEl, '');
    },
    togglePlay() {
      setPlaying(!playing);
    },
    /** Jump to a frame (pauses playback). */
    seek(f) {
      if (!clip) return;
      if (playing) setPlaying(false);
      frame = fmt.clamp(Math.round(f), 0, clip.frames - 1);
      draw();
    },
    get frame() {
      return frame;
    },
  };
}
