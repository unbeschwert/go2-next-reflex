// MJPEG <img> with a placeholder and automatic retry every 2 s on error.
// The stream is closed while the tab is hidden and when the page is left
// (browsers only allow ~6 connections per host; an orphaned MJPEG stream
// would eat one).

import { el, setText, icon, btn } from '../ui.js';

export function createCameraView(scope, { src = '/api/live/camera', alt = 'Live event-camera view', retryMs = 2000 } = {}) {
  const img = el('img', { class: 'camera-img', alt, decoding: 'async' });
  const msg = el('span', { text: 'Connecting to the camera…' });
  const retryBtn = btn('Retry now', { kind: 'secondary', size: 'sm', icon: 'refresh' });
  const ph = el('div', { class: 'camera-placeholder', role: 'status' }, icon('camera', { size: 28 }), msg, retryBtn);
  const root = el('div', { class: 'camera' }, img, ph);

  let stopped = true;
  let retryTimer = null;

  const setLive = (live) => {
    root.classList.toggle('is-live', live);
    ph.hidden = live;
  };

  function start() {
    clearTimeout(retryTimer);
    retryTimer = null;
    stopped = false;
    setLive(false);
    retryBtn.hidden = true;
    setText(msg, 'Connecting to the camera…');
    img.src = `${src}${src.includes('?') ? '&' : '?'}r=${Date.now()}`;
  }

  function stop(text) {
    stopped = true;
    clearTimeout(retryTimer);
    retryTimer = null;
    img.removeAttribute('src');
    setLive(false);
    if (text) setText(msg, text);
  }

  img.addEventListener('load', () => {
    if (!stopped) setLive(true);
  });
  img.addEventListener('error', () => {
    if (stopped) return;
    setLive(false);
    retryBtn.hidden = false;
    setText(msg, 'No camera stream. Retrying every 2 s…');
    clearTimeout(retryTimer);
    retryTimer = setTimeout(() => {
      if (!stopped && !document.hidden) start();
    }, retryMs);
  });
  retryBtn.addEventListener('click', start);

  // Some browsers fire `load` only when an MJPEG stream ends: also watch naturalWidth.
  const liveCheck = setInterval(() => {
    if (!stopped && img.naturalWidth > 0 && !root.classList.contains('is-live')) setLive(true);
  }, 500);

  const onVisibility = () => {
    if (document.hidden) stop('Stream paused while this tab is hidden.');
    else start();
  };
  document.addEventListener('visibilitychange', onVisibility);

  scope.add(() => {
    document.removeEventListener('visibilitychange', onVisibility);
    clearInterval(liveCheck);
    stop();
  });

  if (!document.hidden) start();
  else stop('Stream paused while this tab is hidden.');
  return { el: root, restart: start };
}
