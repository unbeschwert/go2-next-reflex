// fetch wrapper implementing the API.md conventions:
//  - same-origin cookie session; 401 -> login screen (via onUnauthorized handler)
//  - X-CSRF-Token on every non-GET; on 403 refetch /auth/state once and retry once
//  - 409 {confirm} -> modal question; on Confirm resend with "confirm": true (never automatic)
//  - errors -> ApiError with a readable message (422 detail kept for inline display)
//  - network failures flip a global "net down" flag (banner) and probe /api/health until back

import { confirmDialog } from './ui.js';
import { emit } from './events.js';

const DEFAULT_TIMEOUT_MS = 20000;
const PROBE_MS = 2000;

export class ApiError extends Error {
  constructor(message, info = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = info.status ?? 0;
    this.detail = info.detail ?? null;
    this.data = info.data ?? null;
    this.network = !!info.network;
    this.timeout = !!info.timeout;
    this.aborted = !!info.aborted;
    this.cancelled = !!info.cancelled;
    this.unauthorized = !!info.unauthorized;
    this.needsConfirm = !!info.needsConfirm;
  }
}

let csrf = null;
let unauthorizedHandler = null;
let unauthorizedPending = false;
let netDown = false;
let probeTimer = null;

export function setCsrf(token) {
  csrf = token || null;
  unauthorizedPending = false;
}

export function onUnauthorized(fn) {
  unauthorizedHandler = fn;
}

export function isNetworkDown() {
  return netDown;
}

function setNetDown(down) {
  if (down === netDown) return;
  netDown = down;
  emit('net-state', { down });
  if (down) startProbe();
  else stopProbe();
}

function startProbe() {
  if (probeTimer) return;
  probeTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/health', { cache: 'no-store', credentials: 'same-origin' });
      if (res.status > 0) setNetDown(false); // any HTTP answer = server reachable
    } catch {
      /* still down */
    }
  }, PROBE_MS);
}

function stopProbe() {
  clearInterval(probeTimer);
  probeTimer = null;
}

function handleUnauthorized() {
  csrf = null;
  if (unauthorizedPending) return;
  unauthorizedPending = true;
  queueMicrotask(() => {
    if (unauthorizedHandler) unauthorizedHandler();
  });
}

/** Turn FastAPI / contract error details into one readable line. */
export function formatDetail(detail, status) {
  if (detail == null || detail === '') return status ? `Request failed (HTTP ${status})` : 'Request failed';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      if (typeof d === 'string') return d;
      if (d && typeof d === 'object') {
        const loc = Array.isArray(d.loc) ? d.loc.filter((p) => p !== 'body').join('.') : '';
        return [loc, d.msg].filter(Boolean).join(': ');
      }
      return String(d);
    }).join('; ');
  }
  if (typeof detail === 'object' && detail.msg) return String(detail.msg);
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

/** Field names mentioned by a 422 detail (FastAPI loc arrays or "key: ..." strings). */
export function detailKeys(detail) {
  const keys = new Set();
  if (Array.isArray(detail)) {
    for (const d of detail) {
      if (d && Array.isArray(d.loc) && d.loc.length) keys.add(String(d.loc[d.loc.length - 1]));
    }
  }
  return keys;
}

function buildUrl(path, params) {
  let url = `/api${path}`;
  if (params) {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') sp.append(k, String(v));
    }
    const qs = sp.toString();
    if (qs) url += (url.includes('?') ? '&' : '?') + qs;
  }
  return url;
}

async function rawFetch(method, url, { body, signal, timeout, accept = 'json' } = {}) {
  const ctrl = new AbortController();
  let timedOut = false;
  const ms = timeout ?? DEFAULT_TIMEOUT_MS;
  const timer = ms > 0 ? setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, ms) : null;
  const relay = () => ctrl.abort();
  if (signal) {
    if (signal.aborted) ctrl.abort();
    else signal.addEventListener('abort', relay, { once: true });
  }
  const headers = { Accept: accept === 'json' ? 'application/json' : '*/*' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (method !== 'GET' && csrf) headers['X-CSRF-Token'] = csrf;
  try {
    const res = await fetch(url, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      credentials: 'same-origin',
      cache: 'no-store',
      signal: ctrl.signal,
    });
    setNetDown(false);
    let data = null;
    let blob = null;
    if (accept === 'blob' && res.ok) {
      blob = await res.blob();
    } else {
      const text = await res.text();
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = { detail: text.slice(0, 300) };
        }
      }
    }
    return { res, data, blob };
  } catch (err) {
    if (timedOut) throw new ApiError('The server did not answer in time', { timeout: true });
    if (ctrl.signal.aborted) throw new ApiError('Request aborted', { aborted: true });
    setNetDown(true);
    throw new ApiError('Server unreachable', { network: true });
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', relay);
  }
}

/** GET /api/auth/state; also refreshes the CSRF token. */
export async function getAuthState({ signal } = {}) {
  const { res, data } = await rawFetch('GET', '/api/auth/state', { signal, timeout: 8000 });
  if (!res.ok) throw new ApiError(formatDetail(data && data.detail, res.status), { status: res.status, data });
  const state = data || {};
  if (state.logged_in && state.csrf) csrf = state.csrf;
  else if (!state.logged_in) csrf = null;
  return state;
}

/**
 * request(method, path, opts)
 * opts: body, params (query), signal, timeout,
 *       noConfirm (409 confirm -> throw instead of asking; used for 10 Hz moves / E-STOP),
 *       risk ('danger' | 'warn' | 'normal': style of the confirm button),
 *       accept ('json' | 'blob')
 */
export async function request(method, path, opts = {}) {
  const { params, signal, timeout, noConfirm = false, risk = 'warn', accept = 'json' } = opts;
  const url = buildUrl(path, params);
  let body = method === 'GET' ? undefined : (opts.body === undefined ? {} : opts.body);
  const isAuthPath = path.startsWith('/auth/');
  let csrfRetried = false;
  let confirms = 0;

  for (;;) {
    const { res, data, blob } = await rawFetch(method, url, { body, signal, timeout, accept });
    if (res.ok) return accept === 'blob' ? blob : (data ?? {});

    const status = res.status;
    const detail = data && typeof data === 'object' && 'detail' in data ? data.detail : null;

    if (status === 401 && !isAuthPath) {
      handleUnauthorized();
      throw new ApiError('Your session has ended. Please log in again.', { status, unauthorized: true });
    }

    if (status === 403 && !csrfRetried && method !== 'GET') {
      csrfRetried = true;
      let state = null;
      try {
        state = await getAuthState({ signal });
      } catch (err) {
        if (err instanceof ApiError && (err.network || err.aborted || err.timeout)) throw err;
      }
      if (state && state.logged_in) continue; // fresh token, retry once
      if (state && !state.logged_in && !isAuthPath) {
        handleUnauthorized();
        throw new ApiError('Your session has ended. Please log in again.', { status: 401, unauthorized: true });
      }
    }

    if (status === 409 && data && typeof data.confirm === 'string' && data.confirm) {
      if (noConfirm || confirms >= 3) {
        throw new ApiError(data.confirm, { status, detail, data, needsConfirm: true });
      }
      confirms += 1;
      const yes = await confirmDialog({
        title: 'Please confirm',
        message: data.confirm,
        detail: typeof detail === 'string' ? detail : null,
        risk: data.risk === 'danger' || data.risk === 'warn' ? data.risk : risk,
        confirmLabel: 'Confirm',
      });
      if (!yes) throw new ApiError('Cancelled', { status, detail, data, cancelled: true });
      body = { ...(body && typeof body === 'object' ? body : {}), confirm: true };
      continue;
    }

    throw new ApiError(formatDetail(detail, status), { status, detail, data });
  }
}

export const api = {
  get: (path, params, opts = {}) => request('GET', path, { ...opts, params }),
  post: (path, body, opts = {}) => request('POST', path, { ...opts, body }),
  put: (path, body, opts = {}) => request('PUT', path, { ...opts, body }),
  blob: (path, params, opts = {}) => request('GET', path, { ...opts, params, accept: 'blob' }),
};
