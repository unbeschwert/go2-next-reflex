// Lifetime scope for a mounted page (or a sub-component). Everything a page
// starts (timers, polls, bus subscriptions, DOM listeners on window/document,
// in-flight fetches via scope.signal) is registered here and released by
// dispose() when the router leaves the page.

import { on } from './events.js';

export function createScope() {
  const cleanups = [];
  const timeouts = new Set();
  const controller = new AbortController();
  let alive = true;

  const runSafe = (fn) => {
    try {
      fn();
    } catch (err) {
      console.error('[scope] cleanup failed', err);
    }
  };

  const scope = {
    get alive() {
      return alive;
    },
    /** AbortSignal that fires on dispose: pass it to api calls. */
    signal: controller.signal,

    add(fn) {
      if (alive) cleanups.push(fn);
      else runSafe(fn);
    },

    interval(fn, ms) {
      const id = setInterval(() => {
        if (alive) fn();
      }, ms);
      scope.add(() => clearInterval(id));
      return id;
    },

    timeout(fn, ms) {
      const id = setTimeout(() => {
        timeouts.delete(id);
        if (alive) fn();
      }, ms);
      timeouts.add(id);
      return id;
    },

    clearTimeout(id) {
      clearTimeout(id);
      timeouts.delete(id);
    },

    listen(target, type, fn, options) {
      target.addEventListener(type, fn, options);
      scope.add(() => target.removeEventListener(type, fn, options));
    },

    /** Subscribe to a bus / SSE event for the lifetime of the scope. */
    sub(name, fn) {
      scope.add(on(name, (data) => {
        if (alive) fn(data);
      }));
    },

    /** AbortController that is aborted when the scope is disposed. */
    controller() {
      const c = new AbortController();
      if (!alive) c.abort();
      else controller.signal.addEventListener('abort', () => c.abort(), { once: true });
      return c;
    },

    /**
     * Sequential poll: fn runs, then waits ms, never overlapping. Skips while
     * the tab is hidden unless runHidden. Returns {now(), stop(), setDelay(ms)}.
     */
    poll(fn, ms, { runHidden = false, immediate = true } = {}) {
      let timer = null;
      let running = false;
      let again = false;
      let stopped = false;
      let delay = ms;
      const schedule = (d) => {
        clearTimeout(timer);
        timer = null;
        if (!alive || stopped) return;
        timer = setTimeout(tick, d);
      };
      const tick = async () => {
        timer = null;
        if (!alive || stopped) return;
        if (!runHidden && document.hidden) {
          schedule(delay);
          return;
        }
        running = true;
        try {
          await fn();
        } catch {
          /* fn renders its own error state */
        }
        running = false;
        if (again) {
          again = false;
          schedule(0);
        } else {
          schedule(delay);
        }
      };
      const handle = {
        now() {
          if (running) again = true;
          else schedule(0);
        },
        stop() {
          stopped = true;
          clearTimeout(timer);
        },
        setDelay(d) {
          delay = d;
        },
      };
      scope.add(() => handle.stop());
      schedule(immediate ? 0 : delay);
      return handle;
    },

    /** Child scope disposed together with this one (or earlier, on its own). */
    child() {
      const c = createScope();
      scope.add(() => c.dispose());
      return c;
    },

    dispose() {
      if (!alive) return;
      alive = false;
      controller.abort();
      for (const id of timeouts) clearTimeout(id);
      timeouts.clear();
      while (cleanups.length) runSafe(cleanups.pop());
    },
  };
  return scope;
}

/** True when an error only means "the page went away" (no need to show it). */
export function isAbort(err) {
  return !!(err && (err.aborted || err.name === 'AbortError'));
}
