// "Virtual person" segmented control for the simulator (POST /api/sim/person).

import { api } from '../api.js';
import { el, act, segmented } from '../ui.js';

const PERSON = {
  none: { label: 'Idle', icon: 'person', tip: 'A person stands in front of the camera without gesturing' },
  heart: { label: 'Heart', icon: 'heart', tip: 'The virtual person shows the Heart gesture' },
  left: { label: 'Left', icon: 'rotateCcw', tip: 'The virtual person shows Turn left' },
  right: { label: 'Right', icon: 'rotateCw', tip: 'The virtual person shows Turn right' },
  empty: { label: 'Nobody', icon: 'ban', tip: 'Nobody in front of the camera (empty scene)' },
};

/**
 * onState(simState) is called with every sim state answer.
 * Call setState(simState) when the page already has a fresh /sim/state.
 */
export function createVirtualPerson({ onState, compact = false } = {}) {
  const host = el('div', { class: 'virtual-person' });
  let seg = null;
  let peopleKey = '';
  let state = null;

  function build(people) {
    const key = people.join('|');
    if (key === peopleKey) return;
    peopleKey = key;
    seg = segmented(people.map((p) => ({
      value: p,
      label: compact ? null : (PERSON[p] ? PERSON[p].label : p),
      icon: PERSON[p] ? PERSON[p].icon : 'person',
      tip: PERSON[p] ? PERSON[p].tip : p,
    })), {
      label: 'Virtual person',
      cls: 'person-seg',
      onChange: async (p, b) => {
        if (state && state.person === p) return;
        const res = await act(b, () => api.post('/sim/person', { gesture: p }));
        if (res) {
          setState(res);
          if (onState) onState(res);
        }
      },
    });
    host.replaceChildren(seg.el);
  }

  function setState(st) {
    state = st || null;
    const people = state && Array.isArray(state.people) && state.people.length
      ? state.people.map(String) : ['none', 'heart', 'left', 'right', 'empty'];
    build(people);
    seg.set(state ? state.person : null);
    seg.setDisabled(!(state && state.running));
  }

  setState(null);
  return {
    el: host,
    setState,
    get state() {
      return state;
    },
  };
}
