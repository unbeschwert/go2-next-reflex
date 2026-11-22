const tabs = Array.from(document.querySelectorAll('[role="tab"]'));

function selectStage(activeTab) {
  for (const tab of tabs) {
    const selected = tab === activeTab;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
  }
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener('click', () => selectStage(tab));
  tab.addEventListener('keydown', event => {
    const nextIndex = {
      ArrowLeft: (index - 1 + tabs.length) % tabs.length,
      ArrowRight: (index + 1) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    }[event.key];

    if (nextIndex === undefined) return;
    event.preventDefault();
    selectStage(tabs[nextIndex]);
    tabs[nextIndex].focus();
  });
}

const benchmarkButtons = document.querySelectorAll('[data-benchmark]');
for (const button of benchmarkButtons) {
  button.addEventListener('click', () => {
    for (const choice of benchmarkButtons) {
      const selected = choice === button;
      choice.setAttribute('aria-pressed', String(selected));
      document.getElementById(`${choice.dataset.benchmark}-benchmark`).hidden = !selected;
    }
  });
}

const clipVideos = document.querySelectorAll('video[data-recording]');
for (const video of clipVideos) {
  const title = document.getElementById(video.getAttribute('aria-labelledby')).textContent;
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'clip-toggle';

  function updateToggle() {
    const action = video.paused ? 'Play' : 'Pause';
    toggle.textContent = action;
    toggle.setAttribute('aria-label', `${action} ${title.toLowerCase()} clip`);
  }

  toggle.addEventListener('click', () => {
    if (!video.paused) video.pause();
    else video.play().catch(() => {
      video.controls = true;
      toggle.remove();
      video.focus();
    });
  });
  video.addEventListener('play', () => {
    for (const other of clipVideos) {
      if (other !== video && !other.paused) other.pause();
    }
    updateToggle();
  });
  video.addEventListener('pause', updateToggle);
  updateToggle();
  video.controls = false;
  video.after(toggle);
}

const heartMotion = document.getElementById('heart-motion');
const heartStill = document.querySelector('.heart-frame source');
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
let heartPaused = motionPreference.matches;

function updateHeartMotion() {
  heartStill.media = heartPaused ? 'all' : 'not all';
  heartMotion.textContent = heartPaused ? 'Play' : 'Pause';
  heartMotion.setAttribute('aria-label', heartPaused ? 'Play heart animation' : 'Pause heart animation');
}

heartMotion.addEventListener('click', () => {
  heartPaused = !heartPaused;
  updateHeartMotion();
});
motionPreference.addEventListener('change', event => {
  heartPaused = event.matches;
  updateHeartMotion();
});
updateHeartMotion();
heartMotion.hidden = false;

const pipeline = document.getElementById('pipeline');
const present = document.getElementById('present');

if (document.fullscreenEnabled) {
  present.hidden = false;
  present.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement === pipeline) await document.exitFullscreen();
      else await pipeline.requestFullscreen();
    } catch {
      pipeline.scrollIntoView({ block: 'start' });
    }
  });
  document.addEventListener('fullscreenchange', () => {
    present.textContent = document.fullscreenElement === pipeline ? 'Exit full screen' : 'Present pipeline';
  });
}
