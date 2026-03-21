// Configuration
const CHANNEL_NAMES = [
  'Angle',
  'Pitch 1',
  'Pitch Rate 1',
  'Pitch 2',
  'Pitch Rate 2',
  'Input',
  'Prediction'
];

const COLORS = [
  '#4fc3f7',
  '#81c784',
  '#ffb74d',
  '#f06292',
  '#ba68c8',
  '#4db6ac',
  '#fff176'
];

let MAX_POINTS = 300;
let FREQUENCY_HZ = 10;

// State
let liveCards = [];
let replayCards = [];
let liveX = 0;
let sampleCount = 0;
let freqTimer = performance.now();
let freqEst = null;
let isRecording = false;

let replayRecording = null;
let replayX = 0;
let replayIndex = 0;
let replayPlayheadMs = 0;
let replayLastFrameMs = 0;
let replayRaf = null;

const socket = io('http://192.168.30.86:3000');

// Helper functions
function getTimeLabel(sampleIndex, frequencyHz) {
  const seconds = sampleIndex / frequencyHz;
  const mm = Math.floor(seconds / 60);
  const ss = (seconds % 60).toFixed(1);
  return `${mm}:${String(ss).padStart(4, '0')}`;
}

function buildDashboard(gridId, keyPrefix) {
  const grid = document.getElementById(gridId);
  return CHANNEL_NAMES.map((name, i) => {
    const color = COLORS[i];
    const cardId = `${keyPrefix}-${i}`;

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
      <div class="card-header">
        <span class="card-title" style="color:${color}">${name}</span>
        <span class="card-value" id="val-${cardId}">--</span>
        <button class="btn-icon" id="exp-${cardId}" title="Expand">Expand</button>
        <button class="btn-icon" id="col-${cardId}" title="Collapse">Hide</button>
      </div>
      <div class="card-body" id="body-${cardId}">
        <div class="chart-wrap">
          <canvas id="canvas-${cardId}"></canvas>
        </div>
      </div>`;
    grid.appendChild(card);

    const ctx = document.getElementById(`canvas-${cardId}`).getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            data: [],
            borderColor: color,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            fill: false
          }
        ]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: MAX_POINTS - 1,
            ticks: {
              color: '#8e9eb0',
              maxTicksLimit: 6,
              font: { size: 10 },
              callback: function (value) {
                return getTimeLabel(value, FREQUENCY_HZ);
              }
            },
            grid: { color: '#202b36' }
          },
          y: {
            min: -0.05,
            max: 1.05,
            ticks: {
              color: '#8e9eb0',
              maxTicksLimit: 6,
              font: { size: 10 },
              stepSize: 0.2
            },
            grid: { color: '#202b36' }
          }
        }
      }
    });

    const expBtn = document.getElementById(`exp-${cardId}`);
    const colBtn = document.getElementById(`col-${cardId}`);
    const body = document.getElementById(`body-${cardId}`);

    expBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      const expanded = card.classList.toggle('expanded');
      expBtn.textContent = expanded ? 'Contract' : 'Expand';
      chart.resize();
    });

    colBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      const collapsed = body.classList.toggle('collapsed');
      colBtn.textContent = collapsed ? 'Show' : 'Hide';
    });

    card.querySelector('.card-header').addEventListener('click', (event) => {
      if (event.target.classList.contains('btn-icon')) {
        return;
      }
      colBtn.click();
    });

    return { cardId, chart };
  });
}

function updateChartMaxPoints() {
  const allCards = [...liveCards, ...replayCards];
  allCards.forEach((current) => {
    current.chart.options.scales.x.max = MAX_POINTS - 1;
    const dataset = current.chart.data.datasets[0];
    if (dataset.data.length > MAX_POINTS) {
      dataset.data = dataset.data.slice(dataset.data.length - MAX_POINTS);
    }

    // Keep a stable 0..N-1 x-domain so axis never rescales during playback.
    dataset.data = dataset.data.map((point, index) => ({ x: index, y: point.y }));
    current.chart.update('none');
  });
}

function resetDashboard(cards) {
  cards.forEach((current) => {
    current.chart.data.datasets[0].data = [];
    current.chart.update('none');
    document.getElementById(`val-${current.cardId}`).textContent = '--';
  });
}

function pushSample(cards, values) {
  values.forEach((value, i) => {
    const current = cards[i];
    if (!current) {
      return;
    }

    const dataset = current.chart.data.datasets[0];
    dataset.data.push({ x: dataset.data.length, y: value });
    if (dataset.data.length > MAX_POINTS) {
      dataset.data.shift();
      dataset.data = dataset.data.map((point, index) => ({ x: index, y: point.y }));
    }

    const body = document.getElementById(`body-${current.cardId}`);
    if (!body.classList.contains('collapsed')) {
      current.chart.update('none');
    }

    document.getElementById(`val-${current.cardId}`).textContent = Number(value).toFixed(4);
  });
}

// Socket events
function setTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });

  document.getElementById('panel-live').classList.toggle('active', tabName === 'live');
  document.getElementById('panel-recordings').classList.toggle('active', tabName === 'recordings');
}

function updateRecordingButton() {
  const button = document.getElementById('record-btn');
  const state = document.getElementById('recording-state');
  button.textContent = isRecording ? 'Stop Recording' : 'Start Recording';
  button.classList.toggle('danger', isRecording);
  state.textContent = isRecording ? 'Recording in progress' : 'Not recording';
}

function requestRecordingsList() {
  socket.emit('recordings:list', {}, (response) => {
    if (!response || !response.ok) {
      return;
    }
    renderRecordingsList(response.recordings || []);
  });
}

function renderRecordingsList(recordings) {
  const dropdown = document.getElementById('recording-dropdown');
  const currentValue = dropdown.value;

  dropdown.innerHTML = '<option value="">Select a recording...</option>';

  recordings.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.fileName;
    option.textContent = `${item.fileName} (${Math.round(item.sizeBytes / 1024)} KB)`;
    dropdown.appendChild(option);
  });

  if (currentValue && Array.from(dropdown.options).some((o) => o.value === currentValue)) {
    dropdown.value = currentValue;
  }
}

function stopReplayLoop() {
  if (replayRaf) {
    cancelAnimationFrame(replayRaf);
    replayRaf = null;
  }
}

function replayToIndex(targetIndex) {
  if (!replayRecording) {
    return;
  }

  resetDashboard(replayCards);
  replayIndex = 0;
  replayX = 0;

  const finalIndex = Math.max(0, Math.min(targetIndex, replayRecording.samples.length));
  for (let i = 0; i < finalIndex; i++) {
    pushSample(replayCards, replayRecording.samples[i].values, replayX);
    replayX += 1;
  }
  replayIndex = finalIndex;
  document.getElementById('replay-scrub').value = String(replayIndex);
  document.getElementById('replay-info').textContent = `${replayIndex}/${replayRecording.samples.length} samples`;
}

function replayFrame(nowMs) {
  if (!replayRecording || replayIndex >= replayRecording.samples.length) {
    stopReplayLoop();
    return;
  }

  const speed = Number(document.getElementById('replay-speed').value || '1');
  const deltaMs = replayLastFrameMs ? nowMs - replayLastFrameMs : 0;
  replayLastFrameMs = nowMs;
  replayPlayheadMs += deltaMs * speed;

  const firstTs = replayRecording.samples[0].timestamp;
  while (replayIndex < replayRecording.samples.length) {
    const sample = replayRecording.samples[replayIndex];
    const sampleOffset = sample.timestamp - firstTs;
    if (sampleOffset > replayPlayheadMs) {
      break;
    }
    pushSample(replayCards, sample.values);
    replayX += 1;
    replayIndex += 1;
  }

  document.getElementById('replay-scrub').value = String(replayIndex);
  document.getElementById('replay-info').textContent = `${replayIndex}/${replayRecording.samples.length} samples`;

  if (replayIndex < replayRecording.samples.length) {
    replayRaf = requestAnimationFrame(replayFrame);
  } else {
    stopReplayLoop();
  }
}

function replayPlay() {
  if (!replayRecording || replayRaf) {
    return;
  }
  replayLastFrameMs = performance.now();
  replayRaf = requestAnimationFrame(replayFrame);
}

function replayPause() {
  stopReplayLoop();
}

function replayReset() {
  stopReplayLoop();
  replayPlayheadMs = 0;
  replayToIndex(0);
}

function loadRecording(fileName) {
  if (!fileName) {
    replayRecording = null;
    resetDashboard(replayCards);
    document.getElementById('replay-info').textContent = 'No recording selected';
    document.getElementById('replay-scrub').max = '0';
    return;
  }

  socket.emit('recording:get', { fileName }, (response) => {
    if (!response || !response.ok) {
      document.getElementById('replay-info').textContent = response?.error || 'Failed to load recording';
      return;
    }

    replayRecording = response.recording;
    replayPlayheadMs = 0;
    replayToIndex(0);

    const scrub = document.getElementById('replay-scrub');
    scrub.max = String(replayRecording.samples.length);
    scrub.value = '0';
    document.getElementById('replay-info').textContent = `${fileName} | ${replayRecording.samples.length} samples`;
  });
}

// Event listeners (DOM)
document.querySelectorAll('.tab-btn').forEach((button) => {
  button.addEventListener('click', () => setTab(button.dataset.tab));
});

document.getElementById('max-points-input').addEventListener('change', (e) => {
  const val = parseInt(e.target.value, 10);
  if (val > 0) {
    MAX_POINTS = val;
    updateChartMaxPoints();
    resetDashboard(liveCards);
    resetDashboard(replayCards);
    document.getElementById('max-points-display').textContent = val;
  }
});

document.getElementById('frequency-input').addEventListener('change', (e) => {
  const val = parseFloat(e.target.value);
  if (val > 0) {
    FREQUENCY_HZ = val;
    document.getElementById('frequency-display').textContent = val.toFixed(1);
    replayCards.forEach((c) => {
      c.chart.update('none');
    });
    liveCards.forEach((c) => {
      c.chart.update('none');
    });
  }
});

document.getElementById('record-btn').addEventListener('click', () => {
  if (!isRecording) {
    const name = document.getElementById('recording-name').value;
    socket.emit('recording:start', { name }, (response) => {
      if (!response || !response.ok) {
        alert(response?.error || 'Failed to start recording');
      }
    });
  } else {
    socket.emit('recording:stop', {}, (response) => {
      if (!response || !response.ok) {
        alert(response?.error || 'Failed to stop recording');
      }
    });
  }
});

document.getElementById('recording-dropdown').addEventListener('change', (e) => {
  loadRecording(e.target.value);
});

// Handle refresh button in replay toolbar.
const refreshBtn2 = document.getElementById('refresh-recordings-2');
if (refreshBtn2) {
  refreshBtn2.addEventListener('click', requestRecordingsList);
}

document.getElementById('replay-play').addEventListener('click', replayPlay);

document.getElementById('replay-pause').addEventListener('click', replayPause);

document.getElementById('replay-reset').addEventListener('click', replayReset);

document.getElementById('replay-scrub').addEventListener('input', (event) => {
  stopReplayLoop();
  replayToIndex(Number(event.target.value));
  if (replayRecording && replayRecording.samples.length > 0) {
    const firstTs = replayRecording.samples[0].timestamp;
    const idx = Math.max(0, Math.min(replayIndex - 1, replayRecording.samples.length - 1));
    replayPlayheadMs = replayRecording.samples[idx].timestamp - firstTs;
  }
});

// Socket event handlers
socket.on('connect', () => {
  document.getElementById('status-dot').className = 'dot connected';
  document.getElementById('status-text').textContent = 'Connected';
});

socket.on('disconnect', () => {
  document.getElementById('status-dot').className = 'dot disconnected';
  document.getElementById('status-text').textContent = 'Disconnected';
  document.getElementById('freq-display').textContent = '';
});

socket.on('recording:status', (status) => {
  isRecording = !!status.isRecording;
  updateRecordingButton();
});

socket.on('recordings:list', (recordings) => {
  if (Array.isArray(recordings)) {
    renderRecordingsList(recordings);
  }
});

socket.on('sensor-data', (packet) => {
  sampleCount++;
  const now = performance.now();
  const dt = now - freqTimer;
  if (dt >= 1000) {
    const hz = (sampleCount / dt) * 1000;
    freqEst = freqEst === null ? hz : freqEst * 0.7 + hz * 0.3;
    document.getElementById('freq-display').textContent = `${freqEst.toFixed(1)} Hz`;
    sampleCount = 0;
    freqTimer = now;
  }

  if (!packet || !Array.isArray(packet.values)) {
    return;
  }

  pushSample(liveCards, packet.values);
  liveX += 1;
});

// Initialization
window.addEventListener('DOMContentLoaded', () => {
  liveCards = buildDashboard('live-grid', 'live');
  replayCards = buildDashboard('replay-grid', 'replay');

  document.getElementById('max-points-display').textContent = MAX_POINTS;
  document.getElementById('frequency-display').textContent = FREQUENCY_HZ.toFixed(1);
  document.getElementById('max-points-input').value = MAX_POINTS;
  document.getElementById('frequency-input').value = FREQUENCY_HZ;

  updateRecordingButton();
  requestRecordingsList();
});
