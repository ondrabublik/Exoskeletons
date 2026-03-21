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
let currentLayout = '2col';

// ── Grid layout configurations ─────────────────────────────────────────────
const LAYOUT_CONFIGS = {
  '1col': {
    cols: 1,
    rows: [{ label: null, channels: [0, 1, 2, 3, 4, 5, 6], spanFirst: false }]
  },
  '2col': {
    cols: 2,
    rows: [
      { label: 'Angle',   channels: [0],       spanFirst: false },
      { label: 'Pitch 1', channels: [1, 2],    spanFirst: false },
      { label: 'Pitch 2', channels: [3, 4],    spanFirst: false },
      { label: 'Output',  channels: [5, 6],    spanFirst: false }
    ]
  },
  '4col': {
    cols: 4,
    rows: [
      { label: 'Pitch',  channels: [1, 2, 3, 4], spanFirst: false },
      { label: 'Other',  channels: [0, 5, 6],    spanFirst: true  }
    ]
  }
};

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
let replayMode = 'smooth';

const socket = io('http://192.168.30.86:3000');

// Helper functions
function getTimeLabel(sampleIndex, frequencyHz) {
  const seconds = Math.round((sampleIndex - (MAX_POINTS - 1)) / frequencyHz);
  return `${seconds} s`;
}

// ── Render loop ─────────────────────────────────────────────────────────────
// All chart renders are batched through requestAnimationFrame.
// pushSample() only mutates pre-allocated buffers and marks charts dirty;
// no {x,y} objects are ever allocated after startup.
let activeTab = 'live';
const dirtyCharts = new Set();
let rafPending = false;

function scheduleRender() {
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(flushRender);
  }
}

function flushRender() {
  rafPending = false;
  for (const chart of dirtyCharts) {
    chart.update('none');
  }
  dirtyCharts.clear();
}

function buildDashboard(keyPrefix) {
  return CHANNEL_NAMES.map((name, i) => {
    const color = COLORS[i];
    const cardId = `${keyPrefix}-${i}`;

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.dataset.cardId = cardId;
    card.innerHTML = `
      <div class="card-header">
        <span class="card-title" style="color:${color}">${name}</span>
        <span class="card-value" id="val-${cardId}">--</span>
        <button class="btn-icon" id="fsc-${cardId}" title="Fullscreen">⊞</button>
        <button class="btn-icon" id="col-${cardId}" title="Collapse">Hide</button>
      </div>
      <div class="card-body" id="body-${cardId}">
        <div class="chart-wrap">
          <canvas id="canvas-${cardId}"></canvas>
        </div>
      </div>`;

    // Pre-allocate a fixed buffer of {x,y} objects — never reallocated
    const buf = Array.from({ length: MAX_POINTS }, (_, j) => ({ x: j, y: null }));

    const ctx = card.querySelector('canvas').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            data: buf,
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
            min: 0,
            max: 1,
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

    const fscBtn = card.querySelector(`#fsc-${cardId}`);
    const colBtn = card.querySelector(`#col-${cardId}`);
    const body  = card.querySelector(`#body-${cardId}`);

    fscBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      if (document.fullscreenElement === card) {
        document.exitFullscreen();
      } else if (!document.fullscreenElement) {
        card.requestFullscreen();
      }
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

    return { cardId, chart, buf, el: card, rowHidden: false };
  });
}

function applyLayout(gridEl, cards, layoutKey) {
  const config = LAYOUT_CONFIGS[layoutKey];
  gridEl.innerHTML = '';
  gridEl.className = `dashboard layout-${layoutKey}`;
  cards.forEach((c) => { c.rowHidden = false; });

  config.rows.forEach((row) => {
    const section = document.createElement('div');
    section.className = 'row-section' + (row.label === null ? ' no-label' : '');

    if (row.label !== null) {
      const bar = document.createElement('div');
      bar.className = 'row-bar';
      const labelEl = document.createElement('span');
      labelEl.className = 'row-label';
      labelEl.textContent = row.label;
      bar.appendChild(labelEl);

      const toggleBtn = document.createElement('button');
      toggleBtn.className = 'btn-icon';
      toggleBtn.textContent = 'Hide';
      toggleBtn.addEventListener('click', () => {
        const hidden = section.classList.toggle('row-hidden');
        toggleBtn.textContent = hidden ? 'Show' : 'Hide';
        row.channels.forEach((ch) => {
          if (cards[ch]) cards[ch].rowHidden = hidden;
        });
      });
      bar.appendChild(toggleBtn);
      section.appendChild(bar);
    }

    const rowCards = document.createElement('div');
    rowCards.className = `row-cards cols-${config.cols}`;

    row.channels.forEach((ch, idx) => {
      const cardObj = cards[ch];
      if (!cardObj) return;
      // Single card spanning a multi-col row → full width
      if (row.channels.length === 1 && config.cols > 1) {
        cardObj.el.style.gridColumn = '1 / -1';
      } else if (row.spanFirst && idx === 0) {
        cardObj.el.style.gridColumn = 'span 2';
      } else {
        cardObj.el.style.gridColumn = '';
      }
      rowCards.appendChild(cardObj.el);
    });

    section.appendChild(rowCards);
    gridEl.appendChild(section);
  });

  requestAnimationFrame(() => cards.forEach((c) => c.chart.resize()));
}

function setCurrentLayout(layoutKey) {
  currentLayout = layoutKey;
  if (liveCards.length)   applyLayout(document.getElementById('live-grid'),   liveCards,   layoutKey);
  if (replayCards.length) applyLayout(document.getElementById('replay-grid'), replayCards, layoutKey);
  document.querySelectorAll('.layout-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.layout === layoutKey);
  });
}

function updateChartMaxPoints() {
  const allCards = [...liveCards, ...replayCards];
  allCards.forEach((current) => {
    current.chart.options.scales.x.max = MAX_POINTS - 1;
    const oldBuf = current.buf;
    const newBuf = Array.from({ length: MAX_POINTS }, (_, i) => ({ x: i, y: null }));
    const copyCount = Math.min(oldBuf.length, MAX_POINTS);
    const copyFrom = oldBuf.length - copyCount;
    const pad = MAX_POINTS - copyCount;
    for (let j = 0; j < copyCount; j++) newBuf[pad + j].y = oldBuf[copyFrom + j].y;
    current.buf = newBuf;
    current.chart.data.datasets[0].data = newBuf;
    dirtyCharts.add(current.chart);
  });
  scheduleRender();
}

function resetDashboard(cards) {
  cards.forEach((current) => {
    const buf = current.buf;
    for (let j = 0; j < buf.length; j++) buf[j].y = null;
    dirtyCharts.add(current.chart);
    document.getElementById(`val-${current.cardId}`).textContent = '--';
  });
  scheduleRender();
}

function pushSample(cards, values) {
  // Determine visibility: skip dirty-marking for charts on hidden tabs
  const tabPrefix = cards.length > 0 ? cards[0].cardId.split('-')[0] : null;
  const tabVisible = tabPrefix === 'live' ? activeTab === 'live' : activeTab === 'recordings';

  values.forEach((value, i) => {
    const current = cards[i];
    if (!current) return;

    // In-place circular shift — zero allocations
    const buf = current.buf;
    const n = buf.length;
    for (let j = 0; j < n - 1; j++) buf[j].y = buf[j + 1].y;
    buf[n - 1].y = value;

    if (tabVisible) {
      const body = document.getElementById(`body-${current.cardId}`);
      if (!current.rowHidden && !body.classList.contains('collapsed')) {
        dirtyCharts.add(current.chart);
      }
    }

    document.getElementById(`val-${current.cardId}`).textContent = Number(value).toFixed(4);
  });

  if (tabVisible) scheduleRender();
}

// Socket events
function setTab(tabName) {
  activeTab = tabName;
  document.querySelectorAll('.tab-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });

  document.getElementById('panel-live').classList.toggle('active', tabName === 'live');
  document.getElementById('panel-recordings').classList.toggle('active', tabName === 'recordings');
  document.getElementById('panel-files').classList.toggle('active', tabName === 'files');

  // Immediately refresh charts on the newly visible tab
  const cardsToRender = tabName === 'live' ? liveCards
    : tabName === 'recordings' ? replayCards
    : [];
  cardsToRender.forEach((current) => {
    const body = document.getElementById(`body-${current.cardId}`);
    if (!current.rowHidden && !body.classList.contains('collapsed')) dirtyCharts.add(current.chart);
  });
  scheduleRender();
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

  renderFilesTable(recordings);
}

function stopReplayLoop() {
  if (replayRaf) {
    cancelAnimationFrame(replayRaf);
    replayRaf = null;
  }
}

function replayToIndex(targetIndex) {
  if (!replayRecording) return;

  // O(MAX_POINTS) direct buffer fill instead of O(n × MAX_POINTS) pushSample loop
  const finalIndex = Math.max(0, Math.min(targetIndex, replayRecording.samples.length));
  const sliceStart = Math.max(0, finalIndex - MAX_POINTS);
  const slice = replayRecording.samples.slice(sliceStart, finalIndex);
  const pad = MAX_POINTS - slice.length;

  replayCards.forEach((current, i) => {
    const buf = current.buf;
    for (let j = 0; j < pad; j++) buf[j].y = null;
    for (let j = 0; j < slice.length; j++) buf[pad + j].y = slice[j].values[i];

    const lastVal = slice.length > 0 ? slice[slice.length - 1].values[i] : null;
    document.getElementById(`val-${current.cardId}`).textContent =
      lastVal !== null ? Number(lastVal).toFixed(4) : '--';

    const body = document.getElementById(`body-${current.cardId}`);
    if (!current.rowHidden && !body.classList.contains('collapsed')) dirtyCharts.add(current.chart);
  });

  replayIndex = finalIndex;
  replayX = finalIndex;
  scheduleRender();

  document.getElementById('replay-scrub').value = String(replayIndex);
  document.getElementById('replay-info').textContent =
    `${replayIndex}/${replayRecording.samples.length} samples`;
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

  if (replayMode === 'accurate') {
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
  } else {
    const sampleIntervalMs = 1000 / Math.max(FREQUENCY_HZ, 0.001);
    while (replayIndex < replayRecording.samples.length) {
      const sampleOffset = replayIndex * sampleIntervalMs;
      if (sampleOffset > replayPlayheadMs) {
        break;
      }
      pushSample(replayCards, replayRecording.samples[replayIndex].values);
      replayX += 1;
      replayIndex += 1;
    }
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

// ── File manager ───────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function renderFilesTable(recordings) {
  const tbody = document.getElementById('files-tbody');
  const count = document.getElementById('files-count');
  if (!tbody) return;

  tbody.innerHTML = '';
  count.textContent = `${recordings.length} file${recordings.length !== 1 ? 's' : ''}`;

  recordings.forEach((item) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="col-name">${item.fileName}</td>
      <td class="col-size">${formatBytes(item.sizeBytes)}</td>
      <td class="col-date">${new Date(item.modifiedAt).toLocaleString()}</td>
      <td class="col-actions">
        <div class="file-action-row">
          <a class="button" href="/api/recordings/${encodeURIComponent(item.fileName)}" download="${item.fileName}">Download</a>
          <button class="button danger" data-delete="${item.fileName}">Delete</button>
        </div>
      </td>`;

    tr.querySelector('[data-delete]').addEventListener('click', () => {
      if (!confirm(`Delete "${item.fileName}"?`)) return;
      socket.emit('recording:delete', { fileName: item.fileName }, (response) => {
        if (!response || !response.ok) {
          alert(response?.error || 'Failed to delete recording');
        }
      });
    });

    tbody.appendChild(tr);
  });
}

document.getElementById('files-refresh').addEventListener('click', requestRecordingsList);

document.getElementById('files-download-all').addEventListener('click', () => {
  socket.emit('recordings:list', {}, (response) => {
    if (!response || !response.ok || response.recordings.length === 0) return;
    response.recordings.forEach((item, index) => {
      setTimeout(() => {
        const a = document.createElement('a');
        a.href = `/api/recordings/${encodeURIComponent(item.fileName)}`;
        a.download = item.fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }, index * 300);
    });
  });
});

document.getElementById('files-clear-all').addEventListener('click', () => {
  if (!confirm('Delete ALL recordings? This cannot be undone.')) return;
  socket.emit('recording:deleteAll', {}, (response) => {
    if (!response || !response.ok) {
      alert(response?.error || 'Failed to clear recordings');
    }
  });
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

document.getElementById('replay-mode').addEventListener('change', (event) => {
  replayMode = event.target.value === 'accurate' ? 'accurate' : 'smooth';
});

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

// Layout buttons
document.querySelectorAll('.layout-btn').forEach((btn) => {
  btn.addEventListener('click', () => setCurrentLayout(btn.dataset.layout));
});

// Fullscreen
document.addEventListener('fullscreenchange', () => {
  const full = document.fullscreenElement;
  if (full && full.classList.contains('chart-card')) {
    const id = full.dataset.cardId;
    const btn = document.getElementById(`fsc-${id}`);
    if (btn) btn.textContent = '✕';
    const found = [...liveCards, ...replayCards].find((c) => c.cardId === id);
    if (found) requestAnimationFrame(() => found.chart.resize());
  } else {
    document.querySelectorAll('[id^="fsc-"]').forEach((btn) => { btn.textContent = '⊞'; });
    requestAnimationFrame(() => [...liveCards, ...replayCards].forEach((c) => c.chart.resize()));
  }
});

// Initialization
window.addEventListener('DOMContentLoaded', () => {
  liveCards = buildDashboard('live');
  replayCards = buildDashboard('replay');
  applyLayout(document.getElementById('live-grid'),   liveCards,   currentLayout);
  applyLayout(document.getElementById('replay-grid'), replayCards, currentLayout);

  document.getElementById('max-points-display').textContent = MAX_POINTS;
  document.getElementById('frequency-display').textContent = FREQUENCY_HZ.toFixed(1);
  document.getElementById('max-points-input').value = MAX_POINTS;
  document.getElementById('frequency-input').value = FREQUENCY_HZ;
  document.getElementById('replay-mode').value = replayMode;

  updateRecordingButton();
  requestRecordingsList();
});
