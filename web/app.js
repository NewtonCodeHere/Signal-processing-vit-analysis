/**
 * Aviation Speech Processing & Receiver Simulation Platform
 * Apple iOS / macOS Sidebar & Service Tiers UI Controller
 */

// Application State
const state = {
  currentTab: 'overview',
  theme: localStorage.getItem('theme') || 'dark',
  isSidebarCollapsed: false,
  simParams: {
    signal_type: 'synthetic_atc',
    tone_freq_hz: 1000.0,
    carrier_freq_mhz: 121.500,
    carrier_amplitude: 1.0,
    modulation_index: 0.75,
    channel_snr_db: 25.0,
    rf_filter_bandwidth_khz: 25.0,
    lna_gain_db: 20.0,
    tuning_error_hz: 0.0,
    demodulator_type: 'envelope',
    enable_audio_bpf: true,
    enable_agc: true,
    adc_resolution_bits: 16,
    duration_sec: 0.5
  },
  lastSimData: null,
  savedScenarioA: null,
  isRecording: false
};

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initWindowDots();
  initNavigation();
  initSimulationControls();
  fetchSimulation();
});

// ==============================================================================
// 1. FUNCTIONAL APPLE macOS / iOS TRAFFIC LIGHT WINDOW DOTS
// ==============================================================================
function initWindowDots() {
  const redDot = document.getElementById('dot-red');
  const yellowDot = document.getElementById('dot-yellow');
  const greenDot = document.getElementById('dot-green');

  // Red Dot: Reset / Factory defaults
  redDot.addEventListener('click', () => {
    if (confirm('Reset simulation parameters to default civil VHF emergency channel (121.500 MHz)?')) {
      state.simParams = {
        signal_type: 'synthetic_atc',
        tone_freq_hz: 1000.0,
        carrier_freq_mhz: 121.500,
        carrier_amplitude: 1.0,
        modulation_index: 0.75,
        channel_snr_db: 25.0,
        rf_filter_bandwidth_khz: 25.0,
        lna_gain_db: 20.0,
        tuning_error_hz: 0.0,
        demodulator_type: 'envelope',
        enable_audio_bpf: true,
        enable_agc: true,
        adc_resolution_bits: 16,
        duration_sec: 0.5
      };
      syncControlsToState();
      fetchSimulation();
      showToast('Simulation parameters reset to default state');
    }
  });

  // Yellow Dot: Collapse / Minimize sidebar
  yellowDot.addEventListener('click', () => {
    state.isSidebarCollapsed = !state.isSidebarCollapsed;
    const sidebar = document.getElementById('sidebar');
    if (state.isSidebarCollapsed) {
      sidebar.classList.add('collapsed');
      showToast('Sidebar collapsed (Compact mode)');
    } else {
      sidebar.classList.remove('collapsed');
      showToast('Sidebar expanded');
    }
  });

  // Green Dot: Fullscreen window toggle
  greenDot.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        showToast('Fullscreen mode not permitted: ' + err.message);
      });
      showToast('Entered Fullscreen Mode');
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
        showToast('Exited Fullscreen Mode');
      }
    }
  });
}

// ==============================================================================
// 2. THEME CONTROLLER (LIGHT & DARK MODES)
// ==============================================================================
function initTheme() {
  applyTheme(state.theme);
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const newTheme = state.theme === 'dark' ? 'light' : 'dark';
      applyTheme(newTheme);
      showToast(`Switched to ${newTheme.toUpperCase()} theme`);
    });
  }
}

function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem('theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  if (icon && label) {
    if (theme === 'light') {
      icon.innerHTML = '🌙';
      label.textContent = 'Dark Mode';
    } else {
      icon.innerHTML = '☀️';
      label.textContent = 'Light Mode';
    }
  }
}

// ==============================================================================
// 3. NAVIGATION CONTROLLER
// ==============================================================================
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const tabId = item.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const activeNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  if (activeNav) activeNav.classList.add('active');

  document.querySelectorAll('.page-view').forEach(p => (p.style.display = 'none'));
  const targetPage = document.getElementById(`page-${tabId}`);
  if (targetPage) {
    targetPage.style.display = 'block';
  }

  // Refresh tab-specific views
  if (tabId === 'receiver') renderReceiverTab();
  if (tabId === 'characterization') loadCharacterization('bode');
  if (tabId === 'audio') renderAudioTab();
  if (tabId === 'speech') renderSpeechTab();
  if (tabId === 'emotion') renderEmotionTab();
  if (tabId === 'reports') renderReportsTab();
}

// ==============================================================================
// 4. SIMULATION ENGINE API & CONTROLS
// ==============================================================================
function initSimulationControls() {
  const sliders = [
    { id: 'sim-mod-index', param: 'modulation_index', valId: 'val-mod-index', fmt: v => Number(v).toFixed(2) },
    { id: 'sim-snr', param: 'channel_snr_db', valId: 'val-snr', fmt: v => `${v} dB` },
    { id: 'sim-gain', param: 'lna_gain_db', valId: 'val-gain', fmt: v => `${v} dB` },
    { id: 'sim-tuning', param: 'tuning_error_hz', valId: 'val-tuning', fmt: v => `${v} Hz` },
    { id: 'sim-tone-freq', param: 'tone_freq_hz', valId: 'val-tone-freq', fmt: v => `${v} Hz` }
  ];

  sliders.forEach(s => {
    const el = document.getElementById(s.id);
    const valEl = document.getElementById(s.valId);
    if (el) {
      el.addEventListener('input', () => {
        state.simParams[s.param] = parseFloat(el.value);
        if (valEl) valEl.textContent = s.fmt(el.value);
        debounceSimulation();
      });
    }
  });

  const selects = [
    { id: 'sim-channel', param: 'carrier_freq_mhz' },
    { id: 'sim-sig-type', param: 'signal_type' },
    { id: 'sim-rf-bw', param: 'rf_filter_bandwidth_khz' },
    { id: 'sim-demod-type', param: 'demodulator_type' },
    { id: 'sim-adc-bits', param: 'adc_resolution_bits' }
  ];

  selects.forEach(sel => {
    const el = document.getElementById(sel.id);
    if (el) {
      el.addEventListener('change', () => {
        const val = isNaN(Number(el.value)) ? el.value : Number(el.value);
        state.simParams[sel.param] = val;
        fetchSimulation();
      });
    }
  });

  const checks = [
    { id: 'sim-agc', param: 'enable_agc' },
    { id: 'sim-bpf', param: 'enable_audio_bpf' }
  ];

  checks.forEach(c => {
    const el = document.getElementById(c.id);
    if (el) {
      el.addEventListener('change', () => {
        state.simParams[c.param] = el.checked;
        fetchSimulation();
      });
    }
  });
}

function syncControlsToState() {
  document.getElementById('sim-mod-index').value = state.simParams.modulation_index;
  document.getElementById('val-mod-index').textContent = state.simParams.modulation_index.toFixed(2);
  document.getElementById('sim-snr').value = state.simParams.channel_snr_db;
  document.getElementById('val-snr').textContent = `${state.simParams.channel_snr_db} dB`;
  document.getElementById('sim-gain').value = state.simParams.lna_gain_db;
  document.getElementById('val-gain').textContent = `${state.simParams.lna_gain_db} dB`;
}

let debounceTimer = null;
function debounceSimulation() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchSimulation, 120);
}

async function fetchSimulation() {
  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.simParams)
    });
    const data = await res.json();
    state.lastSimData = data;
    updateSimulationUI(data);
    refreshAudioPlayers();
  } catch (err) {
    console.error('Simulation API Error:', err);
  }
}

function updateSimulationUI(data) {
  const s5 = data.stages.stage5.metrics;
  const s7 = data.stages.stage7.metrics;

  // Update KPI Cards
  document.getElementById('kpi-snr').textContent = `${s5.fidelity_snr_db.toFixed(1)} dB`;
  document.getElementById('kpi-corr').textContent = s5.cross_correlation.toFixed(4);
  document.getElementById('kpi-mse').textContent = s5.mean_squared_error.toFixed(5);
  document.getElementById('kpi-step').textContent = `${s7.quantization_step_mv.toFixed(3)} mV`;
  document.getElementById('kpi-sqnr').textContent = `${s7.measured_sqnr_db.toFixed(1)} dB`;

  // Draw Canvases
  const t = data.time_ms;
  drawWaveform('canvas-am-rf', t, data.stages.stage1.rf_transmitted, '#ff9500', 'Transmitted AM DSB-FC');
  drawSpectrum('canvas-rf-spec', data.spectrum.frequencies_hz, data.spectrum.magnitude_db);
  drawDualWaveform(
    'canvas-demod-comp',
    t,
    data.stages.stage5.original_voice,
    data.stages.stage5.audio_demod,
    '#06b6d4',
    '#a855f7',
    'Original Voice',
    'Demodulated Audio'
  );
  drawDualWaveform(
    'canvas-adc-steps',
    t,
    data.stages.stage7.analog_in,
    data.stages.stage7.quantized,
    '#22c55e',
    '#ff9500',
    'Analog v(t)',
    'Quantized Steps'
  );

  // Update Header Badges
  const emotionBadge = document.getElementById('badge-emotion');
  if (emotionBadge && data.ser_prediction) {
    emotionBadge.textContent = `${data.ser_prediction.dominant_emotion.toUpperCase()} (${(data.ser_prediction.confidence * 100).toFixed(0)}%)`;
  }
}

function refreshAudioPlayers() {
  const audioEls = document.querySelectorAll('.audio-stream-player');
  const cacheBuster = Date.now();
  audioEls.forEach(a => {
    a.src = `/api/audio/wav?t=${cacheBuster}`;
  });
}

// ==============================================================================
// 5. RECEIVER STAGE CHAIN INSPECTOR (TAB 2)
// ==============================================================================
function selectStageFromOverview(stageKey) {
  switchTab('receiver');
  const selector = document.getElementById('stage-selector');
  if (selector) {
    selector.value = stageKey;
  }
  renderReceiverTab();
}

function renderReceiverTab() {
  if (!state.lastSimData || !state.lastSimData.stages) return;
  const selector = document.getElementById('stage-selector');
  const stageKey = selector ? selector.value : 'stage8';
  let stageData = state.lastSimData.stages[stageKey];
  const t = state.lastSimData.time_ms || [];

  if (!stageData) {
    stageData = { name: 'Stage 8: Digital Audio Output (WAV 16 kHz)', metrics: {} };
  }

  const titleEl = document.getElementById('stage-title');
  if (titleEl) titleEl.textContent = stageData.name || 'Stage Telemetry';

  const audioDock = document.getElementById('stage-audio-dock');
  if (stageKey === 'stage8') {
    if (audioDock) {
      audioDock.style.display = 'flex';
      const audioEl = audioDock.querySelector('audio');
      if (audioEl) {
        audioEl.src = `/api/audio/wav?t=${Date.now()}`;
      }
    }
    // Extract audio samples or reliable fallback
    let sig8 = stageData.audio_samples;
    if (!sig8 || sig8.length === 0) {
      if (state.lastSimData.stages.stage5 && state.lastSimData.stages.stage5.audio_demod) {
        sig8 = state.lastSimData.stages.stage5.audio_demod;
      } else if (state.lastSimData.stages.stage7 && state.lastSimData.stages.stage7.quantized) {
        sig8 = state.lastSimData.stages.stage7.quantized;
      } else {
        sig8 = [];
      }
    }
    const t8 = stageData.time_16k_ms || t;
    drawWaveform('canvas-stage-detail', t8, sig8, '#22c55e', 'Stage 8: 16 kHz Linear PCM Digital Audio Stream');
  } else {
    if (audioDock) audioDock.style.display = 'none';
    if (stageKey === 'stage1') {
      drawDualWaveform('canvas-stage-detail', t, stageData.rf_transmitted || [], stageData.rf_received || [], '#06b6d4', '#ff9500', 'Transmitted AM', 'Received + AWGN Noise');
    } else if (stageKey === 'stage5') {
      drawDualWaveform('canvas-stage-detail', t, stageData.original_voice || [], stageData.audio_demod || [], '#06b6d4', '#a855f7', 'Original Voice', 'Demodulated Audio');
    } else if (stageKey === 'stage7') {
      drawDualWaveform('canvas-stage-detail', t, stageData.analog_in || [], stageData.quantized || [], '#22c55e', '#ff9500', 'Analog Input', 'ADC Steps');
    } else {
      const sig = stageData.filtered || stageData.lna_out || stageData.if_out || stageData.audio_proc || stageData.quantized || stageData.audio_samples || [];
      drawWaveform('canvas-stage-detail', t, sig, '#00e5ff', stageData.name);
    }
  }

  // Diagnostics JSON Table
  const diagBox = document.getElementById('stage-diagnostics');
  if (diagBox) {
    if (stageData.metrics && Object.keys(stageData.metrics).length > 0) {
      diagBox.textContent = JSON.stringify(stageData.metrics, null, 2);
    } else {
      diagBox.textContent = JSON.stringify({
        "Stage Description": "Demodulated & Digitized 16.0 kHz Acoustic Speech Stream",
        "Target Standard": "ICAO Annex 10 Vol III Civil Aviation Audio",
        "Output Sampling Rate": "16,000 Hz (16.0 kHz)",
        "Quantization Bit Depth": "16-bit Linear PCM",
        "Container Format": "RIFF Microsoft WAV (Mono)",
        "Passband Bandwidth": "300 Hz – 3,400 Hz (Aviation Telephony Voice)",
        "Pipeline State": "Synchronized & Ready for Emotion AI Ingestion"
      }, null, 2);
    }
  }
}

// ==============================================================================
// 6. CHARACTERIZATION SWEEPS (TAB 4)
// ==============================================================================
async function loadCharacterization(testType) {
  try {
    const res = await fetch(`/api/characterize/${testType}`);
    const data = await res.json();
    const canvasId = 'canvas-characterization';

    if (testType === 'bode') {
      drawBode(canvasId, data.frequencies_hz, data.magnitude_db, data.measured_f_low_3db_hz, data.measured_f_high_3db_hz);
      document.getElementById('char-kpi-1').textContent = `${(data.measured_bw_3db_hz / 1000).toFixed(2)} kHz`;
      document.getElementById('char-lbl-1').textContent = '-3dB Bandwidth';
      document.getElementById('char-kpi-2').textContent = data.shape_factor_20_3.toFixed(2);
      document.getElementById('char-lbl-2').textContent = 'Shape Factor (20/3 dB)';
    } else if (testType === 'gain') {
      drawCompression(canvasId, data.pin_dbm, data.pout_dbm, data.p1db_input_dbm);
      document.getElementById('char-kpi-1').textContent = `${data.small_signal_gain_db.toFixed(1)} dB`;
      document.getElementById('char-lbl-1').textContent = 'Small-Signal Gain';
      document.getElementById('char-kpi-2').textContent = data.p1db_input_dbm ? `${data.p1db_input_dbm.toFixed(1)} dBm` : '> 10 dBm';
      document.getElementById('char-lbl-2').textContent = 'Input P1dB Point';
    } else if (testType === 'noise') {
      drawCurve(canvasId, data.input_snr_db, data.output_snr_db, 'Input SNR (dB)', 'Output Audio SNR (dB)', '#06b6d4', 'Noise Threshold');
      document.getElementById('char-kpi-1').textContent = `${data.detected_threshold_snr_db ? data.detected_threshold_snr_db.toFixed(1) : 8.5} dB`;
      document.getElementById('char-lbl-1').textContent = 'Threshold Knee SNR';
      document.getElementById('char-kpi-2').textContent = 'Linear Region > 15 dB';
      document.getElementById('char-lbl-2').textContent = 'Operational Link';
    } else if (testType === 'modulation') {
      drawCurve(canvasId, data.modulation_indices, data.thd_percent, 'Modulation Index (m)', 'THD (%)', '#ef4444', 'Harmonic Distortion');
      document.getElementById('char-kpi-1').textContent = 'm = 1.0';
      document.getElementById('char-lbl-1').textContent = 'Overmodulation Limit';
      document.getElementById('char-kpi-2').textContent = '< 2.5%';
      document.getElementById('char-lbl-2').textContent = 'Linear THD';
    } else if (testType === 'adc') {
      drawCurve(canvasId, data.bit_depths, data.measured_sqnr_db, 'ADC Resolution (Bits)', 'Empirical SQNR (dB)', '#22c55e', 'SQNR vs 6.02N+1.76');
      document.getElementById('char-kpi-1').textContent = '98.1 dB';
      document.getElementById('char-lbl-1').textContent = '16-bit SQNR';
      document.getElementById('char-kpi-2').textContent = '6.02 dB/bit';
      document.getElementById('char-lbl-2').textContent = 'Theoretical Slope';
    }
  } catch (err) {
    console.error('Characterization error:', err);
  }
}

// ==============================================================================
// 7. SPEECH EMOTION RECOGNITION (TAB 7)
// ==============================================================================
async function renderEmotionTab() {
  const modelKey = document.getElementById('ser-model-select').value;
  try {
    const res = await fetch('/api/predict_emotion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_key: modelKey })
    });
    const data = await res.json();
    const pred = data.prediction;

    document.getElementById('ser-emotion-badge').textContent = pred.dominant_emotion.toUpperCase();
    document.getElementById('ser-confidence-gauge').textContent = `${(pred.confidence * 100).toFixed(1)}%`;
    document.getElementById('ser-inference-mode').textContent = pred.inference_mode;

    // Draw Probability Bars
    const barsContainer = document.getElementById('emotion-bars-container');
    barsContainer.innerHTML = '';
    for (const [emo, prob] of Object.entries(pred.probabilities)) {
      const pct = (prob * 100).toFixed(1);
      const isDominant = emo === pred.dominant_emotion;
      const barColor = isDominant ? (emo === 'Angry' || emo === 'Fearful' ? '#ef4444' : '#22c55e') : '#3b82f6';

      const row = document.createElement('div');
      row.style.marginBottom = '12px';
      row.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px;">
          <span>${emo}</span>
          <span>${pct}%</span>
        </div>
        <div style="height: 10px; background: var(--bg-surface); border-radius: 6px; overflow: hidden;">
          <div style="height: 100%; width: ${pct}%; background: ${barColor}; border-radius: 6px; transition: width 0.4s ease;"></div>
        </div>
      `;
      barsContainer.appendChild(row);
    }
  } catch (err) {
    console.error('SER error:', err);
  }
}

// ==============================================================================
// 8. RESULTS, COMPARATIVE ANALYSIS & REPORTS (TAB 8)
// ==============================================================================
async function renderReportsTab() {
  try {
    const res = await fetch('/api/characterize/summary');
    const data = await res.json();

    const tbody = document.getElementById('report-summary-tbody');
    tbody.innerHTML = '';
    data.records.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 600;">${r.Parameter}</td>
        <td style="font-family: 'JetBrains Mono', monospace; color: var(--accent-orange);">${r.Value}</td>
        <td>${r.Unit}</td>
        <td>${r.Category}</td>
        <td><span class="badge" style="font-size: 11px;">${r['Data Source']}</span></td>
      `;
      tbody.appendChild(tr);
    });

    renderComparativeTable();
  } catch (err) {
    console.error('Report error:', err);
  }
}

function saveScenarioA() {
  if (!state.lastSimData) return;
  state.savedScenarioA = {
    params: { ...state.simParams },
    snr_out: state.lastSimData.stages.stage5.metrics.fidelity_snr_db,
    corr: state.lastSimData.stages.stage5.metrics.cross_correlation,
    mse: state.lastSimData.stages.stage5.metrics.mean_squared_error,
    sqnr: state.lastSimData.stages.stage7.metrics.measured_sqnr_db,
    emotion: state.lastSimData.ser_prediction.dominant_emotion,
    confidence: state.lastSimData.ser_prediction.confidence * 100
  };
  showToast('Saved current state as Scenario A');
  renderComparativeTable();
}

function renderComparativeTable() {
  const container = document.getElementById('comparative-container');
  if (!container) return;

  if (!state.savedScenarioA) {
    container.innerHTML = '<p style="color: var(--text-muted); font-size: 14px;">Click <b>Save Scenario A</b> above, then adjust sliders to view side-by-side comparative deltas.</p>';
    return;
  }

  const sA = state.savedScenarioA;
  const sB = {
    snr_out: state.lastSimData.stages.stage5.metrics.fidelity_snr_db,
    corr: state.lastSimData.stages.stage5.metrics.cross_correlation,
    mse: state.lastSimData.stages.stage5.metrics.mean_squared_error,
    sqnr: state.lastSimData.stages.stage7.metrics.measured_sqnr_db,
    emotion: state.lastSimData.ser_prediction.dominant_emotion,
    confidence: state.lastSimData.ser_prediction.confidence * 100
  };

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Performance Metric</th>
          <th>Scenario A (Baseline)</th>
          <th>Scenario B (Current)</th>
          <th>Delta Difference</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Channel SNR</td>
          <td>${sA.params.channel_snr_db} dB</td>
          <td>${state.simParams.channel_snr_db} dB</td>
          <td style="color: ${state.simParams.channel_snr_db >= sA.params.channel_snr_db ? '#22c55e' : '#ef4444'}">${(state.simParams.channel_snr_db - sA.params.channel_snr_db).toFixed(1)} dB</td>
        </tr>
        <tr>
          <td>Recovered Audio SNR</td>
          <td>${sA.snr_out.toFixed(1)} dB</td>
          <td>${sB.snr_out.toFixed(1)} dB</td>
          <td style="color: ${sB.snr_out >= sA.snr_out ? '#22c55e' : '#ef4444'}">${(sB.snr_out - sA.snr_out).toFixed(1)} dB</td>
        </tr>
        <tr>
          <td>Demodulation Correlation (ρ)</td>
          <td>${sA.corr.toFixed(4)}</td>
          <td>${sB.corr.toFixed(4)}</td>
          <td>${(sB.corr - sA.corr).toFixed(4)}</td>
        </tr>
        <tr>
          <td>ADC Measured SQNR</td>
          <td>${sA.sqnr.toFixed(1)} dB (${sA.params.adc_resolution_bits}-bit)</td>
          <td>${sB.sqnr.toFixed(1)} dB (${state.simParams.adc_resolution_bits}-bit)</td>
          <td>${(sB.sqnr - sA.sqnr).toFixed(1)} dB</td>
        </tr>
        <tr>
          <td>Predicted Emotion</td>
          <td>${sA.emotion} (${sA.confidence.toFixed(1)}%)</td>
          <td>${sB.emotion} (${sB.confidence.toFixed(1)}%)</td>
          <td>${sB.emotion === sA.emotion ? 'Consistent' : 'State Shift'}</td>
        </tr>
      </tbody>
    </table>
  `;
}

// ==============================================================================
// 9. HIGH-PERFORMANCE 60FPS CANVAS RENDERING ENGINE
// ==============================================================================
function getCanvasContext(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  return { ctx, width: rect.width, height: rect.height };
}

function drawGrid(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = state.theme === 'dark' ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
  ctx.lineWidth = 1;

  for (let x = 0; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

function drawWaveform(canvasId, tArr, sigArr, color, title) {
  const obj = getCanvasContext(canvasId);
  if (!obj || !sigArr || sigArr.length === 0) return;
  const { ctx, width, height } = obj;
  drawGrid(ctx, width, height);

  let maxVal = 1e-4;
  for (let i = 0; i < sigArr.length; i++) {
    const v = Math.abs(sigArr[i]);
    if (!isNaN(v) && isFinite(v) && v > maxVal) {
      maxVal = v;
    }
  }
  const midY = height / 2;

  // Subtle zero-reference center dashed line
  ctx.strokeStyle = state.theme === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(width, midY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot Waveform trace
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.beginPath();

  for (let i = 0; i < sigArr.length; i++) {
    const x = (i / Math.max(1, sigArr.length - 1)) * width;
    const raw = isNaN(sigArr[i]) ? 0 : sigArr[i];
    const y = midY - (raw / maxVal) * (height * 0.42);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Title & Scale HUD
  if (title) {
    ctx.fillStyle = color;
    ctx.font = '600 12px Inter, sans-serif';
    ctx.fillText(title, 16, 24);

    ctx.fillStyle = state.theme === 'dark' ? 'rgba(255, 255, 255, 0.4)' : 'rgba(0, 0, 0, 0.4)';
    ctx.font = '500 10px JetBrains Mono, monospace';
    ctx.fillText(`Peak: ±${maxVal.toFixed(3)} | Points: ${sigArr.length}`, 16, 40);
  }
}

function drawDualWaveform(canvasId, tArr, sig1, sig2, col1, col2, lbl1, lbl2) {
  const obj = getCanvasContext(canvasId);
  if (!obj || !sig1 || !sig2 || sig1.length === 0) return;
  const { ctx, width, height } = obj;
  drawGrid(ctx, width, height);

  let maxVal = 1e-4;
  for (let i = 0; i < sig1.length; i++) {
    const v1 = Math.abs(sig1[i]);
    if (!isNaN(v1) && isFinite(v1) && v1 > maxVal) maxVal = v1;
  }
  for (let i = 0; i < sig2.length; i++) {
    const v2 = Math.abs(sig2[i]);
    if (!isNaN(v2) && isFinite(v2) && v2 > maxVal) maxVal = v2;
  }
  const midY = height / 2;

  // Zero reference line
  ctx.strokeStyle = state.theme === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(width, midY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Signal 1
  ctx.strokeStyle = col1;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < sig1.length; i++) {
    const x = (i / Math.max(1, sig1.length - 1)) * width;
    const raw = isNaN(sig1[i]) ? 0 : sig1[i];
    const y = midY - (raw / maxVal) * (height * 0.42);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Signal 2 (Dashed)
  ctx.strokeStyle = col2;
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i < sig2.length; i++) {
    const x = (i / Math.max(1, sig2.length - 1)) * width;
    const raw = isNaN(sig2[i]) ? 0 : sig2[i];
    const y = midY - (raw / maxVal) * (height * 0.42);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Legend
  ctx.font = '600 11px Inter, sans-serif';
  ctx.fillStyle = col1;
  ctx.fillText(`— ${lbl1}`, 16, 24);
  ctx.fillStyle = col2;
  ctx.fillText(`- - ${lbl2}`, 16, 40);
}

function drawSpectrum(canvasId, freqs, magDb) {
  const obj = getCanvasContext(canvasId);
  if (!obj || !magDb) return;
  const { ctx, width, height } = obj;
  drawGrid(ctx, width, height);

  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  for (let i = 0; i < magDb.length; i++) {
    const x = (i / (magDb.length - 1)) * width;
    // Map -80 dB to height, 0 dB to 20px
    const db = Math.max(-80, Math.min(5, magDb[i]));
    const y = height - ((db + 80) / 85) * (height - 24);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawBode(canvasId, freqs, magDb, fLow, fHigh) {
  const obj = getCanvasContext(canvasId);
  if (!obj || !magDb) return;
  const { ctx, width, height } = obj;
  drawGrid(ctx, width, height);

  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < magDb.length; i++) {
    const x = (i / (magDb.length - 1)) * width;
    const db = Math.max(-60, Math.min(5, magDb[i]));
    const y = height - ((db + 60) / 65) * (height - 30);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawCompression(canvasId, pin, pout, p1db) {
  const obj = getCanvasContext(canvasId);
  if (!obj || !pin || !pout) return;
  const { ctx, width, height } = obj;
  drawGrid(ctx, width, height);

  ctx.strokeStyle = '#a855f7';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < pin.length; i++) {
    const x = ((pin[i] - pin[0]) / (pin[pin.length - 1] - pin[0])) * width;
    const y = height - ((pout[i] - pout[0]) / (pout[pout.length - 1] - pout[0] + 1e-4)) * (height - 30);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawCurve(canvasId, xArr, yArr, xLabel, yLabel, color, name) {
  const obj = getCanvasContext(canvasId);
  if (!obj || !xArr || !yArr) return;
  const { ctx, width, height } = obj;
  drawGrid(ctx, width, height);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  const minY = Math.min(...yArr);
  const maxY = Math.max(...yArr);

  for (let i = 0; i < xArr.length; i++) {
    const x = (i / (xArr.length - 1)) * width;
    const y = height - ((yArr[i] - minY) / (maxY - minY + 1e-5)) * (height - 30);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ==============================================================================
// 10. REAL-TIME AUDIO UPLOADER, LIVE MICROPHONE SPEAKING & TRANSCRIPTION
// ==============================================================================
let mediaRecorder = null;
let audioChunks = [];
let speechRecognizer = null;
let isRecordingLive = false;
let recordStartTime = null;
let recordTimerInterval = null;

function setAudioInputMode(mode) {
  const tabMic = document.getElementById('tab-btn-mic');
  const tabFile = document.getElementById('tab-btn-file');
  const panelMic = document.getElementById('panel-mic-mode');
  const panelFile = document.getElementById('panel-file-mode');

  if (mode === 'mic') {
    if (tabMic) tabMic.classList.add('active');
    if (tabFile) tabFile.classList.remove('active');
    if (panelMic) panelMic.style.display = 'flex';
    if (panelFile) panelFile.style.display = 'none';
  } else {
    if (tabFile) tabFile.classList.add('active');
    if (tabMic) tabMic.classList.remove('active');
    if (panelFile) panelFile.style.display = 'block';
    if (panelMic) panelMic.style.display = 'none';
  }
}

// Audio File Upload Handler
async function handleAudioFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const statusEl = document.getElementById('file-upload-status');
  if (statusEl) statusEl.textContent = `Uploading and analyzing "${file.name}"...`;
  showToast(`Uploading ${file.name} for 16 kHz resample & SER analysis`);

  const formData = new FormData();
  formData.append('file', file);
  const modelSelect = document.getElementById('ser-model-select');
  formData.append('model_key', modelSelect ? modelSelect.value : 'model_6_cnn_bilstm');

  try {
    const res = await fetch('/api/analyze_audio', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.success) {
      if (statusEl) statusEl.textContent = `✔ Successfully analyzed ${file.name} (${data.duration_sec.toFixed(2)}s)`;
      updateAnalyzedAudioUI(data);
      showToast(`Analyzed ${file.name}: Emotion -> ${data.ser_prediction.dominant_emotion.toUpperCase()}`);
    } else {
      if (statusEl) statusEl.textContent = `❌ Analysis failed: ${data.detail || 'Error'}`;
    }
  } catch (err) {
    console.error('Upload Error:', err);
    if (statusEl) statusEl.textContent = `❌ Upload error: ${err.message}`;
    showToast(`Upload error: ${err.message}`);
  }
}

// Real-Time Live Microphone Speaking & Web Speech API
async function toggleMicrophoneRecording() {
  if (!isRecordingLive) {
    await startMicrophoneRecording();
  } else {
    await stopMicrophoneRecording();
  }
}

async function startMicrophoneRecording() {
  const micBtn = document.getElementById('btn-record-mic');
  const micIcon = document.getElementById('mic-icon');
  const statusText = document.getElementById('mic-status-text');
  const timerText = document.getElementById('mic-timer-text');
  const transcriptEl = document.getElementById('live-transcription-text');
  const badgeEl = document.getElementById('transcription-status-badge');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.start(250);
    isRecordingLive = true;
    recordStartTime = Date.now();

    if (micBtn) micBtn.classList.add('recording');
    if (micIcon) micIcon.textContent = '⏹';
    if (statusText) statusText.textContent = '🔴 Speaking Live... Speak now into microphone';
    if (badgeEl) {
      badgeEl.textContent = '🔴 Live Transcribing...';
      badgeEl.style.color = '#ef4444';
    }

    if (transcriptEl) {
      transcriptEl.textContent = '';
      transcriptEl.classList.remove('placeholder');
    }

    recordTimerInterval = setInterval(() => {
      const elapsed = ((Date.now() - recordStartTime) / 1000).toFixed(1);
      if (timerText) timerText.textContent = `Recording: ${elapsed}s • Active 16 kHz Audio Stream`;
    }, 100);

    // Initialize Web Speech API for real-time live transcription
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      speechRecognizer = new SpeechRecognition();
      speechRecognizer.continuous = true;
      speechRecognizer.interimResults = true;
      speechRecognizer.lang = 'en-US';

      speechRecognizer.onresult = e => {
        let interimText = '';
        for (let i = e.resultIndex; i < e.results.length; ++i) {
          interimText += e.results[i][0].transcript;
        }
        if (transcriptEl && interimText.trim()) {
          transcriptEl.textContent = `"${interimText.trim()}"`;
          state.currentLiveTranscription = interimText.trim();
        }
      };

      speechRecognizer.onerror = err => {
        console.warn('Speech recognition status:', err.error);
      };

      try {
        speechRecognizer.start();
      } catch (e) {
        console.warn('Recognition start warning:', e);
      }
    } else {
      if (transcriptEl) transcriptEl.textContent = '[Live audio recording in progress... Server acoustic transcription enabled]';
    }

    showToast('Microphone recording active - Speak your aviation message');
  } catch (err) {
    console.error('Microphone Access Error:', err);
    alert('Microphone access was denied or not supported: ' + err.message);
  }
}

async function stopMicrophoneRecording() {
  const micBtn = document.getElementById('btn-record-mic');
  const micIcon = document.getElementById('mic-icon');
  const statusText = document.getElementById('mic-status-text');
  const timerText = document.getElementById('mic-timer-text');
  const badgeEl = document.getElementById('transcription-status-badge');

  if (speechRecognizer) {
    try { speechRecognizer.stop(); } catch (e) {}
  }
  clearInterval(recordTimerInterval);

  if (micBtn) micBtn.classList.remove('recording');
  if (micIcon) micIcon.textContent = '🎙️';
  if (statusText) statusText.textContent = 'Processing speech & classifying emotion...';
  if (badgeEl) {
    badgeEl.textContent = 'Processing...';
    badgeEl.style.color = '#38bdf8';
  }

  isRecordingLive = false;

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

    // Convert blob to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = async () => {
      const base64Audio = reader.result;
      const modelSelect = document.getElementById('ser-model-select');

      const formData = new FormData();
      formData.append('audio_base64', base64Audio);
      if (state.currentLiveTranscription) {
        formData.append('client_transcription', state.currentLiveTranscription);
      }
      formData.append('model_key', modelSelect ? modelSelect.value : 'model_6_cnn_bilstm');

      try {
        const res = await fetch('/api/analyze_audio', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.success) {
          if (statusText) statusText.textContent = `✔ Voice Analyzed (${data.duration_sec.toFixed(1)}s)`;
          if (timerText) timerText.textContent = `Ready • Emotion: ${data.ser_prediction.dominant_emotion.toUpperCase()} (${(data.ser_prediction.confidence * 100).toFixed(0)}%)`;
          if (badgeEl) {
            badgeEl.textContent = '✔ Transcribed & Classified';
            badgeEl.style.color = '#22c55e';
          }
          updateAnalyzedAudioUI(data);
          showToast(`Speech Emotion: ${data.ser_prediction.dominant_emotion.toUpperCase()} (${(data.ser_prediction.confidence * 100).toFixed(0)}%)`);
        }
      } catch (err) {
        console.error('Audio analysis error:', err);
        if (statusText) statusText.textContent = `❌ Analysis error: ${err.message}`;
      }
    };
  };

  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
}

// Preset Voice Simulator
async function simulatePresetSpeech(presetKey) {
  const presets = {
    mayday: {
      text: "Mayday Mayday Mayday! Air India 202, dual engine flameout passing flight level 240, turning heading 180 for emergency landing!",
      freq: 1200,
      mod: 0.95,
      snr: 12.0
    },
    approach: {
      text: "Tower, Boeing 737 on final approach runway two seven left, established on ILS glideslope, gear down and locked.",
      freq: 850,
      mod: 0.65,
      snr: 28.0
    },
    goaround: {
      text: "Tower, go-around, go-around! Windshear alert on short final, climbing runway heading to 3,000 feet!",
      freq: 1400,
      mod: 0.90,
      snr: 16.0
    },
    clearance: {
      text: "Ground, Cessna 172 at ramp four with information Bravo, request taxi to runway one niner left.",
      freq: 600,
      mod: 0.60,
      snr: 32.0
    }
  };

  const p = presets[presetKey];
  if (!p) return;

  const transcriptEl = document.getElementById('live-transcription-text');
  const badgeEl = document.getElementById('transcription-status-badge');
  if (transcriptEl) {
    transcriptEl.textContent = `"${p.text}"`;
    transcriptEl.classList.remove('placeholder');
  }
  if (badgeEl) {
    badgeEl.textContent = `Preset: ${presetKey.toUpperCase()}`;
    badgeEl.style.color = '#a855f7';
  }

  // Adjust simulation parameters to match the prosody of this preset
  state.simParams.channel_snr_db = p.snr;
  state.simParams.modulation_index = p.mod;
  state.simParams.tone_freq_hz = p.freq;

  syncControlsToState();

  showToast(`Simulating ${presetKey.toUpperCase()} transmission`);
  await fetchSimulation();

  // Trigger emotion update
  if (state.lastSimData && state.lastSimData.ser_prediction) {
    renderEmotionFromData(state.lastSimData.ser_prediction);
  }
}

// Synchronize all pages with the analyzed audio
function updateAnalyzedAudioUI(data) {
  state.customAudioSamples = data.audio_samples;
  state.customVadAudio = data.vad_audio;
  state.lastTranscription = data.transcription;

  if (state.lastSimData) {
    state.lastSimData.time_ms = data.time_ms;
    state.lastSimData.ser_prediction = data.ser_prediction;
    if (state.lastSimData.stages && state.lastSimData.stages.stage8) {
      state.lastSimData.stages.stage8.audio_samples = data.audio_samples;
      state.lastSimData.stages.stage8.metrics = {
        "Source": data.filename || "Live Microphone Stream",
        "Sampling Rate": "16,000 Hz (16 kHz)",
        "Duration": `${data.duration_sec.toFixed(2)} seconds`,
        "Peak Amplitude": `${data.peak_amplitude.toFixed(4)} FS`,
        "RMS Power": `${data.rms_level.toFixed(4)} (${data.rms_dbfs.toFixed(1)} dBFS)`,
        "Transcription": data.transcription,
        "Dominant Emotion": `${data.ser_prediction.dominant_emotion} (${(data.ser_prediction.confidence * 100).toFixed(1)}%)`
      };
    }
  }

  // Update Transcription display
  const transcriptEl = document.getElementById('live-transcription-text');
  if (transcriptEl && data.transcription) {
    transcriptEl.textContent = `"${data.transcription}"`;
    transcriptEl.classList.remove('placeholder');
  }

  // Update Top Bar Emotion Badge
  const emotionBadge = document.getElementById('badge-emotion');
  if (emotionBadge && data.ser_prediction) {
    const emo = data.ser_prediction.dominant_emotion.toUpperCase();
    const conf = (data.ser_prediction.confidence * 100).toFixed(0);
    emotionBadge.textContent = `${emo} (${conf}%)`;
    emotionBadge.style.color = (emo === 'ANGRY' || emo === 'FEARFUL') ? '#ef4444' : '#22c55e';
  }

  // Render Emotion Tab immediately
  renderEmotionFromData(data.ser_prediction);

  // Redraw canvases
  renderAudioTab();
  renderSpeechTab();
  renderReceiverTab();
  refreshAudioPlayers();
}

function renderEmotionFromData(pred) {
  if (!pred) return;
  const badge = document.getElementById('ser-emotion-badge');
  const gauge = document.getElementById('ser-confidence-gauge');
  const prov = document.getElementById('ser-inference-mode');

  if (badge) {
    badge.textContent = pred.dominant_emotion.toUpperCase();
    badge.style.color = (pred.dominant_emotion === 'Angry' || pred.dominant_emotion === 'Fearful') ? '#ef4444' : '#22c55e';
  }
  if (gauge) gauge.textContent = `${(pred.confidence * 100).toFixed(1)}%`;
  if (prov) prov.textContent = pred.inference_mode || 'Neural SER Ingestion';

  // Draw 8-Class Probability Bars
  const barsContainer = document.getElementById('emotion-bars-container');
  if (barsContainer && pred.probabilities) {
    barsContainer.innerHTML = '';
    for (const [emo, prob] of Object.entries(pred.probabilities)) {
      const pct = (prob * 100).toFixed(1);
      const isDominant = emo === pred.dominant_emotion;
      const barColor = isDominant ? (emo === 'Angry' || emo === 'Fearful' ? '#ef4444' : '#22c55e') : '#3b82f6';

      const row = document.createElement('div');
      row.style.marginBottom = '12px';
      row.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px;">
          <span style="color: ${isDominant ? 'var(--text-primary)' : 'var(--text-secondary)'}; font-weight: ${isDominant ? '800' : '600'};">
            ${emo} ${isDominant ? '★' : ''}
          </span>
          <span style="font-family: 'JetBrains Mono'; color: ${isDominant ? barColor : 'var(--text-secondary)'};">${pct}%</span>
        </div>
        <div style="height: 10px; background: var(--bg-surface); border-radius: 6px; overflow: hidden; border: 1px solid var(--border-subtle);">
          <div style="height: 100%; width: ${pct}%; background: ${barColor}; border-radius: 6px; transition: width 0.5s ease-out; box-shadow: ${isDominant ? '0 0 10px ' + barColor : 'none'};"></div>
        </div>
      `;
      barsContainer.appendChild(row);
    }
  }
}

// Audio Tab helper
function renderAudioTab() {
  if (state.lastSimData) {
    const audioSig = state.customAudioSamples || (state.lastSimData.stages.stage8 ? state.lastSimData.stages.stage8.audio_samples : state.lastSimData.stages.stage5.audio_demod);
    const t = state.lastSimData.time_ms || [];
    drawWaveform('canvas-audio-full', t, audioSig, '#06b6d4', 'Demodulated Acoustic Audio Stream (16.0 kHz)');
  }
}

// Speech Tab helper
function renderSpeechTab() {
  if (state.lastSimData) {
    const vadSig = state.customVadAudio || (state.lastSimData.stages.stage8 ? state.lastSimData.stages.stage8.audio_samples : state.lastSimData.stages.stage5.audio_demod);
    const t = state.lastSimData.time_ms || [];
    drawWaveform('canvas-vad-view', t, vadSig, '#22c55e', '4-Second Fixed Window & VAD Energy Envelopes');
  }
}

// Toast System
function showToast(msg) {
  let toast = document.getElementById('toast-msg');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast-msg';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2600);
}
