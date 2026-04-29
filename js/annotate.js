(function () {
  'use strict';

  // ── Sort key (mirrors Python _sample_sort_key) ────────────────────────────

  function sampleSortKey(v) {
    const t = String(v).trim();
    const n = parseInt(t, 10);
    if (!isNaN(n) && String(n) === t) return [0, n, ''];
    const f = parseFloat(t);
    if (!isNaN(f)) return [1, f, ''];
    return [2, 0, t.toLowerCase()];
  }

  function cmpKeys(a, b) {
    const ka = sampleSortKey(a), kb = sampleSortKey(b);
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    if (ka[0] < 2) return ka[1] - kb[1];
    return ka[2] < kb[2] ? -1 : ka[2] > kb[2] ? 1 : 0;
  }

  // ── Preprocessing (mirrors Python _normalize_df + trim) ───────────────────

  function minMaxNormalize(channels) {
    const nCh = channels[0].length;
    const mins = Array(nCh).fill(Infinity);
    const maxs = Array(nCh).fill(-Infinity);
    for (const row of channels) for (let c = 0; c < nCh; c++) {
      if (row[c] < mins[c]) mins[c] = row[c];
      if (row[c] > maxs[c]) maxs[c] = row[c];
    }
    return channels.map(row =>
      row.map((v, c) => { const r = maxs[c] - mins[c]; return r === 0 ? 0 : (v - mins[c]) / r; })
    );
  }

  function trimNormalize(sample, trimRatio) {
    const n    = sample.channels.length;
    const trim = Math.floor(n * trimRatio);
    const end  = n - trim || n;
    const slicedCh = sample.channels.slice(trim, end);
    const slicedT  = sample.time.slice(trim, end);
    const normed   = minMaxNormalize(slicedCh.length ? slicedCh : sample.channels);
    return { time: slicedT.length === normed.length ? slicedT : Array.from({ length: normed.length }, (_, i) => i), channels: normed };
  }

  // ── Module state ──────────────────────────────────────────────────────────

  let groups     = null;
  let channelCols = [];
  let sortedKeys  = [];
  let traceKey    = null;
  let stateObj    = null;
  let plotReady   = false;

  // ── Init ──────────────────────────────────────────────────────────────────

  async function init() {
    const active = await CnnStore.loadActiveTrace();
    if (!active) { window.location.href = 'source.html'; return; }

    traceKey = active.name;
    const cfg    = CnnStore.loadConfig();
    const parsed = CsvParser.loadAndGroup(active.csv, cfg.group_by_col, cfg.time_index_col, cfg.channel_cols);
    groups      = parsed.groups;
    channelCols = parsed.channelCols;
    sortedKeys  = Object.keys(groups).sort(cmpKeys);

    stateObj = CnnStore.loadAnnotationState();
    if (!stateObj || stateObj.traceKey !== traceKey) {
      stateObj = CnnStore.resetAnnotationState(traceKey, sortedKeys);
    } else {
      stateObj.keys = sortedKeys;
    }

    document.getElementById('active-trace-name').textContent = active.name;
    document.getElementById('active-trace-row').classList.remove('d-none');

    const hasData = sortedKeys.length > 0;
    document.getElementById('annotate-area').classList.toggle('d-none', !hasData);
    document.getElementById('btn-skip').disabled  = !hasData;
    document.getElementById('btn-reset').disabled = !hasData;

    plotReady = false;
    renderSample(stateObj.idx);
  }

  // ── Render ────────────────────────────────────────────────────────────────

  function renderSample(idx) {
    const cfg         = CnnStore.loadConfig();
    const annotations = CnnStore.loadAnnotations(traceKey);

    const inReview   = idx >= sortedKeys.length;
    document.getElementById('review-banner').classList.toggle('d-none', !inReview);

    const clampedIdx = Math.min(Math.max(idx, 0), sortedKeys.length - 1);

    stateObj.idx = idx;
    CnnStore.saveAnnotationState(stateObj);

    const key            = sortedKeys[clampedIdx];
    const annotatedCount = CnnStore.annotationCount(traceKey);

    document.getElementById('sample-key').textContent      = key;
    document.getElementById('sample-idx').textContent      = `${clampedIdx + 1} / ${sortedKeys.length}`;
    document.getElementById('annotated-count').textContent = annotatedCount;
    document.getElementById('btn-prev').disabled           = clampedIdx <= 0;
    document.getElementById('btn-next').disabled           = clampedIdx >= sortedKeys.length - 1;

    const existingLabel = annotations[key] ?? null;
    const existRow = document.getElementById('existing-label-row');
    if (existingLabel !== null) {
      document.getElementById('existing-label-value').textContent = existingLabel;
      existRow.classList.remove('d-none');
    } else {
      existRow.classList.add('d-none');
    }

    const { time, channels } = trimNormalize(groups[key], cfg.trim_ratio);
    const nCh = channels[0]?.length ?? 0;

    const plotTraces = Array.from({ length: nCh }, (_, c) => ({
      x: time, y: channels.map(row => row[c]),
      name: channelCols[c] || `ch${c}`,
      type: 'scatter', mode: 'lines', line: { width: 1.5 },
    }));

    const shapes = existingLabel !== null ? [{
      type: 'line', x0: existingLabel, x1: existingLabel, y0: 0, y1: 1,
      xref: 'x', yref: 'paper', line: { color: '#dc3545', width: 2, dash: 'dot' },
    }] : [];

    Plotly.react('plot', plotTraces, {
      margin: { t: 10, r: 10, b: 40, l: 45 },
      showlegend: true, legend: { orientation: 'h' },
      xaxis: { title: 'Time' }, yaxis: { title: 'Normalized value' },
      shapes,
    }, { responsive: true });

    // Re-register click handler after each react (removes old one first)
    const plotEl = document.getElementById('plot');
    plotEl.removeAllListeners('plotly_click');
    plotEl.on('plotly_click', evt => {
      const xVal = evt.points?.[0]?.x;
      if (xVal === undefined || xVal === null) return;
      CnnStore.upsertAnnotation(traceKey, key, xVal);
      renderSample(stateObj.idx + 1);
    });

    plotReady = true;
  }

  // ── Button handlers ───────────────────────────────────────────────────────

  document.getElementById('btn-prev').addEventListener('click',  () => renderSample(stateObj.idx - 1));
  document.getElementById('btn-next').addEventListener('click',  () => renderSample(stateObj.idx + 1));
  document.getElementById('btn-skip').addEventListener('click',  () => renderSample(stateObj.idx + 1));
  document.getElementById('btn-reset').addEventListener('click', () => {
    stateObj = CnnStore.resetAnnotationState(traceKey, sortedKeys);
    renderSample(0);
  });

  window.addEventListener('keydown', e => {
    if (document.activeElement?.tagName === 'INPUT') return;
    if (e.key?.toLowerCase() === 's') renderSample(stateObj.idx + 1);
  });

  // ── Drop zone ─────────────────────────────────────────────────────────────

  const dropZone  = document.getElementById('trace-drop-zone');
  const fileInput = document.getElementById('trace-file-input');
  const statusEl  = document.getElementById('trace-upload-status');

  document.getElementById('trace-browse-link').addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('click', e => { if (e.target.id !== 'trace-browse-link') fileInput.click(); });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('bg-light'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('bg-light'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('bg-light');
    if (e.dataTransfer.files[0]) handleTraceFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', e => { if (e.target.files[0]) handleTraceFile(e.target.files[0]); });

  async function handleTraceFile(file) {
    statusEl.textContent = `Loading ${file.name}…`;
    statusEl.classList.remove('d-none');
    try {
      const csv = await file.text();
      await CnnStore.saveActiveTrace(file.name, csv);
      await init();
      statusEl.classList.add('d-none');
    } catch (e) {
      statusEl.textContent = 'Error: ' + e.message;
    }
  }

  // ── Start ─────────────────────────────────────────────────────────────────

  document.addEventListener('DOMContentLoaded', init);
})();
