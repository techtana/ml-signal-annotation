(function () {
  const { TRACE_DATA_URL, DEFAULT_DATA_PATH, GROUP_BY_COL, TIME_INDEX_COL, CHANNEL_COLS } =
    window.CNN_PREDICT_CFG;
  const META_KEY  = 'cnn-model-meta';
  const IDB_MODEL = 'indexeddb://cnn-latest';

  // ── State ─────────────────────────────────────────────────────────────────
  let activeModel    = null;
  let modelMeta      = null;
  let traceData      = null;
  let processedCache = {};
  let predictions    = [];
  let groupByCol     = GROUP_BY_COL;
  let currentIdx     = 0;
  let plotReady      = false;
  let hasBrowserModel = false;

  // ── Preprocessing (mirrors Python load_and_group + normalize) ─────────────

  function minMaxNormalize(channels) {
    const nCh = channels[0].length;
    const mins = Array(nCh).fill(Infinity);
    const maxs = Array(nCh).fill(-Infinity);
    for (const row of channels) {
      for (let c = 0; c < nCh; c++) {
        if (row[c] < mins[c]) mins[c] = row[c];
        if (row[c] > maxs[c]) maxs[c] = row[c];
      }
    }
    return channels.map(row =>
      row.map((v, c) => { const r = maxs[c] - mins[c]; return r === 0 ? 0 : (v - mins[c]) / r; })
    );
  }

  function trimAndPad(channels, trimRatio, maxLen) {
    const n = channels.length;
    const trim = Math.floor(n * trimRatio);
    let sliced = channels.slice(trim, n - trim || n);
    if (!sliced.length) throw new Error('All rows trimmed');
    const normed = minMaxNormalize(sliced);
    const last = normed[normed.length - 1];
    while (normed.length < maxLen) normed.push([...last]);
    return normed;
  }

  // ── Client-side CSV parser (mirrors Python load_and_group) ────────────────

  function parseCSVRow(line) {
    const result = [];
    let inQuotes = false, cur = '';
    for (const ch of line) {
      if (ch === '"') { inQuotes = !inQuotes; }
      else if (ch === ',' && !inQuotes) { result.push(cur); cur = ''; }
      else { cur += ch; }
    }
    result.push(cur);
    return result;
  }

  function normalizeKey(value) {
    const text = String(value == null ? '' : value).trim();
    if (!text) return '';
    const num = parseFloat(text);
    if (!isNaN(num) && isFinite(num) && Number.isInteger(num)) return String(num);
    if (!isNaN(num) && isFinite(num)) return text;
    return text;
  }

  function parseTraceData(csvText) {
    const lines = csvText.split('\n').filter(l => l.trim());
    if (lines.length < 2) throw new Error('CSV has no data rows');
    const headers = parseCSVRow(lines[0]).map(h => h.trim());

    const groupIdx = headers.indexOf(GROUP_BY_COL);
    const timeIdx  = headers.indexOf(TIME_INDEX_COL);
    if (groupIdx < 0) throw new Error(`Column "${GROUP_BY_COL}" not found in CSV`);

    let channelCols = CHANNEL_COLS && CHANNEL_COLS.length
      ? CHANNEL_COLS
      : headers.filter(h => h !== GROUP_BY_COL && h !== TIME_INDEX_COL && h !== 'label');
    const chanIndices = channelCols.map(c => {
      const i = headers.indexOf(c);
      if (i < 0) throw new Error(`Channel column "${c}" not found in CSV`);
      return i;
    });

    const rows = lines.slice(1).map(l => parseCSVRow(l));

    if (timeIdx >= 0) {
      rows.sort((a, b) => parseFloat(a[timeIdx]) - parseFloat(b[timeIdx]));
    }

    const grouped = {};
    for (const row of rows) {
      const key = normalizeKey(row[groupIdx]);
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(row);
    }

    let maxLen = 0;
    const samples = {};
    for (const [key, grpRows] of Object.entries(grouped)) {
      const dataRows = grpRows.slice(1); // mirrors Python iloc[1:]
      const time     = dataRows.map(r => timeIdx >= 0 ? parseFloat(r[timeIdx]) : 0);
      const channels = dataRows.map(r => chanIndices.map(ci => {
        const v = parseFloat(r[ci]);
        return isNaN(v) ? 0 : v;
      }));
      samples[key] = { time, channels };
      maxLen = Math.max(maxLen, dataRows.length);
    }

    return { samples, channel_cols: channelCols, max_len: maxLen, annotations: {}, group_by_col: GROUP_BY_COL };
  }

  // ── UI helpers ────────────────────────────────────────────────────────────

  function setModelInfo(html)  { document.getElementById('model-info').innerHTML   = html; }
  function setCompatBadge(html){ document.getElementById('compat-badge').innerHTML = html; }

  function getDataPath() {
    return (document.getElementById('data-path-select').value || '').trim() || DEFAULT_DATA_PATH;
  }

  function updateRunBtn() {
    document.getElementById('run-btn').disabled = !(activeModel && traceData);
  }

  // ── Compat check ──────────────────────────────────────────────────────────

  function checkCompat() {
    if (!modelMeta || !traceData) { setCompatBadge(''); return; }
    const mCh   = modelMeta.nChannels;
    const mCols = modelMeta.channelCols || [];
    const dCh   = traceData.channel_cols.length;
    const dCols = traceData.channel_cols;
    const chOk  = mCh === dCh;
    const colsOk = !mCols.length ||
      JSON.stringify([...dCols].sort()) === JSON.stringify([...mCols].sort());
    if (chOk && colsOk)
      setCompatBadge('<span class="badge bg-success">&#10003; Compatible</span>');
    else if (!chOk)
      setCompatBadge(`<span class="badge bg-danger">&#10007; Channel mismatch — data: ${dCh} ch, model: ${mCh} ch</span>`);
    else
      setCompatBadge(`<span class="badge bg-warning text-dark">&#9888; Column mismatch — data: [${dCols.join(', ')}] &middot; model: [${mCols.join(', ')}]</span>`);
  }

  // ── Load trace data from server ───────────────────────────────────────────

  async function loadTraceData(path) {
    const r = await fetch(`${TRACE_DATA_URL}?path=${encodeURIComponent(path)}`);
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    setTraceData(d);
  }

  function setTraceData(d) {
    traceData      = d;
    groupByCol     = d.group_by_col || groupByCol;
    processedCache = {};
    checkCompat();
    updateRunBtn();
  }

  document.getElementById('data-path-select').addEventListener('change', async () => {
    document.getElementById('csv-file-info').classList.add('d-none');
    try {
      await loadTraceData(getDataPath());
    } catch (e) {
      setCompatBadge(`<span class="badge bg-warning text-dark">&#9888; ${e.message}</span>`);
    }
  });

  // ── CSV drop zone ────────────────────────────────���────────────────────────

  const csvDropZone  = document.getElementById('csv-drop-zone');
  const csvFileInput = document.getElementById('csv-file-input');
  const csvFileInfo  = document.getElementById('csv-file-info');

  document.getElementById('csv-browse-link').addEventListener('click', () => csvFileInput.click());
  csvDropZone.addEventListener('click', e => { if (e.target.id !== 'csv-browse-link') csvFileInput.click(); });
  csvDropZone.addEventListener('dragover',  e => { e.preventDefault(); csvDropZone.classList.add('bg-light'); });
  csvDropZone.addEventListener('dragleave', () => csvDropZone.classList.remove('bg-light'));
  csvDropZone.addEventListener('drop', e => {
    e.preventDefault(); csvDropZone.classList.remove('bg-light');
    if (e.dataTransfer.files[0]) handleCSVFile(e.dataTransfer.files[0]);
  });
  csvFileInput.addEventListener('change', e => { if (e.target.files[0]) handleCSVFile(e.target.files[0]); });

  async function handleCSVFile(file) {
    csvFileInfo.textContent = `Loading ${file.name}…`;
    csvFileInfo.classList.remove('d-none');
    try {
      const text = await file.text();
      const parsed = parseTraceData(text);
      if (!Object.keys(parsed.samples).length) throw new Error('No samples found in CSV');
      document.getElementById('data-path-select').value = '';
      setTraceData(parsed);
      csvFileInfo.innerHTML =
        `<span class="badge bg-secondary me-1">${file.name}</span>` +
        `<span class="text-secondary">${Object.keys(parsed.samples).length} samples &middot; ` +
        `${parsed.channel_cols.length} channels</span>`;
    } catch (e) {
      csvFileInfo.innerHTML = `<span class="text-danger">Error: ${e.message}</span>`;
    }
  }

  // ── Model load helpers ────────────────────────���───────────────────────────

  async function setModel(model, meta) {
    if (activeModel) activeModel.dispose();
    activeModel = model;
    modelMeta   = meta;
    if (!meta) { setModelInfo('<span class="text-warning small">No model loaded.</span>'); updateRunBtn(); return; }

    const parts = [];
    if (meta.nChannels) parts.push(`${meta.nChannels} ch`);
    if (meta.maxLen)    parts.push(`${meta.maxLen} steps`);
    if (meta.trimRatio != null) parts.push(`trim ${meta.trimRatio}`);
    setModelInfo(`<strong>${meta._source || 'Model'}</strong>` +
      (parts.length ? ` <span class="text-muted">&middot; ${parts.join(' &middot; ')}</span>` : ''));
    checkCompat();
    updateRunBtn();
  }

  // ── Load from IndexedDB ───────────────────────────────────────────────────

  async function loadFromIDB() {
    try {
      const model  = await tf.loadLayersModel(IDB_MODEL);
      const stored = JSON.parse(localStorage.getItem(META_KEY) || 'null');
      const meta   = stored
        ? { ...stored, _source: `Browser model (${new Date(stored.trainedAt).toLocaleString()})` }
        : null;
      await setModel(model, meta);
      hasBrowserModel = true;
      document.getElementById('model-stored-row').classList.remove('d-none');
      document.getElementById('model-stored-badge').textContent = 'Browser model';
      document.getElementById('use-browser-model-row').classList.add('d-none');
      if (stored) {
        document.getElementById('model-stored-detail').textContent =
          new Date(stored.trainedAt).toLocaleString();
      }
    } catch {
      // No model in IndexedDB yet
    }
  }

  document.getElementById('clear-model-btn').addEventListener('click', async e => {
    e.preventDefault();
    try { await tf.io.removeModel(IDB_MODEL); } catch {}
    localStorage.removeItem(META_KEY);
    if (activeModel) { activeModel.dispose(); activeModel = null; }
    modelMeta = null;
    hasBrowserModel = false;
    document.getElementById('model-stored-row').classList.add('d-none');
    document.getElementById('use-browser-model-row').classList.add('d-none');
    setModelInfo('<span class="text-secondary small">Model cleared.</span>');
    setCompatBadge('');
    updateRunBtn();
  });

  document.getElementById('use-browser-model-btn').addEventListener('click', async e => {
    e.preventDefault();
    await loadFromIDB();
  });

  // ── Split model drop zones (.json / .bin) ────────────────────────────────

  let pendingJsonFile = null;
  let pendingBinFiles = [];

  function wireDropZone(zoneId, inputId, browseLinkId, onFiles) {
    const zone  = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    document.getElementById(browseLinkId).addEventListener('click', () => input.click());
    zone.addEventListener('click', e => { if (e.target.id !== browseLinkId) input.click(); });
    zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('bg-light'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('bg-light'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('bg-light');
      onFiles(Array.from(e.dataTransfer.files));
    });
    input.addEventListener('change', e => onFiles(Array.from(e.target.files)));
  }

  function setZoneLabel(labelId, files) {
    const names = (Array.isArray(files) ? files : [files]).map(f => f.name).join(', ');
    document.getElementById(labelId).innerHTML =
      `<span class="badge bg-secondary text-truncate" style="max-width:120px" title="${names}">${names}</span>`;
  }

  wireDropZone('json-drop-zone', 'json-file-input', 'json-browse-link', files => {
    const f = files.find(f => f.name.toLowerCase().endsWith('.json'));
    if (!f) return;
    pendingJsonFile = f;
    setZoneLabel('json-drop-label', f);
    tryLoadModel();
  });

  wireDropZone('bin-drop-zone', 'bin-file-input', 'bin-browse-link', files => {
    const bins = files.filter(f => f.name.toLowerCase().endsWith('.bin'));
    if (!bins.length) return;
    pendingBinFiles = bins;
    setZoneLabel('bin-drop-label', bins);
    if (pendingJsonFile) tryLoadModel();
  });

  async function tryLoadModel() {
    if (!pendingJsonFile) return;
    setModelInfo('<span class="text-secondary small">Loading model&hellip;</span>');
    try {
      // Read manifest to learn expected weight basenames, then rename
      // the user's files to match — so any filename works.
      const jsonText = await pendingJsonFile.text();
      const topology = JSON.parse(jsonText);
      const expectedBasenames = (topology.weightsManifest || [])
        .flatMap(g => g.paths)
        .map(p => p.split('/').pop());

      const renamedBins = pendingBinFiles.map((file, i) => {
        const target = expectedBasenames[i];
        return target && target !== file.name
          ? new File([file], target, { type: file.type })
          : file;
      });

      const model = await tf.loadLayersModel(
        tf.io.browserFiles([pendingJsonFile, ...renamedBins])
      );
      const shape = model.inputs[0].shape;
      const meta  = { maxLen: shape[1], nChannels: shape[2], channelCols: [], trimRatio: null,
                      _source: pendingJsonFile.name };
      await setModel(model, meta);
      document.getElementById('model-stored-row').classList.add('d-none');
      if (hasBrowserModel) {
        document.getElementById('use-browser-model-row').classList.remove('d-none');
      }
    } catch (e) {
      if (!pendingBinFiles.length) {
        setModelInfo('<span class="text-secondary small">Drop the <code>.bin</code> weights file to continue.</span>');
      } else {
        setModelInfo(`<span class="text-danger small">Error: ${e.message}</span>`);
      }
    }
  }

  // ── Run prediction ──────────────────────────────────────────────────────���─

  document.getElementById('run-btn').addEventListener('click', async function () {
    if (!activeModel || !traceData) return;
    const btn   = this;
    const errEl = document.getElementById('predict-error');
    btn.disabled = true;
    btn.textContent = 'Running…';
    errEl.classList.add('d-none');

    try {
      const shape     = activeModel.inputs[0].shape;
      const maxLen    = shape[1];
      const nChannels = shape[2];
      const trimRatio = (modelMeta && modelMeta.trimRatio != null) ? modelMeta.trimRatio : 0.1;
      const keys      = Object.keys(traceData.samples);

      for (const k of keys) {
        if (!processedCache[k]) {
          processedCache[k] = trimAndPad(traceData.samples[k].channels, trimRatio, maxLen);
        }
      }

      const flat = new Float32Array(keys.length * maxLen * nChannels);
      let off = 0;
      for (const k of keys) {
        for (const row of processedCache[k]) for (const v of row) flat[off++] = v;
      }
      const X = tf.tensor4d(flat, [keys.length, maxLen, nChannels, 1]);

      const rawPreds = await tf.tidy(() => activeModel.predict(X)).data();
      X.dispose();

      const timeScale = (maxLen - 1) / maxLen;
      predictions = keys.map((k, i) => ({
        [groupByCol]: k,
        predicted_position: rawPreds[i] * maxLen * timeScale,
      }));

      plotReady = false;
      renderResults();
    } catch (e) {
      errEl.textContent = 'Prediction failed: ' + e.message;
      errEl.classList.remove('d-none');
    }
    btn.disabled = false;
    btn.textContent = 'Run prediction';
    updateRunBtn();
  });

  // ── Results ───────────────────────────────────────────────────────────────

  function renderResults() {
    if (!predictions.length) return;
    document.getElementById('results-thead-row').innerHTML =
      `<th>${groupByCol}</th><th>Predicted position</th>`;
    document.getElementById('results-tbody').innerHTML = predictions
      .map(r => `<tr><td>${r[groupByCol]}</td><td>${r.predicted_position.toFixed(4)}</td></tr>`)
      .join('');
    document.getElementById('sample-select').innerHTML = predictions
      .map(r => `<option value="${r[groupByCol]}">${r[groupByCol]}</option>`).join('');
    document.getElementById('results-header').textContent =
      `Results — ${predictions.length} sample${predictions.length !== 1 ? 's' : ''}`;
    document.getElementById('results-card').classList.remove('d-none');
    showView('table');
  }

  document.getElementById('dl-csv-btn').addEventListener('click', () => {
    if (!predictions.length) return;
    const header = `${groupByCol},predicted_position\n`;
    const rows   = predictions.map(r => `${r[groupByCol]},${r.predicted_position}`).join('\n');
    const blob   = new Blob([header + rows], { type: 'text/csv' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a');
    a.href = url; a.download = 'predictions.csv';
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  // ── Plot view ─────────────────────────────────────────────────────────────

  function showView(view) {
    const isTable = view === 'table';
    document.getElementById('view-table').classList.toggle('d-none', !isTable);
    document.getElementById('view-plot').classList.toggle('d-none', isTable);
    document.getElementById('btn-table').className = 'btn btn-sm ' + (isTable ? 'btn-primary' : 'btn-outline-secondary');
    document.getElementById('btn-plot').className  = 'btn btn-sm ' + (!isTable ? 'btn-primary' : 'btn-outline-secondary');
    if (!isTable && !plotReady) { currentIdx = 0; syncControls(); renderTrace(currentIdx); }
  }

  function syncControls() {
    const keys = predictions.map(r => String(r[groupByCol]));
    document.getElementById('sample-select').value       = keys[currentIdx];
    document.getElementById('btn-prev').disabled          = currentIdx === 0;
    document.getElementById('btn-next').disabled          = currentIdx === keys.length - 1;
    document.getElementById('sample-counter').textContent = `${currentIdx + 1} / ${keys.length}`;
  }

  function stepSample(delta) {
    const next = currentIdx + delta;
    if (next < 0 || next >= predictions.length) return;
    currentIdx = next;
    syncControls();
    renderTrace(currentIdx);
  }

  function renderTrace(idx) {
    const row  = predictions[idx];
    const key  = String(row[groupByCol]);
    const pred = row.predicted_position;

    if (!traceData || !traceData.samples[key]) return;
    const shape    = activeModel ? activeModel.inputs[0].shape : [null, null, null, 1];
    const maxLen   = shape[1] || traceData.max_len;
    const trimRatio = (modelMeta && modelMeta.trimRatio != null) ? modelMeta.trimRatio : 0.1;

    const processed = processedCache[key] ||
      trimAndPad(traceData.samples[key].channels, trimRatio, maxLen);
    processedCache[key] = processed;

    const xVals = Array.from({ length: processed.length }, (_, i) => i);
    const nCh   = processed[0].length;
    const cols  = traceData.channel_cols;

    const traces = Array.from({ length: nCh }, (_, c) => ({
      x: xVals,
      y: processed.map(row => row[c]),
      name: cols[c] || `ch${c}`,
      type: 'scatter', mode: 'lines', line: { width: 1.5 },
    }));

    const predIdx = pred / ((maxLen - 1) / maxLen);
    const shapes  = [{ type: 'line', x0: predIdx, x1: predIdx, y0: 0, y1: 1,
      xref: 'x', yref: 'paper', line: { color: '#0d6efd', width: 2, dash: 'dash' } }];
    const annots  = [{ x: predIdx, y: 0.98, xref: 'x', yref: 'paper',
      text: `pred: ${pred.toFixed(2)}`, showarrow: false,
      font: { color: '#0d6efd', size: 11 }, xanchor: 'left', yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.7)', borderpad: 2 }];

    Plotly.react('trace-plot', traces, {
      margin: { t: 16, r: 16, b: 80, l: 50 },
      showlegend: true, legend: { orientation: 'h', x: 0, y: -0.25 },
      xaxis: { title: 'Step' },
      yaxis: { title: 'Normalized value', range: [-0.05, 1.05] },
      shapes, annotations: annots,
    }, { responsive: true });
    plotReady = true;
  }

  document.getElementById('btn-table').addEventListener('click', () => showView('table'));
  document.getElementById('btn-plot').addEventListener('click',  () => showView('plot'));
  document.getElementById('btn-prev').addEventListener('click',  () => stepSample(-1));
  document.getElementById('btn-next').addEventListener('click',  () => stepSample(1));
  document.getElementById('sample-select').addEventListener('change', e => {
    currentIdx = predictions.findIndex(r => String(r[groupByCol]) === e.target.value);
    if (currentIdx < 0) currentIdx = 0;
    syncControls();
    renderTrace(currentIdx);
  });

  // ── Page init ────────────────────────────────────────────────────────��────

  (async () => {
    await loadFromIDB();
    try {
      await loadTraceData(getDataPath());
    } catch { /* non-fatal */ }
  })();
})();
