(function () {
  const { TRAINING_DATA_URL } = window.CNN_TRAIN_CFG;
  const META_KEY  = 'cnn-model-meta';
  const LOSS_KEY  = 'cnn-train-loss';
  const IDB_MODEL = 'indexeddb://cnn-latest';

  // ── Preprocessing ────────────────────────────────────────────────────────

  function minMaxNormalize(channels) {
    // channels: Array[n][nCh]  →  returns same shape, scaled to [0,1] per column
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
      row.map((v, c) => {
        const r = maxs[c] - mins[c];
        return r === 0 ? 0 : (v - mins[c]) / r;
      })
    );
  }

  function trimAndPad(channels, trimRatio, maxLen) {
    const n = channels.length;
    const trim = Math.floor(n * trimRatio);
    let sliced = channels.slice(trim, n - trim || n);
    if (!sliced.length) throw new Error('All rows trimmed — reduce trim_ratio');
    const normed = minMaxNormalize(sliced);
    const last = normed[normed.length - 1];
    while (normed.length < maxLen) normed.push([...last]);
    return normed; // [maxLen, nCh]
  }

  // Seeded Fisher-Yates shuffle (LCG, matches sklearn random_state semantics loosely)
  function seededShuffle(arr, seed) {
    const a = [...arr];
    let s = (seed >>> 0) || 1;
    const next = () => { s = Math.imul(s, 1664525) + 1013904223 >>> 0; return s / 0x100000000; };
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ── Model ────────────────────────────────────────────────────────────────

  function buildModel(maxLen, nChannels) {
    const model = tf.sequential({ layers: [
      tf.layers.conv2d({ inputShape: [maxLen, nChannels, 1], filters: 32, kernelSize: [5, 1], strides: [1, 1], activation: 'relu' }),
      tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }),
      tf.layers.conv2d({ filters: 64, kernelSize: [5, 1], activation: 'relu' }),
      tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }),
      tf.layers.flatten(),
      tf.layers.dense({ units: maxLen, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'linear' }),
    ] });
    model.compile({ loss: 'meanSquaredError', optimizer: 'adam', metrics: ['mse'] });
    return model;
  }

  function makeTensors(keys, processed, annotations, maxLen, nChannels) {
    const flat = new Float32Array(keys.length * maxLen * nChannels);
    let off = 0;
    for (const k of keys) {
      for (const row of processed[k]) {
        for (const v of row) flat[off++] = v;
      }
    }
    const X = tf.tensor4d(flat, [keys.length, maxLen, nChannels, 1]);
    const y = tf.tensor1d(keys.map(k => {
      const rawLabel = annotations[k];
      return Math.min(Math.max(0, Math.round(rawLabel)), maxLen - 1) / maxLen;
    }));
    return { X, y };
  }

  // ── Loss chart ───────────────────────────────────────────────────────────

  const lossHistory = [], valLossHistory = [];

  function initChart() {
    document.getElementById('chart-card').classList.remove('d-none');
    Plotly.newPlot('loss-chart', [
      { x: [], y: [], name: 'train loss', mode: 'lines', line: { color: '#0d6efd', width: 1.5 } },
      { x: [], y: [], name: 'val loss',   mode: 'lines', line: { color: '#fd7e14', width: 1.5 } },
    ], {
      margin: { t: 10, r: 16, b: 60, l: 50 },
      xaxis: { title: 'Epoch' },
      yaxis: { title: 'MSE loss' },
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: -0.35 },
    }, { responsive: true, displayModeBar: false });
  }

  function updateChart() {
    const epochs = lossHistory.map((_, i) => i + 1);
    Plotly.update('loss-chart', { x: [epochs, epochs], y: [lossHistory, valLossHistory] }, {}, [0, 1]);
  }

  // ── Latest-run card ──────────────────────────────────────────────────────

  function renderLatestRun(meta) {
    const el    = document.getElementById('latest-run-content');
    const dlBtn = document.getElementById('download-btn');
    const note  = document.getElementById('download-note');
    if (!meta) {
      el.innerHTML = '<div class="text-secondary">No run yet.</div>';
      dlBtn.classList.add('d-none');
      note.classList.add('d-none');
      return;
    }
    const dt = meta.trainedAt ? new Date(meta.trainedAt).toLocaleString() : '—';
    const rows = [
      ['Trained',    dt],
      ['Samples',    meta.numSamples    ?? '—'],
      ['Annotated',  meta.numAnnotated  ?? '—'],
      ['Final loss', meta.finalLoss     != null ? meta.finalLoss.toFixed(6)    : '—'],
      ['Val loss',   meta.finalValLoss  != null ? meta.finalValLoss.toFixed(6) : '—'],
      ['Channels',   meta.nChannels     ?? '—'],
      ['Max steps',  meta.maxLen        ?? '—'],
      ['Trim ratio', meta.trimRatio     ?? '—'],
    ].map(([k, v]) =>
      `<tr><th class="fw-normal text-secondary pe-3" style="white-space:nowrap;width:1%">${k}</th><td>${v}</td></tr>`
    ).join('');
    el.innerHTML = `<table class="table table-sm mb-0"><tbody>${rows}</tbody></table>`;
    dlBtn.classList.remove('d-none');
    note.classList.remove('d-none');
  }

  // Page load: restore latest-run card
  try {
    renderLatestRun(JSON.parse(localStorage.getItem(META_KEY)));
  } catch {
    renderLatestRun(null);
  }

  // Page load: restore loss chart and progress from last run
  try {
    const stored = JSON.parse(localStorage.getItem(LOSS_KEY));
    if (stored && stored.loss && stored.loss.length) {
      lossHistory.push(...stored.loss);
      valLossHistory.push(...(stored.valLoss || []));
      const numEpochs = stored.numEpochs || lossHistory.length;
      const lastEpoch = lossHistory.length;
      document.getElementById('progress-card').classList.remove('d-none');
      document.getElementById('progress-bar').style.width =
        (lastEpoch / numEpochs * 100).toFixed(0) + '%';
      document.getElementById('epoch-label').textContent =
        `Epoch ${lastEpoch} / ${numEpochs}`;
      document.getElementById('loss-label').textContent =
        `loss ${lossHistory[lastEpoch - 1].toFixed(5)} · val ${valLossHistory[lastEpoch - 1].toFixed(5)}`;
      initChart();
      updateChart();
    }
  } catch { /* no stored history */ }

  // Download button
  document.getElementById('download-btn').addEventListener('click', async () => {
    try {
      const model = await tf.loadLayersModel(IDB_MODEL);
      await model.save('downloads://cnn-model');
      model.dispose();
    } catch (e) {
      alert('Could not load model from browser storage: ' + e.message);
    }
  });

  // ── Training ─────────────────────────────────────────────────────────────

  document.getElementById('train-btn').addEventListener('click', async function () {
    const btn   = this;
    const errEl = document.getElementById('train-error');
    btn.disabled = true;
    btn.textContent = 'Loading data…';
    errEl.classList.add('d-none');
    lossHistory.length = 0;
    valLossHistory.length = 0;
    localStorage.removeItem(LOSS_KEY);

    let data;
    try {
      const r = await fetch(TRAINING_DATA_URL);
      data = await r.json();
      if (data.error) throw new Error(data.error);
    } catch (e) {
      btn.disabled = false;
      btn.textContent = 'Run training';
      errEl.textContent = 'Failed to load data: ' + e.message;
      errEl.classList.remove('d-none');
      return;
    }

    const { samples, annotations, channel_cols, max_len: maxLen, cfg } = data;
    const trimRatio = cfg.trim_ratio;
    const nChannels = channel_cols.length;
    const keys = Object.keys(samples);

    // Preprocess
    const processed = {};
    try {
      for (const k of keys) {
        processed[k] = trimAndPad(samples[k].channels, trimRatio, maxLen);
      }
    } catch (e) {
      btn.disabled = false;
      btn.textContent = 'Run training';
      errEl.textContent = 'Preprocessing failed: ' + e.message;
      errEl.classList.remove('d-none');
      return;
    }

    const annotatedKeys = keys.filter(k => k in annotations);
    if (annotatedKeys.length < 2) {
      btn.disabled = false;
      btn.textContent = 'Run training';
      errEl.textContent = `Need at least 2 annotated samples (found ${annotatedKeys.length}).`;
      errEl.classList.remove('d-none');
      return;
    }

    // Train / test split
    const shuffled = seededShuffle(annotatedKeys, cfg.random_state);
    const splitIdx = Math.max(1, Math.floor(shuffled.length * (1 - cfg.test_size)));
    const trainKeys = shuffled.slice(0, splitIdx);
    const testKeys  = shuffled.slice(splitIdx).length ? shuffled.slice(splitIdx) : [shuffled[0]];

    btn.textContent = 'Training…';
    document.getElementById('progress-card').classList.remove('d-none');
    initChart();

    let model, finalLoss, finalValLoss;
    try {
      model = buildModel(maxLen, nChannels);

      const { X: X_train, y: y_train } = makeTensors(trainKeys, processed, annotations, maxLen, nChannels);
      const { X: X_test,  y: y_test  } = makeTensors(testKeys,  processed, annotations, maxLen, nChannels);

      await model.fit(X_train, y_train, {
        epochs: cfg.num_epochs,
        batchSize: cfg.batch_size,
        validationData: [X_test, y_test],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            lossHistory.push(logs.loss);
            valLossHistory.push(logs.val_loss ?? logs.loss);
            const pct = ((epoch + 1) / cfg.num_epochs * 100).toFixed(0);
            document.getElementById('progress-bar').style.width = pct + '%';
            document.getElementById('epoch-label').textContent =
              `Epoch ${epoch + 1} / ${cfg.num_epochs}`;
            document.getElementById('loss-label').textContent =
              `loss ${logs.loss.toFixed(5)} · val ${(logs.val_loss ?? logs.loss).toFixed(5)}`;
            updateChart();
            localStorage.setItem(LOSS_KEY, JSON.stringify({
              loss: lossHistory,
              valLoss: valLossHistory,
              numEpochs: cfg.num_epochs,
            }));
            await tf.nextFrame();
          },
        },
      });

      finalLoss    = lossHistory[lossHistory.length - 1];
      finalValLoss = valLossHistory[valLossHistory.length - 1];

      X_train.dispose(); y_train.dispose();
      X_test.dispose();  y_test.dispose();

      await model.save(IDB_MODEL);

    } catch (e) {
      btn.disabled = false;
      btn.textContent = 'Run training';
      errEl.textContent = 'Training failed: ' + e.message;
      errEl.classList.remove('d-none');
      if (model) model.dispose();
      return;
    }

    const meta = {
      maxLen, nChannels,
      channelCols: channel_cols,
      trimRatio,
      trainedAt:    new Date().toISOString(),
      numSamples:   keys.length,
      numAnnotated: annotatedKeys.length,
      finalLoss,
      finalValLoss,
    };
    localStorage.setItem(META_KEY, JSON.stringify(meta));

    model.dispose();
    btn.disabled = false;
    btn.textContent = 'Run training';
    renderLatestRun(meta);
  });
})();
