(function () {
  const cfg = window.CNN_ANNOTATE_CFG;

  // ── Trace CSV drop zone (always present) ──────────────────────────────────

  const dropZone  = document.getElementById('trace-drop-zone');
  const fileInput = document.getElementById('trace-file-input');
  const status    = document.getElementById('trace-upload-status');

  document.getElementById('trace-browse-link').addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('click', e => { if (e.target.id !== 'trace-browse-link') fileInput.click(); });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('bg-light'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('bg-light'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('bg-light');
    if (e.dataTransfer.files[0]) handleTraceFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', e => { if (e.target.files[0]) handleTraceFile(e.target.files[0]); });

  async function handleTraceFile(file) {
    status.textContent = `Uploading ${file.name}…`;
    status.classList.remove('d-none');
    dropZone.style.pointerEvents = 'none';
    const fd = new FormData();
    fd.append('trace_file', file);
    try {
      const r = await fetch(cfg.UPLOAD_TRACE_URL, { method: 'POST', body: fd });
      const d = await r.json();
      if (!d.ok) throw new Error(d.error);
      window.location.reload();
    } catch (e) {
      status.textContent = 'Upload failed: ' + e.message;
      dropZone.style.pointerEvents = '';
    }
  }

  // ── Plot (only when a trace is active) ────────────────────────────────────

  if (!cfg.ready) return;

  const { key, x, series, existingLabel, ANNOTATE_LABEL_URL, ANNOTATE_SKIP_URL } = cfg;

  const traces = series.map(s => ({
    x,
    y: s.y,
    name: s.name,
    type: 'scatter',
    mode: 'lines',
    line: { width: 1.5 },
  }));

  const shapes = [];
  if (existingLabel !== null) {
    shapes.push({
      type: 'line',
      x0: existingLabel, x1: existingLabel, y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: { color: '#dc3545', width: 2, dash: 'dot' },
    });
  }

  Plotly.newPlot('plot', traces, {
    margin: { t: 10, r: 10, b: 40, l: 45 },
    showlegend: true,
    legend: { orientation: 'h' },
    xaxis: { title: 'Time' },
    yaxis: { title: 'Normalized value' },
    shapes,
  }, { responsive: true });

  function postLabel(xVal) {
    return fetch(ANNOTATE_LABEL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key, label: xVal }),
    }).then(r => r.json());
  }

  document.getElementById('plot').on('plotly_click', evt => {
    const xVal = evt.points?.[0]?.x;
    if (xVal === undefined || xVal === null) return;
    postLabel(xVal).then(resp => {
      if (resp.ok) window.location.reload();
      else alert(resp.error || 'Failed to save label');
    });
  });

  window.addEventListener('keydown', e => {
    if (e.key?.toLowerCase() === 's') {
      fetch(ANNOTATE_SKIP_URL, { method: 'POST' }).then(() => window.location.reload());
    }
  });
})();
