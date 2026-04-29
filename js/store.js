(function () {
  'use strict';

  const CONFIG_KEY      = 'cnn-config';
  const ANNOT_KEY       = 'cnn-annotations';
  const ANNOT_STATE_KEY = 'cnn-annotation-state';
  const DB_NAME         = 'cnn-db';
  const DB_VERSION      = 1;
  const TRACE_STORE     = 'traces';

  const DEFAULTS = {
    group_by_col:   'run_id',
    time_index_col: 'elapsed_time',
    channel_cols:   null,
    trim_ratio:     0.1,
    test_size:      0.2,
    random_state:   42,
    batch_size:     200,
    num_epochs:     100,
  };

  // ── Config ─────────────────────────────────────────────────────────────────

  function loadConfig() {
    try {
      return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(CONFIG_KEY) || 'null') || {});
    } catch {
      return Object.assign({}, DEFAULTS);
    }
  }

  function saveConfig(cfg) {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
  }

  // ── Annotations ────────────────────────────────────────────────────────────

  function _allAnnotations() {
    try { return JSON.parse(localStorage.getItem(ANNOT_KEY) || '{}'); } catch { return {}; }
  }

  function loadAnnotations(traceKey) {
    return _allAnnotations()[traceKey] || {};
  }

  function upsertAnnotation(traceKey, key, label) {
    const all = _allAnnotations();
    if (!all[traceKey]) all[traceKey] = {};
    all[traceKey][key] = label;
    localStorage.setItem(ANNOT_KEY, JSON.stringify(all));
  }

  function annotationCount(traceKey) {
    return Object.keys(loadAnnotations(traceKey)).length;
  }

  // ── Annotation state ───────────────────────────────────────────────────────

  function loadAnnotationState() {
    try { return JSON.parse(localStorage.getItem(ANNOT_STATE_KEY) || 'null'); } catch { return null; }
  }

  function saveAnnotationState(state) {
    localStorage.setItem(ANNOT_STATE_KEY, JSON.stringify(state));
  }

  function resetAnnotationState(traceKey, keys) {
    const state = { traceKey, keys, idx: 0 };
    saveAnnotationState(state);
    return state;
  }

  // ── Active trace (IndexedDB — avoids 5 MB localStorage cap) ───────────────

  function _openDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = e => e.target.result.createObjectStore(TRACE_STORE);
      req.onsuccess = e => resolve(e.target.result);
      req.onerror   = e => reject(e.target.error);
    });
  }

  async function saveActiveTrace(name, csvText) {
    const db = await _openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(TRACE_STORE, 'readwrite');
      tx.objectStore(TRACE_STORE).put({ name, csv: csvText }, 'active');
      tx.oncomplete = () => resolve();
      tx.onerror    = e => reject(e.target.error);
    });
  }

  async function loadActiveTrace() {
    const db = await _openDb();
    return new Promise((resolve, reject) => {
      const req = db.transaction(TRACE_STORE, 'readonly').objectStore(TRACE_STORE).get('active');
      req.onsuccess = e => resolve(e.target.result || null);
      req.onerror   = e => reject(e.target.error);
    });
  }

  // ── Export ─────────────────────────────────────────────────────────────────

  window.CnnStore = {
    DEFAULTS,
    loadConfig,    saveConfig,
    loadAnnotations, upsertAnnotation, annotationCount,
    loadAnnotationState, saveAnnotationState, resetAnnotationState,
    saveActiveTrace, loadActiveTrace,
  };
})();
