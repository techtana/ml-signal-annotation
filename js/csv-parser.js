(function () {
  'use strict';

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

  // Mirrors Python normalize_sample_key
  function normalizeKey(value) {
    const text = String(value == null ? '' : value).trim();
    if (!text) return '';
    const num = parseFloat(text);
    if (!isNaN(num) && isFinite(num) && Number.isInteger(num)) return String(num);
    if (!isNaN(num) && isFinite(num)) return text;
    return text;
  }

  // Mirrors Python load_and_group (sort by time, group, iloc[1:] skip)
  function loadAndGroup(csvText, groupByCol, timeIndexCol, channelCols) {
    const lines = csvText.split('\n').filter(l => l.trim());
    if (lines.length < 2) throw new Error('CSV has no data rows');
    const headers = parseCSVRow(lines[0]).map(h => h.trim());

    const groupIdx = headers.indexOf(groupByCol);
    const timeIdx  = headers.indexOf(timeIndexCol);
    if (groupIdx < 0) throw new Error(`Column "${groupByCol}" not found`);

    const resolvedCols = (channelCols && channelCols.length)
      ? channelCols
      : headers.filter(h => h !== groupByCol && h !== timeIndexCol && h !== 'label');

    const chanIndices = resolvedCols.map(c => {
      const i = headers.indexOf(c);
      if (i < 0) throw new Error(`Column "${c}" not found`);
      return i;
    });

    const rows = lines.slice(1).map(l => parseCSVRow(l));
    if (timeIdx >= 0) rows.sort((a, b) => parseFloat(a[timeIdx]) - parseFloat(b[timeIdx]));

    const rawGroups = {};
    for (const row of rows) {
      const k = normalizeKey(row[groupIdx]);
      if (!rawGroups[k]) rawGroups[k] = [];
      rawGroups[k].push(row);
    }

    let maxLen = 0;
    const groups = {};
    for (const [k, grpRows] of Object.entries(rawGroups)) {
      const dataRows = grpRows.slice(1); // mirrors Python iloc[1:]
      const time     = dataRows.map(r => timeIdx >= 0 ? parseFloat(r[timeIdx]) : 0);
      const channels = dataRows.map(r =>
        chanIndices.map(ci => { const v = parseFloat(r[ci]); return isNaN(v) ? 0 : v; })
      );
      groups[k] = { time, channels };
      maxLen = Math.max(maxLen, dataRows.length);
    }

    return { groups, channelCols: resolvedCols, maxLen };
  }

  window.CsvParser = { parseCSVRow, normalizeKey, loadAndGroup };
})();
