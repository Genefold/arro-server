// Minimal vanilla-JS frontend for the ArroSpace server.
// No build step. Hits /api/* directly.

const $ = (sel) => document.querySelector(sel);

const state = {
  datasets: [],
  selected: null,        // dataset summary
  windowSize: 200,       // rows per scroll page
  nextOffset: 0,
  loading: false,
  exhausted: false,
  sliceMode: false,      // true when explicit slice spec is in use
};

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return res.json();
}

async function refreshHealth() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    el.textContent = `zarr=${h.zarr_available} arrowspace=${h.arrowspace_backend} roots=${h.data_roots.join(",") || "—"}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = `health: ${e.message}`;
    el.className = "health err";
  }
}

async function refreshDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets;
  renderDatasetList();
}

function renderDatasetList() {
  const ul = $("#dataset-list");
  ul.innerHTML = "";
  const f = $("#filter").value.toLowerCase();
  for (const d of state.datasets) {
    if (f && !d.id.toLowerCase().includes(f)) continue;
    const li = document.createElement("li");
    li.dataset.id = d.id;
    if (state.selected && state.selected.id === d.id) li.classList.add("active");
    const idLine = document.createElement("div");
    idLine.textContent = d.id;
    const meta = document.createElement("div");
    meta.className = "ds-shape";
    meta.textContent = `${d.kind} · [${d.shape.join(",")}] · ${d.dtype || "—"}`;
    li.appendChild(idLine);
    li.appendChild(meta);
    li.addEventListener("click", () => selectDataset(d));
    ul.appendChild(li);
  }
}

async function selectDataset(d) {
  state.selected = d;
  state.nextOffset = 0;
  state.exhausted = false;
  state.sliceMode = false;
  $("#dataset-title").textContent = d.id;
  $("#slice-input").value = "";
  $("#grid").innerHTML = "";
  $("#data-status").textContent = "loading…";
  renderDatasetList();
  try {
    await Promise.all([loadMetadata(d), loadManifold(d), loadStats(d)]);
  } catch (e) {
    console.warn(e);
  }
  if (d.kind === "array") await loadNextPage();
  else $("#data-status").textContent = `(${d.kind} — no data view)`;
}

async function loadMetadata(d) {
  try {
    const m = await api(`/api/datasets/${encodeURI(d.id)}/metadata`);
    $("#metadata-out").textContent = JSON.stringify(m, null, 2);
  } catch (e) {
    $("#metadata-out").textContent = `error: ${e.message}`;
  }
}

async function loadManifold(d) {
  try {
    const m = await api(`/api/datasets/${encodeURI(d.id)}/manifold`);
    $("#manifold-out").textContent = JSON.stringify(m, null, 2);
  } catch (e) {
    $("#manifold-out").textContent = `unavailable: ${e.message}`;
  }
}

async function loadStats(d) {
  try {
    const s = await api(`/api/datasets/${encodeURI(d.id)}/stats`);
    $("#stats-out").textContent = JSON.stringify(s, null, 2);
  } catch (e) {
    $("#stats-out").textContent = `unavailable: ${e.message}`;
  }
}

async function loadNextPage() {
  if (!state.selected || state.loading || state.exhausted || state.sliceMode) return;
  state.loading = true;
  $("#data-status").textContent = `loading rows ${state.nextOffset}…`;
  try {
    const url = `/api/datasets/${encodeURI(state.selected.id)}/data?offset=${state.nextOffset}&limit=${state.windowSize}`;
    const page = await api(url);
    appendRows(page.data);
    if (page.next_offset == null) {
      state.exhausted = true;
      $("#data-status").textContent = `loaded ${page.total} rows (end)`;
    } else {
      state.nextOffset = page.next_offset;
      $("#data-status").textContent = `loaded ${state.nextOffset} of ${page.total}`;
    }
  } catch (e) {
    $("#data-status").textContent = `error: ${e.message}`;
  } finally {
    state.loading = false;
  }
}

async function applySlice() {
  if (!state.selected) return;
  const spec = $("#slice-input").value.trim();
  if (!spec) {
    state.sliceMode = false;
    state.nextOffset = 0;
    state.exhausted = false;
    $("#grid").innerHTML = "";
    return loadNextPage();
  }
  state.sliceMode = true;
  $("#data-status").textContent = `slice ${spec} loading…`;
  try {
    const url = `/api/datasets/${encodeURI(state.selected.id)}/slice?slice=${encodeURIComponent(spec)}`;
    const r = await api(url);
    $("#grid").innerHTML = "";
    appendRows(r.data);
    $("#data-status").textContent = `slice ${spec} → shape [${r.out_shape.join(",")}]`;
  } catch (e) {
    $("#data-status").textContent = `error: ${e.message}`;
  }
}

function appendRows(payload) {
  const grid = $("#grid");
  let table = grid.querySelector("table");
  let baseRow = 0;
  if (!table) {
    table = document.createElement("table");
    const thead = document.createElement("thead");
    const tr = document.createElement("tr");
    const th0 = document.createElement("th");
    th0.className = "row-idx";
    th0.textContent = "#";
    tr.appendChild(th0);
    const ncols = payload.rows
      ? (payload.rows[0]?.length ?? 1)
      : (payload.shape[1] ?? 1);
    for (let c = 0; c < ncols; c++) {
      const th = document.createElement("th");
      th.textContent = c;
      tr.appendChild(th);
    }
    thead.appendChild(tr);
    table.appendChild(thead);
    table.appendChild(document.createElement("tbody"));
    grid.appendChild(table);
  } else {
    baseRow = table.tBodies[0].rows.length;
  }
  const tbody = table.tBodies[0];
  const rows = payload.rows ?? [payload.values ?? []];
  rows.forEach((row, i) => {
    const tr = document.createElement("tr");
    const idx = document.createElement("td");
    idx.className = "row-idx";
    idx.textContent = baseRow + i;
    tr.appendChild(idx);
    const cells = Array.isArray(row) ? row : [row];
    for (const v of cells) {
      const td = document.createElement("td");
      td.textContent = formatCell(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  });
}

function formatCell(v) {
  if (v == null) return "";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return String(v);
    return v.toPrecision(6);
  }
  if (typeof v === "object" && "re" in v && "im" in v) {
    return `${v.re.toPrecision(4)}${v.im >= 0 ? "+" : ""}${v.im.toPrecision(4)}i`;
  }
  return String(v);
}

function attachInfiniteScroll() {
  $("#grid").addEventListener("scroll", (e) => {
    const el = e.currentTarget;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 50) {
      loadNextPage();
    }
  });
}

function wireControls() {
  $("#filter").addEventListener("input", renderDatasetList);
  $("#apply-slice").addEventListener("click", applySlice);
  $("#reset-slice").addEventListener("click", () => {
    $("#slice-input").value = "";
    applySlice();
  });
  $("#slice-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") applySlice();
  });
}

(async function main() {
  wireControls();
  attachInfiniteScroll();
  await refreshHealth();
  try {
    await refreshDatasets();
  } catch (e) {
    $("#dataset-list").innerHTML = `<li>error: ${e.message}</li>`;
  }
})();
