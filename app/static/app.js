/**
 * IntentSight Dashboard JS — v3
 *
 * Enhancements over v2:
 * - Loading skeletons / overlay
 * - Confidence gauge on predictions
 * - Activity log from audit trail
 * - Better error handling with user-facing messages
 * - ARIA live region updates
 */

const state = {
  view: "aggregated",
  metric: "app_usage_time_min",
  category: "gender",
  heatmapX: "gender",
  heatmapY: "app_usage_time_min",
  options: null,
  predictionCount: 0,
};

const colors = ["#3de0d7", "#ff8d72", "#8bd86f", "#f6c860", "#a7b7ff", "#f48ad2"];
const $ = (id) => document.getElementById(id);
const API = "/v1";

// ---------------------------------------------------------------------------
// HTTP helper with error handling
// ---------------------------------------------------------------------------

async function getJson(url, options = {}) {
  try {
    const response = await fetch(url, { ...options, headers: { "Accept": "application/json", ...options.headers } });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`Server error ${response.status}${text ? ": " + text : ""}`);
    }
    return response.json();
  } catch (err) {
    if (err.name === "TypeError") {
      throw new Error("Network error — check the server is running.");
    }
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Confidence gauge
// ---------------------------------------------------------------------------

function showConfidence(result) {
  const container = $("predictionResult");
  const conf = result.confidence;

  let gaugeHTML = "";
  if (conf != null) {
    const pct = (conf * 100).toFixed(1);
    const hue = Math.round((1 - conf) * 120); // green (120) → red (0)
    gaugeHTML = `
      <div class="confidence-gauge">
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill" style="width:${pct}%; background-color:hsl(${hue}, 70%, 45%)"></div>
        </div>
        <div class="confidence-label">
          <span>${result.calibrated_probabilities
            ? `Calibrated: ${JSON.stringify(result.calibrated_probabilities)}`
            : `Confidence: ${pct}%`}</span>
          <span>${conf <= 0.55 ? "Low confidence" : conf <= 0.7 ? "Moderate" : "High confidence"}</span>
        </div>
      </div>`;
  }

  let oodHTML = "";
  if (result.ood_flags && Object.keys(result.ood_flags).length > 0) {
    const flagged = Object.keys(result.ood_flags).join(", ");
    oodHTML = `<div class="ood-warning">⚠ Out-of-distribution input detected: ${flagged}</div>`;
  }

  container.innerHTML = `
    <strong>${result.prediction}</strong> &nbsp;
    <span style="color:var(--muted)">confidence ${conf != null ? (conf * 100).toFixed(1) + "%" : "N/A"}</span>
    <br><small>${result.note || ""}</small>
    ${gaugeHTML}${oodHTML}
  `;
}

function showError(element, message) {
  element.innerHTML = `<span style="color:var(--coral)">✕ ${message}</span>`;
}

// ---------------------------------------------------------------------------
// Loading state
// ---------------------------------------------------------------------------

function setLoading(loading) {
  const overlay = $("loadingOverlay");
  overlay.setAttribute("aria-hidden", loading ? "false" : "true");

  // Also disable form during loading
  const btn = $("scenarioForm")?.querySelector("button[type='submit']");
  if (btn) btn.disabled = loading;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  setLoading(true);

  try {
    const [health, metrics, options] = await Promise.all([
      getJson(`${API}/health`),
      getJson(`${API}/metrics`),
      getJson(`${API}/options`),
    ]);

    state.options = options;

    // Health badge
    $("healthBadge").textContent = health.status === "ok"
      ? "Artifacts ready"
      : "Artifacts degraded";
    $("healthBadge").style.borderColor = health.status === "ok"
      ? "rgba(139,216,111,0.6)"
      : "rgba(246,200,96,0.6)";

    // Metrics tiles
    $("bestAccuracy").textContent = (metrics.best_model.test_accuracy * 100).toFixed(1) + "%";
    $("bestModel").textContent = metrics.best_model.name;
    $("baselineAccuracy").textContent = (metrics.majority_baseline.accuracy * 100).toFixed(1) + "%";
    $("baselineLabel").textContent = `${metrics.majority_baseline.label} baseline (${metrics.majority_baseline.count}/${metrics.majority_baseline.total})`;
    $("weightedF1").textContent = metrics.best_model.weighted_f1.toFixed(4);
    $("testRows").textContent = metrics.majority_baseline.total.toLocaleString();

    // Nested CV info
    if (metrics.nested_cv && metrics.nested_cv.cv_mean) {
      $("nestedCV").textContent =
        `${(metrics.nested_cv.cv_mean * 100).toFixed(1)}% ± ${(metrics.nested_cv.cv_std * 100).toFixed(1)}%`;
    } else {
      $("nestedCV").textContent = "N/A";
    }

    $("defensibilityNote").textContent = metrics.defensibility_note;

    renderComparison(metrics);

    // Populate selects
    const heatmapItems = [...options.categories, ...options.metrics];
    fillSelect($("metricSelect"), options.metrics, state.metric);
    fillSelect($("categorySelect"), options.categories, state.category);
    fillSelect($("heatmapX"), heatmapItems, state.heatmapX);
    fillSelect($("heatmapY"), heatmapItems, state.heatmapY);

    bindEvents();
    await refreshDashboard();
    await loadActivityLog();
  } catch (err) {
    console.error(err);
    showError($("healthBadge"), err.message);
    $("defensibilityNote").textContent = "Failed to load dashboard data: " + err.message;
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Select helpers
// ---------------------------------------------------------------------------

function fillSelect(element, items, selected) {
  element.innerHTML = items
    .map((item) => `<option value="${item.value}" ${item.value === selected ? " selected" : ""}>${item.label}</option>`)
    .join("");
}

// ---------------------------------------------------------------------------
// Event binding
// ---------------------------------------------------------------------------

function bindEvents() {
  document.querySelectorAll(".view-toggle button").forEach((button) => {
    button.addEventListener("click", async () => {
      document.querySelectorAll(".view-toggle button").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      button.classList.add("active");
      button.setAttribute("aria-selected", "true");
      state.view = button.dataset.view;
      await refreshCohorts();
    });
  });

  $("metricSelect").addEventListener("change", async (e) => {
    state.metric = e.target.value;
    state.heatmapY = state.metric;
    $("heatmapY").value = state.heatmapY;
    await refreshDashboard();
  });

  $("categorySelect").addEventListener("change", async (e) => {
    state.category = e.target.value;
    await refreshCohorts();
  });

  $("heatmapX").addEventListener("change", async (e) => {
    state.heatmapX = e.target.value;
    await refreshHeatmap();
  });

  $("heatmapY").addEventListener("change", async (e) => {
    state.heatmapY = e.target.value;
    await refreshHeatmap();
  });

  $("scenarioForm").addEventListener("submit", submitScenario);
}

// ---------------------------------------------------------------------------
// Dashboard refresh
// ---------------------------------------------------------------------------

async function refreshDashboard() {
  await Promise.all([refreshCohorts(), refreshHeatmap()]);
}

async function refreshCohorts() {
  const params = new URLSearchParams({
    view: state.view,
    metric: state.metric,
    category: state.category,
  });

  try {
    const data = await getJson(`${API}/cohorts?${params}`);
    $("chartTitle").textContent = `${data.metric_label} by ${state.view} view`;
    renderLineChart(data.points);
  } catch (err) {
    console.error(err);
    $("lineChart").innerHTML = `<p style="color:var(--coral)">Error loading cohorts: ${err.message}</p>`;
  }
}

async function refreshHeatmap() {
  const params = new URLSearchParams({ x: state.heatmapX, y: state.heatmapY });

  try {
    const data = await getJson(`${API}/heatmap?${params}`);
    $("heatmapTitle").textContent = `${data.y_label} × ${data.x_label}`;
    renderHeatmap(data);
  } catch (err) {
    console.error(err);
    $("heatmap").innerHTML = `<p style="color:var(--coral)">Error loading heatmap: ${err.message}</p>`;
  }
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

function renderLineChart(points) {
  const container = $("lineChart");
  if (!points || points.length === 0) {
    container.innerHTML = `<p style="color:var(--muted)">No data available.</p>`;
    return;
  }

  const width = Math.max(container.clientWidth, 400);
  const height = 292;
  const padding = { top: 20, right: 24, bottom: 46, left: 52 };
  const xs = [...new Set(points.map((p) => p.x))];
  const series = [...new Set(points.map((p) => p.series))].slice(0, 6);
  const maxCount = Math.max(1, ...points.map((p) => p.count));

  const xPos = (x) =>
    padding.left + (xs.indexOf(x) * (width - padding.left - padding.right)) / Math.max(1, xs.length - 1);
  const yPos = (value) =>
    height - padding.bottom - (value / maxCount) * (height - padding.top - padding.bottom);

  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((tick) => {
      const y = yPos(maxCount * tick);
      return `<line class="grid-line" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" />`;
    })
    .join("");

  const axes = xs
    .map((x) => `<text class="axis-label" x="${xPos(x)}" y="${height - 16}" text-anchor="middle">${x}</text>`)
    .join("");

  const lines = series
    .map((name, index) => {
      const rows = xs.map((x) => points.find((p) => p.series === name && p.x === x) || { x, count: 0 });
      const d = rows
        .map((point, i) => `${i === 0 ? "M" : "L"} ${xPos(point.x)} ${yPos(point.count || 0)}`)
        .join(" ");
      const circles = rows
        .map((point) => {
          const detail = encodeURIComponent(
            JSON.stringify({
              series: name,
              cohort: point.x,
              count: point.count || 0,
              datingShare: point.dating_share || 0,
              confidence: point.avg_confidence || 0,
            })
          );
          return `<circle class="chart-point" data-detail="${detail}" cx="${xPos(point.x)}" cy="${yPos(point.count || 0)}" r="4.5" fill="${colors[index % colors.length]}" tabindex="0" role="img" aria-label="${name} ${point.x}: ${point.count} users" />`;
        })
        .join("");
      return `<path d="${d}" fill="none" stroke="${colors[index % colors.length]}" stroke-width="3" stroke-linecap="round" />${circles}`;
    })
    .join("");

  const legend = series
    .map((name, index) =>
      `<span style="color:${colors[index % colors.length]}"><b>${name}</b></span>`
    )
    .join(" ");

  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Cohort trend chart">${grid}${axes}${lines}</svg><div class="legend">${legend}</div>`;
  wireTooltips(container);
}

function wireTooltips(container) {
  const tooltip = $("tooltip");
  container.querySelectorAll(".chart-point").forEach((point) => {
    const show = (e) => {
      const detail = JSON.parse(decodeURIComponent(point.dataset.detail));
      tooltip.hidden = false;
      tooltip.innerHTML = `<strong>${detail.series}</strong><br>${detail.cohort}<br>Volume: ${detail.count}<br>Dating share: ${pct(detail.datingShare)}<br>Avg confidence: ${pct(detail.confidence)}`;
      const rect = container.getBoundingClientRect();
      tooltip.style.left = `${e.clientX - rect.left + 14}px`;
      tooltip.style.top = `${e.clientY - rect.top + 14}px`;
    };
    const hide = () => { tooltip.hidden = true; };

    point.addEventListener("mousemove", show);
    point.addEventListener("mouseleave", hide);
    point.addEventListener("focus", (e) => show(e), false);
    point.addEventListener("blur", hide);
  });
}

function renderHeatmap(data) {
  const heatmap = $("heatmap");
  if (!data.rows || data.rows.length === 0) {
    heatmap.innerHTML = `<p style="color:var(--muted)">No data available.</p>`;
    return;
  }

  const columns = data.columns;
  const maxCount = Math.max(1, data.max_count);
  const header = `<div class="heat-row" style="--cols:${columns.length}"><div class="heat-label"></div>${columns
    .map((c) => `<div class="heat-label">${c}</div>`)
    .join("")}</div>`;

  const rows = data.rows
    .map((row) => {
      const cells = columns
        .map((column) => data.cells.find((cell) => cell.row === row && cell.column === column))
        .map((cell) => {
          const intensity = Math.max(0.05, (cell.count / maxCount) * 0.78).toFixed(2);
          return `<div class="heat-cell" style="--intensity:${intensity}" role="cell" aria-label="${row} ${column}: ${cell.count} users, dating share ${pct(cell.dating_share)}">${cell.count}<small>${pct(cell.avg_confidence)}</small></div>`;
        })
        .join("");
      return `<div class="heat-row" style="--cols:${columns.length}"><div class="heat-label">${row}</div>${cells}</div>`;
    })
    .join("");

  heatmap.innerHTML = header + rows;
}

function renderComparison(metrics) {
  const baseline = metrics.majority_baseline.accuracy;
  const rows = metrics.comparison
    .slice()
    .sort((a, b) => (b["Test Accuracy"] || 0) - (a["Test Accuracy"] || 0))
    .map((row) => {
      const accuracy = Number(row["Test Accuracy"] || 0);
      const color = accuracy <= baseline + 0.001 ? "var(--amber)" : "var(--cyan)";
      return `<div class="bar-row" role="listitem">
        <span>${row.Model}</span>
        <div class="bar-track" role="progressbar" aria-valuenow="${(accuracy * 100).toFixed(1)}" aria-valuemin="0" aria-valuemax="100"><div class="bar-fill" style="width:${accuracy * 100}%; background:${color}"></div></div>
        <strong>${pct(accuracy)}</strong>
      </div>`;
    })
    .join("");
  $("comparisonBars").innerHTML = rows;
}

// ---------------------------------------------------------------------------
// Scenario prediction
// ---------------------------------------------------------------------------

async function submitScenario(event) {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const payload = Object.fromEntries(
    [...formData.entries()].map(([key, value]) => [key, isNaN(Number(value)) ? value : Number(value)])
  );

  const resultDiv = $("predictionResult");
  setLoading(true);

  try {
    const result = await getJson(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    showConfidence(result);
    state.predictionCount++;
    addActivityEntry(result);
  } catch (err) {
    showError(resultDiv, err.message);
    console.error(err);
  } finally {
    setLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Activity log (local from predictions)
// ---------------------------------------------------------------------------

function addActivityEntry(result) {
  const container = $("activityLog");
  const now = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = "activity-entry";
  entry.innerHTML = `<span class="pred">${result.prediction}</span>
    <span class="time">${now} — confidence: ${result.confidence != null ? (result.confidence * 100).toFixed(1) + "%" : "N/A"}</span>`;

  container.prepend(entry);
  $("logCount").textContent = `(${state.predictionCount})`;

  // Keep max 50 entries
  while (container.children.length > 50) {
    container.removeChild(container.lastChild);
  }
}

// ---------------------------------------------------------------------------
// Batch prediction (for programmatic use via API)
// ---------------------------------------------------------------------------

async function submitBatch(scenarios) {
  try {
    const result = await getJson(`${API}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenarios }),
    });
    return result;
  } catch (err) {
    console.error("Batch prediction failed:", err);
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
init().catch((err) => {
  console.error("Fatal init error:", err);
  const badge = $("healthBadge");
  if (badge) badge.textContent = "Dashboard error";
  const note = $("defensibilityNote");
  if (note) note.textContent = "Failed to initialize: " + err.message;
  setLoading(false);
});