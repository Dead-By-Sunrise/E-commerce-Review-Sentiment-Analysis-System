const devCharts = {};

async function j(url, options) {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function initDevCharts() {
  devCharts.bar = echarts.init(document.getElementById("modelBar"));
  devCharts.radar = echarts.init(document.getElementById("modelRadar"));
  devCharts.label = echarts.init(document.getElementById("labelDist"));
  window.addEventListener("resize", () => Object.values(devCharts).forEach((c) => c.resize()));
}

function themeText() {
  return "#d8e4fb";
}

function themeMuted() {
  return "#8f9bb5";
}

function renderMetricsTable(rows) {
  const body = document.getElementById("metricsBody");
  body.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.model}</td><td>${r.accuracy}</td><td>${r.precision_macro}</td><td>${r.recall_macro}</td><td>${r.f1_macro}</td>`;
    body.appendChild(tr);
  });
}

function renderBar(rows) {
  const x = rows.map((r) => r.model);
  devCharts.bar.setOption({
    color: ["#6aa4ff", "#69e3ff", "#8b5cf6", "#34d399"],
    tooltip: { trigger: "axis", backgroundColor: "rgba(8,12,22,0.92)", borderColor: "rgba(255,255,255,0.12)", textStyle: { color: themeText() } },
    legend: { textStyle: { color: themeText() } },
    grid: { left: 12, right: 12, bottom: 28, top: 36, containLabel: true },
    xAxis: { type: "category", data: x, axisLabel: { color: themeMuted(), interval: 0, rotate: x.length > 4 ? 18 : 0 }, axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } } },
    yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: themeMuted() }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } } },
    series: [
      { name: "accuracy", type: "bar", barMaxWidth: 18, data: rows.map((r) => r.accuracy) },
      { name: "precision", type: "bar", barMaxWidth: 18, data: rows.map((r) => r.precision_macro) },
      { name: "recall", type: "bar", barMaxWidth: 18, data: rows.map((r) => r.recall_macro) },
      { name: "f1", type: "line", smooth: true, symbolSize: 8, data: rows.map((r) => r.f1_macro) },
    ],
  });
}

function renderRadar(rows) {
  if (!rows.length) return;
  const indicators = [
    { name: "accuracy", max: 1 },
    { name: "precision", max: 1 },
    { name: "recall", max: 1 },
    { name: "f1", max: 1 },
  ];
  const seriesData = rows.map((r) => ({
    name: r.model,
    value: [r.accuracy, r.precision_macro, r.recall_macro, r.f1_macro],
  }));
  devCharts.radar.setOption({
    color: ["#6aa4ff", "#69e3ff", "#8b5cf6", "#34d399"],
    tooltip: { backgroundColor: "rgba(8,12,22,0.92)", borderColor: "rgba(255,255,255,0.12)", textStyle: { color: themeText() } },
    legend: { textStyle: { color: themeText() } },
    radar: {
      indicator: indicators,
      axisName: { color: themeMuted() },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } },
      splitArea: { areaStyle: { color: ["rgba(255,255,255,0.015)", "rgba(255,255,255,0.03)"] } },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } },
    },
    series: [{ type: "radar", symbolSize: 6, areaStyle: { opacity: 0.18 }, data: seriesData }],
  });
}

function renderLabelDist(rows) {
  devCharts.label.setOption({
    color: ["#6aa4ff", "#69e3ff", "#fbbf24"],
    tooltip: { backgroundColor: "rgba(8,12,22,0.92)", borderColor: "rgba(255,255,255,0.12)", textStyle: { color: themeText() } },
    grid: { left: 12, right: 12, bottom: 18, top: 24, containLabel: true },
    xAxis: { type: "category", data: rows.map((r) => r.label), axisLabel: { color: themeMuted() }, axisLine: { lineStyle: { color: "rgba(255,255,255,0.12)" } } },
    yAxis: { type: "value", axisLabel: { color: themeMuted() }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.08)" } } },
    series: [{ type: "bar", barMaxWidth: 24, data: rows.map((r) => r.count) }],
  });
}

async function loadDev() {
  const [health, metrics, manifest, reports, labelDist] = await Promise.all([
    j("/api/health"),
    j("/api/models/comparison"),
    j("/api/dev/manifest"),
    j("/api/dev/model-reports"),
    j("/api/dev/label-distribution"),
  ]);

  document.getElementById("kpiLoad").textContent = health.last_reload || "-";
  document.getElementById("kpiModels").textContent = health.rows?.models ?? "-";
  document.getElementById("kpiComments").textContent = health.rows?.comments ?? "-";
  document.getElementById("kpiBest").textContent = manifest.best_model_by_f1_macro || "-";
  document.getElementById("reportText").textContent = reports.content || "无报告";

  renderMetricsTable(metrics);
  renderBar(metrics);
  renderRadar(metrics);
  renderLabelDist(labelDist);
}

document.getElementById("reloadInlineBtn").addEventListener("click", async () => {
  await j("/api/reload", { method: "POST" });
  await loadDev();
});

initDevCharts();
loadDev().catch((e) => {
  console.error(e);
  alert("开发者看板加载失败");
});
