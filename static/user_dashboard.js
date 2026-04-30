const state = {
  categoryName: "",
  goodsIds: [],
  goodsNames: [],
  goodsOptions: [],
  keyword: "",
  aspectFocus: "",
  sentiment: "",
  page: 1,
  pageSize: 10,
  suggestedKeywords: [],
};

const goodsChartState = {
  sentiment: [],
  trend: [],
  aspect: [],
};

const charts = {};
let modalChart = null;
const latestOptions = {};

const chartMeta = {
  compare: { elementId: "goodsCompare", title: "多商品对比" },
  pie: { elementId: "sentimentPie", title: "情感分布" },
  trend: { elementId: "trendLine", title: "情感趋势" },
  aspect: { elementId: "aspectStack", title: "方面情感分布" },
  radar: { elementId: "aspectRadar", title: "多商品方面雷达" },
  keyword: { elementId: "keywordBar", title: "关键词统计" },
};

const chartPalette = {
  positive: "#34d399",
  neutral: "#fbbf24",
  negative: "#f87171",
  accent1: "#6aa4ff",
  accent2: "#69e3ff",
  accent3: "#a78bfa",
  grid: "rgba(173, 193, 224, 0.18)",
  axis: "#bcd0f2",
  text: "#dce9ff",
};

function truncateName(name) {
  const s = String(name || "");
  return s.length > 3 ? s.slice(0, 3) : s;
}

async function j(url, options) {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
  return r.json();
}

function qp(extra = {}) {
  const q = new URLSearchParams();
  if (state.categoryName) q.set("category_name", state.categoryName);
  if (state.goodsIds.length) q.set("goods_ids", state.goodsIds.join(","));
  if (!state.goodsIds.length && state.goodsNames.length) q.set("goods_names", state.goodsNames.join(","));
  if (state.keyword) q.set("keyword", state.keyword);
  if (state.aspectFocus) q.set("aspect_focus", state.aspectFocus);
  if (state.sentiment) q.set("sentiment", state.sentiment);
  Object.entries(extra).forEach(([k, v]) => q.set(k, String(v)));
  return q.toString();
}

function getSelectedGoodsQuery() {
  const q = new URLSearchParams();
  if (state.categoryName) q.set("category_name", state.categoryName);
  if (state.goodsIds.length) q.set("goods_ids", state.goodsIds.join(","));
  else if (state.goodsNames.length) q.set("goods_names", state.goodsNames.join(","));
  return q.toString();
}

function sentimentClass(s) {
  if (s === "positive") return "sent-positive";
  if (s === "negative") return "sent-negative";
  return "sent-neutral";
}

function fmtNum(v) {
  const n = Number(v || 0);
  if (Number.isNaN(n)) return "0";
  return n.toLocaleString("zh-CN");
}

function animateValue(el, to, decimals = 0, suffix = "") {
  if (!el) return;
  const target = Number(to || 0);
  if (!Number.isFinite(target)) {
    el.textContent = "-";
    return;
  }
  const from = Number(el.dataset.value || 0);
  const duration = 650;
  const start = performance.now();

  function step(now) {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = from + (target - from) * eased;
    el.textContent = `${val.toFixed(decimals)}${suffix}`;
    if (p < 1) requestAnimationFrame(step);
    else {
      el.textContent = `${target.toFixed(decimals)}${suffix}`;
      el.dataset.value = String(target);
    }
  }

  requestAnimationFrame(step);
}

function baseChartOption() {
  return {
    animationDuration: 700,
    animationEasing: "cubicOut",
    textStyle: { color: chartPalette.text, fontFamily: "Inter, PingFang SC, Microsoft YaHei, sans-serif" },
    grid: { left: 42, right: 24, top: 66, bottom: 42, containLabel: true },
    tooltip: {
      backgroundColor: "rgba(9, 14, 27, 0.92)",
      borderColor: "rgba(110, 231, 255, 0.35)",
      borderWidth: 1,
      textStyle: { color: "#ecf5ff" },
    },
  };
}

function applyChart(name, option) {
  charts[name].setOption(option, true);
  latestOptions[name] = option;
  if (modalChart && document.getElementById("chartModal").classList.contains("show")) {
    const title = document.getElementById("chartModalTitle").textContent || "";
    if (title === chartMeta[name].title) {
      modalChart.setOption(option, true);
    }
  }
}

function initCharts() {
  charts.compare = echarts.init(document.getElementById(chartMeta.compare.elementId));
  charts.pie = echarts.init(document.getElementById(chartMeta.pie.elementId));
  charts.trend = echarts.init(document.getElementById(chartMeta.trend.elementId));
  charts.aspect = echarts.init(document.getElementById(chartMeta.aspect.elementId));
  charts.radar = echarts.init(document.getElementById(chartMeta.radar.elementId));
  charts.keyword = echarts.init(document.getElementById(chartMeta.keyword.elementId));

  Object.entries(charts).forEach(([name, chart]) => {
    chart.getDom().addEventListener("click", () => openChartModal(name));
  });

  window.addEventListener("resize", () => {
    Object.values(charts).forEach((c) => c.resize());
    if (modalChart) modalChart.resize();
  });
}

function openChartModal(chartName) {
  const modal = document.getElementById("chartModal");
  const canvas = document.getElementById("chartModalCanvas");
  document.getElementById("chartModalTitle").textContent = chartMeta[chartName].title;

  modal.classList.add("show");
  modal.setAttribute("aria-hidden", "false");
  canvas.innerHTML = "";

  if (modalChart) {
    modalChart.dispose();
    modalChart = null;
  }

  if ((chartName === "pie" || chartName === "trend" || chartName === "aspect") && state.goodsIds.length > 1) {
    renderMultiChartModal(chartName, canvas);
    return;
  }

  if (!latestOptions[chartName]) return;
  modalChart = echarts.init(canvas);
  modalChart.setOption(latestOptions[chartName], true);
  modalChart.resize();
}

function closeChartModal() {
  const modal = document.getElementById("chartModal");
  modal.classList.remove("show");
  modal.setAttribute("aria-hidden", "true");
  if (modalChart) {
    modalChart.dispose();
    modalChart = null;
  }
}

function bindModal() {
  document.querySelectorAll(".expand-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const name = btn.dataset.chart;
      openChartModal(name);
    });
  });

  document.getElementById("chartModalClose").addEventListener("click", closeChartModal);
  document.querySelector("[data-close='modal']").addEventListener("click", closeChartModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeChartModal();
  });
}

async function loadFilters() {
  const categories = await j("/api/categories");
  const categorySelect = document.getElementById("categorySelect");
  categorySelect.innerHTML = `<option value="">全部类别</option>`;
  categories.forEach((c) => {
    categorySelect.innerHTML += `<option value="${c}">${c}</option>`;
  });
  await loadGoodsMulti();
}

async function loadGoodsMulti() {
  const rows = await j(`/api/goods?${new URLSearchParams({ category_name: state.categoryName }).toString()}`);
  state.goodsOptions = rows;
  renderGoodsPicker(rows);
  renderGoodsChips();
}

function renderGoodsPicker(rows) {
  const kw = (document.getElementById("goodsSearchInput").value || "").trim();
  const list = document.getElementById("goodsPickerList");
  const filtered = rows.filter((r) => {
    const name = r.goods_name || "";
    return !kw || name.includes(kw) || String(r.goods_id || "").includes(kw);
  });
  list.innerHTML = "";
  filtered.forEach((r) => {
    const item = document.createElement("button");
    item.type = "button";
    const name = r.goods_name || r.goods_id;
    const gid = String(r.goods_id || "");
    item.className = `picker-item ${state.goodsIds.includes(gid) ? "active" : ""}`;
    item.innerHTML = `<span>${name}</span><small>${r.comment_count}条 · ${r.avg_rating}分</small>`;
    item.onclick = () => {
      if (state.goodsIds.includes(gid)) {
        const idx = state.goodsIds.indexOf(gid);
        state.goodsIds.splice(idx, 1);
        state.goodsNames.splice(idx, 1);
      } else {
        state.goodsIds.push(gid);
        state.goodsNames.push(name);
      }
      renderGoodsPicker(state.goodsOptions);
      renderGoodsChips();
      renderAll().catch((e) => console.error(e));
    };
    list.appendChild(item);
  });
}

function renderGoodsChips() {
  const wrap = document.getElementById("goodsChips");
  wrap.innerHTML = "";
  if (!state.goodsNames.length) {
    wrap.innerHTML = `<span class="chip muted">未选择商品</span>`;
    return;
  }
  state.goodsNames.forEach((name, idx) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `${name} <button type="button" data-index="${idx}">×</button>`;
    wrap.appendChild(chip);
  });
  wrap.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      const idx = Number(btn.dataset.index);
      state.goodsIds.splice(idx, 1);
      state.goodsNames.splice(idx, 1);
      renderGoodsPicker(state.goodsOptions);
      renderGoodsChips();
      renderAll().catch((e) => console.error(e));
    };
  });
}

function renderHeroMetrics(data) {
  const total = Number(data.comment_count || 0);
  const goods = Number(data.goods_count || 0);
  const positive = Number(data.sentiment_counts?.positive || 0);
  const satisfaction = total === 0 ? 0 : (positive / total) * 100;
  const pulse = data.avg_rating ? Number(data.avg_rating) * 20 : 0;

  document.getElementById("heroCoverage").textContent = `${fmtNum(total)} 条`;
  animateValue(document.getElementById("heroSatisfaction"), satisfaction, 1, "%");
  animateValue(document.getElementById("heroPulse"), pulse, 1);
  document.getElementById("heroGoods").textContent = fmtNum(goods);
}

async function renderKpi() {
  const data = await j(`/api/overview?${qp()}`);
  animateValue(document.getElementById("kpiComments"), Number(data.comment_count || 0), 0);
  animateValue(document.getElementById("kpiGoods"), Number(data.goods_count || 0), 0);
  animateValue(document.getElementById("kpiRating"), Number(data.avg_rating || 0), 2);
  document.getElementById("kpiKeyword").textContent = state.keyword || "无";
  renderHeroMetrics(data);
}

async function renderGoodsCompare() {
  if (!state.goodsIds.length) {
    charts.compare.clear();
    latestOptions.compare = null;
    return;
  }
  const data = await j(`/api/user/goods-compare?${qp()}`);
  const x = data.map((d) => d.goods_name || d.goods_id);
  applyChart("compare", {
    ...baseChartOption(),
    legend: { top: 24, textStyle: { color: chartPalette.text } },
    xAxis: {
      type: "category",
      data: x,
      axisLine: { lineStyle: { color: chartPalette.grid } },
      axisLabel: { color: chartPalette.axis, rotate: 20, formatter: (v) => truncateName(v) },
    },
    yAxis: [
      { type: "value", name: "数量/评分", splitLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
      { type: "value", name: "好评率", min: 0, max: 1, splitLine: { show: false }, axisLabel: { color: chartPalette.axis, formatter: (v) => `${Math.round(v * 100)}%` } },
    ],
    series: [
      { name: "评论量", type: "bar", barMaxWidth: 24, itemStyle: { color: chartPalette.accent1, borderRadius: [8, 8, 0, 0] }, data: data.map((d) => d.comment_count) },
      { name: "平均评分", type: "bar", barMaxWidth: 24, itemStyle: { color: chartPalette.accent2, borderRadius: [8, 8, 0, 0] }, data: data.map((d) => d.avg_rating) },
      { name: "好评率", type: "line", yAxisIndex: 1, data: data.map((d) => d.positive_ratio), smooth: true, symbolSize: 7, lineStyle: { width: 3, color: chartPalette.accent3 }, itemStyle: { color: chartPalette.accent3 } },
    ],
  });
}

async function renderSentimentPie() {
  if (!state.goodsIds.length) {
    charts.pie.clear();
    latestOptions.pie = null;
    return;
  }
  const data = await Promise.all(state.goodsIds.map((gid) => j(`/api/sentiment/distribution?${getGoodsOnlyQuery(gid)}`)));
  if (state.goodsIds.length === 1) {
    applyChart("pie", buildSinglePieOption(state.goodsNames[0], data[0]));
  } else {
    applyChart("pie", buildMultiPieOption(data));
  }
}

async function renderTrend() {
  if (!state.goodsIds.length) {
    charts.trend.clear();
    latestOptions.trend = null;
    return;
  }
  const data = await Promise.all(state.goodsIds.map((gid) => j(`/api/sentiment/trend?${getGoodsOnlyQuery(gid)}`)));
  if (state.goodsIds.length === 1) {
    applyChart("trend", buildSingleTrendOption(state.goodsNames[0], data[0]));
  } else {
    applyChart("trend", buildMultiTrendOption(data));
  }
}

async function renderAspectStack() {
  if (!state.goodsIds.length) {
    charts.aspect.clear();
    latestOptions.aspect = null;
    return;
  }
  const data = await Promise.all(state.goodsIds.map((gid) => j(`/api/aspects/summary?${getGoodsOnlyQuery(gid)}`)));
  if (state.goodsIds.length === 1) {
    applyChart("aspect", buildSingleAspectOption(state.goodsNames[0], data[0]));
  } else {
    applyChart("aspect", buildMultiAspectOption(data));
  }
}

async function renderAspectRadar() {
  if (!state.categoryName || !state.goodsIds.length) {
    charts.radar.clear();
    latestOptions.radar = null;
    state.suggestedKeywords = [];
    renderSuggestedKeywords();
    renderAspectWeights([]);
    return;
  }
  const data = await j(`/api/aspects/compare?${qp()}`);
  if (!data.goods || !data.goods.length) {
    charts.radar.clear();
    latestOptions.radar = null;
    state.suggestedKeywords = [];
    renderSuggestedKeywords();
    renderAspectWeights([]);
    return;
  }
  state.suggestedKeywords = data.suggested_keywords || [];
  renderSuggestedKeywords();
  renderAspectWeights(data.aspect_weights || []);
  const indicators = data.aspects.map((a) => ({ name: a, max: 1, min: -1 }));
  const colors = [chartPalette.accent1, chartPalette.accent2, chartPalette.accent3, "#f472b6", "#f59e0b", "#2dd4bf"];
  const seriesData = data.matrix.map((r, idx) => ({
    name: (data.goods_names && data.goods_names[idx]) ? data.goods_names[idx] : r.goods_id,
    value: data.aspects.map((a) => r[a] || 0),
    lineStyle: { color: colors[idx % colors.length], width: 2 },
    itemStyle: { color: colors[idx % colors.length] },
    areaStyle: { color: `${colors[idx % colors.length]}26` },
  }));
  applyChart("radar", {
    ...baseChartOption(),
    legend: { top: 8, textStyle: { color: chartPalette.text }, formatter: (n) => truncateName(n) },
    radar: {
      indicator: indicators,
      axisName: { color: chartPalette.axis },
      splitLine: { lineStyle: { color: "rgba(173, 193, 224, 0.24)" } },
      splitArea: { areaStyle: { color: ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.01)"] } },
    },
    series: [{ type: "radar", data: seriesData }],
  });
}

function renderSuggestedKeywords() {
  const wrap = document.getElementById("keywordBar").parentElement;
  let panel = document.getElementById("keywordSuggestionPanel");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "keywordSuggestionPanel";
    panel.className = "keyword-suggestion-panel";
    wrap.insertBefore(panel, document.getElementById("keywordBar"));
  }
  panel.innerHTML = "";
  if (!state.suggestedKeywords.length) {
    panel.innerHTML = `<span class="suggestion-empty">暂无推荐特征词，请切换类别或商品。</span>`;
    return;
  }
  state.suggestedKeywords.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `suggestion-pill ${state.aspectFocus === item.label ? "active" : ""}`;
    btn.innerHTML = `<span>${item.label}</span><em>${item.value || 0}</em>`;
    btn.onclick = () => {
      state.aspectFocus = state.aspectFocus === item.label ? "" : item.label;
      state.page = 1;
      renderAll();
    };
    panel.appendChild(btn);
  });
}

function renderAspectWeights(rows) {
  const panelId = "aspectWeightPanel";
  const wrap = document.getElementById("aspectRadar").parentElement;
  let panel = document.getElementById(panelId);
  if (!panel) {
    panel = document.createElement("div");
    panel.id = panelId;
    panel.className = "aspect-weight-panel";
    wrap.insertBefore(panel, document.getElementById("aspectRadar"));
  }
  panel.innerHTML = rows.slice(0, 6).map((r) => `<button type="button" class="weight-pill">${r.aspect} · ${r.count}</button>`).join("");
  panel.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      state.aspectFocus = btn.textContent.split(" · ")[0];
      state.page = 1;
      renderAll();
    };
  });
}

async function renderKeywords() {
  const rows = await j(`/api/keywords?${qp({ top_n: 20 })}`);
  const data = rows.keywords || [];
  const x = data.map((r) => r.name).reverse();
  const y = data.map((r) => r.value).reverse();
  applyChart("keyword", {
    ...baseChartOption(),
    xAxis: { type: "value", splitLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    yAxis: { type: "category", data: x, axisLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    series: [{ type: "bar", data: y, barMaxWidth: 22, itemStyle: { color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [{ offset: 0, color: chartPalette.accent2 }, { offset: 1, color: chartPalette.accent1 }]), borderRadius: [0, 8, 8, 0] } }],
  });
}

function getGoodsOnlyQuery(gid) {
  const q = new URLSearchParams();
  if (state.categoryName) q.set("category_name", state.categoryName);
  q.set("goods_id", gid);
  if (state.keyword) q.set("keyword", state.keyword);
  if (state.aspectFocus) q.set("aspect_focus", state.aspectFocus);
  if (state.sentiment) q.set("sentiment", state.sentiment);
  return q.toString();
}

function setGoodsDetailVisibility() {
  const blur = state.goodsIds.length > 1;
  document.querySelectorAll(".chart-card").forEach((card) => {
    if (card.dataset.chart === "compare" || card.dataset.chart === "pie" || card.dataset.chart === "trend" || card.dataset.chart === "aspect") {
      card.classList.toggle("blurred-card", blur);
      let cover = card.querySelector(".blur-cover");
      if (!cover) {
        cover = document.createElement("div");
        cover.className = "blur-cover";
        cover.innerHTML = `<span>点击放大查看详情</span>`;
        card.appendChild(cover);
      }
      cover.style.display = blur ? "flex" : "none";
    }
  });
}

function buildSinglePieOption(name, rows) {
  return {
    ...baseChartOption(),
    title: { text: name ? `${name} · 情感分布` : "情感分布", left: 8, top: 8, textStyle: { color: chartPalette.text, fontSize: 12 } },
    legend: { top: 28, textStyle: { color: chartPalette.text } },
    series: [{ type: "pie", radius: ["42%", "72%"], center: ["50%", "54%"], label: { color: chartPalette.text }, itemStyle: { borderColor: "#0b1226", borderWidth: 2 }, color: [chartPalette.negative, chartPalette.neutral, chartPalette.positive], data: rows }],
  };
}

function buildMultiPieOption(list) {
  return {
    ...baseChartOption(),
    title: { text: "多商品情感分布", left: 8, top: 8, textStyle: { color: chartPalette.text, fontSize: 12 } },
    legend: { top: 28, textStyle: { color: chartPalette.text } },
    grid: { left: 24, right: 24, top: 70, bottom: 24, containLabel: true },
    series: list.map((rows, idx) => ({
      name: state.goodsNames[idx] || `商品${idx + 1}`,
      type: "pie",
      radius: ["16%", "32%"],
      center: [String(20 + idx * (60 / Math.max(1, list.length))) + "%", "56%"],
      label: { color: chartPalette.text },
      color: [chartPalette.negative, chartPalette.neutral, chartPalette.positive],
      data: rows,
    })),
  };
}

function buildSingleTrendOption(name, data) {
  return {
    ...baseChartOption(),
    title: { text: name ? `${name} · 情感趋势` : "情感趋势", left: 8, top: 8, textStyle: { color: chartPalette.text, fontSize: 12 } },
    legend: { top: 28, textStyle: { color: chartPalette.text } },
    grid: { left: 42, right: 24, top: 74, bottom: 42, containLabel: true },
    xAxis: { type: "category", data: data.dates || [], axisLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    yAxis: { type: "value", splitLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    series: [
      { name: "negative", type: "line", data: data.series?.negative || [], smooth: true, lineStyle: { color: chartPalette.negative } },
      { name: "neutral", type: "line", data: data.series?.neutral || [], smooth: true, lineStyle: { color: chartPalette.neutral } },
      { name: "positive", type: "line", data: data.series?.positive || [], smooth: true, lineStyle: { color: chartPalette.positive } },
    ],
  };
}

function buildMultiTrendOption(list) {
  return {
    ...baseChartOption(),
    title: { text: "多商品情感趋势", left: 8, textStyle: { color: chartPalette.text, fontSize: 12 } },
    grid: { left: 44, right: 24, top: 60, bottom: 42, containLabel: true },
    legend: { top: 8, textStyle: { color: chartPalette.text } },
    xAxis: { type: "category", data: list[0]?.dates || [], axisLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    yAxis: { type: "value", splitLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    series: list.map((data, idx) => ({
      name: state.goodsNames[idx] || `商品${idx + 1}`,
      type: "line",
      data: data.series?.positive || [],
      smooth: true,
      lineStyle: { width: 3 },
    })),
  };
}

function buildSingleAspectOption(name, rows) {
  const aspects = [...new Set(rows.map((r) => r.aspect))];
  const sentiments = ["negative", "neutral", "positive"];
  const map = {};
  rows.forEach((r) => (map[`${r.aspect}-${r.aspect_sentiment}`] = r.count));
  return {
    ...baseChartOption(),
    title: { text: name ? `${name} · 方面情感分布` : "方面情感分布", left: 8, top: 8, textStyle: { color: chartPalette.text, fontSize: 12 } },
    legend: { top: 28, textStyle: { color: chartPalette.text } },
    grid: { left: 42, right: 24, top: 76, bottom: 42, containLabel: true },
    xAxis: { type: "category", data: aspects, axisLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis, rotate: 20 } },
    yAxis: { type: "value", splitLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    series: sentiments.map((s) => ({ name: s, type: "bar", stack: "all", data: aspects.map((a) => map[`${a}-${s}`] || 0) })),
  };
}

function buildMultiAspectOption(list) {
  return {
    ...baseChartOption(),
    title: { text: "多商品方面情感分布", left: 8, top: 8, textStyle: { color: chartPalette.text, fontSize: 12 } },
    legend: { top: 28, textStyle: { color: chartPalette.text } },
    grid: { left: 42, right: 24, top: 76, bottom: 42, containLabel: true },
    xAxis: { type: "category", data: state.goodsNames, axisLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis, rotate: 20 } },
    yAxis: { type: "value", splitLine: { lineStyle: { color: chartPalette.grid } }, axisLabel: { color: chartPalette.axis } },
    series: [{ name: "方面总量", type: "bar", data: list.map((rows) => rows.length) }],
  };
}

function renderMultiChartModal(chartName, canvas) {
  const goods = state.goodsIds.map((gid, idx) => ({ gid, name: state.goodsNames[idx] || gid }));
  canvas.innerHTML = `<div class="multi-chart-grid">${goods.map((g, idx) => `<div class="multi-chart-item"><div id="modal-${chartName}-${idx}" style="width:100%;height:100%;min-height:300px;"></div></div>`).join("")}</div>`;
  goods.forEach((g, idx) => {
    const el = document.getElementById(`modal-${chartName}-${idx}`);
    const chart = echarts.init(el);
    if (chartName === "pie") {
      fetch(`/api/sentiment/distribution?${getGoodsOnlyQuery(g.gid)}`).then((r) => r.json()).then((rows) => {
        chart.setOption(buildSinglePieOption(g.name, rows), true);
      });
    } else if (chartName === "trend") {
      fetch(`/api/sentiment/trend?${getGoodsOnlyQuery(g.gid)}`).then((r) => r.json()).then((rows) => {
        chart.setOption(buildSingleTrendOption(g.name, rows), true);
      });
    } else if (chartName === "aspect") {
      fetch(`/api/aspects/summary?${getGoodsOnlyQuery(g.gid)}`).then((r) => r.json()).then((rows) => {
        chart.setOption(buildSingleAspectOption(g.name, rows), true);
      });
    }
  });
}



async function renderComments() {
  const data = await j(`/api/comments?${qp({ page: state.page, page_size: state.pageSize })}`);
  const body = document.getElementById("commentTableBody");
  body.innerHTML = "";
  data.rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.goods_name || r.goods_id}</td><td>${r.nickname || "-"}</td><td>${r.rating ?? "-"}</td><td class="${sentimentClass(r.sentiment_label_star)}">${r.sentiment_label_star}</td><td>${r.comment_time || "-"}</td><td>${r.comment_text_clean || ""}</td>`;
    body.appendChild(tr);
  });
  const pages = Math.max(1, Math.ceil(data.total / state.pageSize));
  document.getElementById("pageInfo").textContent = `第 ${state.page} 页 / 共 ${pages} 页`;
  document.getElementById("kpiKeyword").textContent = state.aspectFocus || state.keyword || "无";
}

async function renderAll() {
  setGoodsDetailVisibility();
  await renderKpi();
  await Promise.all([
    renderGoodsCompare(),
    renderSentimentPie(),
    renderTrend(),
    renderAspectStack(),
    renderAspectRadar(),
    renderKeywords(),
    renderComments(),
  ]);
}

function bind() {
  document.getElementById("categorySelect").addEventListener("change", async (e) => {
    state.categoryName = e.target.value;
    state.goodsIds = [];
    state.goodsNames = [];
    await loadGoodsMulti();
    state.page = 1;
    await renderAll();
  });

  document.getElementById("goodsSearchInput").addEventListener("input", () => {
    renderGoodsPicker(state.goodsOptions);
  });

  document.getElementById("applyBtn").addEventListener("click", async () => {
    state.keyword = document.getElementById("keywordInput").value.trim();
    state.sentiment = document.getElementById("sentimentSelect").value;
    state.aspectFocus = "";
    state.page = 1;
    await renderAll();
  });

  document.getElementById("clearBtn").addEventListener("click", async () => {
    state.goodsIds = [];
    state.goodsNames = [];
    state.keyword = "";
    state.aspectFocus = "";
    state.sentiment = "";
    state.categoryName = "";
    state.page = 1;
    document.getElementById("keywordInput").value = "";
    document.getElementById("sentimentSelect").value = "";
    document.getElementById("categorySelect").value = "";
    document.getElementById("goodsSearchInput").value = "";
    await loadGoodsMulti();
    await renderAll();
  });

  document.getElementById("reloadBtn").addEventListener("click", async () => {
    await j("/api/reload", { method: "POST" });
    await loadFilters();
    await renderAll();
  });

  document.getElementById("prevPageBtn").addEventListener("click", async () => {
    state.page = Math.max(1, state.page - 1);
    await renderComments();
  });

  document.getElementById("nextPageBtn").addEventListener("click", async () => {
    state.page += 1;
    await renderComments();
  });
}

initCharts();
bindModal();
bind();
loadFilters().then(renderAll).catch((e) => {
  console.error(e);
  alert("用户大屏加载失败");
});
