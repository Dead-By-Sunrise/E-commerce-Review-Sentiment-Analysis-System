const state = {
  categoryName: "",
  goodsName: "",
  page: 1,
  pageSize: 10,
};

const charts = {};

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
  return res.json();
}

function byId(id) {
  return document.getElementById(id);
}

function sentimentClass(s) {
  if (s === "positive") return "sent-positive";
  if (s === "negative") return "sent-negative";
  return "sent-neutral";
}

function truncateName(name) {
  const s = String(name || "");
  return s.length > 3 ? s.slice(0, 3) : s;
}

function initCharts() {
  charts.model = echarts.init(byId("modelChart"));
  charts.pie = echarts.init(byId("sentimentPie"));
  charts.trend = echarts.init(byId("trendLine"));
  charts.aspect = echarts.init(byId("aspectStack"));
  charts.radar = echarts.init(byId("aspectRadar"));
  charts.keyword = echarts.init(byId("keywordBar"));
  window.addEventListener("resize", () => Object.values(charts).forEach((c) => c.resize()));
}

async function loadFilters() {
  const categories = await fetchJSON("/api/categories");
  const categorySelect = byId("categorySelect");
  categorySelect.innerHTML = `<option value="">全部类别</option>`;
  categories.forEach((c) => {
    categorySelect.innerHTML += `<option value="${c}">${c}</option>`;
  });

  await loadGoodsOptions();
}

async function loadGoodsOptions() {
  const q = new URLSearchParams();
  if (state.categoryName) q.set("category_name", state.categoryName);
  const goods = await fetchJSON(`/api/goods?${q.toString()}`);
  const goodsSelect = byId("goodsSelect");
  goodsSelect.innerHTML = `<option value="">全部商品</option>`;
  goods.forEach((g) => {
    const name = g.goods_name || g.goods_id;
    goodsSelect.innerHTML += `<option value="${name}">${name} (${g.comment_count}条)</option>`;
  });
  if (state.goodsName) goodsSelect.value = state.goodsName;
}

async function renderOverview() {
  const q = new URLSearchParams();
  if (state.goodsName) q.set("goods_names", state.goodsName);
  const data = await fetchJSON(`/api/overview?${q.toString()}`);
  byId("kpiComments").textContent = data.comment_count ?? "-";
  byId("kpiGoods").textContent = data.goods_count ?? "-";
  byId("kpiRating").textContent = data.avg_rating ?? "-";
  byId("kpiCategory").textContent = state.categoryName || "全部类别";
}

async function renderModelComparison() {
  const rows = await fetchJSON("/api/models/comparison");
  const x = rows.map((r) => r.model);
  charts.model.setOption({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { top: 4, textStyle: { color: "#c7d8f7" } },
    xAxis: { type: "category", data: x, axisLabel: { color: "#c7d8f7" } },
    yAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "#c7d8f7" } },
    series: [
      { name: "accuracy", type: "bar", data: rows.map((r) => r.accuracy) },
      { name: "f1_macro", type: "bar", data: rows.map((r) => r.f1_macro) },
      { name: "recall_macro", type: "line", data: rows.map((r) => r.recall_macro), smooth: true },
    ],
  });
}

async function renderSentimentPie() {
  const q = new URLSearchParams();
  if (state.goodsName) q.set("goods_names", state.goodsName);
  else if (state.categoryName) q.set("category_name", state.categoryName);
  const rows = await fetchJSON(`/api/sentiment/distribution?${q.toString()}`);
  charts.pie.setOption({
    tooltip: { trigger: "item" },
    legend: { bottom: 0, textStyle: { color: "#c7d8f7" } },
    series: [{ type: "pie", radius: ["40%", "70%"], data: rows }],
  });
}

async function renderTrendLine() {
  const q = new URLSearchParams();
  if (state.goodsName) q.set("goods_names", state.goodsName);
  const data = await fetchJSON(`/api/sentiment/trend?${q.toString()}`);
  charts.trend.setOption({
    tooltip: { trigger: "axis" },
    legend: { top: 0, textStyle: { color: "#c7d8f7" } },
    xAxis: { type: "category", data: data.dates, axisLabel: { color: "#c7d8f7" } },
    yAxis: { type: "value", axisLabel: { color: "#c7d8f7" } },
    series: [
      { name: "negative", type: "line", smooth: true, data: data.series.negative || [] },
      { name: "neutral", type: "line", smooth: true, data: data.series.neutral || [] },
      { name: "positive", type: "line", smooth: true, data: data.series.positive || [] },
    ],
  });
}

async function renderAspectStack() {
  const q = new URLSearchParams();
  if (state.goodsName) q.set("goods_names", state.goodsName);
  else if (state.categoryName) q.set("category_name", state.categoryName);
  const rows = await fetchJSON(`/api/aspects/summary?${q.toString()}`);
  const aspects = [...new Set(rows.map((r) => r.aspect))];
  const sentiments = ["negative", "neutral", "positive"];
  const map = {};
  rows.forEach((r) => {
    map[`${r.aspect}__${r.aspect_sentiment}`] = r.count;
  });
  const series = sentiments.map((s) => ({
    name: s,
    type: "bar",
    stack: "total",
    data: aspects.map((a) => map[`${a}__${s}`] || 0),
  }));

  charts.aspect.setOption({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 0, textStyle: { color: "#c7d8f7" } },
    xAxis: { type: "category", data: aspects, axisLabel: { color: "#c7d8f7", rotate: 20 } },
    yAxis: { type: "value", axisLabel: { color: "#c7d8f7" } },
    series,
  });
}

async function renderAspectRadar() {
  if (!state.categoryName) {
    charts.radar.clear();
    charts.radar.setOption({ title: { text: "请先选择商品类别", left: "center", top: "middle", textStyle: { color: "#c7d8f7" } } });
    return;
  }
  const topN = byId("compareTopN").value || 5;
  const data = await fetchJSON(`/api/aspects/compare?category_name=${encodeURIComponent(state.categoryName)}&top_n=${topN}`);
  const indicators = data.aspects.map((a) => ({ name: a, max: 1, min: -1 }));
  const seriesData = data.matrix.map((row, idx) => ({
    name: (data.goods_names && data.goods_names[idx]) ? data.goods_names[idx] : row.goods_id,
    value: data.aspects.map((a) => row[a] || 0),
  }));
  charts.radar.setOption({
    tooltip: {},
    legend: { top: 0, textStyle: { color: "#c7d8f7" }, formatter: (n) => truncateName(n) },
    radar: { indicator: indicators, splitNumber: 4, axisName: { color: "#c7d8f7" } },
    series: [{ type: "radar", data: seriesData }],
  });
}

async function renderKeywordBar() {
  const q = new URLSearchParams();
  if (state.goodsName) q.set("goods_names", state.goodsName);
  else if (state.categoryName) q.set("category_name", state.categoryName);
  q.set("top_n", "20");
  const rows = await fetchJSON(`/api/keywords?${q.toString()}`);
  const x = rows.map((r) => r.name).reverse();
  const y = rows.map((r) => r.value).reverse();
  charts.keyword.setOption({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: { type: "value", axisLabel: { color: "#c7d8f7" } },
    yAxis: { type: "category", data: x, axisLabel: { color: "#c7d8f7" } },
    series: [{ type: "bar", data: y }],
  });
}

async function renderComments() {
  const q = new URLSearchParams();
  if (state.goodsName) q.set("goods_names", state.goodsName);
  q.set("page", String(state.page));
  q.set("page_size", String(state.pageSize));
  const data = await fetchJSON(`/api/comments?${q.toString()}`);
  byId("pageInfo").textContent = `第 ${state.page} 页 / 共 ${Math.max(1, Math.ceil(data.total / state.pageSize))} 页`;
  const body = byId("commentTableBody");
  body.innerHTML = "";
  data.rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.goods_name || r.goods_id}</td>
      <td>${r.nickname || "-"}</td>
      <td>${r.rating ?? "-"}</td>
      <td class="${sentimentClass(r.sentiment_label_star)}">${r.sentiment_label_star}</td>
      <td>${r.comment_time || "-"}</td>
      <td>${r.comment_text_clean || ""}</td>
    `;
    body.appendChild(tr);
  });
}

async function renderAll() {
  await renderOverview();
  await Promise.all([
    renderModelComparison(),
    renderSentimentPie(),
    renderTrendLine(),
    renderAspectStack(),
    renderAspectRadar(),
    renderKeywordBar(),
    renderComments(),
  ]);
}

function bindEvents() {
  byId("categorySelect").addEventListener("change", async (e) => {
    state.categoryName = e.target.value;
    state.goodsName = "";
    state.page = 1;
    await loadGoodsOptions();
    await renderAll();
  });

  byId("goodsSelect").addEventListener("change", async (e) => {
    state.goodsName = e.target.value;
    state.page = 1;
    await renderAll();
  });

  byId("applyBtn").addEventListener("click", async () => {
    state.page = 1;
    await renderAll();
  });

  byId("clearGoodsBtn").addEventListener("click", async () => {
    state.goodsName = "";
    byId("goodsSelect").value = "";
    state.page = 1;
    await renderAll();
  });

  byId("reloadBtn").addEventListener("click", async () => {
    await fetchJSON("/api/reload", { method: "POST" });
    await loadFilters();
    await renderAll();
  });

  byId("prevPageBtn").addEventListener("click", async () => {
    state.page = Math.max(1, state.page - 1);
    await renderComments();
  });

  byId("nextPageBtn").addEventListener("click", async () => {
    state.page += 1;
    await renderComments();
  });
}

async function bootstrap() {
  initCharts();
  bindEvents();
  await loadFilters();
  await renderAll();
}

bootstrap().catch((err) => {
  console.error(err);
  alert(`页面初始化失败: ${err.message}`);
});
