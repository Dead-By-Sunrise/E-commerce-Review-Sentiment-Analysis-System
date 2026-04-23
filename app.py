import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request

from aspect_config import ASPECT_PRIORITY, ASPECT_SCHEMA


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_CSV = os.path.join(BASE_DIR, "analysis_outputs_full", "model_comparison.csv")
DEFAULT_ASPECT_CSV = os.path.join(BASE_DIR, "analysis_outputs_full", "aspect_sentiment_records.csv")
DEFAULT_COMMENTS_CSV = os.path.join(BASE_DIR, "goods_cleaned", "datasets", "all_comments_clean.csv")
DEFAULT_MANIFEST_JSON = os.path.join(BASE_DIR, "analysis_outputs_full", "analysis_manifest.json")
DEFAULT_REPORT_TXT = os.path.join(BASE_DIR, "analysis_outputs_full", "model_reports.txt")

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
STOP_WORDS = {
    "非常",
    "真的",
    "还是",
    "这个",
    "那个",
    "一个",
    "我们",
    "你们",
    "他们",
    "可以",
    "就是",
    "感觉",
    "东西",
    "产品",
    "使用",
    "总体",
}


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def parse_goods_ids(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def parse_goods_names(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def apply_comment_filters(
    df: pd.DataFrame,
    category_name: str = "",
    goods_ids: Optional[List[str]] = None,
    goods_names: Optional[List[str]] = None,
    keyword: str = "",
    sentiment: str = "",
):
    goods_ids = goods_ids or []
    goods_names = goods_names or []
    out = df
    if goods_ids:
        out = out[out["goods_id"].isin(goods_ids)]
    elif goods_names and "goods_name" in out.columns:
        out = out[out["goods_name"].isin(goods_names)]
    elif category_name:
        valid_ids = [k for k, v in store.goods_major_category.items() if v == category_name]
        out = out[out["goods_id"].isin(valid_ids)]
    if keyword:
        out = out[out["comment_text_clean"].str.contains(re.escape(keyword), na=False)]
    if sentiment:
        out = out[out["sentiment_label_star"] == sentiment]
    return out


def apply_aspect_filters(df: pd.DataFrame, category_name: str = "", goods_ids: Optional[List[str]] = None, keyword: str = ""):
    goods_ids = goods_ids or []
    out = df
    if goods_ids:
        out = out[out["goods_id"].isin(goods_ids)]
    elif category_name:
        out = out[out["category_name"] == category_name]
    if keyword:
        out = out[out["comment_text_clean"].str.contains(re.escape(keyword), na=False)]
    return out


def keyword_extract(texts: List[str], topn: int = 30) -> List[Dict]:
    # 优先使用 jieba，未安装则退化为中文短词抽取
    words = []
    try:
        import jieba  # type: ignore

        for text in texts:
            for w in jieba.lcut(text):
                w = w.strip()
                if len(w) < 2:
                    continue
                if w in STOP_WORDS:
                    continue
                if re.search(r"^[\W_]+$", w):
                    continue
                words.append(w)
    except Exception:
        for text in texts:
            for w in re.findall(r"[\u4e00-\u9fff]{2,4}", text):
                if w in STOP_WORDS:
                    continue
                words.append(w)
    counter = Counter(words)
    return [{"name": k, "value": int(v)} for k, v in counter.most_common(topn)]


def make_keyword_buttons(texts: List[str], category_key: str = "", topn: int = 12) -> List[Dict]:
    if not texts:
        return []
    kw_counter = Counter()
    aspect_counter = Counter()
    keyword_pool = []
    if category_key in ASPECT_SCHEMA:
        keyword_pool.extend(ASPECT_SCHEMA[category_key]["category_keywords"])
        for aspect_name, kws in ASPECT_SCHEMA[category_key]["aspects"].items():
            aspect_counter[aspect_name] += 0
            keyword_pool.extend(kws)
    try:
        import jieba  # type: ignore

        for text in texts:
            for w in jieba.lcut(text):
                w = w.strip()
                if len(w) < 2 or w in STOP_WORDS or re.search(r"^[\W_]+$", w):
                    continue
                kw_counter[w] += 1
    except Exception:
        for text in texts:
            for w in re.findall(r"[\u4e00-\u9fff]{2,4}", text):
                if w in STOP_WORDS:
                    continue
                kw_counter[w] += 1

    for aspect_name, kws in ASPECT_SCHEMA.get(category_key, {}).get("aspects", {}).items():
        for text in texts:
            if any(k in text for k in kws):
                aspect_counter[aspect_name] += 1

    buttons = []
    for kw, cnt in kw_counter.most_common(topn * 2):
        if len(kw) < 2:
            continue
        if kw in STOP_WORDS:
            continue
        buttons.append({"type": "keyword", "label": kw, "value": int(cnt)})
        if len(buttons) >= topn:
            break

    aspect_buttons = []
    for aspect_name, cnt in aspect_counter.most_common():
        aspect_buttons.append({"type": "aspect", "label": aspect_name, "value": int(cnt)})
    buttons.extend(aspect_buttons[: max(0, topn - len(buttons))])

    if category_key in ASPECT_SCHEMA:
        for aspect_name in ASPECT_PRIORITY.get(category_key, []):
            if not any(btn["label"] == aspect_name for btn in buttons):
                buttons.append({"type": "aspect", "label": aspect_name, "value": 0})

    # 去重且保序
    seen = set()
    unique = []
    for item in buttons:
        key = (item["type"], item["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:topn]


class DashboardStore:
    def __init__(self, model_csv: str, aspect_csv: str, comments_csv: str):
        self.model_csv = model_csv
        self.aspect_csv = aspect_csv
        self.comments_csv = comments_csv
        self._last_load_ts = None
        self.model_df = pd.DataFrame()
        self.aspect_df = pd.DataFrame()
        self.comments_df = pd.DataFrame()
        self.goods_major_category = {}
        self.reload()

    def reload(self):
        self.model_df = pd.read_csv(self.model_csv) if os.path.exists(self.model_csv) else pd.DataFrame()
        self.aspect_df = pd.read_csv(self.aspect_csv) if os.path.exists(self.aspect_csv) else pd.DataFrame()
        self.comments_df = pd.read_csv(self.comments_csv) if os.path.exists(self.comments_csv) else pd.DataFrame()

        if not self.comments_df.empty:
            self.comments_df["comment_text_clean"] = self.comments_df["comment_text_clean"].fillna("").map(normalize_text)
            self.comments_df["comment_time"] = self.comments_df["comment_time"].fillna("")
            self.comments_df["goods_id"] = self.comments_df["goods_id"].astype(str)
            if "goods_name" not in self.comments_df.columns:
                self.comments_df["goods_name"] = ""
            self.comments_df["goods_name"] = self.comments_df["goods_name"].fillna("").map(normalize_text)
            if "major_category" not in self.comments_df.columns:
                self.comments_df["major_category"] = ""
        if not self.aspect_df.empty:
            self.aspect_df["goods_id"] = self.aspect_df["goods_id"].astype(str)
            self.aspect_df["comment_text_clean"] = self.aspect_df["comment_text_clean"].fillna("").map(normalize_text)

        # 商品分类：优先使用方面分析结果；否则使用清洗阶段提供的 major_category；再否则退化为未知
        self.goods_major_category = {}
        if not self.aspect_df.empty:
            grp = self.aspect_df.groupby(["goods_id", "category_name"]).size().reset_index(name="cnt")
            for goods_id, sub in grp.groupby("goods_id"):
                top = sub.sort_values("cnt", ascending=False).iloc[0]
                self.goods_major_category[goods_id] = top["category_name"]

        # 无论是否存在方面分析结果，都用清洗阶段的 major_category 为“未覆盖商品”补齐
        if not self.comments_df.empty and "major_category" in self.comments_df.columns:
            def _first_non_empty(series: pd.Series) -> str:
                vals = series.dropna().astype(str)
                vals = vals[vals != ""]
                return str(vals.iloc[0]) if len(vals) else ""

            mp = self.comments_df.groupby("goods_id")["major_category"].apply(_first_non_empty).to_dict()
            for gid, cat in mp.items():
                if gid not in self.goods_major_category:
                    self.goods_major_category[gid] = cat or "未分类"

        self._last_load_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_goods_meta(self):
        if self.comments_df.empty:
            return []
        goods_ids = sorted(self.comments_df["goods_id"].dropna().astype(str).unique().tolist())
        rows = []
        for gid in goods_ids:
            subset = self.comments_df[self.comments_df["goods_id"] == gid]
            goods_name = ""
            if "goods_name" in subset and not subset["goods_name"].empty:
                goods_name = str(subset["goods_name"].dropna().astype(str).iloc[0]) if subset["goods_name"].dropna().any() else ""
            rows.append(
                {
                    "goods_id": gid,
                    "goods_name": goods_name,
                    "comment_count": int(len(subset)),
                    "category_name": self.goods_major_category.get(gid, "未分类"),
                    "avg_rating": round(float(subset["rating"].fillna(0).mean()), 3) if "rating" in subset else 0.0,
                }
            )
        return rows


store = DashboardStore(DEFAULT_MODEL_CSV, DEFAULT_ASPECT_CSV, DEFAULT_COMMENTS_CSV)
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("user_dashboard.html")


@app.route("/dev")
def dev_dashboard():
    return render_template("dev_dashboard.html")


@app.route("/user")
def user_dashboard():
    return render_template("user_dashboard.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "last_reload": store._last_load_ts,
            "rows": {
                "models": int(len(store.model_df)),
                "aspects": int(len(store.aspect_df)),
                "comments": int(len(store.comments_df)),
            },
        }
    )


@app.route("/api/reload", methods=["POST"])
def reload_data():
    store.reload()
    return jsonify({"ok": True, "last_reload": store._last_load_ts})


@app.route("/api/models/comparison")
def model_comparison():
    if store.model_df.empty:
        return jsonify([])
    cols = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    out = store.model_df[cols].copy()
    for c in cols[1:]:
        out[c] = out[c].astype(float).round(4)
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/dev/model-reports")
def model_reports():
    if not os.path.exists(DEFAULT_REPORT_TXT):
        return jsonify({"content": ""})
    with open(DEFAULT_REPORT_TXT, "r", encoding="utf-8") as f:
        return jsonify({"content": f.read()})


@app.route("/api/dev/manifest")
def dev_manifest():
    if not os.path.exists(DEFAULT_MANIFEST_JSON):
        return jsonify({})
    return jsonify(pd.read_json(DEFAULT_MANIFEST_JSON, typ="series").to_dict())


@app.route("/api/dev/label-distribution")
def dev_label_distribution():
    df = store.comments_df
    grouped = df.groupby("sentiment_label_star").size().reindex(SENTIMENT_ORDER, fill_value=0)
    return jsonify([{"label": k, "count": int(v)} for k, v in grouped.to_dict().items()])


@app.route("/api/categories")
def categories():
    cats = set(store.goods_major_category.values())
    # 兜底：即便 goods_major_category 未覆盖到，也从 comments_df 里补全类别候选
    if not store.comments_df.empty and "major_category" in store.comments_df.columns:
        cats |= set(store.comments_df["major_category"].dropna().astype(str).tolist())
    cats = sorted([c for c in cats if c and c.strip()])
    return jsonify(cats)


@app.route("/api/goods")
def goods():
    category_name = request.args.get("category_name", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    rows = store.get_goods_meta()
    if category_name:
        rows = [r for r in rows if r["category_name"] == category_name]
    if goods_ids:
        rows = [r for r in rows if r["goods_id"] in goods_ids]
    if goods_names:
        rows = [r for r in rows if r.get("goods_name", "") in goods_names]
    return jsonify(rows)


@app.route("/api/overview")
def overview():
    category_name = request.args.get("category_name", "").strip()
    keyword = request.args.get("keyword", "").strip()
    sentiment = request.args.get("sentiment", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    one_goods = request.args.get("goods_id", "").strip()
    if one_goods:
        goods_ids = [one_goods]
    df = apply_comment_filters(
        store.comments_df,
        category_name=category_name,
        goods_ids=goods_ids,
        goods_names=goods_names,
        keyword=keyword,
        sentiment=sentiment,
    )
    if df.empty:
        return jsonify({})
    sentiment_counts = (
        df.groupby("sentiment_label_star").size().reindex(SENTIMENT_ORDER, fill_value=0).to_dict()
    )
    return jsonify(
        {
            "comment_count": int(len(df)),
            "goods_count": int(df["goods_id"].nunique()),
            "avg_rating": round(float(df["rating"].fillna(0).mean()), 3),
            "sentiment_counts": {k: int(v) for k, v in sentiment_counts.items()},
        }
    )


@app.route("/api/sentiment/distribution")
def sentiment_distribution():
    category_name = request.args.get("category_name", "").strip()
    keyword = request.args.get("keyword", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    one_goods = request.args.get("goods_id", "").strip()
    if one_goods:
        goods_ids = [one_goods]
    df = apply_comment_filters(store.comments_df, category_name=category_name, goods_ids=goods_ids, goods_names=goods_names, keyword=keyword)
    grouped = df.groupby("sentiment_label_star").size().reindex(SENTIMENT_ORDER, fill_value=0)
    return jsonify([{"name": k, "value": int(v)} for k, v in grouped.to_dict().items()])


@app.route("/api/sentiment/trend")
def sentiment_trend():
    category_name = request.args.get("category_name", "").strip()
    keyword = request.args.get("keyword", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    one_goods = request.args.get("goods_id", "").strip()
    if one_goods:
        goods_ids = [one_goods]
    df = apply_comment_filters(store.comments_df, category_name=category_name, goods_ids=goods_ids, goods_names=goods_names, keyword=keyword)
    if df.empty:
        return jsonify({"dates": [], "series": {}})
    tmp = df.copy()
    tmp["date"] = tmp["comment_time"].astype(str).str.slice(0, 10)
    tmp = tmp[tmp["date"] != ""]
    grp = tmp.groupby(["date", "sentiment_label_star"]).size().reset_index(name="count")
    dates = sorted(grp["date"].unique().tolist())
    series = {}
    for s in SENTIMENT_ORDER:
        mp = {r["date"]: int(r["count"]) for _, r in grp[grp["sentiment_label_star"] == s].iterrows()}
        series[s] = [mp.get(d, 0) for d in dates]
    return jsonify({"dates": dates, "series": series})


@app.route("/api/aspects/summary")
def aspects_summary():
    category_name = request.args.get("category_name", "").strip()
    keyword = request.args.get("keyword", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    one_goods = request.args.get("goods_id", "").strip()
    if one_goods:
        goods_ids = [one_goods]
    df = apply_aspect_filters(store.aspect_df, category_name=category_name, goods_ids=goods_ids, keyword=keyword)
    if df.empty:
        return jsonify([])
    grp = df.groupby(["aspect", "aspect_sentiment"]).size().reset_index(name="count")
    return jsonify(grp.to_dict(orient="records"))


@app.route("/api/aspects/compare")
def aspects_compare():
    category_name = request.args.get("category_name", "").strip()
    if not category_name:
        return jsonify({"error": "category_name is required"}), 400
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    if not goods_ids:
        return jsonify({"category_name": category_name, "goods": [], "goods_names": [], "aspects": [], "matrix": [], "suggested_keywords": [], "aspect_weights": []})
    keyword = request.args.get("keyword", "").strip()
    aspect_focus = request.args.get("aspect_focus", "").strip()
    df = apply_aspect_filters(store.aspect_df, category_name=category_name, goods_ids=goods_ids, keyword=keyword)
    if df.empty:
        return jsonify({"category_name": category_name, "goods": [], "goods_names": [], "aspects": [], "matrix": [], "suggested_keywords": [], "aspect_weights": []})

    if aspect_focus:
        df = df[(df["aspect"] == aspect_focus) | (df["comment_text_clean"].str.contains(re.escape(aspect_focus), na=False))]
        if df.empty:
            return jsonify({"category_name": category_name, "goods": [], "goods_names": [], "aspects": [], "matrix": [], "suggested_keywords": [], "aspect_weights": []})

    goods_ids = [gid for gid in goods_ids if gid in df["goods_id"].astype(str).unique().tolist()]
    if not goods_ids:
        return jsonify({"category_name": category_name, "goods": [], "goods_names": [], "aspects": [], "matrix": [], "suggested_keywords": [], "aspect_weights": []})
    sub = df[df["goods_id"].isin(goods_ids)]
    aspect_names = sorted(sub["aspect"].dropna().unique().tolist())
    if aspect_focus and aspect_focus not in aspect_names:
        aspect_names = [aspect_focus] + aspect_names
    if category_name in ASPECT_SCHEMA:
        preferred = ASPECT_PRIORITY.get(category_name, [])
        if preferred:
            merged = []
            for a in preferred + aspect_names:
                if a not in merged:
                    merged.append(a)
            aspect_names = merged
    matrix = []

    for gid in goods_ids:
        gdf = sub[sub["goods_id"] == gid]
        row = {"goods_id": gid}
        for aspect in aspect_names:
            adf = gdf[gdf["aspect"] == aspect]
            total = len(adf)
            pos = int((adf["aspect_sentiment"] == "positive").sum())
            neg = int((adf["aspect_sentiment"] == "negative").sum())
            score = 0.0 if total == 0 else (pos - neg) / total
            row[aspect] = round(score, 4)
        matrix.append(row)
    goods_names = []
    if not store.comments_df.empty and "goods_name" in store.comments_df.columns:
        name_map = (
            store.comments_df.groupby("goods_id")["goods_name"]
            .apply(lambda s: str(s.dropna().astype(str)[s.dropna().astype(str) != ""].iloc[0]) if ((s.dropna().astype(str) != "").any()) else "")
            .to_dict()
        )
        goods_names = [name_map.get(gid, "") for gid in goods_ids]
    else:
        goods_names = ["" for _ in goods_ids]

    texts = sub["comment_text_clean"].fillna("").tolist()
    suggested_keywords = make_keyword_buttons(texts, category_key=category_name, topn=12)
    aspect_weights = (
        sub.groupby("aspect").size().reset_index(name="count").sort_values("count", ascending=False).to_dict(orient="records")
    )

    return jsonify(
        {
            "category_name": category_name,
            "goods": goods_ids,
            "goods_names": goods_names,
            "aspects": aspect_names,
            "matrix": matrix,
            "suggested_keywords": suggested_keywords,
            "aspect_weights": aspect_weights,
        }
    )


@app.route("/api/keywords")
def keywords():
    category_name = request.args.get("category_name", "").strip()
    keyword = request.args.get("keyword", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    one_goods = request.args.get("goods_id", "").strip()
    aspect_focus = request.args.get("aspect_focus", "").strip()
    if one_goods:
        goods_ids = [one_goods]
    top_n = safe_int(request.args.get("top_n", 40), 40)
    df = apply_comment_filters(store.comments_df, category_name=category_name, goods_ids=goods_ids, goods_names=goods_names, keyword=keyword)
    if aspect_focus and not df.empty and "comment_text_clean" in df.columns:
        df = df[df["comment_text_clean"].str.contains(re.escape(aspect_focus), na=False) | df["comment_text_clean"].str.contains(re.escape(aspect_focus.replace("/", "")), na=False)]
    texts = df["comment_text_clean"].fillna("").tolist()
    return jsonify({
        "keywords": keyword_extract(texts, topn=top_n),
        "suggested": make_keyword_buttons(texts, category_key=category_name, topn=12),
    })


@app.route("/api/comments")
def comments():
    category_name = request.args.get("category_name", "").strip()
    keyword = request.args.get("keyword", "").strip()
    sentiment = request.args.get("sentiment", "").strip()
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    one_goods = request.args.get("goods_id", "").strip()
    if one_goods:
        goods_ids = [one_goods]
    page = max(1, safe_int(request.args.get("page", 1), 1))
    page_size = min(50, max(5, safe_int(request.args.get("page_size", 10), 10)))
    df = apply_comment_filters(
        store.comments_df,
        category_name=category_name,
        goods_ids=goods_ids,
        goods_names=goods_names,
        keyword=keyword,
        sentiment=sentiment,
    )
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    sub = df.iloc[start:end].copy()
    cols = ["goods_id", "goods_name", "major_category", "nickname", "rating", "sentiment_label_star", "comment_time", "comment_text_clean"]
    if sub.empty:
        return jsonify({"total": total, "page": page, "page_size": page_size, "rows": []})
    return jsonify(
        {
            "total": int(total),
            "page": page,
            "page_size": page_size,
            "rows": sub[cols].to_dict(orient="records"),
        }
    )


@app.route("/api/user/goods-compare")
def user_goods_compare():
    goods_ids = parse_goods_ids(request.args.get("goods_ids", "").strip())
    goods_names = parse_goods_names(request.args.get("goods_names", "").strip())
    keyword = request.args.get("keyword", "").strip()
    sentiment = request.args.get("sentiment", "").strip()
    if len(goods_ids) < 1:
        # 允许仅传 goods_names 的场景：由名称反查ID
        if goods_names and not store.comments_df.empty and "goods_name" in store.comments_df.columns:
            ids = (
                store.comments_df[store.comments_df["goods_name"].isin(goods_names)]["goods_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            goods_ids = ids
        else:
            return jsonify([])
    df = apply_comment_filters(store.comments_df, goods_ids=goods_ids, goods_names=goods_names, keyword=keyword, sentiment=sentiment)
    rows = []
    for gid in goods_ids:
        sub = df[df["goods_id"] == gid]
        if sub.empty:
            rows.append({"goods_id": gid, "goods_name": "", "comment_count": 0, "avg_rating": 0, "positive_ratio": 0})
            continue
        gname = ""
        if "goods_name" in sub.columns and sub["goods_name"].dropna().astype(str).any():
            gname = str(sub["goods_name"].dropna().astype(str).iloc[0])
        pos = int((sub["sentiment_label_star"] == "positive").sum())
        rows.append(
            {
                "goods_id": gid,
                "goods_name": gname,
                "comment_count": int(len(sub)),
                "avg_rating": round(float(sub["rating"].fillna(0).mean()), 3),
                "positive_ratio": round(pos / max(1, len(sub)), 4),
            }
        )
    return jsonify(rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
