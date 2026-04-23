import argparse
import csv
import glob
import html
import json
import os
import re
import sqlite3
import sys
import time
import hashlib
from datetime import datetime

import pandas as pd

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MAJOR_CATEGORIES = [
    "3C数码与家电",
    "服饰鞋包与配饰",
    "美妆个护与健康",
    "家居生活与家具",
    "食品饮料与生鲜",
    "母婴用品与玩具",
    "运动户外与器材",
    "图书文具与办公",
]

# 基于商品名称的轻量级分类（对应 classification.md 的 8 大类）
_CATEGORY_KEYWORDS = {
    "3C数码与家电": [
        "手机",
        "iPhone",
        "Mate",
        "电脑",
        "笔记本",
        "主机",
        "显卡",
        "键盘",
        "耳机",
        "空调",
        "冰箱",
        "电视",
        "洗衣机",
    ],
    # 尽量避免 1 字词（如“裤/鞋/包”）带来的误判；用更具体的词覆盖
    "服饰鞋包与配饰": ["短袖", "T恤", "外套", "长袖", "卫衣", "运动鞋", "帽子", "背包", "双肩包", "项链", "手链"],
    # 移除过于泛化的“水”，否则如“香水菠萝”会被误判到美妆
    "美妆个护与健康": ["防晒", "面霜", "精华", "乳液", "爽肤", "洗面奶", "口红", "面膜", "保湿", "美白", "护肤"],
    "家居生活与家具": ["四件套", "床垫", "床褥", "被套", "床单", "枕套", "家具", "沙发", "收纳", "家纺"],
    "食品饮料与生鲜": ["方便面", "牛肉面", "白酒", "汾酒", "茅台", "酒", "水果", "凤梨", "菠萝", "生鲜", "零食"],
    "母婴用品与玩具": ["玩具", "积木", "宝宝", "婴儿", "纸尿裤", "奶粉", "益智"],
    "运动户外与器材": ["健身", "哑铃", "臂力", "跑步", "运动", "户外", "瑜伽", "器材"],
    "图书文具与办公": ["书籍", "算法", "竞赛", "订书机", "订书钉", "文具", "办公", "纸张", "笔记本"],
}


def classify_major_category(goods_name: str) -> str:
    s = normalize_text(goods_name)
    if not s:
        return "未分类"

    # 1) 强规则：食品生鲜优先（解决“香水菠萝”等歧义词）
    food_strong = ["菠萝", "凤梨", "水果", "生鲜", "牛肉面", "方便面", "白酒", "茅台", "汾酒"]
    if any(w in s for w in food_strong):
        return "食品饮料与生鲜"

    # 2) 词元化（可选 jieba），降低“子串误判”
    tokens = []
    try:
        import jieba  # type: ignore

        tokens = [t.strip() for t in jieba.lcut(s) if t and t.strip()]
    except Exception:
        tokens = []

    # 3) 打分：优先命中更长、更具体的关键词
    best_cat = "未分类"
    best_score = 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        score = 0
        for kw in kws:
            kw = str(kw or "").strip()
            if len(kw) < 2:
                continue
            if kw in s:
                score += 3 if len(kw) >= 3 else 2
            if tokens and kw in tokens:
                score += 2
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat if best_score > 0 else "未分类"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_rating(rating_val):
    if rating_val is None:
        return None
    if isinstance(rating_val, int):
        return rating_val
    s = str(rating_val).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_datetime(dt_str: str):
    if not dt_str:
        return None, None
    s = str(dt_str).strip()
    # 兼容“2026-03-09 16:09:09”这种格式
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.isoformat(sep=" "), int(dt.timestamp())
        except Exception:
            continue
    return None, None


_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")

# 仅移除常见表情符号区间，尽量不误删汉字/常见标点。
_EMOJI_RE = re.compile(
    r"["  # type: ignore
    r"\U0001F300-\U0001F5FF"
    r"\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF"
    r"\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002600-\U000027BF"
    r"]+",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = html.unescape(s)  # 处理 &nbsp; 等 HTML 实体
    s = s.replace("\u3000", " ")  # 全角空格
    s = s.replace("\r", " ").replace("\n", " ")
    s = _URL_RE.sub("", s)
    s = _EMOJI_RE.sub("", s)  # 去掉表情符号，降低噪声
    s = s.replace("\t", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()


def star_to_sentiment_label(rating: int):
    # 你后续可以按论文/实验需要调整阈值
    if rating is None:
        return None
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def comment_hash(goods_id: str, nickname: str, comment_time: str, comment_text_raw: str) -> str:
    raw = f"{goods_id}|{nickname}|{comment_time}|{comment_text_raw}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def iter_raw_comments_from_file(path: str):
    """
    支持两种输入：
    1) jd_comments_{goods_id}.json：list
    2) jd_comments_{goods_id}.jsonl：每行一个 dict
    """
    basename = os.path.basename(path)
    if basename.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                raise ValueError(f"Unexpected JSON root for {path}")


def extract_goods_id_from_filename(filename: str):
    """
    兼容两类命名：
    1) 旧：jd_comments_{goods_id}.json(.l)
    2) 新：{goods_id}_{title}_{YYYYmmdd_HHMMSS}.json  （crawler-final.py）
    """
    m = re.search(r"jd_comments_(.+?)\.(?:jsonl|json)$", filename)
    if m:
        return m.group(1)
    # 新版文件名：以数字商品ID开头，后接下划线
    m = re.match(r"(\d+)_.*\.(?:jsonl|json)$", filename)
    if m:
        return m.group(1)
    return None


def init_db(db_path: str, rebuild: bool = False):
    ensure_dir(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if rebuild:
        cur.execute("DROP TABLE IF EXISTS comments")
        cur.execute("DROP TABLE IF EXISTS ingest_log")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            comment_hash TEXT PRIMARY KEY,
            goods_id TEXT NOT NULL,
            goods_name TEXT,
            major_category TEXT,
            nickname TEXT,
            purchase_product TEXT,
            buy_count INTEGER,
            rating INTEGER,
            sentiment_label_star TEXT,
            comment_time TEXT,
            comment_time_ts INTEGER,
            comment_text_raw TEXT,
            comment_text_clean TEXT,
            source_file TEXT,
            ingest_ts TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            goods_id TEXT NOT NULL,
            total_rows INTEGER,
            inserted_rows INTEGER,
            skipped_rows INTEGER,
            status TEXT,
            finished_ts TEXT
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_comments_goods_time ON comments(goods_id, comment_time_ts)")
    conn.commit()

    # 对老库做一次轻量迁移（不要求 rebuild）
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(comments)").fetchall()]
        if "goods_name" not in cols:
            cur.execute("ALTER TABLE comments ADD COLUMN goods_name TEXT")
        if "major_category" not in cols:
            cur.execute("ALTER TABLE comments ADD COLUMN major_category TEXT")
        conn.commit()
    except Exception:
        # 迁移失败不阻断主流程（rebuild 时会重新建表）
        pass
    return conn


def clean_one_comment(raw: dict, goods_id: str, source_file: str, ingest_ts: str):
    # 新版爬虫的每条评论里也带有 商品ID，优先使用内容里的，避免文件名解析失败
    goods_id = str(raw.get("商品ID") or raw.get("goods_id") or goods_id or "").strip()
    goods_name = raw.get("商品名称", "") or raw.get("goods_name", "")
    goods_name = normalize_text(goods_name)
    major_category = classify_major_category(goods_name)
    nickname = raw.get("昵称", "") or raw.get("nickname", "")
    purchase_product = raw.get("购买产品", "") or raw.get("purchase_product", "")
    buy_count = raw.get("购买次数", None)
    rating = parse_rating(raw.get("评分", raw.get("rating", None)))
    comment_text_raw = raw.get("评论内容", "") or raw.get("comment_text", "")
    comment_time_str = raw.get("评论时间(日期)", "") or raw.get("comment_time", "")

    buy_count = parse_rating(buy_count)
    comment_time_iso, comment_time_ts = parse_datetime(comment_time_str)
    comment_text_clean = normalize_text(comment_text_raw)

    sent_label = star_to_sentiment_label(rating)
    h = comment_hash(goods_id, str(nickname), comment_time_iso or "", str(comment_text_raw))

    return {
        "comment_hash": h,
        "goods_id": goods_id,
        "goods_name": goods_name,
        "major_category": major_category,
        "nickname": str(nickname),
        "purchase_product": str(purchase_product).replace("已购", "").strip(),
        "buy_count": buy_count,
        "rating": rating,
        "sentiment_label_star": sent_label,
        "comment_time": comment_time_iso,
        "comment_time_ts": comment_time_ts,
        "comment_text_raw": comment_text_raw,
        "comment_text_clean": comment_text_clean,
        "source_file": source_file,
        "ingest_ts": ingest_ts,
    }


def init_manifest(out_dir: str):
    ensure_dir(out_dir)
    return {
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "source_dir": None,
        "total_source_files": 0,
        "total_rows": 0,
        "total_inserted": 0,
        "total_skipped": 0,
        "files": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Clean and store scraped comments.")
    parser.add_argument("--source-dir", default=os.path.join(PROJECT_ROOT, "goods"))
    parser.add_argument("--db-path", default=os.path.join(PROJECT_ROOT, "goods_cleaned", "comments.sqlite"))
    parser.add_argument("--clean-out-dir", default=os.path.join(PROJECT_ROOT, "goods_cleaned"))
    parser.add_argument("--rebuild-db", action="store_true", help="Rebuild sqlite schema (loses data).")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export a combined CSV for sklearn (under goods_cleaned).",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    db_path = args.db_path
    clean_out_dir = args.clean_out_dir

    ensure_dir(clean_out_dir)
    datasets_dir = os.path.join(clean_out_dir, "datasets")
    ensure_dir(datasets_dir)

    conn = init_db(db_path=db_path, rebuild=args.rebuild_db)
    cur = conn.cursor()

    # 兼容：
    # - 旧：goods/jd_comments_*.json(.l)
    # - 新：goods/{goods_id}_{title}_{timestamp}.json
    json_files_old = glob.glob(os.path.join(source_dir, "jd_comments_*.json"))
    jsonl_files_old = glob.glob(os.path.join(source_dir, "jd_comments_*.jsonl"))
    json_files_new = glob.glob(os.path.join(source_dir, "*.json"))
    jsonl_files_new = glob.glob(os.path.join(source_dir, "*.jsonl"))
    files = sorted(set(json_files_old + jsonl_files_old + json_files_new + jsonl_files_new))
    if not files:
        print(f"[WARN] No comment files found under: {source_dir}")
        return

    combined_jsonl_path = os.path.join(datasets_dir, "all_comments_clean.jsonl")
    combined_csv_path = os.path.join(datasets_dir, "all_comments_clean.csv")
    category_trace_path = os.path.join(datasets_dir, "category_assignment_trace.csv")
    category_summary_path = os.path.join(datasets_dir, "category_assignment_summary.csv")
    manifest_path = os.path.join(clean_out_dir, "manifest.json")
    manifest = init_manifest(clean_out_dir)
    manifest["source_dir"] = source_dir
    manifest["total_source_files"] = len(files)

    category_trace_rows = []

    # CSV 头部只写一次（当开启 export-csv 且文件不存在时）
    csv_writer = None
    csv_file = None
    try:
        if args.export_csv:
            csv_file = open(combined_csv_path, "w", encoding="utf-8", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "goods_id",
                    "goods_name",
                    "major_category",
                    "nickname",
                    "purchase_product",
                    "buy_count",
                    "rating",
                    "sentiment_label_star",
                    "comment_time",
                    "comment_time_ts",
                    "comment_text_clean",
                    "comment_hash",
                ]
            )

        # 清洗输出文件按商品拆分 + 同时写入 combined
        combined_jsonl_f = open(combined_jsonl_path, "w", encoding="utf-8")

        for file_path in files:
            goods_id = extract_goods_id_from_filename(os.path.basename(file_path))
            # 新版数据即使文件名解析失败，也可能在内容里带商品ID；因此先不直接跳过
            ingest_ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            out_goods_jsonl = os.path.join(datasets_dir, f"clean_comments_{goods_id or 'unknown'}.jsonl")

            total_rows = 0
            inserted_rows = 0
            skipped_rows = 0
            started_ts = time.time()

            # 覆盖式输出：每次运行重新生成该商品的清洗文件
            with open(out_goods_jsonl, "w", encoding="utf-8") as out_f:
                try:
                    for raw in iter_raw_comments_from_file(file_path):
                        total_rows += 1
                        cleaned = clean_one_comment(raw, goods_id=goods_id, source_file=file_path, ingest_ts=ingest_ts)
                        # 如果 goods_id 原本未解析到，这里用首条可用记录补齐，并修正输出文件名
                        if not goods_id and cleaned.get("goods_id"):
                            goods_id = cleaned["goods_id"]
                            out_goods_jsonl = os.path.join(datasets_dir, f"clean_comments_{goods_id}.jsonl")
                            # 由于文件已打开，unknown 文件名会保留；这里不重开文件以避免复杂的迁移逻辑
                            # 后续你如果希望严格按 goods_id 命名，我可以再做“首条记录后重命名输出文件”的增强

                        # 幂等插入：comment_hash 为主键
                        try:
                            cur.execute(
                                """
                                INSERT INTO comments (
                                    comment_hash, goods_id, goods_name, major_category, nickname, purchase_product, buy_count, rating,
                                    sentiment_label_star, comment_time, comment_time_ts,
                                    comment_text_raw, comment_text_clean, source_file, ingest_ts
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    cleaned["comment_hash"],
                                    cleaned["goods_id"],
                                    cleaned["goods_name"],
                                    cleaned["major_category"],
                                    cleaned["nickname"],
                                    cleaned["purchase_product"],
                                    cleaned["buy_count"],
                                    cleaned["rating"],
                                    cleaned["sentiment_label_star"],
                                    cleaned["comment_time"],
                                    cleaned["comment_time_ts"],
                                    cleaned["comment_text_raw"],
                                    cleaned["comment_text_clean"],
                                    cleaned["source_file"],
                                    cleaned["ingest_ts"],
                                ),
                            )
                            inserted_rows += 1
                        except sqlite3.IntegrityError:
                            skipped_rows += 1

                        out_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                        combined_jsonl_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                        trace_row = {
                            "goods_id": cleaned["goods_id"],
                            "goods_name": cleaned["goods_name"],
                            "major_category": cleaned["major_category"],
                            "category_rule": "goods_name_keywords",
                            "comment_hash": cleaned["comment_hash"],
                            "source_file": cleaned["source_file"],
                        }
                        category_trace_rows.append(trace_row)
                        if csv_writer is not None:
                            csv_writer.writerow(
                                [
                                    cleaned["goods_id"],
                                    cleaned["goods_name"],
                                    cleaned["major_category"],
                                    cleaned["nickname"],
                                    cleaned["purchase_product"],
                                    cleaned["buy_count"],
                                    cleaned["rating"],
                                    cleaned["sentiment_label_star"],
                                    cleaned["comment_time"],
                                    cleaned["comment_time_ts"],
                                    cleaned["comment_text_clean"],
                                    cleaned["comment_hash"],
                                ]
                            )

                    status = "success"
                except Exception as e:
                    status = f"failed: {e}"

            # 单文件事务提交，提高可靠性和速度
            conn.commit()

            finished_ts = datetime.now().isoformat(sep=" ", timespec="seconds")
            cur.execute(
                """
                INSERT INTO ingest_log (source_file, goods_id, total_rows, inserted_rows, skipped_rows, status, finished_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (file_path, goods_id, total_rows, inserted_rows, skipped_rows, status, finished_ts),
            )
            conn.commit()
            manifest["files"].append(
                {
                    "goods_id": goods_id,
                    "source_file": file_path,
                    "clean_file": out_goods_jsonl,
                    "total_rows": total_rows,
                    "inserted_rows": inserted_rows,
                    "skipped_rows": skipped_rows,
                    "status": status,
                    "finished_ts": finished_ts,
                }
            )
            manifest["total_rows"] += total_rows
            manifest["total_inserted"] += inserted_rows
            manifest["total_skipped"] += skipped_rows

            elapsed = time.time() - started_ts
            print(
                f"[OK] {os.path.basename(file_path)} | goods_id={goods_id} | total={total_rows} inserted={inserted_rows} skipped={skipped_rows} time={elapsed:.2f}s"
            )

    finally:
        try:
            combined_jsonl_f.close()
        except Exception:
            pass
        try:
            if csv_file:
                csv_file.close()
        except Exception:
            pass
        conn.close()

    if category_trace_rows:
        category_trace_df = pd.DataFrame(category_trace_rows)
        category_trace_df.to_csv(category_trace_path, index=False, encoding="utf-8")
        category_summary_df = (
            category_trace_df.groupby(["goods_id", "goods_name", "major_category", "category_rule"])
            .size()
            .reset_index(name="comment_count")
            .sort_values(["major_category", "goods_id"])
        )
        category_summary_df.to_csv(category_summary_path, index=False, encoding="utf-8")
        print(f"[DONE] category trace csv: {category_trace_path}")
        print(f"[DONE] category summary csv: {category_summary_path}")

    print(f"[DONE] sqlite db: {db_path}")
    print(f"[DONE] clean jsonl dir: {clean_out_dir}")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

