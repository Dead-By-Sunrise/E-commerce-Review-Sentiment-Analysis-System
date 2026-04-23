import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


from aspect_config import ASPECT_SCHEMA


POSITIVE_WORDS = set(
    ["好", "很好", "不错", "满意", "喜欢", "流畅", "清晰", "稳定", "划算", "新鲜", "值得", "推荐", "给力", "舒适"]
)
NEGATIVE_WORDS = set(
    ["差", "很差", "失望", "不好", "卡顿", "模糊", "异味", "破损", "贵", "难吃", "过敏", "漏", "慢", "问题", "故障"]
)
NEGATION_WORDS = set(["不", "没", "无", "并非", "不是"])


def lexicon_predict(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "neutral"
    pos_score = 0.0
    neg_score = 0.0
    for w in POSITIVE_WORDS:
        if w in text:
            pos_score += 1
    for w in NEGATIVE_WORDS:
        if w in text:
            neg_score += 1
    for n in NEGATION_WORDS:
        if n in text:
            pos_score, neg_score = neg_score, pos_score
            break
    score = pos_score - neg_score
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def score_category_text(text: str) -> Tuple[str, int, Dict[str, int]]:
    text = normalize_text(text)
    score_map: Dict[str, int] = {}
    best_key = "3c_home_appliance"
    best_score = -1
    for c_key, c_val in ASPECT_SCHEMA.items():
        score = sum(1 for kw in c_val["category_keywords"] if kw in text)
        score_map[c_key] = score
        if score > best_score:
            best_score = score
            best_key = c_key
    return best_key, best_score, score_map


def infer_category(text: str) -> str:
    best_key, _, _ = score_category_text(text)
    return best_key


def category_key_from_display(display_name: str) -> str:
    for k, v in ASPECT_SCHEMA.items():
        if v["display_name"] == display_name:
            return k
    display_map = {
        "3C数码与家电": "3c_home_appliance",
        "服饰鞋包与配饰": "fashion",
        "美妆个护与健康": "beauty_health",
        "家居生活与家具": "home_furniture",
        "食品饮料与生鲜": "food_fresh",
        "母婴用品与玩具": "mother_baby",
        "运动户外与器材": "sports_outdoor",
        "图书文具与办公": "books_office",
    }
    return display_map.get(display_name, "3c_home_appliance")


def infer_category_with_trace(row: pd.Series) -> Dict[str, Any]:
    goods_id = str(row.get("goods_id", ""))
    goods_name = normalize_text(row.get("goods_name", ""))
    purchase_product = normalize_text(row.get("purchase_product", ""))
    comment_text = normalize_text(row.get("comment_text_clean", ""))
    major_category = normalize_text(row.get("major_category", ""))

    merged_text = " ".join([purchase_product, goods_name, comment_text]).strip()
    key_from_text, score_from_text, score_map = score_category_text(merged_text)
    result_key = key_from_text
    rule = "text_keywords"
    if major_category:
        result_key = category_key_from_display(major_category)
        rule = "major_category"

    return {
        "goods_id": goods_id,
        "goods_name": goods_name,
        "purchase_product": purchase_product,
        "comment_text_clean": comment_text,
        "major_category": major_category,
        "category_key": result_key,
        "category_name": ASPECT_SCHEMA[result_key]["display_name"],
        "category_rule": rule,
        "category_text_score": int(score_from_text),
        "category_score_map": json.dumps(score_map, ensure_ascii=False),
    }


def extract_aspect_sentiment(text: str, category_key: str) -> List[Dict]:
    text = normalize_text(text)
    cat = ASPECT_SCHEMA.get(category_key, ASPECT_SCHEMA["3c_home_appliance"])
    result = []
    global_label = lexicon_predict(text)
    for aspect_name, keywords in cat["aspects"].items():
        hit = any(k in text for k in keywords)
        if not hit:
            continue
        local_text = text
        # 在简化场景下，方面情感用本地词典打分，未命中则退化为全局标签
        local_label = lexicon_predict(local_text)
        if local_label == "neutral":
            local_label = global_label
        result.append({"aspect": aspect_name, "sentiment": local_label})
    return result


@dataclass
class DatasetBundle:
    train_x: List[str]
    val_x: List[str]
    test_x: List[str]
    train_y: List[int]
    val_y: List[int]
    test_y: List[int]


def build_dataset(df: pd.DataFrame) -> DatasetBundle:
    df = df.copy()
    df = df[df["sentiment_label_star"].isin(LABEL_TO_ID.keys())]
    df["text"] = df["comment_text_clean"].fillna("").map(normalize_text)
    df = df[df["text"].str.len() > 0]
    df["label_id"] = df["sentiment_label_star"].map(LABEL_TO_ID)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label_id"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df["label_id"]
    )
    return DatasetBundle(
        train_x=train_df["text"].tolist(),
        val_x=val_df["text"].tolist(),
        test_x=test_df["text"].tolist(),
        train_y=train_df["label_id"].tolist(),
        val_y=val_df["label_id"].tolist(),
        test_y=test_df["label_id"].tolist(),
    )


def evaluate_predictions(y_true: List[int], y_pred: List[int], model_name: str) -> Dict:
    labels = [0, 1, 2]
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["negative", "neutral", "positive"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def build_class_weights(labels: List[int]) -> List[float]:
    counts = np.bincount(labels, minlength=3).astype(float)
    total = counts.sum() if counts.sum() > 0 else 1.0
    weights = []
    for c in counts:
        if c <= 0:
            weights.append(1.0)
        else:
            weights.append(total / (len(counts) * c))
    return weights


def save_confusion_matrix_plot(cm: List[List[int]], labels: List[str], path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_metric_plot(metrics_df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 4.5))
    plot_df = metrics_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].copy()
    plot_df = plot_df.melt(id_vars="model", var_name="metric", value_name="score")
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Model Comparison Metrics")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_f1_ranking_plot(metrics_df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 4.5))
    rank_df = metrics_df[["model", "f1_macro"]].sort_values(by="f1_macro", ascending=True)
    sns.barplot(data=rank_df, x="f1_macro", y="model", orient="h", color="#4C72B0")
    plt.xlim(0, 1)
    plt.title("F1 Macro Ranking")
    plt.xlabel("f1_macro")
    plt.ylabel("model")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_best_model_artifact(
    best_model: str,
    metrics_df: pd.DataFrame,
    output_dir: str,
    model_errors: Dict[str, str],
    model_artifacts: Dict[str, Dict[str, Any]],
) -> str:
    artifact = {
        "best_model": best_model,
        "selected_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "metrics": metrics_df.to_dict(orient="records"),
        "model_errors": model_errors,
        "model_artifacts": model_artifacts,
        "best_model_artifact": model_artifacts.get(best_model, {}),
    }
    path = os.path.join(output_dir, "best_model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return path


def run_lexicon_model(data: DatasetBundle, save_path: Optional[str] = None) -> Tuple[Dict, List[int], Optional[str]]:
    pred = [LABEL_TO_ID[lexicon_predict(t)] for t in data.test_x]
    saved = None
    if save_path:
        lexicon_artifact = {
            "positive_words": sorted(list(POSITIVE_WORDS)),
            "negative_words": sorted(list(NEGATIVE_WORDS)),
            "negation_words": sorted(list(NEGATION_WORDS)),
            "label_to_id": LABEL_TO_ID,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(lexicon_artifact, f, ensure_ascii=False, indent=2)
        saved = save_path
    return evaluate_predictions(data.test_y, pred, "lexicon"), pred, saved


def run_textcnn_model(
    data: DatasetBundle,
    epochs: int = 6,
    batch_size: int = 32,
    max_len: int = 128,
    save_path: Optional[str] = None,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError("TextCNN需要安装PyTorch。") from e

    class CharDataset(Dataset):
        def __init__(self, texts, labels, vocab, seq_len):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.seq_len = seq_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            ids = [self.vocab.get(ch, 1) for ch in text[: self.seq_len]]
            if len(ids) < self.seq_len:
                ids += [0] * (self.seq_len - len(ids))
            return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, emb_dim=128, num_classes=3, kernel_sizes=(3, 4, 5), num_filters=64):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        def forward(self, x):
            x = self.emb(x).transpose(1, 2)
            feats = [torch.max(torch.relu(conv(x)), dim=2).values for conv in self.convs]
            x = torch.cat(feats, dim=1)
            x = self.dropout(x)
            return self.fc(x)

    all_chars = "".join(data.train_x)
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in all_chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    train_ds = CharDataset(data.train_x, data.train_y, vocab, max_len)
    val_ds = CharDataset(data.val_x, data.val_y, vocab, max_len)
    test_ds = CharDataset(data.test_x, data.test_y, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = torch.tensor(build_class_weights(data.train_y), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val_f1 = -1.0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                val_pred.extend(pred)
                val_true.extend(y.numpy().tolist())
        f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_pred = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            test_pred.extend(pred)

    saved = None
    if save_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "max_len": max_len,
                "num_classes": 3,
                "label_to_id": LABEL_TO_ID,
            },
            save_path,
        )
        saved = save_path

    return evaluate_predictions(data.test_y, test_pred, "textcnn"), test_pred, saved


def run_bilstm_model(
    data: DatasetBundle,
    epochs: int = 6,
    batch_size: int = 32,
    max_len: int = 128,
    save_path: Optional[str] = None,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError("BiLSTM需要安装PyTorch。") from e

    class CharDataset(Dataset):
        def __init__(self, texts, labels, vocab, seq_len):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.seq_len = seq_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            ids = [self.vocab.get(ch, 1) for ch in text[: self.seq_len]]
            if len(ids) < self.seq_len:
                ids += [0] * (self.seq_len - len(ids))
            return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, emb_dim=128, hidden=128, num_classes=3):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, bidirectional=True, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden * 2, num_classes)

        def forward(self, x):
            x = self.emb(x)
            out, _ = self.lstm(x)
            feat = out[:, -1, :]
            feat = self.dropout(feat)
            return self.fc(feat)

    all_chars = "".join(data.train_x)
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch in all_chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    train_ds = CharDataset(data.train_x, data.train_y, vocab, max_len)
    val_ds = CharDataset(data.val_x, data.val_y, vocab, max_len)
    test_ds = CharDataset(data.test_x, data.test_y, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = torch.tensor(build_class_weights(data.train_y), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val_f1 = -1.0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                val_pred.extend(pred)
                val_true.extend(y.numpy().tolist())
        f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_pred = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            test_pred.extend(pred)

    saved = None
    if save_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "max_len": max_len,
                "num_classes": 3,
                "label_to_id": LABEL_TO_ID,
            },
            save_path,
        )
        saved = save_path

    return evaluate_predictions(data.test_y, test_pred, "bilstm"), test_pred, saved


def run_bert_model(
    data: DatasetBundle,
    model_name: str = os.path.join("BERT_jd_sentiment_model"),
    epochs: int = 2,
    batch_size: int = 16,
    max_len: int = 128,
    save_path: Optional[str] = None,
):
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError("BERT需要安装transformers和PyTorch。") from e

    class BertDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, seq_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.seq_len = seq_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    model_path = model_name if os.path.exists(model_name) else "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    train_ds = BertDataset(data.train_x, data.train_y, tokenizer, max_len)
    val_ds = BertDataset(data.val_x, data.val_y, tokenizer, max_len)
    test_ds = BertDataset(data.test_x, data.test_y, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    class_weights = torch.tensor(build_class_weights(data.train_y), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_state = None
    best_val_f1 = -1.0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").numpy().tolist()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                val_pred.extend(pred)
                val_true.extend(labels)
        f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            test_pred.extend(pred)

    saved = None
    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        saved = save_path

    return evaluate_predictions(data.test_y, test_pred, "bert"), test_pred, saved


def run_aspect_analysis(df: pd.DataFrame, pred_labels: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    comp_rows = []
    trace_rows = []
    for idx, row in df.iterrows():
        trace = infer_category_with_trace(row)
        text = trace["comment_text_clean"]
        goods_id = trace["goods_id"]
        category_key = trace["category_key"]
        aspects = extract_aspect_sentiment(text, category_key)
        global_label = row.get("sentiment_label_star", "neutral")
        if pred_labels is not None and idx < len(pred_labels):
            global_label = pred_labels[idx]
        if not aspects:
            aspects = [{"aspect": "未命中方面词", "sentiment": global_label}]
        for a in aspects:
            rows.append(
                {
                    "goods_id": goods_id,
                    "goods_name": trace["goods_name"],
                    "purchase_product": trace["purchase_product"],
                    "major_category": trace["major_category"],
                    "category_key": category_key,
                    "category_name": trace["category_name"],
                    "category_rule": trace["category_rule"],
                    "category_text_score": trace["category_text_score"],
                    "category_score_map": trace["category_score_map"],
                    "aspect": a["aspect"],
                    "aspect_sentiment": a["sentiment"],
                    "global_sentiment": global_label,
                    "comment_text_clean": text,
                    "aspect_keywords": ",".join(ASPECT_SCHEMA[category_key]["aspects"].get(a["aspect"], [])),
                }
            )
        trace_rows.append(trace)

    aspect_df = pd.DataFrame(rows)
    trace_df = pd.DataFrame(trace_rows)
    if aspect_df.empty:
        return aspect_df, pd.DataFrame(), trace_df
    grp = (
        aspect_df.groupby(["goods_id", "category_name", "aspect", "aspect_sentiment"])
        .size()
        .reset_index(name="count")
    )
    comp_rows = grp.to_dict(orient="records")
    return aspect_df, pd.DataFrame(comp_rows), trace_df


def main():
    parser = argparse.ArgumentParser(description="Run sentiment models and aspect-level analysis.")
    parser.add_argument(
        "--input-csv",
        default=os.path.join("goods_cleaned", "datasets", "all_comments_clean.csv"),
        help="Cleaned dataset csv path.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("analysis_outputs"),
        help="Output directory for metrics and aspect analysis.",
    )
    parser.add_argument(
        "--models",
        default="lexicon,textcnn,bilstm,bert",
        help="Comma-separated: lexicon,textcnn,bilstm,bert",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Epochs for TextCNN/BiLSTM")
    parser.add_argument("--bert-epochs", type=int, default=2, help="Epochs for BERT")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for TextCNN/BiLSTM")
    parser.add_argument("--bert-batch-size", type=int, default=16, help="Batch size for BERT")
    args = parser.parse_args()

    set_seed()
    ensure_dir(args.output_dir)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input dataset not found: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    data = build_dataset(df)

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    metrics = []
    test_pred_by_model: Dict[str, List[int]] = {}
    model_errors: Dict[str, str] = {}
    model_artifacts: Dict[str, Dict[str, Any]] = {}

    model_store_dir = os.path.join(args.output_dir, "saved_models")
    ensure_dir(model_store_dir)

    if "lexicon" in model_list:
        try:
            lexicon_path = os.path.join(model_store_dir, "lexicon_rules.json")
            m, pred, saved = run_lexicon_model(data, save_path=lexicon_path)
            metrics.append(m)
            test_pred_by_model["lexicon"] = pred
            model_artifacts["lexicon"] = {"type": "rules", "path": saved}
        except Exception as e:
            model_errors["lexicon"] = str(e)

    if "textcnn" in model_list:
        try:
            textcnn_path = os.path.join(model_store_dir, "textcnn.pt")
            m, pred, saved = run_textcnn_model(
                data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_path=textcnn_path,
            )
            metrics.append(m)
            test_pred_by_model["textcnn"] = pred
            model_artifacts["textcnn"] = {"type": "torch_checkpoint", "path": saved}
        except Exception as e:
            model_errors["textcnn"] = str(e)

    if "bilstm" in model_list:
        try:
            bilstm_path = os.path.join(model_store_dir, "bilstm.pt")
            m, pred, saved = run_bilstm_model(
                data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_path=bilstm_path,
            )
            metrics.append(m)
            test_pred_by_model["bilstm"] = pred
            model_artifacts["bilstm"] = {"type": "torch_checkpoint", "path": saved}
        except Exception as e:
            model_errors["bilstm"] = str(e)

    if "bert" in model_list:
        try:
            bert_path = os.path.join(model_store_dir, "bert")
            ensure_dir(bert_path)
            m, pred, saved = run_bert_model(
                data,
                epochs=args.bert_epochs,
                batch_size=args.bert_batch_size,
                save_path=bert_path,
            )
            metrics.append(m)
            test_pred_by_model["bert"] = pred
            model_artifacts["bert"] = {"type": "transformers_pretrained", "path": saved}
        except Exception as e:
            model_errors["bert"] = str(e)

    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by="f1_macro", ascending=False)
    metrics_path = os.path.join(args.output_dir, "model_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    report_path = os.path.join(args.output_dir, "model_reports.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"generated_at: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n\n")
        for m in metrics:
            f.write(f"== {m['model']} ==\n")
            f.write(
                f"accuracy={m['accuracy']:.4f}, precision_macro={m['precision_macro']:.4f}, "
                f"recall_macro={m['recall_macro']:.4f}, f1_macro={m['f1_macro']:.4f}\n"
            )
            f.write(m["report"])
            f.write("\n\n")
        if model_errors:
            f.write("== model_errors ==\n")
            for name, err in model_errors.items():
                f.write(f"{name}: {err}\n")
            f.write("\n")

    best_model = metrics_df.iloc[0]["model"] if not metrics_df.empty else "lexicon"

    cm_dir = os.path.join(args.output_dir, "confusion_matrices")
    ensure_dir(cm_dir)
    cm_paths: Dict[str, str] = {}
    for m in metrics:
        if "confusion_matrix" not in m:
            continue
        model_name = str(m["model"])
        model_cm_path = os.path.join(cm_dir, f"confusion_matrix_{model_name}.png")
        save_confusion_matrix_plot(
            m["confusion_matrix"],
            ["negative", "neutral", "positive"],
            model_cm_path,
            f"Confusion Matrix - {model_name}",
        )
        cm_paths[model_name] = model_cm_path

    metric_plot_path = os.path.join(args.output_dir, "model_comparison_metrics.png")
    f1_rank_plot_path = os.path.join(args.output_dir, "model_f1_ranking.png")
    if not metrics_df.empty:
        save_metric_plot(metrics_df, metric_plot_path)
        save_f1_ranking_plot(metrics_df, f1_rank_plot_path)

    best_model_artifact_path = save_best_model_artifact(
        best_model,
        metrics_df,
        args.output_dir,
        model_errors,
        model_artifacts,
    )

    full_df = df.copy()
    full_df = full_df[full_df["comment_text_clean"].fillna("").map(normalize_text).str.len() > 0].reset_index(drop=True)
    # 方面分析在全量数据上执行，优先使用清洗阶段写入的 major_category
    aspect_df, aspect_summary_df, category_trace_df = run_aspect_analysis(full_df)
    aspect_path = os.path.join(args.output_dir, "aspect_sentiment_records.csv")
    aspect_summary_path = os.path.join(args.output_dir, "aspect_sentiment_summary.csv")
    category_trace_path = os.path.join(args.output_dir, "category_assignment_trace.csv")
    category_summary_path = os.path.join(args.output_dir, "category_assignment_summary.csv")
    aspect_df.to_csv(aspect_path, index=False, encoding="utf-8")
    aspect_summary_df.to_csv(aspect_summary_path, index=False, encoding="utf-8")
    if not category_trace_df.empty:
        category_trace_df.to_csv(category_trace_path, index=False, encoding="utf-8")
        category_summary_df = (
            category_trace_df.groupby(["goods_id", "goods_name", "major_category", "category_key", "category_name", "category_rule"])
            .size()
            .reset_index(name="comment_count")
            .sort_values(["major_category", "goods_id"])
        )
    else:
        category_summary_df = pd.DataFrame()
    category_summary_df.to_csv(category_summary_path, index=False, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "input_csv": args.input_csv,
        "models": model_list,
        "best_model_by_f1_macro": best_model,
        "model_errors": model_errors,
        "model_artifacts": model_artifacts,
        "outputs": {
            "model_comparison_csv": metrics_path,
            "model_reports_txt": report_path,
            "confusion_matrices": cm_paths,
            "model_comparison_metrics_png": metric_plot_path,
            "model_f1_ranking_png": f1_rank_plot_path,
            "best_model_json": best_model_artifact_path,
            "saved_models_dir": model_store_dir,
            "aspect_sentiment_records_csv": aspect_path,
            "aspect_sentiment_summary_csv": aspect_summary_path,
            "category_assignment_trace_csv": category_trace_path,
            "category_assignment_summary_csv": category_summary_path,
        },
    }
    with open(os.path.join(args.output_dir, "analysis_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[DONE] model metrics:", metrics_path)
    print("[DONE] model reports:", report_path)
    print("[DONE] confusion matrices:", cm_dir)
    print("[DONE] model comparison chart:", metric_plot_path)
    print("[DONE] f1 ranking chart:", f1_rank_plot_path)
    print("[DONE] best model artifact:", best_model_artifact_path)
    print("[DONE] saved models dir:", model_store_dir)
    print("[DONE] aspect records:", aspect_path)
    print("[DONE] aspect summary:", aspect_summary_path)
    print("[DONE] category trace:", category_trace_path)
    print("[DONE] category summary:", category_summary_path)
    print("[DONE] best model:", best_model)
    if model_errors:
        print("[WARN] model errors:", model_errors)


if __name__ == "__main__":
    main()
