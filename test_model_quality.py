import argparse
import json
import os
from typing import Dict, Optional

import pandas as pd


DEFAULT_INPUT_DIR = "analysis_outputs_full"
DEFAULT_MIN_F1 = 0.65
DEFAULT_MIN_ACCURACY = 0.70
DEFAULT_PREDICTION_FILE = "bert_aspect_predictions.csv"
DEMO_SAMPLES = [
    {
        "text": "手机电池很耐用，但屏幕有点刺眼。",
        "aspects": ["电池", "屏幕"],
    },
    {
        "text": "衣服版型不错，就是面料有点薄。",
        "aspects": ["版型", "面料"],
    },
]


def load_metrics(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "model_comparison.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型对比文件: {path}")
    return pd.read_csv(path)


def load_best_model(output_dir: str) -> Dict:
    path = os.path.join(output_dir, "best_model.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_aspect_predictions(output_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(output_dir, DEFAULT_PREDICTION_FILE)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_model_table(metrics_df: pd.DataFrame, min_f1: float, min_acc: float) -> None:
    print("\n=== 模型指标总览 ===")
    for _, row in metrics_df.iterrows():
        model = str(row.get("model", "unknown"))
        acc = float(row.get("accuracy", 0.0))
        f1 = float(row.get("f1_macro", 0.0))
        p = float(row.get("precision_macro", 0.0))
        r = float(row.get("recall_macro", 0.0))
        status = verdict(acc >= min_acc and f1 >= min_f1)
        print(f"- {model:<14} | acc={format_float(acc)} | p={format_float(p)} | r={format_float(r)} | f1={format_float(f1)} | {status}")


def print_best_model(best_model: Dict) -> None:
    if not best_model:
        print("\n=== 最佳模型信息 ===\n- 未找到 best_model.json")
        return
    print("\n=== 最佳模型信息 ===")
    print(f"- best_model: {best_model.get('best_model', 'unknown')}")
    print(f"- selected_at: {best_model.get('selected_at', 'unknown')}")


def _normalize_text_for_match(text: str) -> str:
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def print_demo_aspect_samples(aspect_df: Optional[pd.DataFrame]) -> None:
    print("\n=== 代表性句子与方面分析示例 ===")
    if aspect_df is None or aspect_df.empty:
        for sample in DEMO_SAMPLES:
            print(f"\n句子: {sample['text']}")
            for aspect in sample["aspects"]:
                print(f"- aspect: {aspect} | gold: unknown | pred: unknown")
        return

    if "text" not in aspect_df.columns or "aspect" not in aspect_df.columns:
        print("- 预测文件缺少 text/aspect 字段，无法展示示例")
        return

    normalized_text = aspect_df["text"].astype(str).map(_normalize_text_for_match)
    for sample in DEMO_SAMPLES:
        text = sample["text"]
        print(f"\n句子: {text}")
        exact = aspect_df[normalized_text == _normalize_text_for_match(text)]
        if exact.empty:
            exact = aspect_df[normalized_text.str.contains(_normalize_text_for_match(text), regex=False, na=False)]

        if exact.empty:
            print("- 在预测文件中未找到对应样本，展示预设方面示例:")
            for aspect in sample["aspects"]:
                print(f"  · aspect: {aspect} -> 需查看模型输出文件")
            continue

        cols = [c for c in ["aspect", "gold_label", "pred_label", "correct"] if c in exact.columns]
        with pd.option_context("display.max_colwidth", 50, "display.width", 140):
            print(exact[cols].head(10).to_string(index=False))


def print_aspect_predictions(aspect_df: Optional[pd.DataFrame]) -> None:
    print("\n=== Aspect-level 预测文件检查 ===")
    if aspect_df is None or aspect_df.empty:
        print("- 未找到 bert_aspect_predictions.csv，或者文件为空")
        return

    required_cols = ["aspect", "gold_label", "pred_label", "correct"]
    missing = [c for c in required_cols if c not in aspect_df.columns]
    if missing:
        print(f"- 文件存在，但缺少字段: {missing}")
        print(f"- 实际字段: {list(aspect_df.columns)}")
        return

    total = len(aspect_df)
    correct_mask = aspect_df["correct"]
    if correct_mask.dtype != bool:
        correct_mask = correct_mask.astype(str).str.lower().isin(["true", "1", "yes"])
    correct = int(correct_mask.sum())
    accuracy = correct / total if total else 0.0
    print(f"- 总样本数: {total}")
    print(f"- 预测正确: {correct}")
    print(f"- aspect准确率: {format_float(accuracy)}")

    print("\n- 前10条样本预览:")
    preview_cols = [c for c in ["text", "aspect", "gold_label", "pred_label", "correct"] if c in aspect_df.columns]
    preview = aspect_df[preview_cols].head(10)
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(preview.to_string(index=False))

    mismatch = aspect_df[~correct_mask]
    if not mismatch.empty:
        print("\n- 错误样本前5条:")
        print(mismatch[preview_cols].head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="快速检查模型是否达到预期的独立测试脚本")
    parser.add_argument("--output-dir", default=DEFAULT_INPUT_DIR, help="分析输出目录")
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1, help="期望的最低 Macro F1")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY, help="期望的最低 Accuracy")
    args = parser.parse_args()

    print("=== 模型预期测试开始 ===")
    print(f"- output_dir: {args.output_dir}")
    print(f"- min_accuracy: {args.min_accuracy}")
    print(f"- min_f1: {args.min_f1}")

    metrics_df = load_metrics(args.output_dir)
    metrics_df = metrics_df.sort_values(by="f1_macro", ascending=False)
    best_model = load_best_model(args.output_dir)
    aspect_df = load_aspect_predictions(args.output_dir)

    print_model_table(metrics_df, args.min_f1, args.min_accuracy)
    print_best_model(best_model)
    print_demo_aspect_samples(aspect_df)
    print_aspect_predictions(aspect_df)

    top = metrics_df.iloc[0]
    overall_pass = bool(float(top.get("accuracy", 0.0)) >= args.min_accuracy and float(top.get("f1_macro", 0.0)) >= args.min_f1)
    print("\n=== 总体判断 ===")
    print(f"- top_model: {top.get('model', 'unknown')}")
    print(f"- top_accuracy: {format_float(float(top.get('accuracy', 0.0)))}")
    print(f"- top_f1: {format_float(float(top.get('f1_macro', 0.0)))}")
    print(f"- result: {verdict(overall_pass)}")

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


# OUTPUT:
# === 模型预期测试开始 ===
# - output_dir: analysis_outputs_full
# - min_accuracy: 0.7
# - min_f1: 0.65

# === 模型指标总览 ===
# - textcnn_aspect | acc=0.9672 | p=0.9530 | r=0.9707 | f1=0.9606 | PASS
# - bert_aspect    | acc=0.7644 | p=0.6560 | r=0.5893 | f1=0.5747 | FAIL
# - bilstm_aspect  | acc=0.4684 | p=0.7697 | r=0.3714 | f1=0.2795 | FAIL
# - lexicon        | acc=0.2590 | p=0.3560 | r=0.3070 | f1=0.1793 | FAIL

# === 最佳模型信息 ===
# - best_model: textcnn_aspect
# - selected_at: 2026-04-29 17:41:21

# === 代表性句子与方面分析示例 ===

# 句子: 手机电池很耐用，但屏幕有点刺眼。
# - 在预测文件中未找到对应样本，展示预设方面示例:
#   · aspect: 电池 -> 需查看模型输出文件
#   · aspect: 屏幕 -> 需查看模型输出文件

# 句子: 衣服版型不错，就是面料有点薄。
# - 在预测文件中未找到对应样本，展示预设方面示例:
#   · aspect: 版型 -> 需查看模型输出文件
#   · aspect: 面料 -> 需查看模型输出文件

# === Aspect-level 预测文件检查 ===
# - 总样本数: 450
# - 预测正确: 344
# - aspect准确率: 0.7644

# - 前10条样本预览:
#                                                                                 text aspect gold_label pred_label  correct
#                          精华液质地水润轻盈，用了之后肌肤水水的，十分透亮，美白效果蛮明显的，而且熬夜后的暗沉也明显改善，毛孔细腻不少，吸收也快  质地/肤感    neutral   negative        0
#                         好用！静音，制冷也很强。装大厅里，很好用，外观简约很好看。使用方便。安装师傅也专业。真空抽足，安装专业。现在用上了很好！   外观设计   positive   positive        1
# 已经是第四次购买这个品牌的空调了，质量还不错，空调运行效果也很好，为炎热的夏天带来了舒适和清凉。高效、专业的空调安装服务，师傅们的工作态度和技术能力都让我感到非常满意。  性能/功能   negative   positive        0
#                             下单了华硕全家桶主机，对比多日才决定。玩黑神话、三角洲不卡顿，剪视频做设计也能胜任，配置契合需求，价格还在预算内  性能/功能   positive   negative        0
#                 非常惊喜，今年第一次买到飞天，买好之后隔一天就涨价了。感觉现在新茅台更柔、更甜、茅香更淡、青南瓜甜香突出风格有所优化，更加适合商务宴请。  口味/口感   positive   positive        1
#                                      一款非常棒的订书机套装，包含起钉子和钉书钉和起钉器，非常齐全，质量很好，性价比很高，订的结实。    性价比   positive   positive        1
#      酒香浓郁，入口顺滑，回甘持久～ 空杯留香，传统工艺真有料！ 基酒品质在线，喝完还想再来一杯～ 适合送礼，高端大气又体面！ 性价比高，值得回购，已列入常备清单！ 分量/性价比   positive   positive        1
#            无印良品床上四件套真的太舒服了！️面料柔软亲肤，睡感超棒～包装严实，一点没磨损，物流也超快！配送员态度友好，全程贴心服务。整体体验满分，强烈推荐！ 未命中方面词   positive   positive        1
#                                空调非常不错 ，安装师傅来的非常快，安装的也超级快，服务特别好，刚刚试了一下，制冷效果非常好 静音效果：好  性能/功能   negative   positive        0
#                      香水波萝己收到，物流快，包装完整，新鲜，个头大，口感酸甜可口，水分足，价格便宜，营养丰富，份量足，需要的的亲放心购买给个好评！  物流/包装   positive   positive        1

# - 错误样本前5条:
#                                                                                 text aspect gold_label pred_label  correct
#                          精华液质地水润轻盈，用了之后肌肤水水的，十分透亮，美白效果蛮明显的，而且熬夜后的暗沉也明显改善，毛孔细腻不少，吸收也快  质地/肤感    neutral   negative        0
# 已经是第四次购买这个品牌的空调了，质量还不错，空调运行效果也很好，为炎热的夏天带来了舒适和清凉。高效、专业的空调安装服务，师傅们的工作态度和技术能力都让我感到非常满意。  性能/功能   negative   positive        0
#                             下单了华硕全家桶主机，对比多日才决定。玩黑神话、三角洲不卡顿，剪视频做设计也能胜任，配置契合需求，价格还在预算内  性能/功能   positive   negative        0
#                                空调非常不错 ，安装师傅来的非常快，安装的也超级快，服务特别好，刚刚试了一下，制冷效果非常好 静音效果：好  性能/功能   negative   positive        0
#                                                    款式百搭不挑风格，通勤约会都能戴，质感与颜值双双在线。真的超级爱！  外观/款式    neutral   negative        0

# === 总体判断 ===
# - top_model: textcnn_aspect
# - top_accuracy: 0.9672
# - top_f1: 0.9606
# - result: PASS
