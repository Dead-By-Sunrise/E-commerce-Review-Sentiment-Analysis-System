import argparse
import os

import pandas as pd


DEFAULT_INPUT_DIR = "analysis_outputs_full"
REQUIRED_COLUMNS = ["model", "input_format", "accuracy", "precision_macro", "recall_macro", "f1_macro", "output_artifact"]
PAPER_MODEL_ORDER = ["lexicon", "textcnn_aspect", "bilstm_aspect", "bert_aspect"]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查论文实验表格是否生成")
    parser.add_argument("--output-dir", default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()

    path = os.path.join(args.output_dir, "paper_experiment_table.csv")
    print("=== 论文实验表格检查 ===")
    print(f"- path: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少字段: {missing}, 实际字段: {list(df.columns)}")

    if "model" in df.columns:
        df["model"] = pd.Categorical(df["model"], categories=PAPER_MODEL_ORDER, ordered=True)
        df = df.sort_values(by="model")
        df = df[df["model"].notna()]

    print(df[REQUIRED_COLUMNS].to_string(index=False))


if __name__ == "__main__":
    main()


# OUTPUT:
# === 论文实验表格检查 ===
# - path: analysis_outputs_full/paper_experiment_table.csv
#          model  input_format  accuracy  precision_macro  recall_macro  f1_macro                                       output_artifact
#        lexicon          text  0.258993         0.355962      0.307010  0.179254 analysis_outputs_full/saved_models/lexicon_rules.json
# textcnn_aspect text + aspect  0.967213         0.952951      0.970712  0.960643         analysis_outputs_full/saved_models/textcnn.pt
#  bilstm_aspect text + aspect  0.468384         0.769727      0.371370  0.279522          analysis_outputs_full/saved_models/bilstm.pt
#    bert_aspect text + aspect  0.764444         0.656011      0.589322  0.574682               analysis_outputs_full/saved_models/bert