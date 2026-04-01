import argparse
import json
from pathlib import Path

import pandas as pd

from predict_distilbert import load_model, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly test the saved fake-news model on rows from fake.csv and true.csv.")
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Directory containing the trained model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing fake.csv and true.csv")
    parser.add_argument("--samples-per-class", type=int, default=5, help="How many rows to test from each class")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    return parser.parse_args()


def row_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    return "\n\n".join(part for part in [title, text] if part)


def evaluate_frame(frame: pd.DataFrame, expected_label: str, tokenizer, model, max_length: int) -> list[dict]:
    results = []
    for _, row in frame.iterrows():
        article = row_text(row)
        prediction = predict(article, tokenizer, model, max_length)
        results.append(
            {
                "expected": expected_label,
                "predicted": prediction["label"],
                "confidence": prediction["confidence"],
                "title": str(row.get("title", "")),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.model_dir)

    fake_df = pd.read_csv(args.data_dir / "fake.csv").head(args.samples_per_class)
    true_df = pd.read_csv(args.data_dir / "true.csv").head(args.samples_per_class)

    results = evaluate_frame(fake_df, "FAKE", tokenizer, model, args.max_length)
    results.extend(evaluate_frame(true_df, "REAL", tokenizer, model, args.max_length))

    correct = sum(1 for item in results if item["expected"] == item["predicted"])
    summary = {
        "tested": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
