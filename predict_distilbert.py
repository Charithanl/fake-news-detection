import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake news prediction with a trained DistilBERT model.")
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Directory containing the saved model")
    parser.add_argument("--title", default="", help="Optional article title")
    parser.add_argument("--text", default="", help="Article body text")
    parser.add_argument("--input-file", type=Path, default=None, help="Optional text file containing article content")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    return parser.parse_args()


def resolve_input_text(args: argparse.Namespace) -> str:
    file_text = ""
    if args.input_file is not None:
        file_text = args.input_file.read_text(encoding="utf-8").strip()

    stdin_text = ""
    if not args.text and not file_text:
        try:
            import sys

            if not sys.stdin.isatty():
                stdin_text = sys.stdin.read().strip()
        except Exception:
            stdin_text = ""

    parts = [args.title.strip(), args.text.strip(), file_text, stdin_text]
    combined = "\n\n".join(part for part in parts if part)
    if not combined:
        raise ValueError("Provide article text with --text, --input-file, or stdin.")
    return combined


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_length: int) -> dict[str, float | str]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    predicted_label = id_to_label[predicted_id]
    result = {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_id].item()),
    }

    for class_id, class_label in sorted(id_to_label.items()):
        result[f"prob_{class_label.lower()}"] = float(probabilities[class_id].item())

    return result


def main() -> None:
    args = parse_args()
    text = resolve_input_text(args)
    tokenizer, model = load_model(args.model_dir)
    result = predict(text, tokenizer, model, args.max_length)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
