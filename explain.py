import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a DistilBERT fake news prediction with sentence ablation.")
    parser.add_argument("--model-dir", type=Path, default=Path("model"), help="Directory containing the saved model")
    parser.add_argument("--title", default="", help="Optional article title")
    parser.add_argument("--text", default="", help="Article body text")
    parser.add_argument("--input-file", type=Path, default=None, help="Optional text file containing article content")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--top-k", type=int, default=5, help="How many influential sentences to show")
    parser.add_argument("--max-sentences", type=int, default=12, help="Maximum sentences to analyze")
    return parser.parse_args()


def resolve_input_text(args: argparse.Namespace) -> str:
    file_text = ""
    if args.input_file is not None:
        file_text = args.input_file.read_text(encoding="utf-8").strip()

    stdin_text = ""
    try:
        import sys

        if not args.text and not file_text and not sys.stdin.isatty():
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


def predict_probabilities(text: str, tokenizer, model, max_length: int) -> tuple[dict[int, str], torch.Tensor]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    return id_to_label, probabilities


def split_sentences(text: str, max_sentences: int) -> list[str]:
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
    return sentences[:max_sentences]


def main() -> None:
    args = parse_args()
    text = resolve_input_text(args)
    tokenizer, model = load_model(args.model_dir)

    id_to_label, baseline_probs = predict_probabilities(text, tokenizer, model, args.max_length)
    predicted_id = int(torch.argmax(baseline_probs).item())
    predicted_label = id_to_label[predicted_id]
    baseline_confidence = float(baseline_probs[predicted_id].item())

    sentences = split_sentences(text, args.max_sentences)
    impacts: list[dict[str, float | str]] = []

    for index, sentence in enumerate(sentences):
        reduced_sentences = [value for i, value in enumerate(sentences) if i != index]
        reduced_text = " ".join(reduced_sentences).strip()
        if not reduced_text:
            continue

        _, reduced_probs = predict_probabilities(reduced_text, tokenizer, model, args.max_length)
        reduced_confidence = float(reduced_probs[predicted_id].item())
        impacts.append(
            {
                "sentence": sentence,
                "confidence_drop": baseline_confidence - reduced_confidence,
            }
        )

    impacts.sort(key=lambda item: item["confidence_drop"], reverse=True)
    output = {
        "label": predicted_label,
        "confidence": baseline_confidence,
        "top_sentences": impacts[: args.top_k],
        "analyzed_sentences": len(sentences),
        "truncated_for_explanation": len(split_sentences(text, 10_000)) > len(sentences),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
