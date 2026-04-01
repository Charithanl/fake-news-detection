import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABEL_TO_ID = {"FAKE": 0, "REAL": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DistilBERT fake news classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing fake.csv and true.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("model"), help="Directory used to save the trained model")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model checkpoint")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum token length per example")
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional number of shuffled rows to train on")
    parser.add_argument("--local-files-only", action="store_true", help="Load tokenizer/model from local Hugging Face cache only")
    return parser.parse_args()


def build_text_frame(frame: pd.DataFrame, label_name: str) -> pd.DataFrame:
    title = frame["title"].fillna("").astype(str).str.strip()
    text = frame["text"].fillna("").astype(str).str.strip()
    combined = (title + ". " + text).str.strip(". ").str.strip()
    cleaned = combined.replace("", pd.NA).dropna()
    return pd.DataFrame(
        {
            "text": cleaned.tolist(),
            "label": [LABEL_TO_ID[label_name]] * len(cleaned),
        }
    )


def load_dataset(data_dir: Path, sample_size: int | None, seed: int) -> pd.DataFrame:
    fake_path = data_dir / "fake.csv"
    true_path = data_dir / "true.csv"

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    data = pd.concat(
        [
            build_text_frame(fake, "FAKE"),
            build_text_frame(true, "REAL"),
        ],
        ignore_index=True,
    )
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if sample_size is not None:
        data = data.head(sample_size).copy()

    return data


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(eval_pred) -> dict[str, float]:
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_labels = predictions.argmax(axis=1)
    accuracy = accuracy_score(eval_pred.label_ids, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_pred.label_ids,
        predicted_labels,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data = load_dataset(args.data_dir, args.sample_size, args.seed)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["label"].tolist(),
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    train_encodings = tokenizer(train_texts, truncation=True, max_length=args.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=args.max_length)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=args.local_files_only,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = {
        "dataset_size": len(data),
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "label_to_id": LABEL_TO_ID,
        "max_length": args.max_length,
        "model_name": args.model_name,
        "metrics": {key: float(value) for key, value in eval_metrics.items()},
    }
    metadata_path = args.output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
