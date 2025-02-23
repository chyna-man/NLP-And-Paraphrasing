import csv
import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_paraphrase_data(filepath):
    texts1, texts2, labels = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # Skip header if present
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0])
            sent1 = row[1]
            sent2 = row[2]
            texts1.append(sent1)
            texts2.append(sent2)
            labels.append(label)
    return texts1, texts2, labels

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length", 
        truncation=True,
        max_length=128
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def main():
    # 1. Load data
    filepath = "D:/Projects/AI paraphrase datasets/Data/msr_paraphrase_train_cleaned.txt"
    texts1, texts2, labels = load_paraphrase_data(filepath)

    # 2. Split data
    train_texts1, test_texts1, train_texts2, test_texts2, train_labels, test_labels = train_test_split(
        texts1, texts2, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 3. Create Hugging Face datasets
    train_dataset = Dataset.from_dict({"sentence1": train_texts1, "sentence2": train_texts2, "label": train_labels})
    test_dataset = Dataset.from_dict({"sentence1": test_texts1, "sentence2": test_texts2, "label": test_labels})

    # 4. Load tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 5. Tokenize
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # 6. Rename 'label' to 'labels'
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # 7. Set format (PyTorch tensors)
    remove_cols = ["sentence1", "sentence2"]  # keep only tokenized columns + labels
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    # 8. Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 9. Training args
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # 10. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 11. Train
    trainer.train()

    # 12. Evaluate
    results = trainer.evaluate()
    print("Evaluation:", results)

if __name__ == "__main__":
    main()
