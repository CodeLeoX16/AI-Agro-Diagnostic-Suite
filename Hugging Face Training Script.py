import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# 1. Configuration
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
DATA_DIR = "./dataset"   # Path where your train/ and val/ folders live
OUTPUT_DIR = "./custom_vit_crop_model"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

# 2. Load Dataset using ImageFolder format
dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
labels = dataset["train"].features["label"].names
num_labels = len(labels)
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

print(f"Loaded {num_labels} crop disease classes.")

# 3. Preprocessing & Image Augmentation
processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)

def transform(example_batch):
    # Converts PIL images into normalized torch tensors expected by ViT
    inputs = processor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["label"] = example_batch["label"]
    return inputs

prepared_ds = dataset.with_transform(transform)

# 4. Load Pre-trained ViT with a New Custom Classifier Head
model = ViTForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 5. Metrics Computation
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=preds, references=labels)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }

# 6. Training Configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
    report_to="none"
)

# 7. Execute Training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=processor,
)

print("🚀 Starting training...")
trainer.train()

# 8. Save the Final Model and Processor
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved successfully to {OUTPUT_DIR}")

# output getting
# In your app.py:
MODEL_PATH = "./custom_vit_crop_model"
viz_pipe = pipeline("image-classification", model=MODEL_PATH)