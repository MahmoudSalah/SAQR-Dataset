"""
train_gender_classifier.py
---------------------------
Binary gender classification from Arabic handwritten line images.
Labels come from the 'category' column in manifest.csv (boys / girls).

Architecture: Fine-tuned ViT-Base (google/vit-base-patch16-224)
              - Proven strong on image classification tasks
              - Lightweight fine-tuning with frozen backbone

Run with:
    python3 train_gender_classifier.py
"""
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import evaluate

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score, matthews_corrcoef

# ── Configuration ──────────────────────────────────────────────────────────────
MANIFEST_PATH = "/home/salah/Downloads/clean_crops_curated/manifest.csv"
IMAGE_DIR     = "/home/salah/Downloads/clean_crops_curated/hw"
MODEL_NAME    = "google/vit-base-patch16-224"
OUTPUT_DIR    = "/home/salah/.gemini/antigravity/scratch/Arabic_Dataset/gender-classifier-vit"
BATCH_SIZE    = 32
EPOCHS        = 25
SEED          = 42
LABEL2ID      = {"boys": 0, "girls": 1}
ID2LABEL      = {0: "boys", 1: "girls"}
# ───────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-5, 5), fill=255),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


class GenderDataset(Dataset):
    def __init__(self, df, processor, is_train=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMAGE_DIR, os.path.basename(row['hw_path']))
        label = LABEL2ID[row['category']]

        image = Image.open(img_path).convert("RGB")

        if self.is_train:
            image = train_transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "balanced_accuracy": bacc, "f1_macro": f1}


def train():
    df = pd.read_csv(MANIFEST_PATH)

    # Normalize category column
    df['category'] = df['category'].str.strip().str.lower()
    df = df[df['category'].isin(['boys', 'girls'])].reset_index(drop=True)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Train gender dist:", train_df['category'].value_counts().to_dict())
    print("Test  gender dist:", test_df['category'].value_counts().to_dict())

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    train_dataset = GenderDataset(train_df, processor, is_train=True)
    val_dataset   = GenderDataset(val_df,   processor, is_train=False)
    test_dataset  = GenderDataset(test_df,  processor, is_train=False)

    # Compute class weights for training set
    n_boys = (train_df['category'] == 'boys').sum()
    n_girls = (train_df['category'] == 'girls').sum()
    total = n_boys + n_girls
    w_boys = total / (2.0 * n_boys)
    w_girls = total / (2.0 * n_girls)
    class_weights = torch.tensor([w_boys, w_girls], dtype=torch.float)
    print(f"Class weights: boys={w_boys:.4f}, girls={w_girls:.4f}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        fp16=True,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        report_to="none",
        save_total_limit=3,
        logging_steps=50,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    print("\nStarting gender classification training...")
    trainer.train()

    # ── Final Evaluation on Test Set ──────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    preds  = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro')

    print("\n" + "="*40)
    print("  GENDER CLASSIFICATION — TEST RESULTS")
    print("="*40)
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 Macro:  {f1:.4f}")
    print("="*40)
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=["Boys", "Girls"]))

    # Save predictions
    pred_labels = [ID2LABEL[p] for p in preds]
    true_labels = [ID2LABEL[l] for l in labels]
    pd.DataFrame({'true': true_labels, 'pred': pred_labels}).to_csv(
        "gender_results.csv", index=False
    )
    print("Saved to gender_results.csv")
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
