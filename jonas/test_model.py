# %%
from __future__ import annotations

from pathlib import Path
import cv2
# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

# ============================================================
# Configuration & Paths
# ============================================================
BASE_DIR = Path.home() / "ai_summer_school_dataset"

# Test CSV created in previous step
CSV_PATH = BASE_DIR / "augment_test.csv"

# Folder where the augmented/test images live
IMAGE_DIR = BASE_DIR / "test_augmented"

# Path to your saved model weights (.pt or .pth file)
# MODEL_CHECKPOINT_PATH = BASE_DIR / "best_model.pth"
MODEL_CHECKPOINT_PATH = Path.home() / "aisummer" / "aisummer" / "best_model.pt"

# Model Architecture constants
NUM_POSITIONS = 7
NUM_CLASSES = 10
IMG_SIZE = (256, 256)  # Change if your training used a different size (e.g. (128, 256))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# Model Definition
# ============================================================
class SeqRecognizer(nn.Module):
    def __init__(self, num_positions=NUM_POSITIONS, num_classes=NUM_CLASSES):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)

        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        backbone.features[0][0] = new_conv

        backbone.classifier[3] = nn.Identity()
        self.backbone = backbone

        self.heads = nn.ModuleList([
            nn.Linear(1024, num_classes) for _ in range(num_positions)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        logits = [head(feats) for head in self.heads]  # 7 x [B, 10]
        return logits


def predict(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Inference-only: converts logits to probabilities."""
    logits = model(x)
    return [F.softmax(l, dim=1) for l in logits]


# ============================================================
# Image Preprocessing Transform
# ============================================================

def load_and_preprocess(img_path: Path) -> torch.Tensor | None:
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # 1. Resize using INTER_AREA
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Convert to float tensor and normalize to [0, 1] -> shape [1, H, W]
    image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

    # 3. Add batch dimension -> shape [1, 1, H, W]
    return image_tensor.unsqueeze(0)


# ============================================================
# Helper: Convert 7-digit number to array of 7 digits and vice-versa
# ============================================================
DIVISORS = 10 ** np.arange(NUM_POSITIONS - 1, -1, -1)

def number_to_digits(number: int) -> np.ndarray:
    """e.g. 1583375 -> [1, 5, 8, 3, 3, 7, 5]"""
    return (number // DIVISORS) % 10

def digits_to_number(digits: list[int] | np.ndarray) -> int:
    """e.g. [1, 5, 8, 3, 3, 7, 5] -> 1583375"""
    return int(np.sum(np.array(digits) * DIVISORS))


# %%
# ============================================================
# Load Model & Weights
# ============================================================
model = SeqRecognizer(num_positions=NUM_POSITIONS, num_classes=NUM_CLASSES)

if MODEL_CHECKPOINT_PATH.exists():
    state_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
    # Handle if checkpoint was saved as a dict with 'model_state_dict'
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {MODEL_CHECKPOINT_PATH}")
else:
    raise Exception(f"Weights not found at {MODEL_CHECKPOINT_PATH}.")

model.to(DEVICE)
model.eval()

# %%
# ============================================================
# Run Inference on the Test Set
# ============================================================
NUM_SAMPLES = None  # <-- Set to 3, 4, etc. Set to None to run on ALL pictures
df = pd.read_csv(CSV_PATH, delimiter=";")
if NUM_SAMPLES is not None:
    df = df.head(NUM_SAMPLES)
print(f"Running evaluation on {len(df)} images from {CSV_PATH}...")

results = []

i = 1
with torch.no_grad():
    for _, row in df.iterrows():
        print(f"\rIter {i}", end="")
        filename = str(row["filename"]).strip()
        true_num = int(row["number"])
        true_digits = number_to_digits(true_num)

        # Check in test_augmented
        img_path = IMAGE_DIR / filename
        # print(f"Current Image: {img_path}")

        if not img_path.exists():
            print(f"Image not found: {filename}, skipping.")
            continue

        tensor = load_and_preprocess(img_path)
        if tensor is None:
            print(f"Failed to read image: {img_path}, skipping.")
            continue

        tensor = tensor.to(DEVICE)
        probs_list = predict(model, tensor)  # 7 tensors of shape [1, 10]
        # print(f"Prob lists: {probs_list}")

        # Extract argmax digit for each of the 7 positions
        pred_digits = [int(p.argmax(dim=1).item()) for p in probs_list]
        pred_num = digits_to_number(pred_digits)
        is_exact_match = (pred_num == true_num)

        results.append({
            "filename": filename,
            "true_number": true_num,
            "pred_number": pred_num,
            "exact_match": is_exact_match,
            "true_digits": true_digits,
            "pred_digits": pred_digits,
        })
        i+=1

# %%
# ============================================================
# Evaluation Metrics & Per-Position Breakdown
# ============================================================
results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("No images were evaluated.")
else:
    # 1. Full Sequence Accuracy (all 7 digits must match)
    exact_acc = results_df["exact_match"].mean() * 100.0
    print("\n" + "=" * 50)
    print(f"Total Evaluated Images     : {len(results_df)}")
    print(f"Exact Sequence Accuracy    : {exact_acc:.2f}% ({results_df['exact_match'].sum()}/{len(results_df)})")
    print("=" * 50)

    # 2. Per-Position Digit Accuracy
    true_mat = np.stack(results_df["true_digits"].values)  # shape: [N, 7]
    pred_mat = np.stack(results_df["pred_digits"].values)  # shape: [N, 7]

    per_position_acc = (true_mat == pred_mat).mean(axis=0) * 100.0

    print("\nAccuracy by Digit Position (0 to 6):")
    for pos, acc in enumerate(per_position_acc):
        print(f"  Position {pos} (10^{6 - pos}s): {acc:.2f}%")

    # 3. Overall Individual Digit Accuracy
    total_digit_acc = (true_mat == pred_mat).mean() * 100.0
    print(f"\nOverall Digit Accuracy     : {total_digit_acc:.2f}%")

    # 4. Save predictions to CSV
    output_predictions_path = BASE_DIR / "test_predictions.csv"
    results_df[["filename", "true_number", "pred_number", "exact_match"]].to_csv(
        output_predictions_path, sep=";", index=False
    )
    print(f"\nDetailed predictions saved to: {output_predictions_path}")
# %%
