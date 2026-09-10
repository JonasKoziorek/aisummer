# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import cv2
# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================
# 1. Paths & Configuration (Fill in your final test set paths)
# ============================================================
# Path to your final test CSV with columns: "filename";"number"
TEST_CSV_PATH = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset/test_example/test_example/gt.csv")

# Path to your final test images directory
TEST_IMAGE_DIR = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset/test_example/test_example/data")

# Path to your trained model checkpoint
MODEL_CHECKPOINT_PATH = Path.home() / "aisummer" / "aisummer" / "best_model.pt"

# Model and image settings
NUM_POSITIONS = 7
NUM_CLASSES = 10
IMG_SIZE = (256, 256)
NUM_SAMPLES = None   # Set to a number (e.g. 5) to test, or None for all images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# 2. Official Evaluation Metric (Loss in EUR)
# ============================================================
SECONDS_PER_HOUR = 3600
MINUTES_PER_HOUR = 60

@dataclass(frozen=True)
class LineConfig:
    plombs_per_hour: float = 3600.0   # Nominal line speed [plombs/hour] (D2)
    eur_per_minute: float = 0.1       # Delay penalty [EUR/min] (D3)
    technician_fee: float = 1.0       # Technician fee per error [EUR] (D5)
    downtime_per_error: float = 2.0   # Line downtime per error [min] (D6)
    batch_size: int = 50              # Batch size (D7)

    @property
    def time_budget_s(self) -> float:
        return SECONDS_PER_HOUR * self.batch_size / self.plombs_per_hour


@dataclass(frozen=True)
class ScoreBreakdown:
    error_cost: float
    overtime_s: float
    overtime_cost: float

    @property
    def total(self) -> float:
        return self.error_cost + self.overtime_cost


def evaluate(errors: int, processing_time_s: float, config: LineConfig) -> ScoreBreakdown:
    # 1) Cost from errors: errors * D6 * (D2/60 * D3 + D5)
    cost_per_error = config.downtime_per_error * (
        (config.plombs_per_hour / MINUTES_PER_HOUR) * config.eur_per_minute + config.technician_fee
    )
    error_cost = errors * cost_per_error

    # 2) Cost from overtime: D2/3600 * D3 * MAX(0, processing_time - budget)
    overtime_s = max(0.0, processing_time_s - config.time_budget_s)
    overtime_cost = (
        (config.plombs_per_hour / SECONDS_PER_HOUR) * config.eur_per_minute * overtime_s
    )

    return ScoreBreakdown(
        error_cost=error_cost,
        overtime_s=overtime_s,
        overtime_cost=overtime_cost,
    )

# ============================================================
# 3. Model Architecture & Preprocessing
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
        return [head(feats) for head in self.heads]


def predict(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    logits = model(x)
    return [F.softmax(l, dim=1) for l in logits]


def load_and_preprocess(img_path: Path) -> torch.Tensor | None:
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
    return image_tensor.unsqueeze(0)  # [1, 1, H, W]


DIVISORS = 10 ** np.arange(NUM_POSITIONS - 1, -1, -1)

def number_to_digits(number: int) -> np.ndarray:
    return (number // DIVISORS) % 10

def digits_to_number(digits: list[int] | np.ndarray) -> int:
    return int(np.sum(np.array(digits) * DIVISORS))

# ============================================================
# 4. Load Model
# ============================================================
model = SeqRecognizer(num_positions=NUM_POSITIONS, num_classes=NUM_CLASSES)

if not MODEL_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found at: {MODEL_CHECKPOINT_PATH}")

state_dict = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
model.load_state_dict(state_dict)
print(f"Loaded weights from {MODEL_CHECKPOINT_PATH}")

model.to(DEVICE)
model.eval()

# ============================================================
# 5. Run Pure Inference (Timed)
# ============================================================
df = pd.read_csv(TEST_CSV_PATH, delimiter=";")
if NUM_SAMPLES is not None:
    df = df.head(NUM_SAMPLES)

print(f"Evaluating {len(df)} images from: {TEST_IMAGE_DIR}")

results = []

# Synchronize GPU for accurate timing
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
t_start = perf_counter()

with torch.no_grad():
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\rImage {i}/{len(df)}", end="", flush=True)

        filename = str(row["filename"]).strip()
        true_num = int(row["number"])
        true_digits = number_to_digits(true_num)

        img_path = TEST_IMAGE_DIR / filename
        if not img_path.exists():
            print(f"\nWarning: {filename} not found, skipping.")
            continue

        tensor = load_and_preprocess(img_path)
        if tensor is None:
            print(f"\nWarning: could not open {filename}, skipping.")
            continue

        tensor = tensor.to(DEVICE)
        probs_list = predict(model, tensor)

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

if DEVICE.type == "cuda":
    torch.cuda.synchronize()
t_end = perf_counter()
processing_time_s = t_end - t_start
print()  # Move past the \r carriage return

# ============================================================
# 6. Evaluation Metrics & EUR Score Calculation
# ============================================================
results_df = pd.DataFrame(results)
total_evaluated = len(results_df)

if total_evaluated == 0:
    print("No images were evaluated.")
else:
    errors = int((~results_df["exact_match"]).sum())
    exact_accuracy = (results_df["exact_match"].mean()) * 100.0

    # Line configuration set to the actual number of evaluated images
    line_config = LineConfig(
        plombs_per_hour=3600.0,
        eur_per_minute=0.1,
        technician_fee=1.0,
        downtime_per_error=2.0,
        batch_size=total_evaluated,
    )

    score = evaluate(
        errors=errors,
        processing_time_s=processing_time_s,
        config=line_config,
    )

    # Per-position digit accuracy
    true_mat = np.stack(results_df["true_digits"].values)
    pred_mat = np.stack(results_df["pred_digits"].values)
    per_position_acc = (true_mat == pred_mat).mean(axis=0) * 100.0

    print("\n" + "=" * 60)
    print("                 FINAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Evaluated Images   : {total_evaluated}")
    print(f"Exact Sequence Accuracy  : {exact_accuracy:.2f}% ({total_evaluated - errors}/{total_evaluated})")
    print(f"Total Seal Errors (C9)   : {errors}")
    print(f"Processing Time (B9)     : {processing_time_s:.3f} s ({total_evaluated/processing_time_s:.1f} images/s)")
    print(f"Time Budget Allowed      : {line_config.time_budget_s:.3f} s")
    print("-" * 60)
    print(f"Error Penalties Cost     : {score.error_cost:.2f} EUR")
    print(f"Overtime Delay           : {score.overtime_s:.3f} s")
    print(f"Overtime Penalty Cost    : {score.overtime_cost:.2f}")
# %%
