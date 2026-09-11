#!/usr/bin/env python3
"""
Seal Sequence Recognition Submission
Required Python Version: Python >= 3.10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

# ============================================================
# 1. Team & Submission Configuration
# ============================================================
TEAM_NAME = "optimistic_prime"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "best_model.pt"

NUM_POSITIONS = 7
NUM_CLASSES = 10
IMG_SIZE = (256, 256)

# ============================================================
# 2. Model Architecture
# ============================================================
class SeqRecognizer(nn.Module):
    def __init__(self, num_positions: int = NUM_POSITIONS, num_classes: int = NUM_CLASSES):
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(x)
        return [head(feats) for head in self.heads]


# ============================================================
# 3. CLI Parser (Supports both -input-dir and --input-dir)
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Submission Script for Seal Number Recognition."
    )
    parser.add_argument(
        "-input-dir",
        "--input-dir",
        type=Path,
        required=True,
        dest="input_dir",
        help="Input directory containing PNG images to evaluate.",
    )
    parser.add_argument(
        "-output-dir",
        "--output-dir",
        type=Path,
        required=True,
        dest="output_dir",
        help="Output directory where the team_name.csv will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    return parser.parse_args()


# ============================================================
# 4. Main Sequential Evaluation Pipeline
# ============================================================
def main() -> None:
    args = parse_args()

    # Validate directories
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {args.checkpoint}\n"
            f"Ensure best_model.pt is placed in the same folder as main.py."
        )

    # Prepare output path: <output-dir>/<TEAM_NAME>.csv
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = args.output_dir / f"{TEAM_NAME}.csv"

    # Device: GPU if available, else CPU (Rule 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Collect all PNG images (Rule 3)
    image_paths = sorted(
        [p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )
    if not image_paths:
        raise ValueError(f"No PNG images found in {args.input_dir}")

    print(f"Processing {len(image_paths)} PNG images sequentially...")

    # Load Model
    model = SeqRecognizer(num_positions=NUM_POSITIONS, num_classes=NUM_CLASSES)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Pre-computed tensor for digit decoding: [1000000, 100000, ..., 1]
    divisors_tensor = torch.tensor(
        10 ** np.arange(NUM_POSITIONS - 1, -1, -1),
        dtype=torch.int64,
        device=device,
    )

    filenames: list[str] = []
    predicted_numbers: list[int] = []

    # Sequential image-by-image inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = perf_counter()

    with torch.inference_mode():
        for path in image_paths:
            # 1. Load single image
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 2. Preprocess single image
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

            # Convert to tensor: [1, 1, H, W] normalized to [0, 1]
            tensor = (
                torch.from_numpy(img)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .div_(255.0)
                .to(device, non_blocking=True)
            )

            # 3. Model forward pass on single image
            logits = model(tensor)                 # 7 x [1, 10]
            stacked_logits = torch.stack(logits, dim=1)  # [1, 7, 10]
            pred_digits = stacked_logits.argmax(dim=2)   # [1, 7]

            # Reconstruct 7-digit integer
            pred_num = int((pred_digits * divisors_tensor).sum().item())

            filenames.append(path.name)
            predicted_numbers.append(pred_num)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = perf_counter() - t_start

    # 4. Save results to CSV using ';' delimiter
    df_out = pd.DataFrame({
        "filename": filenames,
        "number": predicted_numbers,
    })
    df_out.to_csv(output_csv_path, sep=";", index=False)

    print(f"Successfully processed {len(filenames)} images sequentially in {total_time:.3f} s ({len(filenames)/total_time:.1f} FPS)")
    print(f"Saved results to: {output_csv_path}")


if __name__ == "__main__":
    main()