import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import wandb
import argparse


# ============================================================
# Config — adjust paths as needed
# ============================================================
TRAIN_IMAGES_DIR = r"D:\AiSimmer2026\images\ai_summer_school_dataset\train"
TRAIN_CSV = r"D:\AiSimmer2026\images\ai_summer_school_dataset\train.csv"
VAL_IMAGES_DIR = r"D:\AiSimmer2026\images\ai_summer_school_dataset\val"  
VAL_CSV = r"D:\AiSimmer2026\images\ai_summer_school_dataset\val.csv"     
CHECKPOINT_DIR = ""
WANDB_API_KEY = ""

IMAGE_SIZE = (256, 256)
NUM_POSITIONS = 7
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-images-dir", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="digit-sequence-recognition")

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-positions", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-augs", type=int, default=9)  # весь репертуар доступен
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=15)
    parser.add_argument("--detailed-metrics-every", type=int, default=5)

    return parser.parse_args()


ARGS = parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Augmentation building blocks
# ============================================================
def median_border_value(image: np.ndarray):
    border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    return int(np.median(border))


def add_gradient_brightness(image, rng, strength_range):
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    start_x = rng.uniform(0, max(1, width - 1))
    start_y = rng.uniform(0, max(1, height - 1))
    direction = rng.uniform(0, 2 * np.pi)
    projection = (xx - start_x) * np.cos(direction) + (yy - start_y) * np.sin(direction)
    projection -= projection.min()
    max_projection = projection.max()
    if max_projection > 0:
        projection /= max_projection
    strength = int(rng.integers(strength_range[0], strength_range[1] + 1))
    gradient = projection * strength
    augmented = image.astype(np.float32) + gradient
    return np.clip(augmented, 0, 255).astype(np.uint8)


def random_rotate_scale(image, rng, max_rotation_degrees, scale_range):
    height, width = image.shape[:2]
    angle = rng.uniform(-max_rotation_degrees, max_rotation_degrees)
    scale = rng.uniform(scale_range[0], scale_range[1])
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    border_value = median_border_value(image)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def random_perspective(image, rng, max_warp_ratio):
    height, width = image.shape[:2]
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    max_dx = width * max_warp_ratio
    max_dy = height * max_warp_ratio
    dst = src + rng.uniform(-1, 1, size=src.shape) * [max_dx, max_dy]
    dst = dst.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    border_value = median_border_value(image)
    return cv2.warpPerspective(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def safe_random_translation(image, rng, max_shift_ratio):
    height, width = image.shape[:2]
    tx = rng.uniform(-width * max_shift_ratio, width * max_shift_ratio)
    ty = rng.uniform(-height * max_shift_ratio, height * max_shift_ratio)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    border_value = median_border_value(image)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def add_random_rectangles(image, rng, count_range, brightness_range):
    height, width = image.shape[:2]
    augmented = image.copy()
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    upper_half_height = max(1, height // 2)
    for _ in range(count):
        rect_width = int(rng.integers(max(1, width // 12), max(2, width // 3) + 1))
        rect_height = int(rng.integers(max(1, upper_half_height // 12), max(2, upper_half_height // 3) + 1))
        x1 = int(rng.integers(0, max(1, width - rect_width + 1)))
        y1 = int(rng.integers(0, max(1, upper_half_height - rect_height + 1)))
        x2 = min(width, x1 + rect_width)
        y2 = min(upper_half_height, y1 + rect_height)
        brightness_delta = int(rng.integers(brightness_range[0], brightness_range[1] + 1))
        region = augmented[y1:y2, x1:x2].astype(np.int16) + brightness_delta
        augmented[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
    return augmented


def add_salt_and_pepper_noise(image, rng, amount_range, salt_ratio):
    amount = rng.uniform(amount_range[0], amount_range[1])
    if amount <= 0:
        return image
    noisy = image.copy()
    height, width = image.shape[:2]
    mask = rng.random((height, width))
    pepper_threshold = amount * (1.0 - salt_ratio)
    salt_threshold = 1.0 - (amount * salt_ratio)
    noisy[mask < pepper_threshold] = 0
    noisy[mask > salt_threshold] = 255
    return noisy


def random_blur(image, rng, max_kernel_size):
    odd_kernel_sizes = list(range(3, max_kernel_size + 1, 2))
    kernel_size = int(rng.choice(odd_kernel_sizes))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def random_morphology(image, rng, kernel_size_ratio_range):
    width = image.shape[1]
    ratio = rng.uniform(kernel_size_ratio_range[0], kernel_size_ratio_range[1])
    kernel_size = max(3, int(width * ratio))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if rng.random() < 0.5:
        return cv2.erode(image, kernel, iterations=1)
    return cv2.dilate(image, kernel, iterations=1)


def add_gaussian_noise(image, rng, sigma_range):
    sigma = rng.uniform(sigma_range[0], sigma_range[1])
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ============================================================
# Pipeline: pick 0..max_augs random augmentations per image
# (0 chosen => clean image, guarantees a portion of unaugmented samples)
# ============================================================
AUGMENTATION_FUNCTIONS = {
    "gradient_brightness": lambda img, rng: add_gradient_brightness(img, rng, strength_range=(-40, 40)),
    "rotate_scale": lambda img, rng: random_rotate_scale(img, rng, max_rotation_degrees=8, scale_range=(0.9, 1.1)),
    "perspective": lambda img, rng: random_perspective(img, rng, max_warp_ratio=0.06),
    "translation": lambda img, rng: safe_random_translation(img, rng, max_shift_ratio=0.1),
    "rectangles": lambda img, rng: add_random_rectangles(img, rng, count_range=(1, 3), brightness_range=(-60, 60)),
    "salt_pepper": lambda img, rng: add_salt_and_pepper_noise(img, rng, amount_range=(0.0, 0.02), salt_ratio=0.5),
    "blur": lambda img, rng: random_blur(img, rng, max_kernel_size=5),
    "morphology": lambda img, rng: random_morphology(img, rng, kernel_size_ratio_range=(0.006, 0.015)),
    "gaussian_noise": lambda img, rng: add_gaussian_noise(img, rng, sigma_range=(80, 105)),
}


def apply_augmentation_pipeline(image, rng, max_augs):
    max_augs = min(max_augs, len(AUGMENTATION_FUNCTIONS))
    n_to_apply = rng.integers(0, max_augs + 1)  # can be 0 -> clean image
    if n_to_apply == 0:
        return image
    names = rng.choice(list(AUGMENTATION_FUNCTIONS.keys()), size=n_to_apply, replace=False)
    for name in names:
        image = AUGMENTATION_FUNCTIONS[name](image, rng)
    return image


# ============================================================
# Dataset
# ============================================================
class DigitSequenceDataset(Dataset):
    def __init__(self, csv_path, images_dir, image_size, is_train, max_augs=3, seed=None):
        self.images_dir = images_dir
        self.image_size = (image_size, image_size)
        self.is_train = is_train
        self.max_augs = max_augs
        self.rng = np.random.default_rng(seed)
        self.samples = self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                samples.append((row["filename"], row["number"]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, number_str = self.samples[idx]
        image_path = os.path.join(self.images_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Augment at native resolution, resize LAST — same order as in the visualization script
        if self.is_train:
            image = apply_augmentation_pipeline(image, self.rng, max_augs=self.max_augs)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0  # [1, H, W], normalized to [0,1]
        label_tensor = torch.tensor([int(ch) for ch in number_str], dtype=torch.long)  # [7]

        return image_tensor, label_tensor


def worker_init_fn(worker_id):
    # Reseed each worker's rng so parallel workers don't produce identical augmentations
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    seed = (torch.initial_seed() + worker_id) % (2**32)
    dataset.rng = np.random.default_rng(seed)


# ============================================================
# Model (as decided earlier: MobileNetV3-Small, 1-channel input, 7 heads)
# ============================================================
class SeqRecognizer(nn.Module):
    def __init__(self, num_positions=NUM_POSITIONS, num_classes=NUM_CLASSES):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

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
        # Always returns raw logits — required for CrossEntropyLoss
        feats = self.backbone(x)
        logits = [head(feats) for head in self.heads]  # 7 x [B, 10]
        return logits


def predict(model, x):
    """Inference-only: converts logits to probabilities. Not used during training."""
    logits = model(x)
    return [F.softmax(l, dim=1) for l in logits]


# ============================================================
# Loss and metrics
# ============================================================
def compute_loss(outputs, targets):
    # outputs: list of 7 [B, 10] logit tensors, targets: [B, 7]
    loss = 0
    for i, out in enumerate(outputs):
        loss = loss + F.cross_entropy(out, targets[:, i], label_smoothing=0.1)
    return loss


def compute_accuracy(outputs, targets):
    preds = torch.stack([out.argmax(dim=1) for out in outputs], dim=1)  # [B, 7]
    correct_chars = (preds == targets)
    per_char_acc = correct_chars.float().mean().item()
    exact_match_acc = correct_chars.all(dim=1).float().mean().item()
    return per_char_acc, exact_match_acc, preds


def compute_detailed_metrics(all_preds, all_targets, num_classes=10):
    """
    all_preds, all_targets: numpy arrays of shape [N, 7] accumulated over a full epoch.
    Returns per-position precision/recall/f1 and confusion matrices, plus an
    aggregated confusion matrix across all 7 positions combined.
    """
    num_positions = all_preds.shape[1]
    labels = list(range(num_classes))

    per_position_metrics = []
    for pos in range(num_positions):
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets[:, pos], all_preds[:, pos],
            labels=labels, average="macro", zero_division=0,
        )
        cm = confusion_matrix(all_targets[:, pos], all_preds[:, pos], labels=labels)
        per_position_metrics.append({
            "position": pos,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,  # [10, 10]
        })

    # Aggregated confusion matrix: all positions pooled together (digit confusion regardless of position)
    aggregated_cm = confusion_matrix(
        all_targets.flatten(), all_preds.flatten(), labels=labels
    )
    agg_precision, agg_recall, agg_f1, _ = precision_recall_fscore_support(
        all_targets.flatten(), all_preds.flatten(),
        labels=labels, average="macro", zero_division=0,
    )

    return {
        "per_position": per_position_metrics,
        "aggregated": {
            "precision": agg_precision,
            "recall": agg_recall,
            "f1": agg_f1,
            "confusion_matrix": aggregated_cm,  # [10, 10]
        },
    }


# ============================================================
# Train / validate loops
# ============================================================
def run_epoch(model, loader, optimizer=None, compute_detailed=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_char_acc, total_exact_acc, n_batches = 0.0, 0.0, 0.0, 0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            outputs = model(images)
            loss = compute_loss(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            char_acc, exact_acc, preds = compute_accuracy(outputs, targets)
            total_loss += loss.item()
            total_char_acc += char_acc
            total_exact_acc += exact_acc
            n_batches += 1

            if compute_detailed:
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / n_batches
    avg_char_acc = total_char_acc / n_batches
    avg_exact_acc = total_exact_acc / n_batches

    detailed = None
    if compute_detailed:
        all_preds = np.concatenate(all_preds, axis=0)      # [N, 7]
        all_targets = np.concatenate(all_targets, axis=0)  # [N, 7]
        detailed = compute_detailed_metrics(np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0), num_classes=ARGS.num_classes)

    return avg_loss, avg_char_acc, avg_exact_acc, detailed


def main():

    os.makedirs(ARGS.checkpoint_dir, exist_ok=True)

    if ARGS.wandb_api_key:
        wandb.login(key=ARGS.wandb_api_key)
    else:
        wandb.login()

    wandb.init(project=ARGS.wandb_project, config=vars(ARGS))

    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)

    wandb.init(
        project="digit-sequence-recognition",  # замените на своё название проекта
        config={
            "backbone": "mobilenet_v3_small",
            "num_positions": NUM_POSITIONS,
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "image_size": IMAGE_SIZE,
            "label_smoothing": 0.1,
        },
    )



    train_dataset = DigitSequenceDataset(ARGS.train_csv, ARGS.train_images_dir, ARGS.image_size,
                                      is_train=True, max_augs=ARGS.max_augs, seed=ARGS.seed)
    val_dataset = DigitSequenceDataset(ARGS.val_csv, ARGS.val_images_dir, ARGS.image_size,
                                    is_train=False, max_augs=0)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn, pin_memory=(DEVICE.type == "cuda"), 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )

    model = SeqRecognizer().to(DEVICE)
    wandb.watch(model, log="gradients", log_freq=100)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.features[0][0].parameters(), "lr": 1e-3},
        {"params": [p for n, p in model.backbone.named_parameters() if not n.startswith("features.0.0")], "lr": 1e-4},
        {"params": model.heads.parameters(), "lr": 1e-3},
    ])

    best_val_exact_acc = 0.0
    for epoch in range(1, ARGS.num_epochs + 1):
        train_loss, train_char_acc, train_exact_acc, _ = run_epoch(
            model, train_loader, optimizer, ARGS.label_smoothing)

        compute_detailed = (epoch % ARGS.detailed_metrics_every == 0 or epoch == ARGS.num_epochs)
        val_loss, val_char_acc, val_exact_acc, val_detailed = run_epoch(
            model, val_loader, None, ARGS.label_smoothing, compute_detailed=compute_detailed)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} char_acc={train_char_acc:.4f} exact_acc={train_exact_acc:.4f} | "
            f"val loss={val_loss:.4f} char_acc={val_char_acc:.4f} exact_acc={val_exact_acc:.4f}"
        )

        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/char_acc": train_char_acc,
            "train/exact_acc": train_exact_acc,
            "val/loss": val_loss,
            "val/char_acc": val_char_acc,
            "val/exact_acc": val_exact_acc,
            "lr/new_conv": optimizer.param_groups[0]["lr"],
            "lr/backbone": optimizer.param_groups[1]["lr"],
            "lr/heads": optimizer.param_groups[2]["lr"],
        }

        if val_detailed is not None:
            for pos_metrics in val_detailed["per_position"]:
                pos = pos_metrics["position"]
                log_dict[f"val/position_{pos}/precision"] = pos_metrics["precision"]
                log_dict[f"val/position_{pos}/recall"] = pos_metrics["recall"]
                log_dict[f"val/position_{pos}/f1"] = pos_metrics["f1"]

            log_dict["val/aggregated/precision"] = val_detailed["aggregated"]["precision"]
            log_dict["val/aggregated/recall"] = val_detailed["aggregated"]["recall"]
            log_dict["val/aggregated/f1"] = val_detailed["aggregated"]["f1"]

            class_names = [str(i) for i in range(NUM_CLASSES)]
            raw_preds = val_detailed["raw_preds"]
            raw_targets = val_detailed["raw_targets"]

            for pos in range(NUM_POSITIONS):
                log_dict[f"confusion_matrix/position_{pos}"] = wandb.plot.confusion_matrix(
                    y_true=raw_targets[:, pos].tolist(),
                    preds=raw_preds[:, pos].tolist(),
                    class_names=class_names,
                )

            log_dict["confusion_matrix/aggregated"] = wandb.plot.confusion_matrix(
                y_true=raw_targets.flatten().tolist(),
                preds=raw_preds.flatten().tolist(),
                class_names=class_names,
            )

        wandb.log(log_dict, step=epoch)





        # Periodic checkpoint
        if epoch % ARGS.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_exact_acc": val_exact_acc,
            }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"))

        # Best checkpoint
        if val_exact_acc > best_val_exact_acc:
            best_val_exact_acc = val_exact_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_exact_acc": val_exact_acc,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    wandb.finish()

if __name__ == "__main__":
    main()