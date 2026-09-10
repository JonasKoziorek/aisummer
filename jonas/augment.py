# %%
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd

# Keep inputs on Windows:
BASE_DIR = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset")
INPUT_IMAGE_DIR = BASE_DIR / "test"
CSV_PATH = BASE_DIR / "splits" / "split_seals" / "test.csv"

# Write outputs to native WSL Linux home directory:
OUTPUT_DIR = Path.home() / "ai_summer_school_dataset"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "test_augmented"
OUTPUT_CSV_PATH = OUTPUT_DIR / "augment_test.csv"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Output directory on Linux: {OUTPUT_IMAGE_DIR}")

# %%
def median_border_value(image: np.ndarray) -> int | tuple[int, ...]:
    if image.ndim == 2:
        border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
        return int(np.median(border))

    border = np.concatenate(
        [image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]], axis=0
    )
    return tuple(int(v) for v in np.median(border, axis=0))


def random_rotate_scale(
    image: np.ndarray,
    rng: np.random.Generator,
    max_rotation_degrees: float,
    scale_range: tuple[float, float],
) -> np.ndarray:
    h, w = image.shape[:2]
    angle = rng.uniform(-max_rotation_degrees, max_rotation_degrees)
    scale = rng.uniform(scale_range[0], scale_range[1])
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    border_val = median_border_value(image)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


def add_salt_and_pepper_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    amount_range: tuple[float, float],
    salt_ratio: float,
) -> np.ndarray:
    amount = rng.uniform(amount_range[0], amount_range[1])
    if amount <= 0:
        return image

    noisy = image.copy()
    h, w = image.shape[:2]
    mask = rng.random((h, w))
    pepper_th = amount * (1.0 - salt_ratio)
    salt_th = 1.0 - (amount * salt_ratio)

    if image.ndim == 2:
        noisy[mask < pepper_th] = 0
        noisy[mask > salt_th] = 255
    else:
        noisy[mask < pepper_th, :] = 0
        noisy[mask > salt_th, :] = 255
    return noisy


def random_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    max_kernel_size: int,
) -> np.ndarray:
    if max_kernel_size < 3:
        return image
    kernel_size = int(rng.choice(list(range(3, max_kernel_size + 1, 2))))
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def add_random_rectangles(
    image: np.ndarray,
    rng: np.random.Generator,
    count_range: tuple[int, int],
    brightness_range: tuple[int, int],
) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 2 or w < 2:
        return image

    augmented = image.copy()
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    upper_h = max(1, h // 2)

    for _ in range(count):
        rect_w = int(rng.integers(max(1, w // 12), max(2, w // 3) + 1))
        rect_h = int(rng.integers(max(1, upper_h // 12), max(2, upper_h // 3) + 1))
        x1 = int(rng.integers(0, max(1, w - rect_w + 1)))
        y1 = int(rng.integers(0, max(1, upper_h - rect_h + 1)))
        x2 = min(w, x1 + rect_w)
        y2 = min(upper_h, y1 + rect_h)
        delta = int(rng.integers(brightness_range[0], brightness_range[1] + 1))

        region = augmented[y1:y2, x1:x2].astype(np.int16) + delta
        augmented[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)

    return augmented


def add_gradient_brightness(
    image: np.ndarray,
    rng: np.random.Generator,
    strength_range: tuple[int, int],
) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        return image

    yy, xx = np.mgrid[0:h, 0:w]
    start_x = rng.uniform(0, max(1, w - 1))
    start_y = rng.uniform(0, max(1, h - 1))
    direction = rng.uniform(0, 2 * np.pi)

    proj = (xx - start_x) * np.cos(direction) + (yy - start_y) * np.sin(direction)
    proj -= proj.min()
    max_proj = proj.max()
    if max_proj > 0:
        proj /= max_proj

    strength = int(rng.integers(strength_range[0], strength_range[1] + 1))
    gradient = proj * strength
    if image.ndim == 3:
        gradient = gradient[:, :, None]

    return np.clip(image.astype(np.float32) + gradient, 0, 255).astype(np.uint8)


def augment_image(image: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    augmented = image
    if rng.random() < cfg["geometry_probability"]:
        augmented = random_rotate_scale(
            augmented, rng, cfg["max_rotation_degrees"], (cfg["min_scale"], cfg["max_scale"])
        )
    if rng.random() < cfg["gradient_probability"]:
        augmented = add_gradient_brightness(
            augmented, rng, (cfg["min_gradient_strength"], cfg["max_gradient_strength"])
        )
    if rng.random() < cfg["rectangle_probability"]:
        augmented = add_random_rectangles(
            augmented,
            rng,
            (cfg["min_rectangles"], cfg["max_rectangles"]),
            (cfg["min_rectangle_brightness"], cfg["max_rectangle_brightness"]),
        )
    if rng.random() < cfg["blur_probability"]:
        augmented = random_blur(augmented, rng, cfg["max_blur_kernel_size"])
    if rng.random() < cfg["noise_probability"]:
        augmented = add_salt_and_pepper_noise(
            augmented, rng, (cfg["min_noise_amount"], cfg["max_noise_amount"]), cfg["salt_ratio"]
        )
    return augmented

# %%
# Configuration
CFG = {
    "sample_ratio": 0.15,          # 15% of images will be augmented
    "seed": 42,
    # Probabilities
    "geometry_probability": 0.8,
    "gradient_probability": 0.5,
    "rectangle_probability": 0.4,
    "blur_probability": 0.3,
    "noise_probability": 0.2,
    # Ranges
    "max_rotation_degrees": 10.0,
    "min_scale": 0.9,
    "max_scale": 1.1,
    "min_noise_amount": 0.002,
    "max_noise_amount": 0.015,
    "salt_ratio": 0.5,
    "max_blur_kernel_size": 5,
    "min_rectangles": 1,
    "max_rectangles": 4,
    "min_rectangle_brightness": -80,
    "max_rectangle_brightness": 80,
    "min_gradient_strength": -60,
    "max_gradient_strength": 60,
}

# %%
# Load test CSV
df = pd.read_csv(CSV_PATH, delimiter=";")
print(f"Loaded {len(df)} entries from {CSV_PATH}")

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(CFG["seed"])

# Determine which indices will be augmented (e.g., 15%)
num_to_augment = max(1, int(round(len(df) * CFG["sample_ratio"])))
augmented_indices = set(rng.choice(df.index, size=num_to_augment, replace=False))

print(f"Total images        : {len(df)}")
print(f"Augmenting          : {num_to_augment} ({CFG['sample_ratio']*100:.1f}%)")
print(f"Keeping clean       : {len(df) - num_to_augment}")
print(f"Saving images to    : {OUTPUT_IMAGE_DIR}")
print(f"Saving new CSV to   : {OUTPUT_CSV_PATH}")

# %%
def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Avoids WSL os.sendfile bug by doing direct binary read/write
    dst.write_bytes(src.read_bytes())


new_records = []

for idx, row in df.iterrows():
    filename = str(row["filename"]).strip()
    label = row["number"]

    src_img_path = INPUT_IMAGE_DIR / filename
    dst_img_path = OUTPUT_IMAGE_DIR / filename

    if not src_img_path.exists():
        print(f"Warning: file not found {src_img_path}, skipping.")
        continue

    dst_img_path.parent.mkdir(parents=True, exist_ok=True)

    if idx in augmented_indices:
        # Load and augment
        image = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: could not read {src_img_path}, copying original.")
            safe_copy(src_img_path, dst_img_path)
        else:
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            aug_image = augment_image(image, rng, CFG)
            cv2.imwrite(str(dst_img_path), aug_image)
    else:
        # Copy clean original image safely without shutil/sendfile
        safe_copy(src_img_path, dst_img_path)

    new_records.append({"filename": filename, "number": label})

# Save the new CSV
aug_df = pd.DataFrame(new_records)
aug_df.to_csv(OUTPUT_CSV_PATH, sep=";", index=False)

print(f"Done! Saved {len(aug_df)} images to {OUTPUT_IMAGE_DIR}")
print(f"Created {OUTPUT_CSV_PATH}")

# %%