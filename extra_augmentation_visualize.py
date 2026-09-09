import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Path to the image you want to test augmentations on
# ============================================================
image_path = r"D:\AiSimmer2026\images\ai_summer_school_dataset\train\00000.png"
FINAL_SIZE = (256, 256)  # what the network actually receives, after augmentation

# ============================================================
# Helper (needed by random_perspective for consistent borders)
# ============================================================
def median_border_value(image: np.ndarray) -> int | tuple[int, ...]:
    if image.ndim == 2:
        border = np.concatenate(
            [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
        )
        return int(np.median(border))
    border = np.concatenate(
        [image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]],
        axis=0,
    )
    return tuple(int(value) for value in np.median(border, axis=0))


# ============================================================
# Augmentations (kernel/pixel-based params scaled to image size)
# ============================================================
def random_perspective(
    image: np.ndarray,
    rng: np.random.Generator,
    max_warp_ratio: float,
) -> np.ndarray:
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
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def safe_random_translation(
    image: np.ndarray,
    rng: np.random.Generator,
    max_shift_ratio: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    max_shift_x = width * max_shift_ratio
    max_shift_y = height * max_shift_ratio
    tx = rng.uniform(-max_shift_x, max_shift_x)
    ty = rng.uniform(-max_shift_y, max_shift_y)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    border_value = median_border_value(image)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def random_morphology(
    image: np.ndarray,
    rng: np.random.Generator,
    kernel_size_ratio_range: tuple[float, float],
) -> np.ndarray:
    """Kernel size is derived from image width, so it scales correctly
    regardless of native resolution (1920px vs any other)."""
    width = image.shape[1]
    ratio = rng.uniform(kernel_size_ratio_range[0], kernel_size_ratio_range[1])
    kernel_size = max(3, int(width * ratio))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if rng.random() < 0.5:
        return cv2.erode(image, kernel, iterations=1)
    return cv2.dilate(image, kernel, iterations=1)


def add_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    sigma_range: tuple[float, float],
) -> np.ndarray:
    sigma = rng.uniform(sigma_range[0], sigma_range[1])
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ============================================================
# Visualization only — no files are saved
# ============================================================
def load_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return image


def main():
    rng = np.random.default_rng(42)
    image = load_grayscale(image_path)

    # Kernel ratios: e.g. 0.008 of width ~= 15px on a 1920px image, ~2px on 256px image
    augmentations = {
        "Perspective warp": lambda img: random_perspective(img, rng, max_warp_ratio=0.08),
        "Safe translation": lambda img: safe_random_translation(img, rng, max_shift_ratio=0.1),
        "Erosion/Dilation": lambda img: random_morphology(img, rng, kernel_size_ratio_range=(0.006, 0.015)),
        "Gaussian noise": lambda img: add_gaussian_noise(img, rng, sigma_range=(80, 105)),
    }

    n_samples = 3
    n_rows = len(augmentations) + 1

    fig, axes = plt.subplots(
        n_rows, n_samples,
        figsize=(4 * n_samples, 3.5 * n_rows),
    )

    def show(ax, img, title):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Original, resized to final size for reference
    original_resized = cv2.resize(image, FINAL_SIZE, interpolation=cv2.INTER_AREA)
    for col in range(n_samples):
        show(axes[0, col], original_resized, "Original (resized)" if col == 0 else "")

    # Apply augmentation at native resolution, THEN resize to final network input size
    for row, (name, aug_fn) in enumerate(augmentations.items(), start=1):
        for col in range(n_samples):
            augmented = aug_fn(image)
            augmented_resized = cv2.resize(augmented, FINAL_SIZE, interpolation=cv2.INTER_AREA)
            show(axes[row, col], augmented_resized, name if col == 0 else "")

    fig.subplots_adjust(hspace=0.5, wspace=0.05, top=0.97, bottom=0.02)
    plt.show()


if __name__ == "__main__":
    main()