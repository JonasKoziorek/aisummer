# %%
from csv import DictReader
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Paths & Settings
# ============================================================
CALIBRATION_COUNT = 100  # Number of images to build templates from
NUM_IMAGES_TO_SHOW = 3   # 3 images to visualize in detail

def find_dataset() -> tuple[Path, Path]:
    roots = (
        Path.home() / "jonas" / "ai_summer_school_dataset",
        Path.home() / "ai_summer_school_dataset",
    )
    for root in roots:
        image_dir = root / "test_augmented"
        labels = root / "augment_test.csv"
        if image_dir.is_dir() and labels.is_file():
            return image_dir, labels
    raise FileNotFoundError("Could not find test_augmented and augment_test.csv")

dir_path, labels_path = find_dataset()

# ============================================================
# Helper Functions
# ============================================================
def number_to_digits(number: int) -> np.ndarray:
    return np.asarray([int(d) for d in f"{number:07d}"], dtype=np.int8)

def digits_to_number(digits: np.ndarray) -> int:
    return int("".join(str(int(d)) for d in digits))

def hole_count(image: np.ndarray) -> int:
    binary = (image > 0.35).astype(np.uint8)
    _, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(1 for contour in hierarchy[0] if contour[3] >= 0)

# ============================================================
# Extraction with Intermediate Step Capture
# ============================================================
def extract_digits_with_steps(image_path: Path):
    """Extracts digits and returns all intermediate images for plotting."""
    orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if orig is None:
        return None

    # Step 1 & 2: Deskew via Hough Lines
    edges = cv2.Canny(cv2.GaussianBlur(orig, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=250, maxLineGap=30)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) < 25:
                angles.append(angle)

    rotated = orig.copy()
    if angles:
        angle = float(np.median(angles))
        if abs(angle) > 1:
            center = ((orig.shape[1] - 1) / 2, (orig.shape[0] - 1) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                orig, rot_mat, (orig.shape[1], orig.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

    # Step 3: Character band crop
    band = rotated[650:1100, 430:1450]

    # Step 4: Adaptive thresholding
    mask = cv2.adaptiveThreshold(
        band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9
    )

    # Step 5: Connected components & slot selection
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if height >= 70 and width >= 10 and area >= 500:
            components.append((int(x), int(y), int(width), int(height), int(area)))

    expected_centers = np.arange(105, 826, 120)
    selected = []
    used = set()
    for exp_center in expected_centers:
        candidates = [
            c for c in components
            if c not in used and abs(c[0] + c[2] / 2 - exp_center) < 85
        ]
        if not candidates:
            return None
        comp = min(candidates, key=lambda c: abs(c[0] + c[2] / 2 - exp_center))
        used.add(comp)
        selected.append(comp)

    # Draw bounding boxes on RGB copy of band
    annotated_band = cv2.cvtColor(band, cv2.COLOR_GRAY2RGB)
    digit_crops = []
    for i, (x, y, w, h, _) in enumerate(selected):
        cv2.rectangle(annotated_band, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(annotated_band, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        crop = mask[max(0, y - 3):min(mask.shape[0], y + h + 3), max(0, x - 3):min(mask.shape[1], x + w + 3)]
        digit_crops.append(cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)

    steps = {
        "original": orig,
        "rotated": rotated,
        "band": band,
        "mask": mask,
        "annotated_band": annotated_band,
        "digits": np.asarray(digit_crops),
    }
    return steps

# Standard extract for template building
def extract_digits(image_path: Path):
    steps = extract_digits_with_steps(image_path)
    return steps["digits"] if steps is not None else None

def make_templates(rows: list[dict[str, str]]):
    templates = [[[] for _ in range(10)] for _ in range(7)]
    usable = 0
    for row in rows:
        extracted = extract_digits(dir_path / row["filename"].strip())
        if extracted is None:
            continue
        for pos, digit in enumerate(number_to_digits(int(row["number"]))):
            templates[pos][int(digit)].append(extracted[pos])
        usable += 1
    print(f"Templates created from {usable}/{len(rows)} images.")
    return templates

def predict(extracted: np.ndarray, templates):
    prediction = []
    for pos, digit_image in enumerate(extracted):
        available = [d for d in range(10) if templates[pos][d]]
        image_holes = hole_count(digit_image)
        distances = [
            min(float(np.mean((t - digit_image) ** 2)) for t in templates[pos][d])
            + 0.005 * abs(float(np.median([hole_count(t) for t in templates[pos][d]])) - image_holes)
            for d in available
        ]
        prediction.append(available[int(np.argmin(distances))])
    return np.asarray(prediction, dtype=np.int8)

# ============================================================
# Run & Visualize 3 Images
# ============================================================
with labels_path.open(newline="") as f:
    all_rows = list(DictReader(f, delimiter=";"))

# 1. Build templates from calibration set
print("Calibrating templates...")
templates = make_templates(all_rows[:CALIBRATION_COUNT])

# 2. Pick 3 images from the validation set
test_samples = all_rows[CALIBRATION_COUNT:CALIBRATION_COUNT + NUM_IMAGES_TO_SHOW]

for idx, row in enumerate(test_samples, start=1):
    filename = row["filename"].strip()
    true_num = int(row["number"])
    true_digits = number_to_digits(true_num)
    img_path = dir_path / filename

    steps = extract_digits_with_steps(img_path)
    if steps is None:
        print(f"Failed to segment {filename}")
        continue

    # Predict
    pred_digits = predict(steps["digits"], templates)
    pred_num = digits_to_number(pred_digits)
    is_correct = (true_num == pred_num)

    # --------------------------------------------------------
    # Plot Pipeline Stages Side-by-Side
    # --------------------------------------------------------
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"Image {idx}: {filename}  |  True: {true_num}  |  Pred: {pred_num}  [{'MATCH' if is_correct else 'MISMATCH'}]",
        fontsize=16,
        fontweight="bold",
        color="green" if is_correct else "red",
    )

    # Row 1: Pipeline steps
    ax1 = plt.subplot2grid((3, 7), (0, 0), colspan=2)
    ax1.imshow(steps["original"], cmap="gray")
    ax1.set_title("1. Original Image")
    ax1.axis("off")

    ax2 = plt.subplot2grid((3, 7), (0, 2), colspan=2)
    ax2.imshow(steps["rotated"], cmap="gray")
    ax2.set_title("2. Deskewed (Hough Lines)")
    ax2.axis("off")

    ax3 = plt.subplot2grid((3, 7), (0, 4), colspan=3)
    ax3.imshow(steps["band"], cmap="gray")
    ax3.set_title("3. Cropped Seal Band")
    ax3.axis("off")

    ax4 = plt.subplot2grid((3, 7), (1, 0), colspan=3)
    ax4.imshow(steps["mask"], cmap="gray")
    ax4.set_title("4. Adaptive Threshold Mask")
    ax4.axis("off")

    ax5 = plt.subplot2grid((3, 7), (1, 3), colspan=4)
    ax5.imshow(steps["annotated_band"])
    ax5.set_title("5. Connected Components (7 Slots Detected)")
    ax5.axis("off")

    # Row 2: The 7 extracted digit crops
    for digit_idx in range(7):
        ax = plt.subplot2grid((3, 7), (2, digit_idx))
        ax.imshow(steps["digits"][digit_idx], cmap="gray")
        match_color = "green" if pred_digits[digit_idx] == true_digits[digit_idx] else "red"
        ax.set_title(
            f"Pos {digit_idx}\nTrue: {true_digits[digit_idx]} | Pred: {pred_digits[digit_idx]}",
            color=match_color,
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()

    # Save visualization to disk (safe for headless WSL)
    out_file = f"pipeline_step_image_{idx}_{filename}"
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    print(f"Saved step-by-step visual to: {out_file}")

    plt.show()
# %%
