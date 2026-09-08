# %%
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# %%
base = "/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset"
path = base + "/train/09417.png"

path2 = base + "/ground_truth_chars_balanced"


# %%

bgr_img = cv2.imread(path)

rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_img)
plt.axis('off')
plt.show()

# %%

im = cv2.imread(path)

def show_image(image, cmap=None):
    plt.figure(figsize=(6, 4))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

show_image(im)

print(f"Shape: {im.shape}")
print(f"Data type: {im.dtype}")
print(f"Min pixel value: {im.min()}")
print(f"Max pixel value: {im.max()}")

blue, green, red = cv2.split(im)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, channel, title in zip(axes, [blue, green, red], ["Blue", "Green", "Red"]):
    ax.imshow(channel, cmap="gray")
    ax.set_title(title)
    # ax.axis("off")
plt.tight_layout()
# %%
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
show_image(gray, cmap="gray")

# %%

blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original grayscale")
axes[0].axis("off")
axes[1].imshow(blurred, cmap="gray")
axes[1].set_title("Blurred")
axes[1].axis("off")
plt.tight_layout()


# %%

threshold_value, mask = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

print(f"Otsu threshold: {threshold_value:.1f}")
show_image(mask, cmap="gray")
# %%

edges = cv2.Canny(blurred, threshold1=60, threshold2=160)
show_image(edges, cmap="gray")


# %%

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
component_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    component_mask,
    connectivity=8,
)

min_area = 500
components = []
component_overlay = im.copy()

for label_id in range(1, num_labels):
    x = stats[label_id, cv2.CC_STAT_LEFT]
    y = stats[label_id, cv2.CC_STAT_TOP]
    w = stats[label_id, cv2.CC_STAT_WIDTH]
    h = stats[label_id, cv2.CC_STAT_HEIGHT]
    area = stats[label_id, cv2.CC_STAT_AREA]
    cx, cy = centroids[label_id]

    if area < min_area:
        continue

    components.append(
        {
            "label": label_id,
            "area": int(area),
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
        }
    )

components = sorted(components, key=lambda item: item["area"], reverse=True)

for index, component in enumerate(components, start=1):
    x, y, w, h = component["bbox"]
    cx, cy = component["centroid"]
    cv2.rectangle(component_overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.circle(component_overlay, (int(cx), int(cy)), 5, (0, 0, 255), thickness=-1)
    cv2.putText(
        component_overlay,
        str(index),
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

print(f"Detected {len(components)} components with area >= {min_area} pixels")
for index, component in enumerate(components[:10], start=1):
    print(
        f"{index:>2}: area={component['area']:>6}, "
        f"bbox={component['bbox']}, "
        f"centroid=({component['centroid'][0]:.1f}, {component['centroid'][1]:.1f})"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(component_mask, cmap="gray")
axes[0].set_title("Cleaned component mask")
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(component_overlay, cv2.COLOR_BGR2RGB))
axes[1].set_title("Detected components")
axes[1].axis("off")
plt.tight_layout()
# %%

import re
from pathlib import Path

base = Path(base)
folder = base / "ground_truth_chars_balanced"

digit_files = {i: [] for i in range(10)}

# Regex pattern matching any file containing "number_<digit>.tif"
# e.g., matches "number_0.tif", "sample_number_3.tif", etc.
pattern = re.compile(r"^\d+_(\d)\.tif$")

for file_path in folder.iterdir():
    if file_path.is_file():
        match = pattern.match(file_path.name)
        if match:
            # match.group(1) captures the digit right before .tif
            digit = int(match.group(1))
            digit_files[digit].append(str(file_path))

# Sort file lists (optional, for reproducible ordering)
for digit in digit_files:
    digit_files[digit].sort()
# %%

def load_class_images(file_paths):
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return images

# %%
def compute_average_from_loaded(images, threshold_ratio=0.5):
    # Determine maximal dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    accumulator = np.zeros((max_h, max_w), dtype=np.uint32)

    for img in images:
        h, w = img.shape[:2]
        bin_img = (img > 127).astype(np.uint8)

        y_start = (max_h - h) // 2
        x_start = (max_w - w) // 2

        accumulator[y_start:y_start + h, x_start:x_start + w] += bin_img

    avg_continuous = accumulator / len(images)
    avg_binary = np.where(accumulator >= int(len(images) * threshold_ratio), np.uint8(255), np.uint8(0))

    return avg_continuous, avg_binary

output_dir = Path("average_images")
output_dir.mkdir(parents=True, exist_ok=True)
# %%
number = 0
images = load_class_images(digit_files[number])
result = compute_average_from_loaded(images)
avg_img = result[1]
white_mask = (avg_img == 255).astype(np.uint8)
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_cnt)
cropped_img = avg_img[y:y+h, x:x+w]
show_image(cropped_img, cmap="gray")
save_path = output_dir / f"average_{number}.png"
cv2.imwrite(str(save_path), cropped_img)
# %%

def compute_average_shape(file_paths, threshold_val=0.5):
    """
    Loads binary images, averages them, and returns:
    1. The continuous probability/heatmap (0.0 to 1.0)
    2. The thresholded binary image (0 or 255)
    """
    images = []
    
    i = 0
    for fpath in file_paths:
        print(i)
        # 1. Load as single-channel grayscale
        img = cv2.imread(path)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            continue
            
        # Normalize binary image to float [0.0, 1.0]
        # (Assuming foreground is white (255) and background is black (0))
        img_normalized = (img > 127).astype(np.float32)
        images.append(img_normalized)
        i=i+1
        
    if not images:
        raise ValueError("No valid images found.")

    # 2. Stack into 3D array: shape (N, height, width)
    stack = np.stack(images, axis=0)

    # 3. Compute pixel-wise mean along axis 0
    # Values represent the probability of each pixel being "on" across all samples
    avg_continuous = np.mean(stack, axis=0)

    # 4. Partition by midpoint (or custom threshold)
    # Pixels where >50% of images agree become 255 (white), else 0 (black)
    avg_binary = np.where(avg_continuous >= threshold_val, 255, 0).astype(np.uint8)

    return avg_continuous, avg_binary

avg_img = compute_average_shape(digit_files[0][:100])
# %%

test_images = ["06098.png",
                "02197.png",
                "03400.png",
                "02988.png",
                "03500.png",
                "02699.png",

               ]