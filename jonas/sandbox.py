# %%
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from time import time


# %%
base = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset")
path = base / "train/09417.png"

path2 = base / "ground_truth_chars_balanced"


# %%

bgr_img = cv2.imread(path)

rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_img)
plt.axis('off')
plt.show()

# %%

im = cv2.imread(path)

def show_image(image, cmap="gray"):
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

test_images = ["06098.png",
                "02197.png",
                "03400.png",
                "02988.png",
                "03500.png",
                "02699.png",
               ]

paths = [base / "train" / img for img in test_images]
# %%

paths = [Path("/home/jonas/aisummer/aisummer/jonas/local_images/rectangles.png"), Path("/home/jonas/aisummer/aisummer/jonas/local_images/gradient.png")]

# %%

num_images = len(paths)
fig, axes = plt.subplots(num_images, 2, figsize=(8, 3 * num_images))

for i, p in enumerate(paths):
    # Load as single-channel grayscale
    src = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Warning: Could not read {p}")
        continue

    # Apply histogram equalization
    t1 = time()
    dst = cv2.equalizeHist(src)
    t2 = time()
    print(t2-t1)

    # Plot original on the left
    axes[i, 0].imshow(src, cmap="gray")
    axes[i, 0].set_title(f"Original: {p.name}")
    axes[i, 0].axis("off")

    # Plot equalized on the right
    axes[i, 1].imshow(dst, cmap="gray")
    axes[i, 1].set_title("equalizeHist")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
# %%


test_images = ["06098.png",
                "02197.png",
                "03400.png",
                "02988.png",
                "03500.png",
                "02699.png",
               ]


paths = [base / "train" / img for img in test_images]
# %%

images = load_class_images(paths)
img_gray = images[0]
# result = locate_number_strip_gradient(images[0])
# show_image(result[1], cmap="gray")

# %%

def extract_center_component(img_gray):
    H, W = img_gray.shape[:2]
    cx, cy = W // 2, H // 2

    # Smooth out metal brush textures and noise
    blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check both polarities (plate might be white or black)
    for mask in [thresh, cv2.bitwise_not(thresh)]:
        # Close internal character holes
        k_size = max(9, int(min(H, W) * 0.05))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Label connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

        # Find which component the center pixel (cx, cy) belongs to
        center_label = labels[cy, cx]

        # If center pixel hit a small hole, check the nearest 5x5 neighborhood mode
        if center_label == 0:
            window = labels[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            non_zero = window[window > 0]
            if len(non_zero) > 0:
                center_label = np.bincount(non_zero).argmax()

        if center_label > 0:
            x = stats[center_label, cv2.CC_STAT_LEFT]
            y = stats[center_label, cv2.CC_STAT_TOP]
            w = stats[center_label, cv2.CC_STAT_WIDTH]
            h = stats[center_label, cv2.CC_STAT_HEIGHT]
            return (x, y, w, h), img_gray[y:y+h, x:x+w]

    return None, None

images = load_class_images(paths)
img_gray = images[0]
result = extract_center_component(images[0])
show_image(result[1])

# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_plate_kmeans(src_gray, K=3):
    t1 = time()
    H, W = src_gray.shape[:2]

    # 1. Downscale significantly for fast clustering (< 10 ms)
    scale = 0.25
    small = cv2.resize(src_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (5, 5), 0)

    # 2. Reshape into 1D float array of intensities for OpenCV's kmeans
    pixel_vals = small_blur.reshape((-1, 1)).astype(np.float32)

    # Criteria: stop if 10 iterations reached or epsilon < 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Reshape labels back to 2D image
    label_img = labels.reshape(small.shape)

    # 3. Identify the cluster belonging to the center plate
    # Query the cluster label at the center of the image
    center_y, center_x = int(small.shape[0] / 2), int(small.shape[1] / 2)
    center_label = label_img[center_y, center_x]

    # If the center pixel happened to hit text, take the modal cluster in a 11x11 center box
    box = label_img[center_y - 5 : center_y + 5, center_x - 5 : center_x + 5]
    center_label = np.bincount(box.flatten()).argmax()

    # 4. Create binary mask for the center plate cluster
    plate_mask = (label_img == center_label).astype(np.uint8) * 255

    # 5. Morphological close to bridge interior character cutouts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Extract the largest connected component within this cluster
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    main_cnt = max(contours, key=cv2.contourArea)
    sx, sy, sw, sh = cv2.boundingRect(main_cnt)

    # 7. Scale coordinates back to original image size
    orig_x = int(sx / scale)
    orig_y = int(sy / scale)
    orig_w = int(sw / scale)
    orig_h = int(sh / scale)

    # Clip to boundaries
    orig_x = max(0, orig_x)
    orig_y = max(0, orig_y)
    orig_w = min(W - orig_x, orig_w)
    orig_h = min(H - orig_y, orig_h)

    plate_roi = src_gray[orig_y : orig_y + orig_h, orig_x : orig_x + orig_w]

    # 8. Extract the bottom region for numbers (~bottom 30% of the detected plate)
    num_y1 = int(orig_h * 0.50)
    num_roi = plate_roi[num_y1:orig_h, :]

    t2 = time()
    print(t2-t1)
    return (orig_x, orig_y, orig_w, orig_h), plate_roi, num_roi

fig, axes = plt.subplots(len(paths), 3, figsize=(12, 3 * len(paths)))

for i, p in enumerate(paths):
    src = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if src is None:
        continue

    bbox, plate, numbers = extract_plate_kmeans(src, K=3)

    # Bounding box on original
    x, y, w, h = bbox
    vis = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)

    axes[i, 0].imshow(vis)
    axes[i, 0].set_title(f"K-Means BBox: {p.name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(plate, cmap="gray")
    axes[i, 1].set_title("Segmented Plate")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(numbers, cmap="gray")
    axes[i, 2].set_title("Bottom Numbers")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
# %%

num_images = len(images)
fig, axes = plt.subplots(num_images, 5, figsize=(16, 3 * num_images))

if num_images == 1:
    axes = np.expand_dims(axes, axis=0)

def make_numbers_white(th_img):
    H, W = th_img.shape[:2]

    # Sample a central patch of the plate right above the numbers
    # (roughly middle vertically, center horizontally)
    sample_patch = th_img[int(H * 0.40):int(H * 0.60), int(W * 0.35):int(W * 0.65)]

    # If the background plate is predominantly white, the numbers are black: INVERT
    if np.mean(sample_patch) > 127:
        return cv2.bitwise_not(th_img), True  # Inverted

    return th_img, False  # Already white numbers on dark plate

for i, img in enumerate(images):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    eqgray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eqgray, (5, 5), 0)
    ret3, th3 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # 1. Original / Grayscale
    axes[i, 0].imshow(gray, cmap="gray")
    axes[i, 0].set_title(f"Original [{i}]")
    axes[i, 0].axis("off")

    # 2. Histogram of the Original
    axes[i, 1].hist(gray.ravel(), bins=256, range=[0, 256], color="black", alpha=0.75)
    axes[i, 1].set_title(f"Histogram [{i}]")
    axes[i, 1].set_xlim([0, 256])
    axes[i, 1].grid(True, linestyle="--", alpha=0.4)

    # 3. Equalized Image
    axes[i, 2].imshow(eqgray, cmap="gray")
    axes[i, 2].set_title("Equalized Hist")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(make_numbers_white(gray)[0], cmap="gray")
    axes[i, 3].set_title(f"White Numbers")
    axes[i, 2].axis("off")

    # 4. Thresholded
    axes[i, 4].imshow(th3, cmap="gray")
    axes[i, 4].set_title(f"Thresh (T={int(ret3)})")
    axes[i, 4].axis("off")

plt.tight_layout()
plt.show()

# %%

fig, axes = plt.subplots(len(images), 4, figsize=(14, 3.5 * len(images)))

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))

for i, img in enumerate(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Method 1: Background Division
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = np.clip((gray.astype(np.float32) / (bg.astype(np.float32) + 1e-5)) * 255, 0, 255).astype(np.uint8)
    _, th_norm = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 2: CLAHE + Otsu
    cl = clahe.apply(gray)
    _, th_clahe = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 3: Adaptive Gaussian
    th_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8)

    axes[i, 0].imshow(gray, cmap="gray")
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(th_norm, cmap="gray")
    axes[i, 1].set_title("Morph. Division + Otsu")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(th_clahe, cmap="gray")
    axes[i, 2].set_title("CLAHE + Otsu")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(th_adapt, cmap="gray")
    axes[i, 3].set_title("Adaptive Gaussian")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.show()

# %%
