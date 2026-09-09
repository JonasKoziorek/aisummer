# %%
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from time import time

# %%
test_images = ["06098.png",
                "02197.png",
                "03400.png",
                "02988.png",
                "03500.png",
                "02699.png",
               ]


base = Path("/mnt/c/Users/pepaz/Downloads/ai_summer_school_dataset/ai_summer_school_dataset")
paths = [base / "train" / img for img in test_images]

def show_image(image, cmap="gray"):
    plt.figure(figsize=(6, 4))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

def load_class_images(file_paths):
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return images

images = load_class_images(paths)
show_image(images[0])
# %%

def apply_watershed(image_bgr):
    """Applies marker-based watershed segmentation to an input BGR image.

    Returns the original RGB image and the image overlaid with boundary
    markers.
    """
    img = image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Thresholding to extract binary foreground/background
    # Otsu thresholding handles varied illumination well
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Morphological opening to eliminate salt-and-pepper noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Sure foreground via Euclidean distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Peak regions (distance >= 0.5 * max_distance) become sure foreground
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    # 5. Unknown boundary region: area between sure background and sure foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Shift labels up by 1 so known background is 1, not 0
    markers = markers + 1
    # Mark unknown regions as 0 for the watershed algorithm
    markers[unknown == 255] = 0

    # 7. Apply Watershed
    # OpenCV modifies `markers` in-place; boundaries receive a label of -1
    markers = cv2.watershed(img, markers)

    # Overlay boundaries in bright red on the original image
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result[markers == -1] = [255, 0, 0]

    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return original_rgb, result

def plot_watershed_batch(images):
    """Takes a list/array of BGR images and plots Original vs Watershed side-by-side."""
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))

    # Ensure 2D indexing even if only 1 image is passed
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, img in enumerate(images):
        orig, segmented = apply_watershed(img)

        # Plot original
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Image {i + 1}: Original")
        axes[i, 0].axis("off")

        # Plot watershed boundary overlay
        axes[i, 1].imshow(segmented)
        axes[i, 1].set_title(f"Image {i + 1}: Watershed Segmented (Red Boundaries)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
# If loading from disk:
# paths = ["coins1.jpg", "cells2.png", "sample3.jpg"]
image_list = [cv2.imread(p) for p in paths[0:3]]

plot_watershed_batch(image_list)

# %%
image = images[0]
show_image(image)
# %%

image = images[5]
eroded = cv2.dilate(image, np.ones((5, 5), np.uint8), iterations=4)
show_image(eroded)
# %%

# %%
show_image(image)
_, thresh = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
show_image(thresh)

# %%
image = images[0].copy()
show_image(image)
blurred = cv2.GaussianBlur(image, (3, 3), 0)

sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# show_image(sobel_combined)
# eroded = cv2.dilate(sobel_combined, np.ones((5, 5), np.uint8), iterations=0)
ret, thresh = cv2.threshold(sobel_combined, 30, 255, 0)
show_image(thresh)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
show_image(opening)
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
show_image(image)

# %%

image = images[0].copy()
gray = image

filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

# 3. Detect edges of the groove
# Canny edge detection catches the dark boundary channel around the insert
edges = cv2.Canny(filtered, threshold1=40, threshold2=120)

# 4. Close small gaps along the perimeter groove
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# 5. Find external contours
contours, _ = cv2.findContours(
    closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# 6. Filter contours by size, position, and bounding box properties
h, w = gray.shape
min_area = (w * h) * 0.10  # Must be at least 10% of the image
max_area = (w * h) * 0.70  # Must not be the whole frame boundary

target_contour = None
best_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Verify the box is relatively centered
        center_x = x + bw / 2
        center_y = y + bh / 2
        if 0.3 * w < center_x < 0.7 * w and 0.2 * h < center_y < 0.8 * h:
            if area > best_area:
                best_area = area
                target_contour = cnt

# 7. Draw the detected boundary
result = image.copy()
if target_contour is not None:
    # Draw green boundary contour
    cv2.drawContours(result, [target_contour], -1, (0, 255, 0), 3)

    # Optional: Get the bounding rectangle or polygon approximation
    x, y, bw, bh = cv2.boundingRect(target_contour)
    print(f"Bounding Box: x={x}, y={y}, width={bw}, height={bh}")

# 8. Display with Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")
axes[0].axis("off")

axes[1].imshow(closed, cmap="gray")
axes[1].set_title("Edge & Morphology Mask")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title("Detected Boundary (Green)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
# %%

def find_inner_rectangle(image_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    original_image = image.copy()
    img_h, img_w = image.shape[:2]
    total_image_area = img_h * img_w

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian Blur to reduce the grainy texture and focus on structural edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 4. Apply Canny Edge Detection
    # These thresholds (30, 150) work well for identifying the distinct gap 
    # surrounding the inner plate.
    edges = cv2.Canny(blurred, 30, 150)

    # 5. Dilate the edges slightly to close any small gaps in the boundary line
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # 6. Find Contours
    # Using RETR_LIST to find all contours, not just extreme outer ones
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_area = 0
    best_bbox = None

    # 7. Loop through contours to find the main plate
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        
        # Filter conditions:
        # - Must not be the entire image (bounding box area < 90% of image)
        # - Must be reasonably large (bounding box area > 10% of image)
        if (total_image_area * 0.10) < bbox_area < (total_image_area * 0.90):
            # We want the largest contour that fits this description
            if bbox_area > max_area:
                max_area = bbox_area
                best_contour = cnt
                best_bbox = (x, y, w, h)

    # 8. Draw the result and output coordinates
    if best_bbox is not None:
        x, y, w, h = best_bbox
        
        # Draw a green bounding box (Thickness=3)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Calculate coordinates in [ymin, xmin, ymax, xmax] format
        ymin, xmin, ymax, xmax = y, x, y + h, x + w
        print(f"Found inner rectangle bounding box!")
        print(f"Format [ymin, xmin, ymax, xmax]: [{ymin}, {xmin}, {ymax}, {xmax}]")
        print(f"Format [x, y, w, h]: [{x}, {y}, {w}, {h}]")
    else:
        print("Could not find a matching inner rectangle.")

    # 9. Display the processing steps
    # Resize for display purposes if the image is too large
    # plt.imshow(gray)
    # plt.imshow(edges_dilated)
    plt.imshow(image)

find_inner_rectangle(paths[0])
# %%
