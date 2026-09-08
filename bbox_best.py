import argparse
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import detection_predictor

IMAGE_PATH = r"D:\AiSimmer2026\images\ai_summer_school_dataset\train\00016.png"


def normalize_rect(rect):
    """
    Converts a minAreaRect-style rect ((cx,cy),(w,h),angle) into a canonical
    form: width is always the long side, angle is the tilt of that long side
    relative to horizontal, range (-90, 90].
    """
    (cx, cy), _, _ = rect
    box = cv2.boxPoints(rect)

    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    len1, len2 = float(np.linalg.norm(edge1)), float(np.linalg.norm(edge2))

    if len1 >= len2:
        long_edge, width, height = edge1, len1, len2
    else:
        long_edge, width, height = edge2, len2, len1

    angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
    angle = angle % 180
    if angle > 90:
        angle -= 180

    return (cx, cy), (width, height), angle


def pick_lower_box(boxes):
    """
    boxes: list of rects, either
      - minAreaRect-style ((cx,cy),(w,h),angle), or
      - axis-aligned (x1,y1,x2,y2)
    Normalizes each by rotation angle, then returns the one whose center
    is lowest in the image (largest y).
    """
    normalized = []
    for b in boxes:
        if len(b) == 4:  # axis-aligned (x1,y1,x2,y2) -> convert to minAreaRect first
            x1, y1, x2, y2 = b
            rect = ((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), 0.0
        else:
            rect = b
        normalized.append(normalize_rect(rect))

    # lowest in the image = largest y-coordinate of the center
    return max(normalized, key=lambda r: r[0][1])

def column_projection(gray_crop):
    """
    Binarizes the crop (bright text on dark plate) and returns the
    per-column count of foreground (text) pixels - the projection onto
    the x-axis.
    """
    _, thresh = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile = (thresh == 255).sum(axis=0).astype(np.float32)
    return profile, thresh


def split_into_n_boxes(gray_crop, n, search_frac=0.5):
    """
    Splits a single-line character crop into `n` sequential boxes
    (left to right, order preserved).
 
    Strategy: start from n equal-width slices as a prior, then snap each
    internal boundary to the nearest low-density column (gap between
    characters) within a local search window, using the column projection.
    Falls back to the equal-width boundary if no clear gap is found nearby.
 
    Returns: list of n boxes (x1, 0, x2, h), in left-to-right order, and
    the raw projection profile (for debugging/plotting).
    """
    h, w = gray_crop.shape[:2]
    profile, _ = column_projection(gray_crop)
 
    # light smoothing to avoid snapping to single-column noise
    kernel = np.ones(3, dtype=np.float32) / 3
    profile_smooth = np.convolve(profile, kernel, mode="same")
 
    ideal_width = w / n
    boundaries = [0]
    for i in range(1, n):
        ideal_x = int(round(i * ideal_width))
        radius = max(int(ideal_width * search_frac / 2), 1)
        lo = max(ideal_x - radius, boundaries[-1] + 1)
        hi = min(ideal_x + radius, w - 1)
        if lo >= hi:
            boundaries.append(ideal_x)
            continue
        window = profile_smooth[lo:hi]
        best_x = lo + int(np.argmin(window))
        boundaries.append(best_x)
    boundaries.append(w)
 
    boxes = []
    for i in range(n):
        x1, x2 = boundaries[i], boundaries[i + 1]
        boxes.append((x1, 0, x2, h))
    return boxes, profile


def crop_rotated_rect(img, rect, pad=6):
    """Crops the region and straightens it horizontally (deskew)."""
    (cx, cy), (w, h), angle = normalize_rect(rect)
    w += pad * 2
    h += pad * 2
 
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
 
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, rotated.shape[1]), min(y2, rotated.shape[0])
 
    return rotated[y1:y2, x1:x2]



def main() -> None:

    # detection_predictor() skips the recognition stage entirely - we only
    # need boxes, not the actual decoded text, so this saves real time.
    # fast_tiny is the lightest architecture docTR offers (the FAST family,
    # designed specifically for low latency) - starting here since detection
    # is only part of a larger pipeline with a tight time budget.
    # If accuracy on your engraved digits isn't good enough, step up to
    # "db_mobilenet_v3_large" (heavier, more accurate, still lightweight).
    model = detection_predictor(arch="fast_tiny", pretrained=True)

    doc = DocumentFile.from_images(IMAGE_PATH)

    start = time.perf_counter()
    result = model(doc)  # list of dicts, one per page
    elapsed = time.perf_counter() - start
    print(f"Inference time: {elapsed:.3f}s")
    print("RESULT: ", result, "END PF RESULT")

    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    # result[0]["words"] is an array of shape (N, 5): x_min, y_min, x_max, y_max, score
    # (relative coordinates, 0.0-1.0)
    boxes = []
    for x_min, y_min, x_max, y_max, score in result[0]["words"]:
        boxes.append((
            int(x_min * width), int(y_min * height),
            int(x_max * width), int(y_max * height),
        ))

    print(f"Found {len(boxes)} text regions")
    for box in boxes:
        print(box)

    if not boxes:
        print("No text regions found - nothing to pick.")
        return

    # normalize all boxes by rotation angle, keep only the lower one
    # (e.g. the serial number line, below the brand name)
    lower_rect = pick_lower_box(boxes)
    (cx, cy), (w, h), angle = lower_rect
    print(f"Lower box (normalized): center=({cx:.0f},{cy:.0f}) size=({w:.0f}x{h:.0f}) angle={angle:.1f}")

    # --- visualize: only the selected lower box ---
    vis = image.copy()
    box_pts = cv2.boxPoints(lower_rect)
    box_pts = np.intp(box_pts)
    cv2.drawContours(vis, [box_pts], 0, (0, 0, 255), 2)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_rgb)
    plt.title("Selected lower text region (normalized)")
    plt.axis("off")
    plt.show()


    crop = crop_rotated_rect(image, lower_rect)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
 
    # NOTE: our test plate reads "1585422" = 7 digits. Use n matching the
    # real number of characters on your actual label (you said 6).
    crop = crop_rotated_rect(image, lower_rect)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
 
    # NOTE: our test plate reads "1585422" = 7 digits. Use n matching the
    # real number of characters on your actual label (you said 6).
    N_CHARS = 7
    boxes, profile = split_into_n_boxes(gray_crop, n=N_CHARS)
 
    print(f"Crop size: {gray_crop.shape[1]}x{gray_crop.shape[0]}")
    print(f"Split into {len(boxes)} boxes (in order):")
    for i, b in enumerate(boxes):
        print(f"  char {i}: {b}")
 
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(3, N_CHARS, height_ratios=[2, 1, 1], hspace=0.5)
 
    # 1) crop with cut boxes drawn on top
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(gray_crop, cmap="gray")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax0.add_patch(plt.Rectangle((x1, y1), x2 - x1 - 1, y2 - y1 - 1,
                                     fill=False, edgecolor="red", linewidth=1.5))
        ax0.text(x1 + 2, 12, str(i), color="lime", fontsize=10, weight="bold")
    ax0.set_title(f"Split into {N_CHARS} sequential boxes")
    ax0.axis("off")
 
    # 2) column projection with cut points
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(profile)
    for x1, _, _, _ in boxes[1:]:
        ax1.axvline(x1, color="red", linestyle="--", linewidth=1)
    ax1.set_title("Column projection (x-axis) with cut points")
    ax1.set_xlabel("x (column)")
    ax1.set_ylabel("foreground px")
    ax1.set_xlim(0, gray_crop.shape[1])
 
    # 3) the individual character crops, laid out in order left to right
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        sub_ax = fig.add_subplot(gs[2, i])
        sub_ax.imshow(gray_crop[y1:y2, x1:x2], cmap="gray")
        sub_ax.set_title(str(i), fontsize=9)
        sub_ax.axis("off")
 
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()