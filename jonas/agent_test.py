from csv import DictReader, DictWriter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


# Keep the small staged evaluation here so experimentation does not immediately
# scan the complete test directory.
CALIBRATION_COUNT = 512
VALIDATION_COUNT = 64
RUN_FULL_EVALUATION = False


def find_dataset() -> tuple[Path, Path]:
	roots = (Path.home() / "jonas" / "ai_summer_school_dataset", Path.home() / "ai_summer_school_dataset")
	for root in roots:
		image_dir = root / "test_augmented"
		labels = root / "augment_test.csv"
		if image_dir.is_dir() and labels.is_file():
			return image_dir, labels
	raise FileNotFoundError("Could not find test_augmented and augment_test.csv")


dir_path, labels_path = find_dataset()


def number_to_digits(number: int) -> np.ndarray:
	return np.asarray([int(digit) for digit in f"{number:07d}"], dtype=np.int8)


def digits_to_number(digits: np.ndarray) -> int:
	return int("".join(str(int(digit)) for digit in digits))


def extract_digits(image_path: Path) -> np.ndarray | None:
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		return None

	# Correct the moderate rotations introduced by augmentation before locating
	# the character band. Long near-horizontal plate edges provide a cheap angle.
	edges = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 0), 50, 150)
	lines = cv2.HoughLinesP(
		edges, 1, np.pi / 180, 80, minLineLength=250, maxLineGap=30
	)
	angles = []
	if lines is not None:
		for x1, y1, x2, y2 in lines:
			angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
			if abs(angle) < 25:
				angles.append(angle)
	if angles:
		angle = float(np.median(angles))
		if abs(angle) > 1:
			center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
			rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
			image = cv2.warpAffine(
				image,
				rotation,
				(image.shape[1], image.shape[0]),
				flags=cv2.INTER_LINEAR,
				borderMode=cv2.BORDER_REPLICATE,
			)

	# The augmentations move the strip, so use overlapping expected windows
	# instead of splitting the image into rigid, non-overlapping slots.
	band = image[650:1100, 430:1450]
	mask = cv2.adaptiveThreshold(
		band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9
	)
	count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
	components = []
	for index in range(1, count):
		x, y, width, height, area = stats[index]
		if height >= 70 and width >= 10 and area >= 500:
			components.append((int(x), int(y), int(width), int(height), int(area)))

	expected_centers = np.arange(105, 826, 120)
	selected = []
	used = set()
	for expected_center in expected_centers:
		candidates = [
			component
			for component in components
			if component not in used
			and abs(component[0] + component[2] / 2 - expected_center) < 85
		]
		if not candidates:
			return None
		component = min(candidates, key=lambda value: abs(value[0] + value[2] / 2 - expected_center))
		used.add(component)
		selected.append(component)

	digit_images = []
	for x, y, width, height, _ in selected:
		crop = mask[max(0, y - 3):min(mask.shape[0], y + height + 3), max(0, x - 3):min(mask.shape[1], x + width + 3)]
		digit_images.append(cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
	return np.asarray(digit_images)


def make_templates(rows: list[dict[str, str]]) -> list[list[list[np.ndarray]]]:
	templates = [[[] for _ in range(10)] for _ in range(7)]
	usable = 0
	for row in rows:
		extracted = extract_digits(dir_path / row["filename"].strip())
		if extracted is None:
			continue
		for position, digit in enumerate(number_to_digits(int(row["number"]))):
			templates[position][int(digit)].append(extracted[position])
		usable += 1
	print(f"Calibration: {usable}/{len(rows)} images produced seven digit crops")
	return templates


def hole_count(image: np.ndarray) -> int:
	binary = (image > 0.35).astype(np.uint8)
	_, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	if hierarchy is None:
		return 0
	return sum(1 for contour in hierarchy[0] if contour[3] >= 0)


def predict(extracted: np.ndarray, templates: list[list[list[np.ndarray]]]) -> np.ndarray:
	prediction = []
	for position, digit_image in enumerate(extracted):
		available = [digit for digit in range(10) if templates[position][digit]]
		image_holes = hole_count(digit_image)
		distances = [
			min(float(np.mean((template - digit_image) ** 2)) for template in templates[position][digit])
			+ 0.005
			* abs(float(np.median([hole_count(template) for template in templates[position][digit]])) - image_holes)
			for digit in available
		]
		prediction.append(available[int(np.argmin(distances))])
	return np.asarray(prediction, dtype=np.int8)


def evaluate(
	rows: list[dict[str, str]],
	templates: list[list[list[np.ndarray]]],
	name: str,
	write_predictions: bool = False,
) -> None:
	start = perf_counter()
	exact = 0
	digit_hits = 0
	digit_total = 0
	processed = 0
	prediction_rows = []
	for row in rows:
		extracted = extract_digits(dir_path / row["filename"].strip())
		prediction_row = {"filename": row["filename"].strip(), "number": ""}
		if extracted is None:
			prediction_rows.append(prediction_row)
			continue
		truth = number_to_digits(int(row["number"]))
		predicted = predict(extracted, templates)
		prediction_row["number"] = str(digits_to_number(predicted))
		prediction_rows.append(prediction_row)
		exact += int(np.array_equal(truth, predicted))
		digit_hits += int(np.sum(truth == predicted))
		digit_total += 7
		processed += 1
	elapsed = perf_counter() - start
	print(
		f"{name}: {processed}/{len(rows)} images, "
		f"digit accuracy={digit_hits / max(1, digit_total):.2%}, "
		f"exact accuracy={exact / max(1, processed):.2%}, "
		f"{elapsed / max(1, processed):.4f}s/image"
	)
	if write_predictions:
		predictions_path = labels_path.with_name("augment_test_predictions.csv")
		with predictions_path.open("w", newline="") as predictions_file:
			writer = DictWriter(predictions_file, fieldnames=["filename", "number"], delimiter=";")
			writer.writeheader()
			writer.writerows(prediction_rows)
		print(f"Predictions written to {predictions_path}")


with labels_path.open(newline="") as labels_file:
	all_rows = list(DictReader(labels_file, delimiter=";"))

calibration_rows = all_rows[:CALIBRATION_COUNT]
validation_rows = all_rows[CALIBRATION_COUNT:CALIBRATION_COUNT + VALIDATION_COUNT]
templates = make_templates(calibration_rows)
evaluate(validation_rows, templates, "Small holdout")

if RUN_FULL_EVALUATION:
	evaluate(all_rows, templates, "Full evaluation", write_predictions=True)