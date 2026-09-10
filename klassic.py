"""
Pipeline: image -> AKAZE descriptors -> BoVW (fixed-length vector)
          -> per-position supervised LDA reduction -> per-position kNN classifier

Important note on k-means here:
    K-means IS still used, but only to build a visual-word "vocabulary" so that
    a variable number of AKAZE descriptors per image can be turned into a
    single fixed-length vector (a histogram of visual words). This is the
    standard Bag-of-Visual-Words trick and has nothing to do with using
    k-means as a *classifier* (which we deliberately avoid, per the earlier
    discussion) — the actual classification is done later by LDA + kNN,
    which use the true labels.

Structure:
    1. extract_akaze_descriptors(image)      -> Nx61 array (binary descriptors)
    2. build_vocabulary(all_descriptors, k)  -> KMeans vocabulary (unsupervised, encoding only)
    3. image_to_bovw(descriptors, vocab)     -> fixed-length histogram vector
    4. PositionModel                         -> LDA (supervised) + kNN for one of the 7 positions
    5. PipelineSevenPositions                -> trains/predicts all 7 positions together
"""

import os
import csv
import time
import argparse

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# ---------- 1. Descriptor extraction ----------

_akaze = cv2.AKAZE_create()


def extract_akaze_descriptors(image_gray: np.ndarray) -> np.ndarray:
    """
    image_gray: single-channel (grayscale) uint8 image.
    Returns: (N, 61) uint8 array of AKAZE descriptors, or an empty array
             of shape (0, 61) if no keypoints were found.
    """
    keypoints, descriptors = _akaze.detectAndCompute(image_gray, None)
    if descriptors is None:
        return np.empty((0, 61), dtype=np.uint8)
    return descriptors


# ---------- 2. Vocabulary (BoVW codebook) ----------

def build_vocabulary(all_descriptors: list[np.ndarray], n_words: int = 200,
                      random_state: int = 42) -> MiniBatchKMeans:
    """
    all_descriptors: list of per-image descriptor arrays (output of step 1),
                      collected from the TRAINING set only.
    n_words: vocabulary size (typical range 100-500; tune on validation data).
    """
    stacked = np.vstack([d for d in all_descriptors if len(d) > 0]).astype(np.float32)
    vocab = MiniBatchKMeans(n_clusters=n_words, random_state=random_state,
                             batch_size=1024, n_init=10)
    vocab.fit(stacked)
    return vocab


# ---------- 3. Encode one image as a fixed-length BoVW vector ----------

def image_to_bovw(descriptors: np.ndarray, vocab: MiniBatchKMeans) -> np.ndarray:
    """
    Converts a variable-length set of descriptors into a single fixed-length
    (n_words,) L2-normalized histogram vector.
    """
    n_words = vocab.n_clusters
    if len(descriptors) == 0:
        return np.zeros(n_words, dtype=np.float32)

    words = vocab.predict(descriptors.astype(np.float32))
    hist = np.bincount(words, minlength=n_words).astype(np.float32)
    hist = normalize(hist.reshape(1, -1))[0]  # L2 normalize -> robust to descriptor count
    return hist


# ---------- 4. One supervised classifier for one of the 7 positions ----------

class PositionModel:
    """
    Supervised dimensionality reduction (LDA) + kNN classifier for ONE
    of the 7 label positions. LDA uses the true labels while fitting,
    so the reduced space is built to separate classes — not an unsupervised
    guess like PCA/k-means would be.
    """

    def __init__(self, n_components: int | None = None, k_neighbors: int = 5):
        # LDA components are capped at (n_classes - 1); leave None to let sklearn pick max.
        self.n_components = n_components
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PositionModel":
        X_reduced = self.lda.fit_transform(X, y)
        self.knn.fit(X_reduced, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self.lda.transform(X)
        return self.knn.predict(X_reduced)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self.lda.transform(X)
        return self.knn.predict_proba(X_reduced)


# ---------- 5. Full pipeline over all 7 positions ----------

class PipelineSevenPositions:
    """
    Owns: one shared AKAZE vocabulary + 7 independent PositionModel instances
    (one per digit/character position).
    """

    def __init__(self, n_words: int = 200, k_neighbors: int = 5,
                 n_positions: int = 7):
        self.n_words = n_words
        self.k_neighbors = k_neighbors
        self.n_positions = n_positions
        self.vocab: MiniBatchKMeans | None = None
        self.position_models: list[PositionModel] = []

    def fit(self, images_gray: list[np.ndarray], labels: np.ndarray):
        """
        images_gray: list of grayscale images (training set).
        labels: array of shape (n_samples, 7) — true class per position.
        """
        # 1. Extract descriptors for every image once.
        all_descriptors = [extract_akaze_descriptors(img) for img in images_gray]

        # 2. Build one shared vocabulary from all training descriptors.
        self.vocab = build_vocabulary(all_descriptors, n_words=self.n_words)

        # 3. Encode every image as a fixed-length BoVW vector.
        X = np.vstack([image_to_bovw(d, self.vocab) for d in all_descriptors])

        # 4. Fit one supervised model per position.
        self.position_models = []
        for pos in range(self.n_positions):
            model = PositionModel(k_neighbors=self.k_neighbors)
            model.fit(X, labels[:, pos])
            self.position_models.append(model)

        return self

    def predict(self, images_gray: list[np.ndarray]) -> np.ndarray:
        """
        Returns array of shape (n_samples, 7) with predicted class per position.
        """
        descriptors = [extract_akaze_descriptors(img) for img in images_gray]
        X = np.vstack([image_to_bovw(d, self.vocab) for d in descriptors])

        predictions = np.column_stack([
            model.predict(X) for model in self.position_models
        ])
        return predictions


# ---------- 6. Loading the dataset (same CSV format as train_loop.py) ----------

def load_dataset(csv_path: str, images_dir: str, num_positions: int = 7):
    """
    Reads a "filename;number" CSV (same format/delimiter as DigitSequenceDataset)
    and loads the corresponding grayscale images.

    Returns:
        images_gray: list[np.ndarray]
        labels: np.ndarray of shape (n_samples, num_positions), dtype=int
    """
    images_gray, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            filename, number_str = row["filename"], row["number"]
            image_path = os.path.join(images_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            images_gray.append(image)
            labels.append([int(ch) for ch in number_str])

    labels = np.array(labels, dtype=np.int64)
    assert labels.shape[1] == num_positions, (
        f"Expected {num_positions} positions, got {labels.shape[1]} in {csv_path}"
    )
    return images_gray, labels


# ---------- 7. Evaluation (mirrors the metrics used in train_loop.py) ----------

def evaluate(preds: np.ndarray, targets: np.ndarray, num_classes: int = 10):
    """
    preds, targets: [N, num_positions] integer arrays.
    Returns per-position precision/recall/f1 + confusion matrix, plus the
    same aggregated metrics and char/exact accuracy reported during NN training,
    so the two approaches are directly comparable.
    """
    num_positions = preds.shape[1]
    labels = list(range(num_classes))

    correct_chars = (preds == targets)
    char_acc = correct_chars.mean()
    exact_acc = correct_chars.all(axis=1).mean()

    per_position = []
    for pos in range(num_positions):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets[:, pos], preds[:, pos], labels=labels,
            average="macro", zero_division=0,
        )
        cm = confusion_matrix(targets[:, pos], preds[:, pos], labels=labels)
        per_position.append({"position": pos, "precision": precision,
                              "recall": recall, "f1": f1, "confusion_matrix": cm})

    agg_precision, agg_recall, agg_f1, _ = precision_recall_fscore_support(
        targets.flatten(), preds.flatten(), labels=labels,
        average="macro", zero_division=0,
    )
    aggregated_cm = confusion_matrix(targets.flatten(), preds.flatten(), labels=labels)

    return {
        "char_acc": char_acc,
        "exact_acc": exact_acc,
        "per_position": per_position,
        "aggregated": {"precision": agg_precision, "recall": agg_recall,
                        "f1": agg_f1, "confusion_matrix": aggregated_cm},
    }


def print_report(results: dict):
    print(f"char_acc={results['char_acc']:.4f}  exact_acc={results['exact_acc']:.4f}")
    print("\nper-position (precision / recall / f1):")
    for pos_metrics in results["per_position"]:
        print(f"  pos {pos_metrics['position']}: "
              f"P={pos_metrics['precision']:.4f}  "
              f"R={pos_metrics['recall']:.4f}  "
              f"F1={pos_metrics['f1']:.4f}")
    agg = results["aggregated"]
    print(f"\naggregated: P={agg['precision']:.4f}  R={agg['recall']:.4f}  F1={agg['f1']:.4f}")
    print("\naggregated confusion matrix (rows=true digit, cols=predicted digit):")
    print(agg["confusion_matrix"])


# ---------- 8. Train on train.csv, evaluate on val.csv ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-images-dir", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--num-positions", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--n-words", type=int, default=200)
    parser.add_argument("--k-neighbors", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading train set...")
    t0 = time.time()
    train_images, train_labels = load_dataset(
        args.train_csv, args.train_images_dir, num_positions=args.num_positions)
    print(f"  {len(train_images)} images loaded in {time.time() - t0:.1f}s")

    print("Loading val set...")
    t0 = time.time()
    val_images, val_labels = load_dataset(
        args.val_csv, args.val_images_dir, num_positions=args.num_positions)
    print(f"  {len(val_images)} images loaded in {time.time() - t0:.1f}s")

    print("\nFitting AKAZE + BoVW + LDA + kNN pipeline on train...")
    t0 = time.time()
    pipeline = PipelineSevenPositions(
        n_words=args.n_words, k_neighbors=args.k_neighbors,
        n_positions=args.num_positions,
    )
    pipeline.fit(train_images, train_labels)
    print(f"  fit done in {time.time() - t0:.1f}s")

    print("\n--- TRAIN metrics (sanity check, not a generalization estimate) ---")
    train_preds = pipeline.predict(train_images)
    print_report(evaluate(train_preds, train_labels, num_classes=args.num_classes))

    print("\n--- VAL metrics ---")
    val_preds = pipeline.predict(val_images)
    print_report(evaluate(val_preds, val_labels, num_classes=args.num_classes))


if __name__ == "__main__":
    main()