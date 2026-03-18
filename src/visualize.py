"""
Standalone visualization script that draws predictions + ground truth boxes
from JSON files produced by eval.py. No GPU, model, or Hydra required.

Usage:
    python src/visualize.py <dataset_dir> <eval_dir> [options]

Example:
    python src/visualize.py datasets/my_dataset outputs/18-08-48/eval_my_dataset_best_model_pth
    python src/visualize.py datasets/my_dataset outputs/18-08-48/eval_my_dataset_best_model_pth \
        --confidence-threshold 0.5 --max-images 20
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_file(directory: Path, patterns: list[str]) -> Path | None:
    """Return the first file matching any of the glob patterns."""
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_ground_truth(gt_path: Path):
    """Load COCO ground-truth JSON and build lookup structures."""
    with open(gt_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    return images, categories, gt_by_image


def load_predictions(pred_path: Path):
    """Load predictions JSON and group by image_id."""
    with open(pred_path) as f:
        predictions = json.load(f)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        preds_by_image[pred["image_id"]].append(pred)

    return preds_by_image


def draw_image(
    img: np.ndarray,
    preds: list[dict],
    gt_anns: list[dict],
    categories: dict[int, str],
    confidence_threshold: float,
    title: str,
    output_path: Path,
    filter_labels: set[str] | None = None,
):
    """Draw predictions (red) and ground truth (lime) on an image and save.

    If filter_labels is provided, only boxes whose category name is in the set
    are drawn (applies to both predictions and ground truth).
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    def _cat_name(cat_id):
        return categories.get(cat_id, str(cat_id))

    # Predictions: red solid boxes (COCO bbox format [x, y, w, h])
    for pred in preds:
        score = pred["score"]
        if score <= confidence_threshold:
            continue
        if filter_labels and _cat_name(pred["category_id"]) not in filter_labels:
            continue
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x, y - 5,
            f'{_cat_name(pred["category_id"])} {score:.2f}',
            color="red", fontsize=7,
        )

    # Ground truth: lime dashed boxes (COCO bbox format [x, y, w, h])
    for ann in gt_anns:
        if filter_labels and _cat_name(ann["category_id"]) not in filter_labels:
            continue
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1,
            edgecolor="lime", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(x, y + h + 12, _cat_name(ann["category_id"]), color="lime", fontsize=7)

    plt.title(title)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Draw predictions + ground truth from eval JSON files (no GPU needed).",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset directory (must contain data/ with images)",
    )
    parser.add_argument(
        "eval_dir",
        type=Path,
        help="Path to eval output directory (containing predictions + ground truth JSONs)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Only draw predictions above this score (default: 0.7)",
    )
    parser.add_argument(
        "--predictions-file", type=Path, default=None,
        help="Predictions JSON path (default: auto-detect *_predictions.json in eval_dir)",
    )
    parser.add_argument(
        "--gt-file", type=Path, default=None,
        help="Ground truth COCO JSON path (default: auto-detect *_coco.json in eval_dir)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for visualizations (default: eval_dir/visualizations/)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Maximum number of images to visualize (default: all)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Only draw boxes for these label names (default: all labels)",
    )
    parser.add_argument(
        "--only-with-predictions", action="store_true",
        help="Only save images that have at least one prediction above the confidence threshold",
    )
    args = parser.parse_args()

    # --- Validate dataset_dir ---
    data_dir = args.dataset_dir / "data"
    if not data_dir.is_dir():
        log.error(f"data/ directory not found in {args.dataset_dir}")
        sys.exit(1)

    # --- Locate JSON files ---
    if args.predictions_file:
        pred_path = args.predictions_file
    else:
        pred_path = find_file(args.eval_dir, ["*_predictions.json"])
    if pred_path is None or not pred_path.exists():
        log.error(f"Predictions JSON not found in {args.eval_dir}")
        sys.exit(1)

    if args.gt_file:
        gt_path = args.gt_file
    else:
        gt_path = find_file(args.eval_dir, ["ground_truth_coco.json", "*_coco.json"])
    if gt_path is None or not gt_path.exists():
        log.error(f"Ground truth COCO JSON not found in {args.eval_dir}")
        sys.exit(1)

    output_dir = args.output_dir or (args.eval_dir / "visualizations")
    filter_labels = set(args.labels) if args.labels else None

    log.info(f"Predictions : {pred_path}")
    log.info(f"Ground truth: {gt_path}")
    log.info(f"Images from : {data_dir}")
    log.info(f"Output dir  : {output_dir}")
    log.info(f"Confidence  : {args.confidence_threshold}")
    if filter_labels:
        log.info(f"Filter labels: {filter_labels}")

    # --- Load data ---
    images, categories, gt_by_image = load_ground_truth(gt_path)
    preds_by_image = load_predictions(pred_path)

    def _cat_name(cat_id):
        return categories.get(cat_id, str(cat_id))

    log.info(f"Loaded {len(images)} images, {len(categories)} categories, "
             f"{sum(len(v) for v in preds_by_image.values())} predictions")

    # --- Draw ---
    image_ids = sorted(images.keys())
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    drawn = 0
    skipped = 0
    for image_id in tqdm(image_ids, desc="Drawing"):
        img_info = images[image_id]
        file_name = img_info["file_name"]
        img_path = data_dir / file_name

        if not img_path.exists():
            log.warning(f"Image not found: {img_path}, skipping")
            skipped += 1
            continue

        img = np.array(Image.open(img_path).convert("RGB"))

        preds = preds_by_image.get(image_id, [])
        gt_anns = gt_by_image.get(image_id, [])

        if args.only_with_predictions:
            has_pred = any(
                p["score"] > args.confidence_threshold
                and (not filter_labels or _cat_name(p["category_id"]) in filter_labels)
                for p in preds
            )
            if not has_pred:
                skipped += 1
                continue

        stem = Path(file_name).stem
        out_path = output_dir / f"Image_{image_id}_{stem}.png"

        draw_image(
            img=img,
            preds=preds,
            gt_anns=gt_anns,
            categories=categories,
            confidence_threshold=args.confidence_threshold,
            title=f"Image {image_id}",
            output_path=out_path,
            filter_labels=filter_labels,
        )
        drawn += 1

    log.info(f"Done. Drew {drawn} images, skipped {skipped}. Output: {output_dir}")


if __name__ == "__main__":
    main()
