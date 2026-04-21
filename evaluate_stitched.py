#!/usr/bin/env python3
"""
Stitched-pair evaluation for the omni-detector.

Finds single-view source images in a Viam dataset (entries tagged
``single_view`` in ``classification_annotations``), randomly pairs them,
stitches each pair side-by-side into a synthetic "one screen, two views"
image, and runs the ONNX model three ways:

1. On the stitched image (one forward pass).
2. On each source image alone (two forward passes), with predictions remapped
   into stitched coordinates.

It reports two metric sets:

- ``vs_ground_truth``: stitched predictions vs. the transposed ground-truth
  bboxes. Absolute accuracy on the stitched regime.
- ``vs_per_view_predictions``: stitched predictions vs. per-view predictions
  (treated as pseudo-GT after confidence filtering). Whether stitching itself
  costs accuracy relative to how the model behaves on each view alone.
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils.coco_eval import (
    compute_precision_recall,
    convert_to_xywh,
    evaluate_coco_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SINGLE_VIEW_TAG = "single_view"


def _resolve_image_path(dataset_dir: Path, image_path_str: str) -> Path:
    """Resolve a dataset.jsonl image_path relative to the dataset directory."""
    data_dir = dataset_dir / "data"
    if os.path.isabs(image_path_str):
        return Path(image_path_str)
    elif image_path_str.startswith(dataset_dir.name + "/"):
        return dataset_dir / image_path_str[len(dataset_dir.name) + 1 :]
    else:
        return data_dir / os.path.basename(image_path_str)


def find_single_view_samples(jsonl_path: Path, dataset_dir: Path) -> List[Dict]:
    """Return entries tagged ``single_view`` (in ``classification_annotations``)
    whose image exists on disk."""
    samples: List[Dict] = []
    total = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tags = {
                a.get("annotation_label")
                for a in entry.get("classification_annotations", []) or []
            }
            if SINGLE_VIEW_TAG not in tags:
                continue
            image_path_str = entry.get("image_path", "")
            if not image_path_str:
                continue
            resolved = _resolve_image_path(dataset_dir, image_path_str)
            if not resolved.exists():
                continue
            entry["_resolved_path"] = str(resolved)
            samples.append(entry)
    log.info(
        f"Scanned {total} JSONL entries; found {len(samples)} single-view samples "
        f"(tagged '{SINGLE_VIEW_TAG}', image exists)."
    )
    return samples


def pair_samples(samples: List[Dict], seed: int) -> List[Tuple[Dict, Dict]]:
    """Shuffle deterministically and pair consecutive samples."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    pairs: List[Tuple[Dict, Dict]] = [
        (samples[indices[i]], samples[indices[i + 1]])
        for i in range(0, len(indices) - 1, 2)
    ]
    if len(indices) % 2 == 1:
        dropped = samples[indices[-1]]["_resolved_path"]
        log.warning(f"Odd number of single-view samples; dropped {dropped}")
    log.info(f"Formed {len(pairs)} stitched pairs (seed={seed})")
    return pairs


def resize_to(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """Resize a PIL image to (target_h, target_w). Return HWC uint8 numpy."""
    resized = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(resized)


def stitch_images(
    left_img: Image.Image, right_img: Image.Image, input_h: int, input_w: int
) -> np.ndarray:
    """Resize both images to (input_h, input_w // 2) and concat along width."""
    half_w = input_w // 2
    left = resize_to(left_img, input_h, half_w)
    right = resize_to(right_img, input_h, half_w)
    stitched = np.concatenate([left, right], axis=1)
    return stitched


def onnx_inference(
    session: ort.InferenceSession,
    input_name: str,
    expects_float: bool,
    img_hwc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ONNX on HWC uint8 image already sized to the model input.

    Returns (boxes [N,4] in x1y1x2y2 pixel coords, labels [N], scores [N]).
    """
    img_nchw = img_hwc.transpose(2, 0, 1)[np.newaxis, ...]
    if expects_float:
        img_nchw = img_nchw.astype(np.float32)
    outputs = session.run(None, {input_name: img_nchw})
    boxes, labels, scores = outputs
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    valid = scores > 0
    return boxes[valid], labels[valid], scores[valid]


def boxes_to_coco_preds(
    boxes_x1y1x2y2: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    image_id: int,
) -> List[Dict]:
    """Convert [N,4] x1y1x2y2 boxes to a list of COCO prediction dicts (xywh)."""
    if len(boxes_x1y1x2y2) == 0:
        return []
    boxes_xywh = convert_to_xywh(torch.from_numpy(boxes_x1y1x2y2).float()).numpy()
    return [
        {
            "image_id": image_id,
            "category_id": int(label),
            "bbox": box.tolist(),
            "score": float(score),
        }
        for box, label, score in zip(boxes_xywh, labels, scores)
    ]


def transpose_pixel_boxes(
    boxes_x1y1x2y2: np.ndarray, half: str, input_w: int
) -> np.ndarray:
    """Map pixel boxes from per-view inference coords into stitched coords.

    Per-view inference runs at (input_h, input_w); the view occupies half the
    stitched image horizontally, so x coordinates are halved, and the right
    half is additionally offset by input_w / 2.
    """
    if boxes_x1y1x2y2.size == 0:
        return boxes_x1y1x2y2
    out = boxes_x1y1x2y2.copy().astype(np.float32)
    out[:, [0, 2]] = out[:, [0, 2]] / 2.0
    if half == "right":
        out[:, [0, 2]] = out[:, [0, 2]] + (input_w / 2.0)
    return out


def build_gt_coco(
    pairs: List[Tuple[Dict, Dict]],
    classes: List[str],
    input_h: int,
    input_w: int,
) -> Dict:
    """Build a COCO ground-truth dict for the stitched pairs.

    Each stitched image is assigned image_id = pair index. Bboxes from the
    left source get x halved; bboxes from the right source get x halved and
    shifted by input_w / 2. Only labels in ``classes`` are kept (sonar_view
    is excluded because it is not in ``classes``).
    """
    label_to_id = {name: idx + 1 for idx, name in enumerate(classes)}
    coco: Dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx + 1, "name": name, "supercategory": "none"}
            for idx, name in enumerate(classes)
        ],
    }
    annotation_id = 1
    for pair_idx, (left, right) in enumerate(pairs):
        image_id = pair_idx
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"stitched_{pair_idx:05d}.jpg",
                "width": input_w,
                "height": input_h,
            }
        )
        for source, half in ((left, "left"), (right, "right")):
            x_offset_norm = 0.0 if half == "left" else 0.5
            for bbox in source.get("bounding_box_annotations", []) or []:
                label = bbox.get("annotation_label")
                if label not in label_to_id:
                    continue
                x_min_norm = bbox.get("x_min_normalized")
                y_min_norm = bbox.get("y_min_normalized")
                x_max_norm = bbox.get("x_max_normalized")
                y_max_norm = bbox.get("y_max_normalized")
                if None in (x_min_norm, y_min_norm, x_max_norm, y_max_norm):
                    continue
                x_min_s_norm = x_min_norm / 2.0 + x_offset_norm
                x_max_s_norm = x_max_norm / 2.0 + x_offset_norm
                x_min = x_min_s_norm * input_w
                y_min = y_min_norm * input_h
                x_max = x_max_s_norm * input_w
                y_max = y_max_norm * input_h
                if x_max <= x_min or y_max <= y_min:
                    continue
                w = x_max - x_min
                h = y_max - y_min
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": label_to_id[label],
                        "bbox": [x_min, y_min, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
    return coco


def build_pseudo_gt_coco(
    per_view_predictions: List[Dict],
    classes: List[str],
    n_pairs: int,
    input_h: int,
    input_w: int,
    confidence_threshold: float,
) -> Dict:
    """Wrap per-view predictions (already in stitched xywh coords) as COCO GT."""
    coco: Dict = {
        "images": [
            {
                "id": i,
                "file_name": f"stitched_{i:05d}.jpg",
                "width": input_w,
                "height": input_h,
            }
            for i in range(n_pairs)
        ],
        "annotations": [],
        "categories": [
            {"id": idx + 1, "name": name, "supercategory": "none"}
            for idx, name in enumerate(classes)
        ],
    }
    annotation_id = 1
    for pred in per_view_predictions:
        if pred["score"] < confidence_threshold:
            continue
        _, _, w, h = pred["bbox"]
        if w <= 0 or h <= 0:
            continue
        coco["annotations"].append(
            {
                "id": annotation_id,
                "image_id": pred["image_id"],
                "category_id": pred["category_id"],
                "bbox": pred["bbox"],
                "area": w * h,
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    return coco


def draw_visualization(
    stitched_hwc: np.ndarray,
    gt_annotations: List[Dict],
    stitched_preds: List[Dict],
    per_view_preds: List[Dict],
    cat_id_to_name: Dict[int, str],
    confidence_threshold: float,
    output_path: Path,
    input_w: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.imshow(stitched_hwc)
    ax.axvline(input_w / 2, color="yellow", linestyle=":", linewidth=1, alpha=0.6)

    for ann in gt_annotations:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
        ax.text(x, y - 4, f"GT:{name}", color="lime", fontsize=8)

    for pred in stitched_preds:
        if pred["score"] < confidence_threshold:
            continue
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        name = cat_id_to_name.get(pred["category_id"], str(pred["category_id"]))
        ax.text(
            x, y + h + 12, f"Stitched:{name} {pred['score']:.2f}", color="red", fontsize=7
        )

    for pred in per_view_preds:
        if pred["score"] < confidence_threshold:
            continue
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor="orange", facecolor="none", linestyle=":"
        )
        ax.add_patch(rect)
        name = cat_id_to_name.get(pred["category_id"], str(pred["category_id"]))
        ax.text(
            x + w,
            y - 4,
            f"PerView:{name} {pred['score']:.2f}",
            color="orange",
            fontsize=7,
            ha="right",
        )

    ax.set_title(
        "Stitched-pass preds (red solid)  |  Per-view-pass preds (orange dotted)  |  GT (lime dashed)"
    )
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_pr(label: str, pr_results: Dict) -> None:
    ov = pr_results["overall"]
    class_names = sorted(pr_results["per_class"])
    col_w = max([len(n) for n in class_names + ["OVERALL"]] + [7])
    log.info(f"  [{label}]")
    log.info(
        f"  {'Class':<{col_w}s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  "
        f"{'TP':>5s}  {'FP':>5s}  {'FN':>5s}"
    )
    log.info(f"  {'-'*col_w}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for name in class_names:
        c = pr_results["per_class"][name]
        log.info(
            f"  {name:<{col_w}s}  {c['precision']:6.3f}  {c['recall']:6.3f}  "
            f"{c['f1']:6.3f}  {c['tp']:5d}  {c['fp']:5d}  {c['fn']:5d}"
        )
    log.info(f"  {'-'*col_w}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    log.info(
        f"  {'OVERALL':<{col_w}s}  {ov['precision']:6.3f}  {ov['recall']:6.3f}  "
        f"{ov['f1']:6.3f}  {ov['tp']:5d}  {ov['fp']:5d}  {ov['fn']:5d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the omni-detector on synthetic stitched pairs of "
            "single-view screenshots."
        )
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--labels", required=True, help="Path to labels.txt (one class per line)"
    )
    parser.add_argument(
        "--test-data", required=True, help="Path to dataset directory (contains dataset.jsonl, data/)"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for precision/recall reporting",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for precision/recall reporting",
    )
    parser.add_argument(
        "--pseudo-gt-confidence",
        type=float,
        default=0.5,
        help="Per-view predictions >= this score are treated as pseudo-GT",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducible pairing"
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of sample stitched images to save with overlays",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [l.strip() for l in Path(args.labels).read_text().splitlines() if l.strip()]
    log.info(f"Classes (category_id 1..N, in order): {classes}")
    cat_id_to_name = {idx + 1: name for idx, name in enumerate(classes)}

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type
    expects_float = "float" in input_type
    input_shape = input_info.shape
    try:
        input_h = int(input_shape[2])
        input_w = int(input_shape[3])
    except (TypeError, ValueError):
        log.error(
            f"ONNX model has dynamic input spatial dims {input_shape}; "
            f"cannot determine stitch size."
        )
        sys.exit(1)
    if input_w % 2 != 0:
        log.error(
            f"ONNX input width {input_w} is odd; cannot split evenly for side-by-side stitching."
        )
        sys.exit(1)
    log.info(
        f"ONNX input: name={input_name}, type={input_type}, shape=({input_h}, {input_w})"
    )

    jsonl_path = dataset_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        log.error(f"dataset.jsonl not found at {jsonl_path}")
        sys.exit(1)

    samples = find_single_view_samples(jsonl_path, dataset_dir)
    if len(samples) < 2:
        log.error(
            f"Need at least 2 single-view samples; found {len(samples)}. Exiting."
        )
        sys.exit(1)

    pairs = pair_samples(samples, args.seed)
    pairs_record = {
        str(i): [left["_resolved_path"], right["_resolved_path"]]
        for i, (left, right) in enumerate(pairs)
    }
    (output_dir / "pairs.json").write_text(json.dumps(pairs_record, indent=2))

    gt_coco_dict = build_gt_coco(pairs, classes, input_h, input_w)
    gt_path = output_dir / "stitched_ground_truth_coco.json"
    gt_path.write_text(json.dumps(gt_coco_dict, indent=2))
    coco_gt = COCO(str(gt_path))
    log.info(
        f"Stitched GT: {len(gt_coco_dict['images'])} images, "
        f"{len(gt_coco_dict['annotations'])} annotations"
    )

    gt_by_image: Dict[int, List[Dict]] = {}
    for ann in gt_coco_dict["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    stitched_predictions: List[Dict] = []
    per_view_predictions: List[Dict] = []

    viz_dir = None
    if args.visualize > 0:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    for pair_idx, (left, right) in enumerate(tqdm(pairs, desc="Inference")):
        left_img = Image.open(left["_resolved_path"]).convert("RGB")
        right_img = Image.open(right["_resolved_path"]).convert("RGB")

        stitched_hwc = stitch_images(left_img, right_img, input_h, input_w)
        sb, sl, ss = onnx_inference(session, input_name, expects_float, stitched_hwc)
        stitched_for_image = boxes_to_coco_preds(sb, sl, ss, pair_idx)
        stitched_predictions.extend(stitched_for_image)

        left_hwc = resize_to(left_img, input_h, input_w)
        right_hwc = resize_to(right_img, input_h, input_w)
        lb, ll, lsc = onnx_inference(session, input_name, expects_float, left_hwc)
        rb, rl, rsc = onnx_inference(session, input_name, expects_float, right_hwc)
        lb_t = transpose_pixel_boxes(lb, "left", input_w)
        rb_t = transpose_pixel_boxes(rb, "right", input_w)
        per_view_for_image = boxes_to_coco_preds(lb_t, ll, lsc, pair_idx)
        per_view_for_image += boxes_to_coco_preds(rb_t, rl, rsc, pair_idx)
        per_view_predictions.extend(per_view_for_image)

        if viz_dir is not None and pair_idx < args.visualize:
            draw_visualization(
                stitched_hwc=stitched_hwc,
                gt_annotations=gt_by_image.get(pair_idx, []),
                stitched_preds=stitched_for_image,
                per_view_preds=per_view_for_image,
                cat_id_to_name=cat_id_to_name,
                confidence_threshold=args.confidence_threshold,
                output_path=viz_dir / f"pair_{pair_idx:05d}.png",
                input_w=input_w,
            )

    (output_dir / "stitched_predictions.json").write_text(
        json.dumps(stitched_predictions)
    )
    (output_dir / "per_view_predictions.json").write_text(
        json.dumps(per_view_predictions)
    )
    log.info(
        f"Stitched predictions: {len(stitched_predictions)} | "
        f"Per-view predictions: {len(per_view_predictions)}"
    )

    log.info("=" * 60)
    log.info("Metric set 1: stitched predictions vs. ground truth")
    log.info("=" * 60)
    metrics_gt = evaluate_coco_predictions(stitched_predictions, coco_gt, verbose=True)
    for key in ("AP", "AP50", "AP75", "AR100"):
        log.info(f"  {key}: {metrics_gt[key]:.4f}")
    pr_gt = compute_precision_recall(
        predictions=stitched_predictions,
        coco_gt=coco_gt,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    _log_pr("vs_ground_truth", pr_gt)
    metrics_gt["precision_recall"] = pr_gt

    log.info("=" * 60)
    log.info("Metric set 2: stitched predictions vs. per-view predictions (pseudo-GT)")
    log.info("=" * 60)
    pseudo_gt_dict = build_pseudo_gt_coco(
        per_view_predictions,
        classes,
        len(pairs),
        input_h,
        input_w,
        args.pseudo_gt_confidence,
    )
    pseudo_gt_path = output_dir / "per_view_pseudo_gt_coco.json"
    pseudo_gt_path.write_text(json.dumps(pseudo_gt_dict, indent=2))
    coco_pseudo_gt = COCO(str(pseudo_gt_path))
    log.info(
        f"Pseudo-GT: {len(pseudo_gt_dict['images'])} images, "
        f"{len(pseudo_gt_dict['annotations'])} annotations "
        f"(per-view preds filtered at score >= {args.pseudo_gt_confidence})"
    )
    metrics_pseudo = evaluate_coco_predictions(
        stitched_predictions, coco_pseudo_gt, verbose=True
    )
    for key in ("AP", "AP50", "AP75", "AR100"):
        log.info(f"  {key}: {metrics_pseudo[key]:.4f}")
    pr_pseudo = compute_precision_recall(
        predictions=stitched_predictions,
        coco_gt=coco_pseudo_gt,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    _log_pr("vs_per_view_predictions", pr_pseudo)
    metrics_pseudo["precision_recall"] = pr_pseudo

    all_metrics = {
        "model": str(Path(args.model).resolve()),
        "dataset": str(dataset_dir.resolve()),
        "n_pairs": len(pairs),
        "n_single_view_samples_found": len(samples),
        "input_size": [input_h, input_w],
        "seed": args.seed,
        "iou_threshold": args.iou_threshold,
        "confidence_threshold": args.confidence_threshold,
        "pseudo_gt_confidence": args.pseudo_gt_confidence,
        "vs_ground_truth": metrics_gt,
        "vs_per_view_predictions": metrics_pseudo,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    log.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
