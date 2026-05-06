#!/usr/bin/env python3
"""
Evaluate an ONNX object detection model on a test dataset.

Caches ALL raw predictions (unfiltered) so that metrics can be recomputed
at any threshold later (e.g., for ROC curves) without re-running inference.
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils.coco_converter import jsonl_to_coco
from utils.coco_eval import compute_det_curves, compute_precision_recall, convert_to_xywh, evaluate_coco_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def plot_roc_curve(curve_data: dict, output_dir: Path, iou_threshold: float) -> Path:
    """Plot and save per-class and overall precision-recall curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    class_names = sorted(curve_data['per_class'].keys())
    for idx, cls_name in enumerate(class_names):
        cls = curve_data['per_class'][cls_name]
        if not cls['recall']:
            continue
        color = colors[idx % len(colors)]
        ax.plot(
            cls['recall'], cls['precision'],
            color=color, linewidth=1.5,
            label=f"{cls_name} (AP={cls['ap']:.3f}, n={cls['n_gt']})",
        )

    overall = curve_data['overall']
    if overall['recall']:
        ax.plot(
            overall['recall'], overall['precision'],
            color='black', linewidth=2.5, linestyle='--',
            label=f"Overall (AP={overall['ap']:.3f}, n={overall['n_gt']})",
        )

    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title(f'Precision\u2013Recall Curve (IoU \u2265 {iou_threshold})', fontsize=14)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = output_dir / "precision_recall_curve.png"
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved precision-recall curve to {save_path}")
    return save_path


def _resolve_image_path(dataset_dir: Path, image_path_str: str) -> Path:
    """Resolve an image path from dataset.jsonl relative to the dataset directory."""
    data_dir = dataset_dir / "data"
    if os.path.isabs(image_path_str):
        return Path(image_path_str)
    elif image_path_str.startswith(dataset_dir.name + "/"):
        path_suffix = image_path_str[len(dataset_dir.name) + 1 :]
        return dataset_dir / path_suffix
    else:
        return data_dir / os.path.basename(image_path_str)


def run_inference(
    model_path: str,
    dataset_dir: Path,
    classes: list[str],
    input_h: int,
    input_w: int,
    vertical_flip_p: float = 0.0,
) -> list[dict]:
    """
    Run ONNX inference on all images in a dataset and return COCO-format predictions.

    Returns ALL predictions with no confidence filtering applied.
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type  # e.g. 'tensor(uint8)' or 'tensor(float)'
    expects_float = "float" in input_type

    log.info(f"Model input: name={input_name}, type={input_type}")
    if expects_float:
        log.info("Model expects float32 input (will convert from uint8)")

    # Build label->category_id mapping (1-based, matching jsonl_to_coco)
    label_to_id = {name: idx + 1 for idx, name in enumerate(classes)}

    jsonl_path = dataset_dir / "dataset.jsonl"
    coco_results = []
    image_count = 0

    with open(jsonl_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for image_id, line in enumerate(tqdm(lines, desc="Running inference")):
        sample = json.loads(line)
        image_path_str = sample.get("image_path", "")
        if not image_path_str:
            continue

        img_path = _resolve_image_path(dataset_dir, image_path_str)
        if not img_path.exists():
            continue

        # Load and resize image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((input_w, input_h), Image.BILINEAR)
        img_np = np.array(img_resized).transpose(2, 0, 1)[np.newaxis, ...]  # [1, C, H, W]

        if expects_float:
            img_np = img_np.astype(np.float32)

        # Vertical flip with probability p (test-time augmentation)
        flipped = vertical_flip_p > 0 and random.random() < vertical_flip_p
        if flipped:
            img_np = img_np[:, :, ::-1, :].copy()

        # Run inference
        outputs = session.run(None, {input_name: img_np})
        boxes, labels, scores = outputs

        # Handle batch dimension
        if boxes.ndim == 3 and boxes.shape[0] == 1:
            boxes = boxes[0]
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels[0]
        if scores.ndim == 2 and scores.shape[0] == 1:
            scores = scores[0]

        # Filter invalid (score <= 0)
        valid = scores > 0
        boxes = boxes[valid]
        labels = labels[valid]
        scores = scores[valid]

        if len(boxes) == 0:
            image_count += 1
            continue

        # Unflip boxes back to original orientation so they match ground truth
        if flipped:
            y1 = boxes[:, 1].copy()
            y2 = boxes[:, 3].copy()
            boxes[:, 1] = input_h - y2
            boxes[:, 3] = input_h - y1

        # Scale boxes from model input coords to original image coords
        scale_w = orig_w / input_w
        scale_h = orig_h / input_h
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] *= scale_w
        boxes_scaled[:, [1, 3]] *= scale_h

        # Convert [x1, y1, x2, y2] -> COCO [x, y, w, h]
        boxes_tensor = torch.from_numpy(boxes_scaled).float()
        boxes_xywh = convert_to_xywh(boxes_tensor)

        for box, score, label in zip(boxes_xywh, scores, labels):
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(score),
                }
            )

        image_count += 1

    log.info(f"Processed {image_count} images, collected {len(coco_results)} predictions")
    return coco_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an ONNX model and cache all raw predictions"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--labels",
        default="omni-detector-0_0_1-rc8/labels.txt",
        help="Path to labels.txt (one class per line)",
    )
    parser.add_argument(
        "--test-data",
        default="omni_2.17_test",
        help="Path to test dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save predictions cache and metrics",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.2,
        help="IoU threshold for precision/recall computation",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for precision/recall computation",
    )
    parser.add_argument(
        "--vertical-flip",
        type=float,
        default=0.0,
        metavar="P",
        help="Apply random vertical flip with probability P before inference (default: 0, disabled)",
    )
    parser.add_argument(
        "--plot-roc",
        action="store_true",
        help="Save a precision-recall curve plot to the output directory",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_dir = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read labels
    labels_path = Path(args.labels)
    classes = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
    log.info(f"Classes: {classes}")

    # Get model input shape
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    input_h, input_w = input_shape[2], input_shape[3]
    log.info(f"Model input size: {input_h}x{input_w}")
    del session

    # Convert ground truth to COCO format
    gt_path = output_dir / "ground_truth_coco.json"
    data_dir = dataset_dir / "data"
    jsonl_path = dataset_dir / "dataset.jsonl"
    jsonl_to_coco(
        jsonl_path=str(jsonl_path),
        data_dir=str(data_dir),
        output_path=str(gt_path),
        classes=classes,
    )
    coco_gt = COCO(str(gt_path))
    log.info(f"Ground truth: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")

    # Run inference and collect ALL predictions (unfiltered)
    if args.vertical_flip > 0:
        log.info(f"Vertical flip enabled with p={args.vertical_flip}")
    predictions = run_inference(
        model_path=str(model_path),
        dataset_dir=dataset_dir,
        classes=classes,
        input_h=input_h,
        input_w=input_w,
        vertical_flip_p=args.vertical_flip,
    )

    # Cache all predictions
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f)
    log.info(f"Cached {len(predictions)} predictions to {predictions_path}")

    # Compute COCO metrics
    log.info("=" * 60)
    log.info("COCO Evaluation")
    log.info("=" * 60)
    metrics = evaluate_coco_predictions(predictions, coco_gt, verbose=True)

    for key in ["AP", "AP50", "AP75", "AR100"]:
        log.info(f"  {key}: {metrics[key]:.4f}")

    # Compute precision/recall at specified thresholds
    log.info("=" * 60)
    log.info(
        f"Precision/Recall (IoU>={args.iou_threshold}, conf>={args.confidence_threshold})"
    )
    log.info("=" * 60)
    pr_results = compute_precision_recall(
        predictions=predictions,
        coco_gt=coco_gt,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    ov = pr_results["overall"]
    class_names = sorted(pr_results["per_class"])
    col_w = max(len(n) for n in class_names + ["OVERALL"])
    col_w = max(col_w, 7)
    log.info(
        f"  {'Class':<{col_w}s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}"
    )
    log.info(
        f"  {'-'*col_w}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}"
    )
    for cls_name in class_names:
        c = pr_results["per_class"][cls_name]
        log.info(
            f"  {cls_name:<{col_w}s}  {c['precision']:6.3f}  {c['recall']:6.3f}  "
            f"{c['f1']:6.3f}  {c['tp']:5d}  {c['fp']:5d}  {c['fn']:5d}"
        )
    log.info(
        f"  {'-'*col_w}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}"
    )
    log.info(
        f"  {'OVERALL':<{col_w}s}  {ov['precision']:6.3f}  {ov['recall']:6.3f}  "
        f"{ov['f1']:6.3f}  {ov['tp']:5d}  {ov['fp']:5d}  {ov['fn']:5d}"
    )

    # Precision-recall curve
    if args.plot_roc:
        log.info("Computing precision-recall curves...")
        curve_data = compute_det_curves(
            predictions=predictions,
            coco_gt=coco_gt,
            iou_threshold=args.iou_threshold,
        )
        plot_roc_curve(curve_data, output_dir, args.iou_threshold)

    # Save metrics
    metrics["precision_recall"] = pr_results
    metrics["model"] = str(model_path)
    metrics["dataset"] = str(dataset_dir)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
