#!/usr/bin/env python3
"""
Compare evaluation results between two ONNX models (e.g., original vs quantized).

Loads cached predictions from evaluate_onnx.py runs and recomputes metrics
at specified thresholds. No inference needed.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from pycocotools.coco import COCO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils.coco_eval import compute_precision_recall, evaluate_coco_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_cached(eval_dir: Path) -> tuple[list[dict], COCO]:
    """Load cached predictions and ground truth from an evaluate_onnx.py output dir."""
    predictions_path = eval_dir / "predictions.json"
    gt_path = eval_dir / "ground_truth_coco.json"

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.json not found in {eval_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"ground_truth_coco.json not found in {eval_dir}")

    with open(predictions_path) as f:
        predictions = json.load(f)
    coco_gt = COCO(str(gt_path))

    return predictions, coco_gt


def main():
    parser = argparse.ArgumentParser(
        description="Compare two ONNX model evaluation results from cached predictions"
    )
    parser.add_argument(
        "--original-dir",
        default="quantization_results/original",
        help="Directory with cached predictions for the original model",
    )
    parser.add_argument(
        "--quantized-dir",
        default="quantization_results/quantized",
        help="Directory with cached predictions for the quantized model",
    )
    parser.add_argument(
        "--original-model",
        default=None,
        help="Path to original ONNX model (optional, for file size comparison)",
    )
    parser.add_argument(
        "--quantized-model",
        default=None,
        help="Path to quantized ONNX model (optional, for file size comparison)",
    )
    parser.add_argument(
        "--output",
        default="quantization_results/comparison.json",
        help="Path to save comparison JSON",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.2,
        help="IoU threshold for precision/recall",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for precision/recall",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    quantized_dir = Path(args.quantized_dir)

    # Load cached results
    log.info("Loading cached predictions...")
    orig_preds, orig_gt = load_cached(original_dir)
    quant_preds, quant_gt = load_cached(quantized_dir)
    log.info(f"  Original:  {len(orig_preds)} predictions")
    log.info(f"  Quantized: {len(quant_preds)} predictions")

    # Compute COCO metrics for both
    log.info("Computing COCO metrics...")
    orig_coco = evaluate_coco_predictions(orig_preds, orig_gt, verbose=False)
    quant_coco = evaluate_coco_predictions(quant_preds, quant_gt, verbose=False)

    # Compute precision/recall at specified thresholds
    orig_pr = compute_precision_recall(
        orig_preds, orig_gt, args.iou_threshold, args.confidence_threshold
    )
    quant_pr = compute_precision_recall(
        quant_preds, quant_gt, args.iou_threshold, args.confidence_threshold
    )

    # Model sizes (optional)
    orig_size_mb = None
    quant_size_mb = None
    if args.original_model:
        orig_model = Path(args.original_model)
        if orig_model.exists():
            orig_size_mb = orig_model.stat().st_size / (1024 * 1024)
    if args.quantized_model:
        quant_model = Path(args.quantized_model)
        if quant_model.exists():
            quant_size_mb = quant_model.stat().st_size / (1024 * 1024)

    # Print comparison
    print()
    print("=" * 70)
    print("MODEL COMPARISON: Original vs Quantized")
    print("=" * 70)

    if orig_size_mb and quant_size_mb:
        reduction = (1 - quant_size_mb / orig_size_mb) * 100
        print(f"\nModel Size:")
        print(f"  Original:  {orig_size_mb:.1f} MB")
        print(f"  Quantized: {quant_size_mb:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")

    # COCO metrics comparison
    print(f"\nCOCO Metrics:")
    print(f"  {'Metric':<8s}  {'Original':>10s}  {'Quantized':>10s}  {'Diff':>10s}  {'Diff%':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    coco_keys = ["AP", "AP50", "AP75", "AR100"]
    for key in coco_keys:
        orig_val = orig_coco.get(key, 0.0)
        quant_val = quant_coco.get(key, 0.0)
        diff = quant_val - orig_val
        diff_pct = (diff / orig_val * 100) if orig_val > 0 else 0
        print(
            f"  {key:<8s}  {orig_val:10.4f}  {quant_val:10.4f}  {diff:+10.4f}  {diff_pct:+7.2f}%"
        )

    # Precision/Recall comparison
    print(
        f"\nPrecision/Recall (IoU>={args.iou_threshold}, conf>={args.confidence_threshold}):"
    )

    orig_ov = orig_pr["overall"]
    quant_ov = quant_pr["overall"]

    # Per-class
    class_names = sorted(
        set(orig_pr["per_class"].keys()) | set(quant_pr["per_class"].keys())
    )
    col_w = max(len(n) for n in class_names + ["OVERALL"])
    col_w = max(col_w, 7)

    header = (
        f"  {'Class':<{col_w}s}  "
        f"{'Prec(O)':>7s}  {'Prec(Q)':>7s}  "
        f"{'Rec(O)':>7s}  {'Rec(Q)':>7s}  "
        f"{'F1(O)':>7s}  {'F1(Q)':>7s}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for cls_name in class_names:
        oc = orig_pr["per_class"].get(cls_name, {"precision": 0, "recall": 0, "f1": 0})
        qc = quant_pr["per_class"].get(cls_name, {"precision": 0, "recall": 0, "f1": 0})
        print(
            f"  {cls_name:<{col_w}s}  "
            f"{oc['precision']:7.3f}  {qc['precision']:7.3f}  "
            f"{oc['recall']:7.3f}  {qc['recall']:7.3f}  "
            f"{oc['f1']:7.3f}  {qc['f1']:7.3f}"
        )

    print(f"  {'-' * (len(header) - 2)}")
    print(
        f"  {'OVERALL':<{col_w}s}  "
        f"{orig_ov['precision']:7.3f}  {quant_ov['precision']:7.3f}  "
        f"{orig_ov['recall']:7.3f}  {quant_ov['recall']:7.3f}  "
        f"{orig_ov['f1']:7.3f}  {quant_ov['f1']:7.3f}"
    )
    print()

    # Save comparison JSON
    comparison = {
        "thresholds": {
            "iou": args.iou_threshold,
            "confidence": args.confidence_threshold,
        },
        "model_sizes": {
            "original_mb": orig_size_mb,
            "quantized_mb": quant_size_mb,
        },
        "coco_metrics": {
            key: {
                "original": orig_coco.get(key, 0.0),
                "quantized": quant_coco.get(key, 0.0),
                "difference": quant_coco.get(key, 0.0) - orig_coco.get(key, 0.0),
            }
            for key in coco_keys
        },
        "precision_recall": {
            "original": orig_pr,
            "quantized": quant_pr,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=4)
    log.info(f"Comparison saved to {output_path}")


if __name__ == "__main__":
    main()
