#!/usr/bin/env python3
"""
Static INT8 quantization of an ONNX object detection model.

Uses calibration images from a dataset (JSONL format) to compute
activation ranges for accurate quantization. Pre-screens images to
ensure they produce detections (avoids empty tensor errors in ROI pooling
observer nodes during calibration).
"""
import argparse
import json
import logging
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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


def _load_image_paths(dataset_dir: Path) -> list[Path]:
    """Read image paths from dataset.jsonl."""
    jsonl_path = dataset_dir / "dataset.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {dataset_dir}")

    paths = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            image_path_str = sample.get("image_path", "")
            if not image_path_str:
                continue
            resolved = _resolve_image_path(dataset_dir, image_path_str)
            if resolved.exists():
                paths.append(resolved)
    return paths


def _load_image_as_uint8(img_path: Path, input_h: int, input_w: int) -> np.ndarray:
    """Load an image, resize, and return as uint8 [1, 3, H, W] numpy array."""
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((input_w, input_h), Image.BILINEAR)
    img_np = np.array(img_resized)  # [H, W, C] uint8
    img_np = img_np.transpose(2, 0, 1)  # [C, H, W]
    return img_np[np.newaxis, ...]  # [1, C, H, W]


def _prescreen_calibration_images(
    all_paths: list[Path],
    model_path: str,
    input_name: str,
    input_h: int,
    input_w: int,
    num_needed: int,
    score_threshold: float = 0.1,
    max_candidates: int = 500,
    seed: int = 42,
) -> list[Path]:
    """
    Pre-screen images to find ones that produce detections.

    During static quantization calibration, observer nodes (ReduceMax/Min)
    are inserted into the graph. These crash on empty tensors when ROI pooling
    receives no proposals. Pre-screening ensures all calibration images
    produce at least one detection.
    """
    rng = random.Random(seed)
    candidates = list(all_paths)
    rng.shuffle(candidates)
    candidates = candidates[:max_candidates]

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    good_images = []
    checked = 0

    log.info(
        f"Pre-screening images for calibration (need {num_needed}, "
        f"checking up to {len(candidates)})..."
    )

    for img_path in candidates:
        checked += 1
        try:
            img_np = _load_image_as_uint8(img_path, input_h, input_w)
            outputs = session.run(None, {input_name: img_np})
            _, _, scores = outputs

            # Check if any detection has score above threshold
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
            if len(scores) > 0 and np.any(scores > score_threshold):
                good_images.append(img_path)
                if len(good_images) >= num_needed:
                    break
        except Exception:
            continue

    log.info(
        f"Found {len(good_images)} images with detections "
        f"(checked {checked}/{len(candidates)})"
    )

    if len(good_images) < num_needed:
        log.warning(
            f"Only found {len(good_images)} images with detections, "
            f"needed {num_needed}. Proceeding with what we have."
        )

    return good_images


class DatasetCalibrationReader(CalibrationDataReader):
    """Yields pre-screened calibration images from a list of paths."""

    def __init__(
        self,
        image_paths: list[Path],
        input_name: str,
        input_height: int,
        input_width: int,
    ):
        self.input_name = input_name
        self.input_height = input_height
        self.input_width = input_width
        self.image_paths = image_paths
        self.index = 0

        log.info(
            f"Calibration reader: {len(self.image_paths)} pre-screened images, "
            f"input size {input_height}x{input_width}"
        )

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1

        img_batch = _load_image_as_uint8(img_path, self.input_height, self.input_width)
        return {self.input_name: img_batch}


def main():
    parser = argparse.ArgumentParser(
        description="Quantize an ONNX object detection model using static INT8 quantization"
    )
    parser.add_argument(
        "--model",
        default="omni-detector-0_0_1-rc8/model.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--calibration-data",
        default="omni_2.17_train",
        help="Path to calibration dataset directory (contains dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for quantized model (default: model_quantized.onnx alongside input)",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=200,
        help="Number of calibration images to use",
    )
    parser.add_argument(
        "--exclude-head",
        action="store_true",
        help="Exclude detection head nodes from quantization (keeps heads in float32)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path = Path(args.output) if args.output else model_path.parent / "model_quantized.onnx"
    dataset_dir = Path(args.calibration_data)

    # Read model input shape
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # [1, 3, H, W]
    input_h, input_w = input_shape[2], input_shape[3]
    log.info(f"Model input: name={input_name}, shape={input_shape}, type={input_info.type}")
    del session

    # Pre-screen calibration images (only keep ones that produce detections)
    all_paths = _load_image_paths(dataset_dir)
    good_paths = _prescreen_calibration_images(
        all_paths=all_paths,
        model_path=str(model_path),
        input_name=input_name,
        input_h=input_h,
        input_w=input_w,
        num_needed=args.num_calibration,
    )

    if not good_paths:
        raise RuntimeError("No calibration images produced detections. Cannot proceed.")

    # Preprocess model for quantization (shape inference + optimization)
    log.info("Preprocessing model for quantization...")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        preprocessed_path = tmp.name

    try:
        quant_pre_process(
            str(model_path),
            preprocessed_path,
            skip_symbolic_shape=True,
        )
        log.info("Preprocessed model saved to temp file")

        # Create calibration reader with pre-screened images
        reader = DatasetCalibrationReader(
            image_paths=good_paths,
            input_name=input_name,
            input_height=input_h,
            input_width=input_w,
        )

        # Static quantization targeting Conv/MatMul/Gemm ops.
        # NMS post-processing can produce empty tensors (zero detections)
        # which crashes calibration observer nodes. Restricting to
        # Conv/MatMul/Gemm avoids placing observers on post-processing.
        #
        # Optionally exclude detection head nodes (--exclude-head) to
        # preserve score calibration. Quantizing heads aggressively can
        # collapse scores to zero, killing all detections.
        nodes_to_exclude = []
        if args.exclude_head:
            import onnx as _onnx
            _model = _onnx.load(preprocessed_path)
            nodes_to_exclude = [
                n.name for n in _model.graph.node
                if "head" in n.name
            ]
            log.info(f"Excluding {len(nodes_to_exclude)} head nodes from quantization")
            del _model

        log.info("Running static quantization (Conv/MatMul/Gemm ops)...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        quantize_static(
            model_input=preprocessed_path,
            model_output=str(output_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            nodes_to_exclude=nodes_to_exclude,
        )
    finally:
        os.unlink(preprocessed_path)

    # Report sizes
    orig_size = model_path.stat().st_size / (1024 * 1024)
    quant_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    log.info("=" * 60)
    log.info("Quantization complete!")
    log.info(f"  Original:  {model_path} ({orig_size:.1f} MB)")
    log.info(f"  Quantized: {output_path} ({quant_size:.1f} MB)")
    log.info(f"  Reduction: {reduction:.1f}%")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
