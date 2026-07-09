"""Class-balanced classification loss for Faster R-CNN.

Injects a per-class ``weight`` tensor into the ROI head's softmax cross-entropy
so that rare classes contribute more to the gradient (counteracting class
imbalance). Faster R-CNN only — the other detectors use focal loss / hard
negative mining and are not handled here.

The weight vector has length ``num_classes + 1`` with index 0 = background and
indices ``1..N`` = the sorted foreground classes (matching
``ViamDataset.label_to_id``).
"""
import logging
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision.models.detection.roi_heads as _roi_heads_mod

log = logging.getLogger(__name__)


def compute_class_annotation_counts(train_dataset) -> Dict[int, int]:
    """Count annotations per foreground class id over the training split.

    Annotation-level counts (not image-level) are the right basis here: the ROI
    classification loss operates per sampled proposal, roughly per object.

    Handles both dataset shapes: a torch ``Subset`` (auto-split) or a bare
    ``ViamDataset`` (separate val_dir). Counts are computed over exactly the
    images the training loader will see.

    Args:
        train_dataset: The final train dataset (Subset or ViamDataset).

    Returns:
        Dict mapping foreground class id (1..N) -> annotation count. Every id in
        ``1..N`` is present (zero-count classes map to 0).
    """
    from torch.utils.data import Subset

    if isinstance(train_dataset, Subset):
        base = train_dataset.dataset          # ViamDataset (clone)
        order = list(train_dataset.indices)
    else:
        base = train_dataset                  # ViamDataset
        order = list(range(len(base.samples)))

    num_classes = len(base.label_to_id)
    counts = {cid: 0 for cid in range(1, num_classes + 1)}

    for idx in order:
        for bbox in base.samples[idx]['boxes']:
            cid = base.label_to_id.get(bbox.get('annotation_label'))
            if cid is not None:
                counts[cid] += 1

    return counts


def compute_class_weights(
    counts: Dict[int, int],
    num_classes: int,
    background_weight: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Build an inverse-frequency class-weight vector for cross-entropy.

    Foreground weights are ``1 / N_c`` normalized so their mean is 1.0 (this
    keeps the overall loss scale close to the unweighted loss, so the existing
    learning rate stays valid). The background weight is set explicitly.

    Args:
        counts: Annotation count per foreground class id (1..N).
        num_classes: Number of foreground classes (N).
        background_weight: Weight for the background class (index 0).
        device: Device for the returned tensor.

    Returns:
        Float32 tensor of length ``num_classes + 1``; index 0 = background.
    """
    ids = list(range(1, num_classes + 1))

    zero_classes = [cid for cid in ids if counts.get(cid, 0) == 0]
    if zero_classes:
        log.warning(
            f"class_balanced_loss: class ids {zero_classes} have 0 annotations "
            f"in the training split; their weight is clamped (treated as count=1)."
        )

    # Inverse frequency with a count>=1 guard to avoid division by zero.
    inv = torch.tensor(
        [1.0 / max(counts.get(cid, 0), 1) for cid in ids],
        dtype=torch.float32,
    )
    # Normalize foreground weights to mean 1.0 (preserve loss scale).
    inv = inv * (inv.numel() / inv.sum())

    weights = torch.empty(num_classes + 1, dtype=torch.float32)
    weights[0] = background_weight
    weights[1:] = inv

    if device is not None:
        weights = weights.to(device)
    return weights


def apply_weighted_fastrcnn_loss(class_weights: torch.Tensor) -> None:
    """Monkeypatch torchvision's ``fastrcnn_loss`` to weight the classification CE.

    torchvision's ``RoIHeads.forward`` calls the module-level ``fastrcnn_loss``
    by bare name, resolving it from the module globals at call time — so
    replacing the module attribute is picked up by the existing model. This is a
    process-global side effect, which is fine: one model is trained per process,
    and weighting both train and val loss is the intended behavior.

    The replacement mirrors torchvision 0.23.0's ``fastrcnn_loss`` exactly except
    for adding ``weight=`` to the cross-entropy.

    Args:
        class_weights: Tensor of length ``num_classes + 1`` (index 0 = background).
    """

    def weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(
            class_logits, labels, weight=class_weights.to(class_logits.device)
        )

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    _roi_heads_mod.fastrcnn_loss = weighted_fastrcnn_loss
    log.info(
        f"Patched fastrcnn_loss with class-balanced weights "
        f"(background={class_weights[0].item():.4f}, "
        f"foreground={[round(w, 4) for w in class_weights[1:].tolist()]})"
    )
