import logging

import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
)
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

log = logging.getLogger(__name__)


class FCOSDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FCOSDetector, self).__init__()

        input_h, input_w = cfg.model.transform.input_size

        if cfg.training.pretrained:
            weights = FCOS_ResNet50_FPN_Weights.COCO_V1
            self.model = fcos_resnet50_fpn(weights=weights)

            log.info("Loaded FCOS-ResNet50-FPN with COCO_V1 pretrained weights")

            # Override transform to match target deployment resolution,
            # keeping the pretrained normalization stats.
            self.model.transform = GeneralizedRCNNTransform(
                min_size=input_h,
                max_size=input_w,
                image_mean=self.model.transform.image_mean,
                image_std=self.model.transform.image_std,
            )
            log.info(
                f"Overrode transform to {input_h}x{input_w} "
                f"(mean: {self.model.transform.image_mean}, std: {self.model.transform.image_std})"
            )
        else:
            self.model = fcos_resnet50_fpn(weights=None)

            self.model.transform = GeneralizedRCNNTransform(
                min_size=input_h,
                max_size=input_w,
                image_mean=cfg.model.transform.get("image_mean", [0.485, 0.456, 0.406]),
                image_std=cfg.model.transform.get("image_std", [0.229, 0.224, 0.225]),
            )
            log.info(
                f"Training from scratch with custom transform - "
                f"mean: {self.model.transform.image_mean}, std: {self.model.transform.image_std}"
            )

        # Replace classification head for custom number of classes
        self.num_classes = cfg.model.num_classes
        if cfg.model.num_classes != 90:
            num_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
            in_channels = self.model.backbone.out_channels  # 256 for ResNet50-FPN

            if cfg.training.pretrained:
                log.info("Transfer learning: Replacing classification head")
                log.info("  - COCO classes: 91 (90 + background)")
                log.info(
                    f"  - Your classes: {cfg.model.num_classes + 1} "
                    f"({cfg.model.num_classes} + background)"
                )
                log.info(
                    "  - Keeping pretrained: Backbone (ResNet50), FPN, box regression head"
                )
                log.info("  - Replacing with random init: Classification head only")

            self.model.head.classification_head = FCOSClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=cfg.model.num_classes + 1,
            )
        else:
            if cfg.training.pretrained:
                log.info("Using COCO classes (90), keeping all pretrained weights")

    def to(self, device):
        """Override to method to ensure all components are moved to device."""
        self.model = self.model.to(device)
        return super().to(device)

    def forward(self, data, targets=None):
        return self.model(data, targets)
