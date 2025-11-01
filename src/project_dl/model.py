from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import timm
import torch
from torch import Tensor, nn


@dataclass
class MultiTaskOutput:
    class_logits: Tensor
    attr_logits: Tensor
    embedding: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "class_logits": self.class_logits,
            "attr_logits": self.attr_logits,
            "embedding": self.embedding,
        }


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_attributes: int,
        *,
        freeze_backbone: bool = True,
        retrieval_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )

        if hasattr(self.backbone, "num_features"):
            self.feature_dim = int(self.backbone.num_features)
        else:
            raise AttributeError("Backbone must expose num_features attribute")

        self.class_head = nn.Linear(self.feature_dim, num_classes)
        self.attr_head = nn.Linear(self.feature_dim, num_attributes)

        self.retrieval_head = retrieval_head if retrieval_head is not None else nn.Identity()

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        features = self.backbone.forward_features(images)
        pooled = (
            self.backbone.forward_head(features, pre_logits=True)
            if hasattr(self.backbone, "forward_head")
            else features
        )

        class_logits = self.class_head(pooled)
        attr_logits = self.attr_head(pooled)
        embedding = self.retrieval_head(pooled)

        return MultiTaskOutput(class_logits, attr_logits, embedding).as_dict()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def backbone_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.backbone.parameters()

    def head_parameters(self) -> Iterable[torch.nn.Parameter]:
        params = [*self.class_head.parameters(), *self.attr_head.parameters()]
        params.extend(self.retrieval_head.parameters())
        return params
