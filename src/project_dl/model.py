from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import timm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn


@dataclass
class MultiTaskOutput:
    class_logits: Tensor
    attr_logits: Tensor
    image_embeddings: Tensor
    text_embeddings: Optional[Tensor] = None

    def as_dict(self) -> Dict[str, Tensor]:
        payload: Dict[str, Tensor] = {
            "class_logits": self.class_logits,
            "attr_logits": self.attr_logits,
            "image_embeddings": self.image_embeddings,
        }
        if self.text_embeddings is not None:
            payload["text_embeddings"] = self.text_embeddings
        # Backward compatibility alias for any downstream code still expecting `embedding`.
        payload["embedding"] = self.image_embeddings
        return payload


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_attributes: int,
        *,
        freeze_backbone: bool = True,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 256,
        freeze_text_backbone: bool = True,
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

        self.embedding_dim = int(embedding_dim)

        self.image_projection = nn.Linear(self.feature_dim, self.embedding_dim)

        self.text_encoder = SentenceTransformer(text_model_name)
        self.freeze_text_backbone = freeze_text_backbone
        if self.freeze_text_backbone:
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

        text_embedding_dim = int(self.text_encoder.get_sentence_embedding_dimension())
        self.text_projection = nn.Linear(text_embedding_dim, self.embedding_dim)

        if freeze_backbone:
            self.freeze_backbone()

    def forward(
        self,
        images: Tensor,
        captions: Optional[List[str]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tensor]:
        features = self.backbone.forward_features(images)
        pooled = (
            self.backbone.forward_head(features, pre_logits=True)
            if hasattr(self.backbone, "forward_head")
            else features
        )

        class_logits = self.class_head(pooled)
        attr_logits = self.attr_head(pooled)
        image_embeddings = F.normalize(self.image_projection(pooled), p=2, dim=-1)

        text_embeddings: Optional[Tensor] = None
        if captions is not None:
            text_embeddings = self.encode_text(captions, device=device or images.device)

        return MultiTaskOutput(class_logits, attr_logits, image_embeddings, text_embeddings).as_dict()

    def encode_text(self, captions: List[str], device: torch.device) -> Tensor:
        if not captions:
            raise ValueError("Captions list must not be empty when requesting text embeddings.")

        if self.freeze_text_backbone:
            with torch.no_grad():
                raw_embeddings = self.text_encoder.encode(
                    captions,
                    convert_to_tensor=True,
                    device=device,
                    show_progress_bar=False,
                )
        else:
            tokenized = self.text_encoder.tokenize(captions)
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            outputs = self.text_encoder(tokenized)
            raw_embeddings = outputs["sentence_embedding"]

        projected = self.text_projection(raw_embeddings.to(device))
        return F.normalize(projected, p=2, dim=-1)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def backbone_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.backbone.parameters()

    def head_parameters(self) -> Iterable[torch.nn.Parameter]:
        params = [
            *self.class_head.parameters(),
            *self.attr_head.parameters(),
            *self.image_projection.parameters(),
            *self.text_projection.parameters(),
        ]
        return params
