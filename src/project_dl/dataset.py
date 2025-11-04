from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import csv


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _parse_attributes_yaml(path: Path) -> Dict[str, List[str]]:
    entries: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, raw_values = stripped.split(":", 1)
            cleaned = raw_values.strip()
            if not (cleaned.startswith("[") and cleaned.endswith("]")):
                raise ValueError(f"Attribute entry for '{key}' must be a list literal.")
            inner = cleaned[1:-1].strip()
            values = [segment.strip() for segment in inner.split(",") if segment.strip()]
            entries[key.strip()] = values
    return entries


def _build_default_transforms(split: str) -> transforms.Compose:
    pipeline: List[Callable] = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)
    ]
    if split == "train":
        pipeline.append(transforms.RandomHorizontalFlip())
    pipeline.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(pipeline)


class MultiTaskDataset(Dataset):
    """Dataset returning image, class id, attribute vector, and metadata."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split

        self.data_dir = self.root / "data_pooled"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Expected data directory at {self.data_dir}")

        self.classes_path = self.data_dir / "classes.txt"
        self.attributes_path = self.data_dir / "attributes.yaml"
        self.labels_path = self.data_dir / "labels.csv"

        self.class_to_idx, self.idx_to_class = self._load_classes(self.classes_path)
        self.attr_to_idx, self.idx_to_attr = self._load_attributes(self.attributes_path)

        self.samples = self._load_labels(self.labels_path, split)
        self._attr_positive_counts = self._compute_attribute_counts(len(self.attr_to_idx))

        self.transform = transform if transform is not None else _build_default_transforms(split)

    def _load_classes(self, path: Path) -> Tuple[Dict[str, int], List[str]]:
        with path.open("r", encoding="utf-8") as handle:
            class_names = [line.strip() for line in handle if line.strip()]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        return class_to_idx, class_names

    def _load_attributes(self, path: Path) -> Tuple[Dict[str, int], List[str]]:
        yaml_entries = _parse_attributes_yaml(path)
        attr_keys: List[str] = []
        for category, values in yaml_entries.items():
            for value in values:
                attr_keys.append(f"{category}:{value}")
        attr_to_idx = {key: idx for idx, key in enumerate(attr_keys)}
        return attr_to_idx, attr_keys

    def _load_labels(self, path: Path, split: str) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") != split:
                    continue
                attr_pairs = [segment.strip() for segment in row["attributes"].split(";") if segment]
                attr_indices: List[int] = []
                for pair in attr_pairs:
                    if pair not in self.attr_to_idx:
                        raise KeyError(f"Attribute '{pair}' not defined in {self.attributes_path}")
                    attr_indices.append(self.attr_to_idx[pair])
                records.append(
                    {
                        "image_rel_path": Path(row["image_path"]),
                        "class_label": row["class_label"],
                        "class_id": self.class_to_idx[row["class_label"]],
                        "attr_indices": attr_indices,
                        "caption": row.get("caption", ""),
                        "split": row["split"],
                        "instance_id": row.get("instance_id", ""),
                    }
                )
        if not records:
            raise ValueError(f"No samples found for split='{split}' in {path}")
        return records

    def _compute_attribute_counts(self, num_attributes: int) -> torch.Tensor:
        counts = torch.zeros(num_attributes, dtype=torch.long)
        for sample in self.samples:
            for idx in sample["attr_indices"]:
                counts[idx] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = self.data_dir / sample["image_rel_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image) if self.transform else image

        class_id = torch.tensor(sample["class_id"], dtype=torch.long)
        attr_vec = torch.zeros(len(self.attr_to_idx), dtype=torch.float32)
        for attr_idx in sample["attr_indices"]:
            attr_vec[attr_idx] = 1.0

        metadata = {
            "caption": sample["caption"],
            "instance_id": sample["instance_id"],
            "class_label": sample["class_label"],
            "image_path": str(image_path),
        }

        return image_tensor, class_id, attr_vec, metadata

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_class)

    @property
    def num_attributes(self) -> int:
        return len(self.idx_to_attr)

    def class_names(self) -> Sequence[str]:
        return tuple(self.idx_to_class)

    def attribute_names(self) -> Sequence[str]:
        return tuple(self.idx_to_attr)

    def attribute_positive_counts(self) -> torch.Tensor:
        return self._attr_positive_counts.clone()
