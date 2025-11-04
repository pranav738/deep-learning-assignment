"""Training routine for classification, attribute, and text-retrieval heads."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MultilabelAccuracy,
    MultilabelF1Score,
)

from project_dl.dataset import MultiTaskDataset
from project_dl.model import MultiTaskModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


CONFIG: Dict[str, Any] = {
    "project_root": PROJECT_ROOT,
    "experiment_tag": "own_dataset",
    
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "pin_memory": True,
    "backbone_name": "deit_tiny_patch16_224",
    "freeze_backbone_epochs": 2,
    "text_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 256,
    "freeze_text_backbone": True,
    "attr_loss_weight": 1.0,
    "attr_threshold": 0.5,
    "grad_clip_norm": 1.0,
    "retrieval_loss_weight": 0.1,
    "retrieval_temperature": 0.07,
    "retrieval_recall_at_k": [],
    "checkpoint_f1_weights": {
        "classification": 0.5,
        "attributes": 0.5,
    },
    "scheduler": {
        "type": "cosine",
        "eta_min": 1e-5,
    },
    "runs_csv_path": PROJECT_ROOT / "runs.csv",
    "seed": 1337,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(config: Dict[str, Any]) -> Tuple[MultiTaskDataset, MultiTaskDataset, DataLoader, DataLoader]:
    train_dataset = MultiTaskDataset(config["project_root"], split="train")
    val_dataset = MultiTaskDataset(config["project_root"], split="val")

    common_loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"] and torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(config: Dict[str, Any], num_classes: int, num_attributes: int) -> MultiTaskModel:
    return MultiTaskModel(
        backbone_name=config["backbone_name"],
        num_classes=num_classes,
        num_attributes=num_attributes,
        freeze_backbone=config["freeze_backbone_epochs"] > 0,
        text_model_name=config["text_model_name"],
        embedding_dim=config["embedding_dim"],
        freeze_text_backbone=config["freeze_text_backbone"],
    )


def build_optimizer(model: MultiTaskModel, config: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_cfg = config.get("scheduler", {})
    name = scheduler_cfg.get("type", "none").lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config["num_epochs"]),
            eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
        )
    if name == "step":
        step_size = int(scheduler_cfg.get("step_size", 5))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


def init_metric_bundle(num_classes: int, num_attributes: int, attr_threshold: float, device: torch.device) -> Dict[str, Any]:
    bundle = {
        "class_acc_overall": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "class_acc_per_class": MulticlassAccuracy(num_classes=num_classes, average="none"),
        "class_f1_overall": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "class_f1_per_class": MulticlassF1Score(num_classes=num_classes, average="none"),
        "attr_acc_overall": MultilabelAccuracy(num_labels=num_attributes, average="micro", threshold=attr_threshold),
        "attr_acc_per_label": MultilabelAccuracy(num_labels=num_attributes, average="none", threshold=attr_threshold),
        "attr_f1_overall": MultilabelF1Score(num_labels=num_attributes, average="macro", threshold=attr_threshold),
        "attr_f1_per_label": MultilabelF1Score(num_labels=num_attributes, average="none", threshold=attr_threshold),
    }
    for metric in bundle.values():
        metric.to(device)
    return bundle


def reset_metrics(metrics: Dict[str, Any]) -> None:
    for metric in metrics.values():
        metric.reset()


def compute_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_accuracy_overall": float(metrics["class_acc_overall"].compute().item()),
        "class_accuracy_per_class": metrics["class_acc_per_class"].compute().cpu().tolist(),
        "class_f1_overall": float(metrics["class_f1_overall"].compute().item()),
        "class_f1_per_class": metrics["class_f1_per_class"].compute().cpu().tolist(),
        "attr_accuracy_overall": float(metrics["attr_acc_overall"].compute().item()),
        "attr_accuracy_per_label": metrics["attr_acc_per_label"].compute().cpu().tolist(),
        "attr_f1_overall": float(metrics["attr_f1_overall"].compute().item()),
        "attr_f1_per_label": metrics["attr_f1_per_label"].compute().cpu().tolist(),
    }


def run_epoch(
    model: MultiTaskModel,
    dataloader: DataLoader,
    device: torch.device,
    metrics: Dict[str, Any],
    classification_loss: nn.Module,
    attribute_loss: nn.Module,
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    retrieval_weight = float(config.get("retrieval_loss_weight", 0.0))
    retrieval_temperature = float(config.get("retrieval_temperature", 0.07))
    recall_targets = [int(k) for k in config.get("retrieval_recall_at_k", []) if int(k) > 0]
    recall_targets.sort()
    max_recall_k = recall_targets[-1] if recall_targets else 0
    compute_retrieval = retrieval_weight > 0.0 or bool(recall_targets)

    retrieval_loss_sum = 0.0
    retrieval_sample_count = 0
    retrieval_recall_hits = {k: 0 for k in recall_targets}
    retrieval_reciprocal_rank_sum = 0.0

    for batch in dataloader:
        images, class_ids, attr_targets, metadata = batch
        images = images.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)
        attr_targets = attr_targets.to(device, non_blocking=True)
        if isinstance(metadata, dict):
            raw_captions = metadata.get("caption")
            if isinstance(raw_captions, list):
                captions = [text if isinstance(text, str) else "" for text in raw_captions]
            else:
                captions = ["" for _ in range(images.size(0))]
        else:
            captions = [entry.get("caption", "") if isinstance(entry, dict) else "" for entry in metadata]

        optimizer_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with optimizer_ctx:
            outputs = model(images, captions=captions if compute_retrieval else None, device=device)
            class_logits = outputs["class_logits"]
            attr_logits = outputs["attr_logits"]

            loss_class = classification_loss(class_logits, class_ids)
            loss_attr = attribute_loss(attr_logits, attr_targets)
            loss = loss_class + config["attr_loss_weight"] * loss_attr

            if compute_retrieval:
                image_embeddings = outputs.get("image_embeddings")
                text_embeddings = outputs.get("text_embeddings")
                if image_embeddings is None or text_embeddings is None:
                    raise RuntimeError("Model did not return retrieval embeddings while retrieval is enabled.")
                logits = image_embeddings @ text_embeddings.t()
                if retrieval_temperature <= 0:
                    raise ValueError("retrieval_temperature must be positive.")
                logits = logits / retrieval_temperature
                labels = torch.arange(images.size(0), device=device)

                loss_i2t = F.cross_entropy(logits, labels)
                loss_t2i = F.cross_entropy(logits.t(), labels)
                retrieval_loss = 0.5 * (loss_i2t + loss_t2i)
                loss = loss + retrieval_weight * retrieval_loss

                retrieval_loss_sum += retrieval_loss.item() * images.size(0)
                retrieval_sample_count += images.size(0)

                with torch.no_grad():
                    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
                    matches = sorted_indices.eq(labels.unsqueeze(1))
                    match_positions = torch.argmax(matches.float(), dim=1)
                    ranks = match_positions + 1
                    retrieval_reciprocal_rank_sum += torch.sum(1.0 / ranks.float()).item()

                    if max_recall_k > 0:
                        topk_indices = logits.topk(k=max_recall_k, dim=-1).indices
                        for k in recall_targets:
                            hits = (topk_indices[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
                            retrieval_recall_hits[k] += hits

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.get("grad_clip_norm"):
                    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        class_preds = class_logits.argmax(dim=-1)
        metrics["class_acc_overall"].update(class_preds, class_ids)
        metrics["class_acc_per_class"].update(class_preds, class_ids)
        metrics["class_f1_overall"].update(class_preds, class_ids)
        metrics["class_f1_per_class"].update(class_preds, class_ids)

        attr_prob = torch.sigmoid(attr_logits)
        attr_targets_int = attr_targets.int()
        metrics["attr_acc_overall"].update(attr_prob, attr_targets_int)
        metrics["attr_acc_per_label"].update(attr_prob, attr_targets_int)
        metrics["attr_f1_overall"].update(attr_prob, attr_targets_int)
        metrics["attr_f1_per_label"].update(attr_prob, attr_targets_int)

    avg_loss = total_loss / max(1, total_samples)
    summary = {"loss": avg_loss, **compute_metrics(metrics)}
    if retrieval_sample_count > 0:
        summary["retrieval_loss"] = retrieval_loss_sum / retrieval_sample_count
        summary["retrieval_mrr"] = retrieval_reciprocal_rank_sum / retrieval_sample_count
        summary["retrieval_recall_at_k"] = {
            k: retrieval_recall_hits[k] / retrieval_sample_count for k in recall_targets
        }
    else:
        summary["retrieval_loss"] = 0.0
        summary["retrieval_mrr"] = 0.0
        summary["retrieval_recall_at_k"] = {k: 0.0 for k in recall_targets}

    return summary


def build_epoch_record(
    epoch: int,
    split: str,
    metrics: Dict[str, Any],
    class_names: Iterable[str],
    attribute_names: Iterable[str],
) -> Dict[str, Any]:
    retrieval_recall = metrics.get("retrieval_recall_at_k", {})

    record = {
        "epoch": epoch,
        "split": split,
        "loss": metrics["loss"],
        "classification": {
            "accuracy": {
                "overall": metrics["class_accuracy_overall"],
                "per_class": {name: value for name, value in zip(class_names, metrics["class_accuracy_per_class"])}
            },
            "f1": {
                "overall": metrics["class_f1_overall"],
                "per_class": {name: value for name, value in zip(class_names, metrics["class_f1_per_class"])}
            },
        },
        "attributes": {
            "accuracy": {
                "overall": metrics["attr_accuracy_overall"],
                "per_attribute": {name: value for name, value in zip(attribute_names, metrics["attr_accuracy_per_label"])}
            },
            "f1": {
                "overall": metrics["attr_f1_overall"],
                "per_attribute": {name: value for name, value in zip(attribute_names, metrics["attr_f1_per_label"])}
            },
        },
    }

    record["retrieval"] = {
        "loss": metrics.get("retrieval_loss", 0.0),
        "mrr": metrics.get("retrieval_mrr", 0.0),
        "recall_at_k": {str(k): float(retrieval_recall.get(k, 0.0)) for k in sorted(retrieval_recall)},
    }

    return record


def append_json_records(path: Path, records: List[Dict[str, Any]]) -> None:
    existing: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    existing.extend(records)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)


def write_run_results(
    path: Path,
    run_id: str,
    epoch: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    checkpoint_score: float,
    training_seconds: float,
) -> None:
    train_recall = train_metrics.get("retrieval_recall_at_k", {})
    val_recall = val_metrics.get("retrieval_recall_at_k", {})

    payload = {
        "run_id": run_id,
        "best_epoch": epoch,
        "checkpoint_score": checkpoint_score,
        "training_seconds": training_seconds,
        "train": {
            "loss": train_metrics["loss"],
            "classification_accuracy": train_metrics["class_accuracy_overall"],
            "classification_f1": train_metrics["class_f1_overall"],
            "attribute_accuracy": train_metrics["attr_accuracy_overall"],
            "attribute_f1": train_metrics["attr_f1_overall"],
            "retrieval_loss": train_metrics.get("retrieval_loss", 0.0),
            "retrieval_mrr": train_metrics.get("retrieval_mrr", 0.0),
            "retrieval_recall_at_k": {str(k): float(train_recall.get(k, 0.0)) for k in sorted(train_recall)},
        },
        "val": {
            "loss": val_metrics["loss"],
            "classification_accuracy": val_metrics["class_accuracy_overall"],
            "classification_f1": val_metrics["class_f1_overall"],
            "attribute_accuracy": val_metrics["attr_accuracy_overall"],
            "attribute_f1": val_metrics["attr_f1_overall"],
            "retrieval_loss": val_metrics.get("retrieval_loss", 0.0),
            "retrieval_mrr": val_metrics.get("retrieval_mrr", 0.0),
            "retrieval_recall_at_k": {str(k): float(val_recall.get(k, 0.0)) for k in sorted(val_recall)},
        },
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_runs_csv(path: Path, headers: List[str], row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    config = {key: (value.copy() if isinstance(value, dict) else value) for key, value in CONFIG.items()}
    run_id = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y%m%d-%H%M%S")
    backbone_tag = config["backbone_name"].replace("/", "_")
    dataset_tag = config["experiment_tag"]

    run_dir = PROJECT_ROOT / "artifacts" / dataset_tag / backbone_tag / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config["run_id"] = run_id
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config["pin_memory"] = config["pin_memory"] and device.type == "cuda"

    checkpoint_path = run_dir / "best_multitask.pt"
    metrics_history_path = run_dir / "metrics_history.json"
    results_path = run_dir / "results.json"
    runs_csv_path = Path(config["runs_csv_path"])

    config_snapshot = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in config.items()
        if key != "runs_csv_path"
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2)

    train_ds, val_ds, train_loader, val_loader = create_dataloaders(config)
    model = build_model(config, train_ds.num_classes, train_ds.num_attributes).to(device)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    classification_loss = nn.CrossEntropyLoss()
    train_pos_counts = train_ds.attribute_positive_counts().to(torch.float32)
    total_train = float(len(train_ds))
    neg_counts = torch.clamp(total_train - train_pos_counts, min=0.0)
    pos_weight = torch.where(train_pos_counts > 0, neg_counts / train_pos_counts, torch.ones_like(train_pos_counts))
    attribute_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    class_names = train_ds.class_names()
    attribute_names = train_ds.attribute_names()

    best_val_score = float("-inf")
    best_val_metrics: Dict[str, Any] | None = None
    best_epoch = -1
    last_train_metrics: Dict[str, Any] | None = None

    start_time = time.perf_counter()

    for epoch in range(1, config["num_epochs"] + 1):
        if config["freeze_backbone_epochs"] and epoch == config["freeze_backbone_epochs"] + 1:
            model.unfreeze_backbone()

        train_metrics_bundle = init_metric_bundle(train_ds.num_classes, train_ds.num_attributes, config["attr_threshold"], device)
        val_metrics_bundle = init_metric_bundle(train_ds.num_classes, train_ds.num_attributes, config["attr_threshold"], device)

        reset_metrics(train_metrics_bundle)
        last_train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            metrics=train_metrics_bundle,
            classification_loss=classification_loss,
            attribute_loss=attribute_loss,
            config=config,
            optimizer=optimizer,
        )

        reset_metrics(val_metrics_bundle)
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            metrics=val_metrics_bundle,
            classification_loss=classification_loss,
            attribute_loss=attribute_loss,
            config=config,
        )

        metric_records = [
            build_epoch_record(epoch, "train", last_train_metrics, class_names, attribute_names),
            build_epoch_record(epoch, "val", val_metrics, class_names, attribute_names),
        ]
        append_json_records(metrics_history_path, metric_records)

        f1_weights = config.get("checkpoint_f1_weights", {})
        class_weight = float(f1_weights.get("classification", 0.5))
        attr_weight = float(f1_weights.get("attributes", 0.5))
        val_score = (
            class_weight * val_metrics.get("class_f1_overall", 0.0)
            + attr_weight * val_metrics.get("attr_f1_overall", 0.0)
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = val_metrics
            best_epoch = epoch
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_snapshot,
            }
            torch.save(checkpoint_payload, checkpoint_path)

        if scheduler is not None:
            scheduler.step()

        retrieval_recall_val = val_metrics.get("retrieval_recall_at_k", {})
        val_recall_at1 = retrieval_recall_val.get(1, 0.0)
        val_mrr = val_metrics.get("retrieval_mrr", 0.0)
        print(
            f"Epoch {epoch:03d}/{config['num_epochs']}: "
            f"train_loss={last_train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['class_accuracy_overall']:.4f}, val_attr_f1={val_metrics['attr_f1_overall']:.4f}, "
            f"val_recall@1={val_recall_at1:.4f}, val_mrr={val_mrr:.4f}, "
            f"checkpoint_score={val_score:.4f}"
        )

    if last_train_metrics is None or best_val_metrics is None:
        raise RuntimeError("Training did not produce metrics; please check configuration.")

    training_seconds = time.perf_counter() - start_time
    print(f"Run {run_id} completed in {training_seconds:.1f} seconds")

    write_run_results(
        results_path,
        run_id,
        best_epoch,
        last_train_metrics,
        best_val_metrics,
        best_val_score,
        training_seconds,
    )

    # Extend config with result summaries for CSV export.
    config["result_best_epoch"] = best_epoch
    config["result_checkpoint_score"] = best_val_score
    config["result_train_loss"] = last_train_metrics["loss"]
    config["result_train_classification_f1"] = last_train_metrics["class_f1_overall"]
    config["result_train_attribute_f1"] = last_train_metrics["attr_f1_overall"]
    config["result_val_loss"] = best_val_metrics["loss"]
    config["result_val_classification_f1"] = best_val_metrics["class_f1_overall"]
    config["result_val_attribute_f1"] = best_val_metrics["attr_f1_overall"]
    config["result_val_retrieval_mrr"] = best_val_metrics.get("retrieval_mrr", 0.0)
    config["result_training_seconds"] = training_seconds
    config["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))

    csv_headers = sorted(config.keys())
    csv_row: Dict[str, Any] = {}
    for key in csv_headers:
        value = config[key]
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        elif isinstance(value, (list, tuple)):
            value = json.dumps(list(value))
        csv_row[key] = value

    append_runs_csv(runs_csv_path, csv_headers, csv_row)


if __name__ == "__main__":  # pragma: no cover
    main()
