"""Torch-based orientation classifier (0/90/180/270)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm.auto import tqdm

from ..constants import ANGLE_TO_INDEX, CARDINAL_ANGLES, INDEX_TO_ANGLE
from ..memory_profile import format_memory, log_memory, snapshot_memory
from ..pdf_ops import rotate_image_clockwise


TORCH_MEAN = (0.5, 0.5, 0.5)
TORCH_STD = (0.5, 0.5, 0.5)


def _require_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    return torch, nn, DataLoader, Dataset


def _normalize_image(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(TORCH_MEAN, dtype=np.float32)
    std = np.asarray(TORCH_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def _augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    out = image
    if rng.random() < 0.6:
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.6:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.4:
        out = ImageEnhance.Color(out).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.25:
        out = ImageEnhance.Sharpness(out).enhance(rng.uniform(0.8, 1.3))
    if rng.random() < 0.25:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.8)))
    return out


@dataclass(slots=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    image_size: int
    val_split: float
    seed: int
    num_workers: int
    device: str
    log_every_batches: int
    early_stopping_patience: int
    early_stopping_min_delta: float


def select_device(device: str):
    torch, _, _, _ = _require_torch()

    if device != "auto":
        return torch.device(device)

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(num_classes: int = 4):
    torch, nn, _, _ = _require_torch()

    class SmallOrientationCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(192),
                nn.SiLU(),
                nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.dropout = nn.Dropout(p=0.2)
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return self.classifier(x)

    return SmallOrientationCNN()


def _split_entries_by_document(
    entries: list[dict[str, Any]], val_split: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in entries:
        grouped.setdefault(str(row["doc_id"]), []).append(row)

    doc_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    val_docs = max(1, int(math.ceil(len(doc_ids) * val_split))) if len(doc_ids) > 1 else 1
    val_doc_set = set(doc_ids[:val_docs])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        target = val_rows if doc_id in val_doc_set else train_rows
        target.extend(grouped[doc_id])

    def _class_set(rows: list[dict[str, Any]]) -> set[int]:
        return {int(r["rotation_deg"]) % 360 for r in rows}

    if not train_rows and val_rows:
        split_idx = int(len(val_rows) * 0.8)
        train_rows = val_rows[:split_idx]
        val_rows = val_rows[split_idx:]

    overall_classes = _class_set(entries)
    train_classes = _class_set(train_rows)
    val_classes = _class_set(val_rows)

    needs_stratified_fallback = (
        len(overall_classes) > 1
        and (train_classes != overall_classes or val_classes != overall_classes)
    )
    if needs_stratified_fallback:
        grouped_by_angle: dict[int, list[dict[str, Any]]] = {}
        for row in entries:
            angle = int(row["rotation_deg"]) % 360
            grouped_by_angle.setdefault(angle, []).append(row)

        rng = random.Random(seed)
        train_rows = []
        val_rows = []
        for rows in grouped_by_angle.values():
            rng.shuffle(rows)
            if len(rows) == 1:
                train_rows.extend(rows)
                continue
            n_val = max(1, int(round(len(rows) * val_split)))
            n_val = min(n_val, len(rows) - 1)
            val_rows.extend(rows[:n_val])
            train_rows.extend(rows[n_val:])

        rng.shuffle(train_rows)
        rng.shuffle(val_rows)

    return train_rows, val_rows


class LabeledPageDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        dataset_root: Path,
        image_size: int,
        augment: bool,
        seed: int,
    ):
        self.rows = rows
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        torch, _, _, _ = _require_torch()

        row = self.rows[index]
        image_path = self.dataset_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        base_label_angle = int(row["rotation_deg"]) % 360

        if self.augment:
            delta = self.rng.choice(CARDINAL_ANGLES)
            image = rotate_image_clockwise(image, delta)
            image = _augment_image(image, self.rng)
            label_angle = (base_label_angle + delta) % 360
        else:
            label_angle = base_label_angle

        arr = _normalize_image(image, image_size=self.image_size)
        tensor = torch.from_numpy(arr)
        label = ANGLE_TO_INDEX[label_angle]
        return tensor, label


def _run_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    train_mode: bool,
    *,
    epoch: int,
    phase_name: str,
    log_every_batches: int,
) -> tuple[float, float]:
    torch, _, _, _ = _require_torch()

    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0
    total_correct = 0

    total_batches = len(dataloader)

    progress = tqdm(
        dataloader,
        total=total_batches,
        desc=f"[{phase_name}] epoch={epoch:03d}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    try:
        last_batch_end = time.perf_counter()
        for batch_idx, (images, labels) in enumerate(progress, start=1):
            waited_s = time.perf_counter() - last_batch_end
            batch_started = time.perf_counter()

            images = images.to(device)
            labels = labels.to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train_mode):
                logits = model(images)
                loss = criterion(logits, labels)
                if train_mode:
                    loss.backward()
                    optimizer.step()

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())

            running_loss = total_loss / max(total_count, 1)
            running_acc = total_correct / max(total_count, 1)
            progress.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

            batch_elapsed_s = time.perf_counter() - batch_started
            if waited_s > 20.0 or batch_elapsed_s > 20.0:
                mem = snapshot_memory(device=device)
                print(
                    f"[{phase_name}] epoch={epoch:03d} batch={batch_idx}/{total_batches} "
                    f"slow wait_s={waited_s:.1f} step_s={batch_elapsed_s:.1f} {format_memory(mem)}"
                )

            should_log = (
                log_every_batches > 0
                and (batch_idx % log_every_batches == 0 or batch_idx == total_batches)
            )
            if should_log:
                mem = snapshot_memory(device=device)
                print(
                    f"[{phase_name}] epoch={epoch:03d} batch={batch_idx}/{total_batches} "
                    f"loss={running_loss:.4f} acc={running_acc:.4f} {format_memory(mem)}"
                )

            last_batch_end = time.perf_counter()
    finally:
        progress.close()

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def train_orientation_model(
    labels: list[dict[str, Any]],
    dataset_root: Path,
    checkpoint_path: Path,
    config: TrainConfig,
    val_labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train the lightweight 4-class orientation model."""
    torch, nn, DataLoader, _ = _require_torch()

    if len(labels) < 8:
        raise RuntimeError("Need at least 8 labeled pages to train the model.")

    device = select_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if val_labels is None:
        train_rows, val_rows = _split_entries_by_document(labels, config.val_split, config.seed)
        used_explicit_val_split = False
    else:
        train_rows = list(labels)
        val_rows = list(val_labels)
        used_explicit_val_split = True

    if not val_rows:
        train_rows, val_rows = _split_entries_by_document(train_rows, config.val_split, config.seed)
        used_explicit_val_split = False

    if not train_rows or not val_rows:
        raise RuntimeError("Could not create a non-empty train/val split.")

    train_ds = LabeledPageDataset(
        train_rows,
        dataset_root,
        config.image_size,
        augment=True,
        seed=config.seed,
    )
    val_ds = LabeledPageDataset(
        val_rows,
        dataset_root,
        config.image_size,
        augment=False,
        seed=config.seed,
    )

    effective_num_workers = config.num_workers
    if effective_num_workers > 0:
        print(
            "[train] num_workers>0 requested but shared-memory workers are unstable in this "
            "environment; falling back to num_workers=0"
        )
        effective_num_workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = _build_model(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    epochs_since_improvement = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    log_memory("[train] init", device=device)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=True,
            epoch=epoch,
            phase_name="train",
            log_every_batches=config.log_every_batches,
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=False,
            epoch=epoch,
            phase_name="val",
            log_every_batches=config.log_every_batches,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        row["memory"] = snapshot_memory(device=device)
        history.append(row)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print(f"[train] epoch={epoch:03d} {format_memory(row['memory'])}")

        is_better = (val_acc > best_val_acc + config.early_stopping_min_delta) or (
            abs(val_acc - best_val_acc) <= config.early_stopping_min_delta
            and val_loss < best_val_loss - config.early_stopping_min_delta
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": config.image_size,
                    "angles": list(CARDINAL_ANGLES),
                    "mean": TORCH_MEAN,
                    "std": TORCH_STD,
                    "best_val_accuracy": best_val_acc,
                    "history": history,
                    "train_config": {
                        "epochs": config.epochs,
                        "batch_size": config.batch_size,
                        "learning_rate": config.learning_rate,
                        "weight_decay": config.weight_decay,
                        "val_split": config.val_split,
                        "seed": config.seed,
                        "num_workers": config.num_workers,
                        "effective_num_workers": effective_num_workers,
                        "log_every_batches": config.log_every_batches,
                        "early_stopping_patience": config.early_stopping_patience,
                        "early_stopping_min_delta": config.early_stopping_min_delta,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_since_improvement += 1
            if config.early_stopping_patience > 0:
                print(
                    f"[train] no improvement for {epochs_since_improvement} epoch(s); "
                    f"patience={config.early_stopping_patience}"
                )
                if epochs_since_improvement >= config.early_stopping_patience:
                    print(f"[train] early stopping at epoch={epoch:03d}")
                    break

    print(f"best_val_accuracy={best_val_acc:.4f} checkpoint={checkpoint_path}")
    log_memory("[train] done", device=device)
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_accuracy": best_val_acc,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "used_explicit_val_split": used_explicit_val_split,
        "device": str(device),
        "history": history,
    }


def load_trained_model(checkpoint_path: Path, device: str = "auto"):
    """Load a checkpoint for inference."""
    torch, _, _, _ = _require_torch()

    selected_device = select_device(device)
    payload = torch.load(checkpoint_path, map_location=selected_device)

    model = _build_model(num_classes=4)
    model.load_state_dict(payload["model_state_dict"])
    model.to(selected_device)
    model.eval()
    return model, payload, selected_device


def predict_rotation_torch(image: Image.Image, model, device, payload: dict[str, Any]) -> dict[str, Any]:
    """Predict page clockwise orientation with the trained torch model."""
    torch, _, _, _ = _require_torch()

    image_size = int(payload.get("image_size", 320))
    array = _normalize_image(image, image_size=image_size)
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float)

    best_idx = int(np.argmax(probs))
    predicted_angle = INDEX_TO_ANGLE[best_idx]

    return {
        "predicted_rotation_deg": predicted_angle,
        "confidence": float(probs[best_idx]),
        "probabilities": {str(INDEX_TO_ANGLE[idx]): float(value) for idx, value in enumerate(probs)},
    }
