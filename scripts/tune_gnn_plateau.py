#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "gnn_tuning"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def next_run_id(today: str) -> str:
    prefix = f"RUN-{today}-"
    ensure_dir(RESULTS_DIR)
    nums = []
    for p in RESULTS_DIR.glob(f"{prefix}*"):
        if not p.is_dir():
            continue
        m = re.match(rf"{re.escape(prefix)}(\d{{3}})$", p.name)
        if m:
            nums.append(int(m.group(1)))
    seq = max(nums) + 1 if nums else 1
    return f"{prefix}{seq:03d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph(path: Path):
    return torch.load(path, weights_only=False)


def load_graphs(paths: list[Path]) -> list[Any]:
    return [load_graph(p) for p in paths if p.exists()]


def maybe_swap_to_embedded(path: Path, use_codebert: bool) -> Path:
    if not use_codebert:
        return path
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        return path
    if rel.startswith("data/pyg/"):
        swapped = ROOT / rel.replace("data/pyg/", "data/pyg_embedded/", 1)
        if swapped.exists():
            return swapped
    return path


class GNNClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        codebert_fusion: bool,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.codebert_fusion = codebert_fusion
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        head_in = hidden_dim + (in_dim if codebert_fusion else 0)
        self.head = nn.Linear(head_in, 2)

    def _pool_codebert(self, data: Any) -> torch.Tensor:
        x = data.x
        batch = data.batch
        src = getattr(data, "x_source", None)
        if isinstance(src, torch.Tensor) and src.numel() == x.shape[0]:
            mask = (src == 1).to(x.dtype).unsqueeze(1)
            masked_x = x * mask
            summed = global_add_pool(masked_x, batch)
            counts = global_add_pool(mask, batch).clamp_min(1.0)
            return summed / counts
        return global_mean_pool(x, batch)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        gnn_emb = global_mean_pool(x, batch)
        if self.codebert_fusion:
            codebert_emb = self._pool_codebert(data)
            gnn_emb = torch.cat([gnn_emb, codebert_emb], dim=1)
        return self.head(gnn_emb)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(batch.y.view(-1).cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    if not y_true:
        return {"f1": 0.0, "accuracy": 0.0, "n": 0.0}
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": float(len(y_true)),
    }


def train_one_trial(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    trial_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    set_seed(seed)
    model = GNNClassifier(
        in_dim=int(trial_cfg["in_dim"]),
        hidden_dim=int(trial_cfg["hidden_dim"]),
        dropout=float(trial_cfg["dropout"]),
        codebert_fusion=bool(trial_cfg.get("codebert_fusion", False)),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = str(trial_cfg["optimizer"]).lower()
    lr = float(trial_cfg["lr"])
    weight_decay = float(trial_cfg["weight_decay"])
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    max_epochs = int(trial_cfg["max_epochs"])
    epoch_patience = int(trial_cfg["epoch_patience"])
    epoch_min_delta = float(trial_cfg["epoch_min_delta"])

    best_val = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

        improved = val_metrics["f1"] > best_val + epoch_min_delta
        if improved:
            best_val = val_metrics["f1"]
            best_epoch = epoch
            stale = 0
            ensure_dir(checkpoint_path.parent)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg": trial_cfg,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= epoch_patience:
            break

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    return {
        "trial_cfg": trial_cfg,
        "best_val_f1": float(best_val),
        "best_epoch": int(best_epoch),
        "history": history,
        "test": test_metrics,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }


def generate_trial_cfg(
    trial_idx: int,
    in_dim: int,
    max_epochs: int,
    epoch_patience: int,
    epoch_min_delta: float,
    codebert_fusion: bool,
) -> dict[str, Any]:
    base = [
        {
            "hidden_dim": 64,
            "dropout": 0.2,
            "lr": 1e-3,
            "optimizer": "adam",
            "weight_decay": 0.0,
        },
        {
            "hidden_dim": 128,
            "dropout": 0.2,
            "lr": 1e-3,
            "optimizer": "adam",
            "weight_decay": 0.0,
        },
        {
            "hidden_dim": 256,
            "dropout": 0.2,
            "lr": 1e-3,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
        },
        {
            "hidden_dim": 128,
            "dropout": 0.3,
            "lr": 5e-4,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
        },
        {
            "hidden_dim": 192,
            "dropout": 0.25,
            "lr": 7e-4,
            "optimizer": "adam",
            "weight_decay": 0.0,
        },
    ]
    if trial_idx < len(base):
        cfg = dict(base[trial_idx])
    else:
        cfg = {
            "hidden_dim": int(random.choice([64, 96, 128, 160, 192, 256, 320])),
            "dropout": float(random.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])),
            "lr": float(random.choice([3e-4, 5e-4, 7e-4, 1e-3, 2e-3])),
            "optimizer": str(random.choice(["adam", "adamw"])),
            "weight_decay": float(random.choice([0.0, 1e-5, 1e-4, 5e-4])),
        }
    cfg["in_dim"] = in_dim
    cfg["max_epochs"] = max_epochs
    cfg["epoch_patience"] = epoch_patience
    cfg["epoch_min_delta"] = epoch_min_delta
    cfg["codebert_fusion"] = codebert_fusion
    return cfg


def run(
    split_path: Path,
    max_train: int,
    max_val: int,
    max_test: int,
    max_trials: int,
    trial_patience: int,
    min_delta: float,
    max_epochs: int,
    epoch_patience: int,
    epoch_min_delta: float,
    codebert_augment: bool,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    split_obj = load_json(split_path)
    train_paths = [
        maybe_swap_to_embedded(ROOT / p, codebert_augment)
        for p in split_obj.get("train", [])
    ]
    val_paths = [
        maybe_swap_to_embedded(ROOT / p, codebert_augment)
        for p in split_obj.get("val", [])
    ]
    test_paths = [
        maybe_swap_to_embedded(ROOT / p, codebert_augment)
        for p in split_obj.get("test", [])
    ]

    split_train_raw = [ROOT / p for p in split_obj.get("train", [])]
    split_val_raw = [ROOT / p for p in split_obj.get("val", [])]
    split_test_raw = [ROOT / p for p in split_obj.get("test", [])]
    swapped_count = sum(a != b for a, b in zip(split_train_raw, train_paths))
    swapped_count += sum(a != b for a, b in zip(split_val_raw, val_paths))
    swapped_count += sum(a != b for a, b in zip(split_test_raw, test_paths))

    def cap(paths: list[Path], n: int) -> list[Path]:
        if n <= 0 or len(paths) <= n:
            return paths
        idx = list(range(len(paths)))
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
        return [paths[i] for i in idx]

    train_paths = cap(train_paths, max_train)
    val_paths = cap(val_paths, max_val)
    test_paths = cap(test_paths, max_test)

    train_data = load_graphs(train_paths)
    val_data = load_graphs(val_paths)
    test_data = load_graphs(test_paths)
    if not train_data or not val_data or not test_data:
        raise RuntimeError("empty dataset after loading")

    in_dim = int(train_data[0].x.shape[1])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = next_run_id(datetime.now().strftime("%Y%m%d"))
    run_dir = RESULTS_DIR / run_id
    ensure_dir(run_dir)
    ckpt_dir = CHECKPOINTS_DIR / run_id
    ensure_dir(ckpt_dir)

    best_val = -1.0
    best_trial = -1
    stale_trials = 0
    trial_results: list[dict[str, Any]] = []

    for trial_idx in range(max_trials):
        cfg = generate_trial_cfg(
            trial_idx=trial_idx,
            in_dim=in_dim,
            max_epochs=max_epochs,
            epoch_patience=epoch_patience,
            epoch_min_delta=epoch_min_delta,
            codebert_fusion=codebert_augment,
        )
        ckpt_path = ckpt_dir / f"trial_{trial_idx:03d}.pt"
        trial_seed = seed + trial_idx
        trial = train_one_trial(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            trial_cfg=cfg,
            device=device,
            seed=trial_seed,
            checkpoint_path=ckpt_path,
        )
        trial["trial_index"] = trial_idx
        trial["seed"] = trial_seed
        trial_results.append(trial)

        val_f1 = float(trial["best_val_f1"])
        improved = val_f1 > best_val + min_delta
        if improved:
            best_val = val_f1
            best_trial = trial_idx
            stale_trials = 0
        else:
            stale_trials += 1
        if stale_trials >= trial_patience:
            break

    if best_trial < 0:
        raise RuntimeError("no successful trial")

    best_row = next(t for t in trial_results if int(t["trial_index"]) == best_trial)
    report = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "split_path": str(split_path),
        "codebert_augment": codebert_augment,
        "feature_source": "pyg_embedded" if codebert_augment else "pyg",
        "swapped_to_embedded_count": swapped_count,
        "dataset_sizes": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "plateau_policy": {
            "max_trials": max_trials,
            "trial_patience": trial_patience,
            "min_delta": min_delta,
            "max_epochs": max_epochs,
            "epoch_patience": epoch_patience,
            "epoch_min_delta": epoch_min_delta,
        },
        "stop_reason": "plateau" if len(trial_results) < max_trials else "max_trials",
        "trials_executed": len(trial_results),
        "best_trial_index": best_trial,
        "best_val_f1": best_row["best_val_f1"],
        "best_test": best_row["test"],
        "best_checkpoint": best_row["checkpoint"],
        "trial_results": trial_results,
        "spec_refs": [
            "SPEC:docs/spec/traceability.yaml#experiments[1] (ID:E-001)",
            "SPEC:docs/spec/traceability.yaml#experiments[4] (ID:E-004)",
            "SPEC:docs/spec/quality.yaml#requirements[3] (ID:R-004)",
            "SPEC:docs/spec/workflows.yaml#commands[2] (ID:CMD-003)",
        ],
    }
    save_json(run_dir / "gnn_tuning_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plateau-based GNN tuning")
    parser.add_argument("--split-path", type=Path, default=ROOT / "data" / "split.json")
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=400)
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--trial-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--epoch-patience", type=int, default=3)
    parser.add_argument("--epoch-min-delta", type=float, default=0.001)
    parser.add_argument("--codebert-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run(
        split_path=args.split_path,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        max_trials=args.max_trials,
        trial_patience=args.trial_patience,
        min_delta=args.min_delta,
        max_epochs=args.max_epochs,
        epoch_patience=args.epoch_patience,
        epoch_min_delta=args.epoch_min_delta,
        codebert_augment=args.codebert_augment,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "trials_executed": report["trials_executed"],
                "stop_reason": report["stop_reason"],
                "best_trial_index": report["best_trial_index"],
                "best_val_f1": report["best_val_f1"],
                "best_test_f1": report["best_test"]["f1"],
                "codebert_augment": report["codebert_augment"],
                "feature_source": report["feature_source"],
                "report_path": str(
                    ROOT / "results" / report["run_id"] / "gnn_tuning_report.json"
                ),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
