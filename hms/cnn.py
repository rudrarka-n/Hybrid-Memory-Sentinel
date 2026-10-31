"""Byte channel: multi-scale 1D-CNN over raw 4 KB memory pages (report section 5B).

    Stage 1  Conv1D(k=8, 64)   microscopic instruction patterns
    Stage 2  Conv1D(k=4, 128)  function-call prefixes
    Stage 3  Conv1D(k=2, 256)  high-dimensional byte pairings
    Dense FC + Dropout(0.5) -> malicious probability per page

Probabilities are calibrated with temperature scaling: after training, one
temperature T is fitted on the held-out validation pages to minimise log
loss, so the probabilities are better calibrated. T is stored in the model and
applied by ``predict_pages``. Dividing by T never changes the sign of a logit,
so it cannot change which pages pass the 0.5 cutoff; an undertrained model
that flags clean pages still does after calibration. ``train_cnn`` therefore
warns when the validation false-positive rate or recall is poor.

The report fixes kernel sizes and channel counts but not strides or pooling.
Stage 1 uses stride 4 and each stage is followed by max pooling so the model
trains on a CPU-only laptop.

Training data is two folders of raw byte files (memory dumps, process dumps or
plain binaries): one benign, one malicious. Every file goes through the same
page filter used at scan time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import PAGE_SIZE
from .metrics import binary_metrics, format_metrics
from .preprocess import FilterConfig, collect_candidates, dump_files


class PageCNN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, page_size) floats in [0, 1]. Returns logits, shape (batch,)."""
        return self.classifier(self.features(x.unsqueeze(1))).squeeze(1)


def to_tensor(pages: np.ndarray) -> torch.Tensor:
    """uint8 pages -> float32 in [0.0, 1.0]."""
    return torch.from_numpy(np.ascontiguousarray(pages)).float().div_(255.0)


@torch.no_grad()
def page_logits(model: PageCNN, pages: np.ndarray, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    out = [model(to_tensor(pages[i:i + batch_size])) for i in range(0, len(pages), batch_size)]
    return torch.cat(out) if out else torch.zeros(0)


def predict_pages(model: PageCNN, pages: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Calibrated malicious probability for each page."""
    logits = page_logits(model, pages, batch_size)
    return torch.sigmoid(logits / model.temperature).numpy()


# train_cnn warns when the validation pages give worse than this.
MAX_VAL_FPR = 0.10
MIN_VAL_RECALL = 0.50


def fit_temperature(logits: torch.Tensor, labels: np.ndarray) -> float:
    """Temperature that minimises log loss on held-out pages (Guo et al., 2017)."""
    y = torch.from_numpy(np.asarray(labels)).float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def step():
        opt.zero_grad()
        loss = loss_fn(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(step)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def build_page_dataset(
    benign_dir: str | Path,
    malicious_dir: str | Path,
    cfg: FilterConfig | None = None,
    max_pages_per_file: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pages uint8, labels, file index per page)."""
    cfg = cfg or FilterConfig()
    rng = np.random.default_rng(seed)
    pages, labels, groups = [], [], []
    file_id = 0
    for label, folder in ((0, benign_dir), (1, malicious_dir)):
        files = dump_files(folder)
        if not files:
            raise ValueError(f"no files found in {folder}")
        for f in files:
            batch, _ = collect_candidates(f, cfg=cfg, max_pages=max_pages_per_file, rng=rng)
            if len(batch):
                pages.append(batch.pages)
                labels.append(np.full(len(batch), label, dtype=np.int64))
                groups.append(np.full(len(batch), file_id, dtype=np.int64))
            file_id += 1
    if not pages:
        raise ValueError("no candidate pages survived filtering")
    return np.concatenate(pages), np.concatenate(labels), np.concatenate(groups)


def train_cnn(
    benign_dir: str | Path,
    malicious_dir: str | Path,
    out_path: str | Path | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    max_pages_per_file: int = 512,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Train the page classifier. Validation files are never seen in training."""
    torch.manual_seed(seed)
    X, y, groups = build_page_dataset(
        benign_dir, malicious_dir, max_pages_per_file=max_pages_per_file, seed=seed
    )

    # Split by file, per class, so pages of one file never straddle the split.
    rng = np.random.default_rng(seed)
    val_files: list[int] = []
    for label in (0, 1):
        files = np.unique(groups[y == label])
        if len(files) < 2:
            raise ValueError("need at least 2 files per class to hold one out for validation")
        rng.shuffle(files)
        val_files.extend(files[: max(1, round(len(files) * val_fraction))])
    val_mask = np.isin(groups, val_files)
    X_tr, y_tr, X_va, y_va = X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]

    model = PageCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Weight the positive class so an unbalanced page count does not bias the loss.
    pos_weight = torch.tensor(float((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X_tr))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(to_tensor(X_tr[idx])), torch.from_numpy(y_tr[idx]).float())
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        metrics = binary_metrics(y_va, predict_pages(model, X_va))
        metrics["train_loss"] = total / len(order)
        history.append(metrics)
        if verbose:
            print(f"epoch {epoch}/{epochs} loss={metrics['train_loss']:.4f} "
                  + format_metrics("val", metrics))

    # Calibrate on the validation pages, then report the calibrated metrics.
    model.temperature.fill_(fit_temperature(page_logits(model, X_va), y_va))
    calibrated = binary_metrics(y_va, predict_pages(model, X_va))
    if verbose:
        print(f"temperature {float(model.temperature):.3f}  " + format_metrics("calibrated", calibrated))
    warnings = []
    if calibrated["fpr"] > MAX_VAL_FPR:
        warnings.append(f"the model flags {calibrated['fpr']:.0%} of the clean validation pages")
    if calibrated["recall"] < MIN_VAL_RECALL:
        warnings.append(f"the model finds only {calibrated['recall']:.0%} of the malicious validation pages")
    if warnings and verbose:
        print("WARNING: " + "; ".join(warnings) + ". Train for more epochs or add more dumps "
              "before relying on this model.", flush=True)

    results = {
        "n_train_pages": int(len(X_tr)), "n_val_pages": int(len(X_va)),
        "n_files": int(len(np.unique(groups))), "epochs": epochs, "history": history,
        "temperature": float(model.temperature), "final": calibrated, "warnings": warnings,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "page_size": PAGE_SIZE,
                    "results": results}, out_path)
    return results


def load_cnn(path: str | Path) -> PageCNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PageCNN()
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
