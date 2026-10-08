"""Evaluation loop and non-parametric bootstrap confidence intervals.

With 81 test images, a point estimate of accuracy or macro-F1 carries very
little information on its own: resampling the test set shows how much of the
reported number is attributable to which images happened to land in it. Every
headline metric in this project is reported with a 95% bootstrap interval for
that reason.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run the model over a loader and return loss, metrics and raw predictions."""
    model.eval()
    losses, all_preds, all_targets = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        losses.append(criterion(logits, y).item())

        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        all_targets.extend(y.cpu().numpy().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "preds": preds,
        "targets": targets,
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """Percentile bootstrap interval for a metric over (y_true, y_pred).

    Resamples prediction/label pairs with replacement, so the interval reflects
    sampling variability of the test set itself, not of model training.

    Returns (mean, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        stats[i] = metric_fn(y_true[sample], y_pred[sample])

    return (
        float(stats.mean()),
        float(np.quantile(stats, alpha / 2)),
        float(np.quantile(stats, 1 - alpha / 2)),
    )


def summarise(model_name, result, extra=None):
    """Assemble a JSON-serialisable summary with bootstrap intervals."""
    targets, preds = result["targets"], result["preds"]

    acc_mean, acc_lo, acc_hi = bootstrap_ci(
        targets, preds, lambda t, p: accuracy_score(t, p)
    )
    f1_mean, f1_lo, f1_hi = bootstrap_ci(
        targets, preds, lambda t, p: f1_score(t, p, average="macro")
    )

    summary = {
        "model": model_name,
        "test_accuracy": result["accuracy"],
        "test_macro_f1": result["macro_f1"],
        "test_accuracy_ci95": [acc_lo, acc_hi],
        "test_macro_f1_ci95": [f1_lo, f1_hi],
        "bootstrap_mean_accuracy": acc_mean,
        "bootstrap_mean_macro_f1": f1_mean,
        "n_test": int(len(targets)),
    }
    if extra:
        summary.update(extra)
    return summary
