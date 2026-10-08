"""Training entry point for both models.

Usage
-----
    python -m src.train --model efficientnet_b0 --images-root /path/to/images
    python -m src.train --model smallcnn        --images-root /path/to/images

Both models are trained with identical data, loss, optimiser, schedule and
early-stopping criterion. Model selection uses validation macro-F1; the test
split is touched exactly once, at the end, by the selected checkpoint.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .data import CervixDataset, NUM_CLASSES, build_transforms, load_labels
from .evaluate import evaluate, summarise
from .losses import FocalLoss, inverse_frequency_alpha
from .models import SmallCNN, build_efficientnet_b0, count_parameters

DEFAULTS = dict(
    image_size=224,
    batch_size=32,
    epochs=40,
    lr=5e-5,
    weight_decay=3e-5,
    patience=10,
    num_workers=2,
    seed=42,
    unfreeze_last_n=3,
    focal_gamma=2.0,
    label_smoothing=0.05,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["smallcnn", "efficientnet_b0"],
                        default="efficientnet_b0")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--images-root", required=True,
                        help="Directory containing the case folders.")
    parser.add_argument("--out-dir", default="results")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}",
                            type=type(value), default=value)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data -------------------------------------------------------------
    df, missing = load_labels(args.labels, args.images_root)
    if len(missing):
        print(f"WARNING: {len(missing)} labelled images were not found on disk "
              f"and are excluded from all splits.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Images  train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Cases   train/val/test: {train_df['case_id'].nunique()}/"
          f"{val_df['case_id'].nunique()}/{test_df['case_id'].nunique()}")

    train_tfms, eval_tfms = build_transforms(args.image_size)
    loaders = {
        "train": DataLoader(CervixDataset(train_df, train_tfms),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True),
        "val": DataLoader(CervixDataset(val_df, eval_tfms),
                          batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(CervixDataset(test_df, eval_tfms),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True),
    }

    # --- Model, loss, optimiser ------------------------------------------
    if args.model == "smallcnn":
        model = SmallCNN(num_classes=NUM_CLASSES).to(device)
    else:
        model = build_efficientnet_b0(
            num_classes=NUM_CLASSES,
            unfreeze_last_n_blocks=args.unfreeze_last_n,
        ).to(device)

    total, trainable, frac = count_parameters(model)
    print(f"Parameters: total={total:,} trainable={trainable:,} ({frac:.1%})")

    counts = train_df["class"].value_counts().sort_index().values
    alpha = inverse_frequency_alpha(counts)
    print(f"Train class counts: {counts.tolist()}  alpha: {np.round(alpha, 3).tolist()}")

    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma,
                          label_smoothing=args.label_smoothing).to(device)
    monitor_criterion = nn.CrossEntropyLoss()  # for comparable loss curves only

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # --- Training loop ----------------------------------------------------
    run_dir = Path(args.out_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    best_val_f1, best_epoch, patience_counter = -1.0, -1, 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer,
                                     criterion, device)
        val = evaluate(model, loaders["val"], monitor_criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])

        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
              f"val_macro_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_val_f1:
            best_val_f1, best_epoch, patience_counter = val["macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val_macro_f1": best_val_f1}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}).")
                break

        scheduler.step()

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # --- Single pass over the held-out test split -------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test = evaluate(model, loaders["test"], monitor_criterion, device)

    print("\nClassification report:")
    print(classification_report(test["targets"], test["preds"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(test["targets"], test["preds"]))

    summary = summarise(args.model, test, extra={
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "confusion_matrix": confusion_matrix(
            test["targets"], test["preds"]
        ).tolist(),
        "hyperparameters": {k: getattr(args, k) for k in DEFAULTS},
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ntest accuracy  {summary['test_accuracy']:.4f}  "
          f"95% CI {np.round(summary['test_accuracy_ci95'], 4).tolist()}")
    print(f"test macro-F1  {summary['test_macro_f1']:.4f}  "
          f"95% CI {np.round(summary['test_macro_f1_ci95'], 4).tolist()}")
    print(f"Saved -> {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
