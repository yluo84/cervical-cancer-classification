"""Focal loss with class-weighted alpha and optional label smoothing.

Class imbalance here is severe: Cancerous is roughly 10% of the data and only a
handful of images reach the test split. Focal loss down-weights the many easy
Normal examples so the gradient is dominated by the cases the model is still
getting wrong, and the per-class alpha adds an explicit inverse-frequency
reweighting on top of that.

Note that class imbalance is handled *only* at the loss level in this project;
no resampling is applied to the data loader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Parameters
    ----------
    alpha : sequence of float or None
        Per-class weights, indexed by class id. Typically inverse class
        frequency normalised to mean 1.
    gamma : float
        Focusing parameter. gamma=0 reduces to (weighted) cross entropy;
        larger values put more weight on hard examples.
    label_smoothing : float
        Mass redistributed from the target class to the others.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        num_classes = logits.size(1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(
                    1, targets.unsqueeze(1), 1 - self.label_smoothing
                )
            log_probs = torch.log_softmax(logits, dim=1)
            ce = -(true_dist * log_probs).sum(dim=1)
        else:
            ce = F.cross_entropy(logits, targets, reduction="none")

        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            focal = self.alpha.to(logits.device)[targets] * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def inverse_frequency_alpha(class_counts):
    """Inverse-frequency class weights normalised to mean 1.

    class_counts is an array indexed by class id.
    """
    import numpy as np

    counts = np.asarray(class_counts, dtype=float)
    return counts.sum() / (len(counts) * counts)
