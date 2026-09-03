"""Abdelnabi & Salem's linear probe, their recipe (``mlp_train.py``): a 2-output linear layer
``fc`` on span-mean hidden states, cross-entropy, plain SGD lr 0.008, batch 128, 300 epochs, seeds 0.
The steering direction is ``fc.weight[1]``, the positive-class row. Classes are balanced by
subsampling negatives, and the split is by prompt so both sides of an item land in the same fold.

We also fit sklearn's ``LogisticRegression`` (their ``probe.py``) as a cross-check and report AUROC,
which they do not. A shuffled-label probe is the control: it must sit at chance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_fc(X: torch.Tensor, y: torch.Tensor, seed: int = 0, epochs: int = 300, lr: float = 0.008, batch: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Their loop. Returns ``(weight [2, d], bias [2])`` on CPU, float32."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fc = torch.nn.Linear(X.shape[1], 2).to(dev)
    opt = torch.optim.SGD(fc.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    X = X.float().to(dev)
    y = y.long().to(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), batch):
            idx = perm[s : s + batch].to(dev)
            opt.zero_grad()
            loss = lossf(fc(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return fc.weight.detach().cpu().clone(), fc.bias.detach().cpu().clone()


def fc_scores(w: torch.Tensor, b: torch.Tensor, X: torch.Tensor) -> np.ndarray:
    """Logit of the positive class minus the negative class."""
    logits = X.float() @ w.T + b
    return (logits[:, 1] - logits[:, 0]).numpy()


@dataclass
class ProbeResult:
    layer: int
    n_train: int
    n_test: int
    acc: float
    auroc: float
    sk_acc: float
    sk_auroc: float
    direction: torch.Tensor  # fc.weight[1]
    weight: torch.Tensor | None = None  # full fc.weight [2, d]


def split_by_group(groups: np.ndarray, test_frac: float = 1 / 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = set(uniq[:n_test].tolist())
    is_test = np.array([g in test_groups for g in groups])
    return np.where(~is_test)[0], np.where(is_test)[0]


def balance(y: np.ndarray, seed: int = 4) -> np.ndarray:
    """Their ``get_classes``: shuffle negatives with seed 4 and keep as many as positives."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    neg = rng.permutation(neg)[: len(pos)]
    return np.sort(np.concatenate([pos, neg]))


def fit_layer(X: torch.Tensor, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, layer: int, seed: int = 0) -> ProbeResult:
    yt = torch.tensor(y)
    w, b = train_fc(X[train_idx], yt[train_idx], seed=seed)
    s = fc_scores(w, b, X[test_idx])
    yte = y[test_idx]
    acc = float(((s > 0).astype(int) == yte).mean())
    auroc = float(roc_auc_score(yte, s)) if len(set(yte.tolist())) == 2 else math.nan
    sk = LogisticRegression(max_iter=2000).fit(X[train_idx].numpy(), y[train_idx])
    sk_s = sk.decision_function(X[test_idx].numpy())
    sk_acc = float(((sk_s > 0).astype(int) == yte).mean())
    sk_auroc = float(roc_auc_score(yte, sk_s)) if len(set(yte.tolist())) == 2 else math.nan
    return ProbeResult(layer=layer, n_train=len(train_idx), n_test=len(test_idx), acc=acc, auroc=auroc, sk_acc=sk_acc, sk_auroc=sk_auroc, direction=w[1].clone(), weight=w.clone())


def probe_all_layers(feats: torch.Tensor, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[list[ProbeResult], dict]:
    """``feats``: [n, n_hidden_states, d]. Balances, splits by group, fits every layer, and fits a
    shuffled-label control at the best layer."""
    keep = balance(y)
    feats, y, groups = feats[keep], y[keep], groups[keep]
    train_idx, test_idx = split_by_group(groups, seed=seed)
    results = [fit_layer(feats[:, k], y, train_idx, test_idx, k, seed) for k in range(feats.shape[1])]
    best = max(results, key=lambda r: (r.acc, r.auroc))
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    y_shuf[train_idx] = rng.permutation(y_shuf[train_idx])
    shuf = fit_layer(feats[:, best.layer], y_shuf, train_idx, test_idx, best.layer, seed)
    control = {"shuffled_layer": best.layer, "shuffled_acc": shuf.acc, "shuffled_auroc": shuf.auroc, "shuffled_direction": shuf.direction, "layer0_acc": results[0].acc, "layer0_auroc": results[0].auroc, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    return results, control
