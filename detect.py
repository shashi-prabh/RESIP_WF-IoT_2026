# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Shashi Prabh <shashi.prabh@ahduni.edu.in>
#
# Simulation harness for:
#   S. Prabh, "Fast or Stealthy, Not Both: The QoS Cost of Evading
#   Residential-Proxy Detection," IEEE World Forum on the Internet of
#   Things (WF-IoT), 2026.
#
# Supported by Ahmedabad University grant URBSEASE20A6/SUG/20-21/03-SP-08.23.

"""
Detector and its operating points.

Binary task: relayed (positive) vs background. Class
imbalance is handled with balanced sample weights, applied by the caller.

Operating points: Since the operator cannot touch the device owner's 
background traffic, the negative-score distribution is identical across 
evasion configs and the FPR stays put by construction.
This makes cost to halve at a target FPR well defined.

"""

from __future__ import annotations
import numpy as np


def make_classifier(seed):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             # tree_method="hist",       # no measurable difference at this size
                             subsample=0.9, eval_metric="logloss", n_jobs=4, random_state=seed)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=4, max_iter=250,
                                              learning_rate=0.1, random_state=seed)


def scores(clf, X):
    """Positive-class score, for either backend."""
    return clf.predict_proba(X)[:, 1]


def threshold_at_fpr(neg_scores, target_fpr):
    """
    Smallest threshold admitting at most `target_fpr` of the negatives. Classify positive iff score > tau.
    """
    neg = np.sort(np.asarray(neg_scores, float))[::-1]      # descending 
    n = len(neg)
    if n == 0:
        return float("inf"), 0.0, 0
    m = int(np.floor(target_fpr * n))                       # budget of false positives
    if m >= n:
        return float("-inf"), 1.0, n
    tau = float(neg[m])
    return tau, float((neg > tau).sum()) / n, n


def recall_at(sc, y, tau):
    """Recall of the positive class at a fixed threshold."""
    pos = np.asarray(y, int) == 1
    return float((np.asarray(sc, float)[pos] > tau).mean() * 100) if pos.any() else 0.0

