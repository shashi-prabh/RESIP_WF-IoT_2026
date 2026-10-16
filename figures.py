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
Figures 2 and 3, drawn with matplotlib defaults.
"""

from __future__ import annotations
import os

import numpy as np

from experiments import DETECTORS, cost_at_recall, _rk


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # plt.rcParams.update({"font.family": "serif", "font.size": 9})  # paper style, not shipped
    return plt


def fig2_primitives(res, outdir, bin_s=0.1):
    """Fig. 2: both primitives against the QoS they cost.

    The x axis is the COST. Plotted against their own
    intensity the two primitives collapse at a similar displacement (sigma* and
    tau* are both about one bin); plotted against what the operator actually
    pays.

    The shift arm is TRUNCATED AT ONE BIN WIDTH. simlib._make_packets draws
    packet times iid uniform over the connection, so the binned volume series is
    near-white and a shift of a full bin lands on an uncorrelated part of it.
    """
    
    plt = _plt()
    j = res[res.name.str.startswith("sched+jit")
            | (res.name == "sched")].sort_values("added_latency_ms")
    sh = res[(res.name.str.startswith("sched+shift") & (res.shift_s <= bin_s + 1e-9))
             | (res.name == "sched")].sort_values("added_latency_ms")

    fig, ax = plt.subplots(figsize=(7, 4.2))
 
    for frame, col, mk, lab in ((j, "tab:green", "o", "IPD jitter"),
                                (sh, "tab:red", "s", "constant shift")):
        ax.plot(frame.added_latency_ms, frame["recall_TA+Corr"], marker=mk,
                color=col, label=f"TA+Corr, {lab}")
        ax.plot(frame.added_latency_ms, frame["recall_Corr"], marker=mk, ls="--",
                color=col, lw=1.1, ms=4, mfc="none", alpha=0.85,
                label=f"Corr, {lab}")
    ax.plot(j.added_latency_ms, j["recall_RTT"], ls=":", color="tab:blue",
            label="RTT (free scheduling)")
    ax.set_xlim(0, 620)
    ax.set_ylim(-4, 104)
    ax.axhline(50, ls=":", color="0.35")
    ax.text(600, 53, "recall halved", color="0.35", ha="right")
    
    # Drop lines mark where each primitive crosses 50%
    for pref, col in (("sched+shift", "tab:red"), ("sched+jit", "tab:green")):
        c = cost_at_recall(res, "TA+Corr", prefix=pref)
        if np.isfinite(c):
            ax.plot([c, c], [0, 50], ls=":", color=col, lw=1.0)
            
    # Mark where the shift arm stops, so the truncation is not read as the curve
    # having reached zero and ended.
    tau_max = float(sh.added_latency_ms.max())
    ax.annotate(r"$\tau=\Delta$", xy=(tau_max, 2), xytext=(tau_max + 14, 11),
                color="tab:red",
                arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.8))
    ax.set_xlabel("Added latency (ms)")
    ax.set_ylabel("Deployed-detector recall (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_rq1_jitter.pdf")
    fig.savefig(path)
    # fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def fig3_frontier(res, outdir, bin_s=0.1):
    """Fig. 3: recall against each of the two cost axes; lower-left is the goal.
    Truncated at one bin width for the same reason as
    Fig. 2, and the points past tau = Delta are dropped.
    """
    
    plt = _plt()
    is_shift = res.name.str.startswith("sched+shift") & (res.shift_s <= bin_s + 1e-9)
    drop = res.name.str.startswith("sched+shift") & ~is_shift

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)
    
    for ax, (col, lab) in zip(axes, [("added_latency_ms", "Added latency (ms)"),
                                     ("bw_overhead", "Bandwidth overhead ($\\times$)")]):
        for d in DETECTORS:
            ax.scatter(res.loc[~is_shift & ~drop, col],
                       res.loc[~is_shift & ~drop, _rk(d)],
                       s=16, alpha=0.8, label=d)
        ax.scatter(res.loc[is_shift, col], res.loc[is_shift, "recall_TA+Corr"],
                   s=30, marker="D", facecolors="none", edgecolors="tab:red",
                   linewidths=1.1, label=r"TA+Corr, shift ($\tau\!\leq\!\Delta$)")
        ax.axhline(50, ls=":", color="grey")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("Deployed-detector recall (%)")
    axes[0].legend(loc="center right", fontsize="small")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_frontier.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path
