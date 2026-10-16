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
Experiment drivers: sweeps, detector calibration, and the derived costs.

The detector is hardened against the FREE scheduling attack by training on clean as well as
scheduled traffic) so that it must rely on correlation. The operator then spends
an evasion budget, and we measure the deployed detector's recall against the QoS cost the operator pays. 
The cost is measured in added latency, bandwidth overhead, and goodput loss.


"""


from __future__ import annotations
import numpy as np
import pandas as pd

from simlib import SimConfig, WorkloadConfig, generate_dataset
from features import build_features, featurize_one, RTT_COLS, TA_COLS, CORR_COLS
from evasion import Evasion, apply_evasion, qos_cost
from detect import make_classifier, scores, threshold_at_fpr, recall_at
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import recall_score

# Seeds used for every number reported in the paper.
PAPER_SEEDS = [7, 11, 13, 17, 19, 23]

FEATURE_SETS = {"RTT": RTT_COLS, "Corr": CORR_COLS, "TA+Corr": TA_COLS + CORR_COLS}
DETECTORS = list(FEATURE_SETS)
FPR_TARGETS = (1e-2, 1e-3)          # pinned operating points
SLA_MS = 1000.0                     # commercial per-request latency target
CALIB_SCALE = 6                     # calibration windows / workload windows
# CALIB_SCALE = 10                  # 1e-4 would need this, runtime does not justify it

# Jitter grid the Sec. V-B cost is interpolated on
SIGMAS_MAIN = [0.0125, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.6]

# ---------------------------------------------------------------- 
# Table III grids
#
# The scaling sweep CANNOT use SIGMAS_MAIN. The 50% crossing moves with Delta, so
# a grid fixed in milliseconds brackets it tightly at one bin width and loosely at
# another, and interpolating across the loose gaps produces per-seed noise that
# looks like a real effect. Expressing the grid as MULTIPLES OF Delta holds the
# bracket at 4-8% of the cost at every bin width, which is what makes an exponent
# fittable. Each grid is dense where its own crossing lands: sigma*/Delta near
# 0.8-1.0, and tau*/Delta near 0.5-0.65.
# ------------------------------------------------------------------

SIGMA_FRACS = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
               1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0]
TAU_FRACS = [0.2, 0.3, 0.4, 0.45, 0.5, 0.525, 0.55, 0.575, 0.6,
             0.625, 0.65, 0.7, 0.8, 1.0, 1.5, 2.0]

# Table III's columns, which are also the range Eq. (6) is fitted over. Delta =
# 800 ms is measured in the full harness but excluded from both: it leaves only
# 11 correlation bins in a 9 s flow, and including it takes the shift fit from
# chi2/dof 0.99 to 9.74.
SCALING_BINS = [0.025, 0.05, 0.1, 0.2, 0.4]

# Documents Table I's swept backhaul range. The sweep itself is Sec. V-C, which
# is outside this release's scope.
BACKHAUL_MEANS_MS = (12.5, 130.0)


def _rk(det, fpr=None):
    """Column name for a recall series: default-threshold, or pinned at `fpr`."""
    return f"recall_{det}" if fpr is None else f"recall_{det}@fpr{fpr:g}"


def shift_configs(taus):
    """Constant-lag sweep: delay the whole relayed series by tau seconds."""
    return ([Evasion("clean"), Evasion("sched", schedule=True)]
            + [Evasion(f"sched+shift{t}", schedule=True, shift_s=t) for t in taus])


def jitter_configs(sigmas, variable=False):
    """Focused set for resolving the Corr-evasion cost (scheduling always on)."""
    tag = "vjit" if variable else "jit"
    return ([Evasion("clean"), Evasion("sched", schedule=True)]
            + [Evasion(f"sched+{tag}{s}", schedule=True, jitter_sigma=s, jitter_variable=variable)
               for s in sigmas])


# Table II. SC = Scheduling, TP = Targeted Padding, PR = Packet Reshaping,
# IPD = Inter-Packet Delays. Representative intensities: TP = 100% chaff,
# PR = 3-way split, IPD = sigma at the detector's 0.1 s bin width.
#
# SC+IPD is not listed here because it IS a point of the jitter sweep
# (sched+jit0.1); exp_combined picks it up from there under its Table II name.
# The two settings without SC are omitted from the table since leaving the RTT
# feature intact only raises recall.
NDSS_COMBOS = [
    ("SC+TP",           dict(schedule=True, pad_frac=1.0)),
    ("TP+IPD",          dict(pad_frac=1.0, jitter_sigma=0.1)),
    ("SC+TP+PR",        dict(schedule=True, pad_frac=1.0, reshape_mult=3)),
    ("SC+TP+IPD",       dict(schedule=True, pad_frac=1.0, jitter_sigma=0.1)),
    ("SC+PR+IPD",       dict(schedule=True, reshape_mult=3, jitter_sigma=0.1)),
    ("TP+PR+IPD",       dict(pad_frac=1.0, reshape_mult=3, jitter_sigma=0.1)),
    ("SC+TP+PR+IPD",    dict(schedule=True, pad_frac=1.0, reshape_mult=3, jitter_sigma=0.1)),
]
IPD_SIGMA = 0.1                     # the sigma the IPD combos use
TABLE2_ROWS = ["SC+TP", "SC+TP+PR", "SC+IPD", "SC+PR+IPD", "SC+TP+IPD", "SC+TP+PR+IPD"]


def combined_configs(sigmas):
    """Table II settings plus the single-primitive and mixed sweeps of Figs. 2-3.

    Single-primitive sweeps populate the bandwidth axis (padding/reshaping alone).
    The mixed sweeps test whether bandwidth can substitute for latency at the
    margin once jitter is already deployed.
    """
    cfgs = [Evasion("clean"), Evasion("sched", schedule=True)]
    cfgs += [Evasion(name, **kw) for name, kw in NDSS_COMBOS]
    cfgs += [Evasion(f"sched+jit{s}", schedule=True, jitter_sigma=s) for s in sigmas]
    cfgs += [Evasion(f"sched+pad{p}", schedule=True, pad_frac=p) for p in [0.25, 0.5, 1.0, 2.0]]
    cfgs += [Evasion(f"sched+resh{m}", schedule=True, reshape_mult=m) for m in [2, 3, 5]]
    for p, m in [(1.0, 1.0), (0.0, 3.0), (1.0, 3.0)]:       # mix each with the jitter sweep
        for s in sigmas:
            cfgs.append(Evasion(f"sched+pad{p}+resh{m}+jit{s}", schedule=True,
                                pad_frac=p, reshape_mult=m, jitter_sigma=s))
    cfgs += [Evasion(f"sched+vjit{s}", schedule=True, jitter_sigma=s, jitter_variable=True)
             for s in sigmas]                                # variable-sigma jitter
    cfgs += [Evasion(f"sched+shift{t}", schedule=True, shift_s=t)
             for t in sigmas]                                # constant lag, same grid as sigma
    return cfgs


def eval_config(ev, test_conns, clfs, gw_by_window, edges, bin_s, link, cseed,
                taus=None, corr_pkt_limit=None):
    rng = np.random.default_rng(cseed)
    rows, qrows = [], []
    for c in test_conns:
        if c.relayed:
            t, b, d = apply_evasion(c.pkt_t, c.pkt_bytes, c.pkt_dir, c.start_s, c.end_s,
                                    ev, rng, cid=c.cid)
            qrows.append(qos_cost(c, link, ev))
        else:
            t, b, d = c.pkt_t, c.pkt_bytes, c.pkt_dir
        rows.append(featurize_one(c, t, b, d, gw_by_window[c.window_id], edges, bin_s,
                                  ev.schedule, corr_pkt_limit))
    df = pd.DataFrame(rows)
    q = pd.DataFrame(qrows)
    lat_ms = q.total_latency.to_numpy() * 1000.0            # per-FETCH latency distribution
    out = dict(name=ev.name, jitter_sigma=ev.jitter_sigma, pad_frac=ev.pad_frac,
               reshape_mult=ev.reshape_mult, schedule=int(ev.schedule),
               jitter_variable=int(ev.jitter_variable), shift_s=ev.shift_s,
               added_latency_ms=float(q.added_latency.mean() * 1000),
               bw_overhead=float(q.bw_overhead.mean()),
               goodput_loss=float(q.goodput_loss.mean()),
               added_jitter_ms=float(q.added_jitter.mean() * 1000),
               # QoS as the customer sees it, not just the mean
               sla_violation_pct=float((lat_ms > SLA_MS).mean() * 100),
               lat_p50_ms=float(np.percentile(lat_ms, 50)),
               lat_p95_ms=float(np.percentile(lat_ms, 95)),
               lat_p99_ms=float(np.percentile(lat_ms, 99)))
    y = df["relayed"].to_numpy(int)
    for name, (clf, cols) in clfs.items():
        X = df[cols].to_numpy(float)
        # NOTE: this is the classifier's own threshold (p > 0.5), not a pinned
        # one. It is the operating point behind the jitter cost; Sec. V-B
        # reports its calibrated rate (3.2e-3 for TA+Corr) and brackets it with
        # the two pinned points below.
        out[_rk(name)] = float(recall_score(y, clf.predict(X), pos_label=1, zero_division=0) * 100)
        if taus:
            sc = scores(clf, X)
            neg = sc[y == 0]
            for f, tau in taus[name].items():               # threshold pinned on CLEAN traffic
                out[_rk(name, f)] = recall_at(sc, y, tau)
                out[f"fpr_{name}@fpr{f:g}"] = float((neg > tau).mean() * 100)
    return out


def calibrate(cfg, clfs, bin_s, corr_pkt_limit=None, scale=CALIB_SCALE):
    """Pin a threshold per (detector, target FPR) on an independent clean dataset.

    Uses its own larger draw so the FPR estimate is not limited by the test split:
    at 1e-3 a few hundred negatives cannot resolve the target at all.
    """
    ccfg = SimConfig(seed=cfg.seed + 500, link=cfg.link,
                     work=WorkloadConfig(**{**cfg.work.__dict__,
                                            "n_windows": cfg.work.n_windows * scale}))
    conns, gw, edges = generate_dataset(ccfg)
    bg = [c for c in conns if not c.relayed]                # operator cannot touch these
    df = build_features(bg, gw, edges, bin_s, Evasion("clean"),
                        np.random.default_rng(ccfg.seed), corr_pkt_limit)
    taus, meta = {}, {}
    for name, (clf, cols) in clfs.items():
        neg = scores(clf, df[cols].to_numpy(float))
        taus[name] = {}
        for f in FPR_TARGETS:
            tau, ach, n = threshold_at_fpr(neg, f)
            taus[name][f] = tau
            meta[(name, f)] = (tau, ach, n)
    return taus, meta


def run_sweep(cfg: SimConfig, configs, corr_pkt_limit=None, pin_fpr=False):
    """Generate -> harden detectors against scheduling -> evaluate configs -> DataFrame."""
    conns, gw_by_window, edges = generate_dataset(cfg)
    bin_s, link = cfg.work.bin_s, cfg.link
    y = np.array([c.relayed for c in conns], int)
    tr, te = train_test_split(np.arange(len(conns)), test_size=0.3, random_state=cfg.seed, stratify=y)
    train_conns = [conns[i] for i in tr]
    test_conns = [conns[i] for i in te]

    tr_clean = build_features(train_conns, gw_by_window, edges, bin_s, Evasion("clean"),
                              np.random.default_rng(cfg.seed), corr_pkt_limit)
    tr_sched = build_features(train_conns, gw_by_window, edges, bin_s, Evasion("sched", schedule=True),
                              np.random.default_rng(cfg.seed + 1), corr_pkt_limit)
    pool = pd.concat([tr_clean, tr_sched], ignore_index=True)
    clfs = {}
    for name, cols in FEATURE_SETS.items():
        clf = make_classifier(cfg.seed)
        ytr = pool["relayed"].to_numpy(int)
        clf.fit(pool[cols].to_numpy(float), ytr, sample_weight=compute_sample_weight("balanced", ytr))
        clfs[name] = (clf, cols)

    taus, meta = (calibrate(cfg, clfs, bin_s, corr_pkt_limit) if pin_fpr else (None, {}))
    res = pd.DataFrame([eval_config(ev, test_conns, clfs, gw_by_window, edges, bin_s, link,
                                    1000 + i, taus, corr_pkt_limit)
                        for i, ev in enumerate(configs)])
    res.attrs["calib"] = meta
    return res


def cost_at_recall(res, detector, target=50.0, fpr=None, prefix="sched+jit"):
    """Interpolated added-latency (ms) at which `detector` recall crosses `target`,
    along one evasion arm. NaN if never crossed.

    Linear interpolation between the two bracketing grid points; the reported
    cost is therefore an interpolated quantity, not a measured grid value.
    """
    d = res[res.name.str.startswith(prefix) | (res.name == "sched")].sort_values("added_latency_ms")
    x = d["added_latency_ms"].to_numpy()
    r = d[_rk(detector, fpr)].to_numpy()
    below = np.where(r < target)[0]
    if not len(below):
        return float("nan")
    i = below[0]
    if i == 0:
        return float(x[0])
    r0, r1, x0, x1 = r[i - 1], r[i], x[i - 1], x[i]
    frac = (r0 - target) / (r0 - r1) if r0 != r1 else 1.0
    return float(x0 + frac * (x1 - x0))


def fit_power_law(df, col, seeds_axis="seed"):
    """Eq. (6): weighted log-log fit. Returns (exponent, stderr, prefactor, chi2/dof).

    Weights are 1/var of the log-mean, and the stderr is inflated by
    sqrt(chi2/dof) when the seed scatter exceeds the s.e.m., so a poor fit widens
    the bar rather than hiding in it.
    """
    p = df.pivot(index=seeds_axis, columns="bin_s", values=col)
    x = np.log(np.array([c * 1000 for c in p.columns], float))
    m = p.mean().to_numpy()
    sem = (p.std() / np.sqrt(len(p))).to_numpy()
    w = 1.0 / (sem / m) ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    C = np.linalg.inv(X.T @ np.diag(w) @ X)
    beta = C @ (X.T @ np.diag(w) @ np.log(m))
    resid = np.log(m) - X @ beta
    dof = max(1, len(x) - 2)
    chi2 = float(resid @ np.diag(w) @ resid) / dof
    se = np.sqrt(np.diag(C) * max(chi2, 1.0))
    return beta[0], se[0], float(np.exp(beta[1])), chi2

# ---------------------------------------------------------------- 
# experiments

def exp_frontier(seeds=None):
    """Figs. 2-3 and Table II: every evasion configuration against every detector.

    """
    frames = []
    for s in (seeds or PAPER_SEEDS):
        res = run_sweep(SimConfig(seed=s), combined_configs(SIGMAS_MAIN), pin_fpr=True)
        res.insert(0, 'seed', s)
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def exp_combined(seeds=None, frontier=None):
    """Table II: the combined-attack settings of the reference work.

    """
    res = frontier if frontier is not None else exp_frontier(seeds)
    named = res[res.name.isin([n for n, _ in NDSS_COMBOS])].copy()
    # SC+IPD is the pure-jitter point at the detector's own bin width; it carries
    # a sweep name, so relabel it into the table's vocabulary.
    ipd = res[res.name == f"sched+jit{IPD_SIGMA}"].copy()
    ipd["name"] = "SC+IPD"
    keep = ["seed", "name", "added_latency_ms", "bw_overhead", "sla_violation_pct"] \
        + [_rk(d) for d in DETECTORS]
    out = pd.concat([named, ipd], ignore_index=True)[keep]
    return out[out.name.isin(TABLE2_ROWS)].reset_index(drop=True)


def exp_scaling(seeds=None, deltas=None):
    """Table III and Eq. (6): cost to halve TA+Corr recall vs the bin width.

    Both primitives, each measured on a grid scaled to its own Delta (see
    SIGMA_FRACS / TAU_FRACS) so the interpolation bracket is a constant fraction
    of the cost at every bin width. Emits per-seed rows. The table quotes the
    mean and s.e.m. over seeds.

    pin_fpr is off: Table III reads the default-threshold column only, and the
    calibration draw does not touch the RNG stream that column depends on.
    """
    rows = []

    for b in (deltas or SCALING_BINS):
        for s in (seeds or PAPER_SEEDS):
            cfg = SimConfig(seed=s, work=WorkloadConfig(bin_s=b))
            rj = run_sweep(cfg, jitter_configs([round(f * b, 6) for f in SIGMA_FRACS]))
            rs = run_sweep(cfg, shift_configs(sorted({round(f * b, 6) for f in TAU_FRACS})))
            rows.append(dict(bin_s=b, seed=s,
                             jitter=cost_at_recall(rj, "TA+Corr"),
                             shift=cost_at_recall(rs, "TA+Corr", prefix="sched+shift"),
                             clean_Corr=float(rj.loc[rj.name == "clean", "recall_Corr"].iloc[0])))
    return pd.DataFrame(rows)


def exp_jitter(seeds=None):
    """Sec. V-B: the jitter surcharge and its two pinned operating points.

    """
    rows = []
    for s in (seeds or PAPER_SEEDS):
        res = run_sweep(SimConfig(seed=s), jitter_configs(SIGMAS_MAIN), pin_fpr=True)
        r = dict(seed=s, cost50_TAcorr=cost_at_recall(res, "TA+Corr"),
                 cost50_Corr=cost_at_recall(res, "Corr"),
                 clean_Corr=float(res.loc[res.name == "clean", "recall_Corr"].iloc[0]))
        for f in FPR_TARGETS:
            r[f"cost50_TAcorr@fpr{f:g}"] = cost_at_recall(res, "TA+Corr", fpr=f)
            r[f"clean_TAcorr@fpr{f:g}"] = float(
                res.loc[res.name == "clean", _rk("TA+Corr", f)].iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)
