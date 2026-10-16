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
Feature extraction for the frontier sweep.

Mirrors the NDSS'26 feature families:
  - RTT / TA "time" features: cross-layer RTT discrepancy is the dominant one
    (the Huang-et-al. / RTT signal the scheduling attack targets).
  - TA "volume/structural" features: size & timing summaries.
  - Corr features: NDSS Algorithm 1 — bin connection vs gateway flow at 0.1s,
    z-score each, take the element-wise product, summarise.

"""

from __future__ import annotations
import numpy as np
import pandas as pd

from evasion import apply_evasion


def _zscore(x):
    s = x.std()
    return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s


def rtt_features(c, schedule):
    tcp_rtt = 2 * c.last_hop_ms + c.jit_tcp                      # TCP terminates at exit (short)
    if c.relayed and not schedule:
        app_rtt = 2 * c.last_hop_ms + 2 * c.backhaul_ms + c.jit_app   # TLS loops through backhaul
    elif c.relayed and schedule:
        app_rtt = 2 * c.last_hop_ms + c.jit_resid                # backhaul masked in handshake window
    else:                                                        # background
        app_rtt = 2 * c.last_hop_ms + c.jit_app
    disc = app_rtt - tcp_rtt
    return dict(tcp_rtt=tcp_rtt, app_rtt=app_rtt, rtt_discrepancy=disc,
                rtt_ratio=app_rtt / tcp_rtt if tcp_rtt > 0 else 1.0)


def ta_size_timing(t, b, d):
    up = float(b[d == 1].sum())
    down = float(b[d == -1].sum())
    ts = np.sort(t)
    iat = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    return dict(total_up=up, total_down=down, n_pkts=int(len(t)),
                mean_pkt=float(b.mean()) if len(b) else 0.0,
                down_up_ratio=down / (up + 1.0),
                duration=float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
                mean_iat=float(iat.mean()), std_iat=float(iat.std()), max_iat=float(iat.max()))


def _corr_stats(prod, valid=1):
    """The nine statistics the released NDSS extractor emits over the product vector."""
    return dict(corr_count=float(prod.size), corr_sum=float(np.sum(prod)),
                corr_mean=float(np.mean(prod)), corr_median=float(np.median(prod)),
                corr_min=float(np.min(prod)), corr_max=float(np.max(prod)),
                corr_range=float(np.max(prod) - np.min(prod)),
                corr_var=float(np.var(prod)), corr_std=float(np.std(prod)),
                corr_valid=valid)


_ZERO_CORR = dict(corr_count=0.0, corr_sum=0.0, corr_mean=0.0, corr_median=0.0,
                  corr_min=0.0, corr_max=0.0, corr_range=0.0, corr_var=0.0,
                  corr_std=0.0, corr_valid=0)


# Lag search window, in SECONDS (0 disables it and reproduces the zero-lag
# correlator of the reference work). Specified as a time rather than a bin count
# so that sweeping the bin width does not change the search range as side effect.
#
# The reference work considered lag and dropped it, having tested it only on
# traffic where the misalignment is small. Against an operator who
# deliberately delays the relayed series by one bin width that choice is seriously suboptimal.
# Recall falls to ~0.5% for a constant 100 ms shift. Searching lags restores it.
CORR_LAG_BINS = 0
# CORR_LAG_BINS = 3       # +/- 3 bins, see the lag discussion


def corr_features(t, b, gw, edges, bin_s, start_s, end_s, pkt_limit=None):
    """Binned gateway-vs-relay correlation (NDSS's Algorithm 1).

    `pkt_limit=None` (default) correlates over the whole connection. Setting it to
    20 reproduces the *released* NDSS extractor, which keeps only connections with
    >= pkt_limit packets, correlates over the first pkt_limit packets only, and
    bounds the window at (last kept packet + 1 s). That narrower window is the one
    substantive difference between our reimplementation and theirs.
    """
    t = np.asarray(t, float)
    b = np.asarray(b, float)
    if pkt_limit is not None:
        if len(t) < pkt_limit:      # NDSS ref drops short connections
            return dict(_ZERO_CORR)
        keep = np.argsort(t, kind="stable")[:pkt_limit]
        t, b = t[keep], b[keep]
        start_s, end_s = float(t.min()), float(t.max()) + 1.0   # bound_range = 1.0 s
    lo = max(0, int(np.floor(start_s / bin_s)))
    hi = min(len(gw), int(np.ceil(end_s / bin_s)) + 1)
    if hi - lo < 3:
        return dict(_ZERO_CORR)
    conn_bins, _ = np.histogram(t, bins=edges, weights=b)
    cz = _zscore(conn_bins[lo:hi])
    
    L = int(CORR_LAG_BINS)
    if not L:
        return _corr_stats(cz * _zscore(gw[lo:hi]))
    
    # Slide the gateway window against the connection window and keep the lag
    # that maximises the mean product: a constant relay delay is then absorbed
    # by the alignment rather than read as decorrelation.
    best, best_m = None, -np.inf
    for k in range(-L, L + 1):
        a, z = lo + k, hi + k
        if a < 0 or z > len(gw):
            continue
        pr = cz * _zscore(gw[a:z])
        m = float(np.mean(pr))
        if m > best_m:
            best, best_m = pr, m
    return _corr_stats(best) if best is not None else dict(_ZERO_CORR)


RTT_COLS = ["tcp_rtt", "app_rtt", "rtt_discrepancy", "rtt_ratio"]
TA_COLS = RTT_COLS + ["total_up", "total_down", "n_pkts", "mean_pkt",
                      "down_up_ratio", "duration", "mean_iat", "std_iat", "max_iat"]
CORR_COLS = ["corr_mean", "corr_median", "corr_std", "corr_max"]


def featurize_one(c, t, b, d, gw, edges, bin_s, schedule, corr_pkt_limit=None):
    row = {"cid": c.cid, "relayed": int(c.relayed)}
    row.update(rtt_features(c, schedule))
    row.update(ta_size_timing(t, b, d))
    e_end = max(c.end_s, float(t.max()) if len(t) else c.end_s)
    row.update(corr_features(t, b, gw, edges, bin_s, c.start_s, e_end, corr_pkt_limit))
    return row


def build_features(conns, gw_by_window, edges, bin_s, ev, rng=None, corr_pkt_limit=None):
    # TODO: expose corr_pkt_limit on the CLI, it is only reachable from Python now
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    for c in conns:
        if c.relayed:
            t, b, d = apply_evasion(c.pkt_t, c.pkt_bytes, c.pkt_dir, c.start_s, c.end_s,
                                    ev, rng, cid=c.cid)
        else:
            t, b, d = c.pkt_t, c.pkt_bytes, c.pkt_dir
        rows.append(featurize_one(c, t, b, d, gw_by_window[c.window_id], edges, bin_s,
                                  ev.schedule, corr_pkt_limit))
    return pd.DataFrame(rows)
