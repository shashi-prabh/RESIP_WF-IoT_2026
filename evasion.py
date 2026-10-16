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
Evasion primitives and their QoS cost

Each primitive is *one* physical transformation of the relay's observed packet
series, which simultaneously (a) perturbs detector features and (b) costs QoS,
both derived from the same transformation:

  schedule      re-time one handshake packet     -> masks RTT discrepancy   | cost: none
  jitter_sigma  add IPD jitter (seconds)         -> smears volume bins      | cost: latency
  pad_frac      add chaff bytes at random times  -> perturbs volume bins    | cost: bandwidth
  reshape_mult  split packets into sub-packets   -> alters size sequence    | cost: header bytes

"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

HEADER_BYTES = 70.0   # IP+TCP+TLS-record overhead per packet
CHAFF_PKT = 1200.0    # bytes per chaff packet
VAR_JITTER_SEED = 90210   # fixes the per-connection sigma draw for variable jitter

# Grid for the exact E[max of n half-normals]; built once, reused for every n.
_X = np.linspace(0.0, 12.0, 40001)
_F = np.vectorize(math.erf)(_X / math.sqrt(2.0))
_EMAX_CACHE: dict[int, float] = {}


def emax_halfnormal(n):
    """E[max of n iid |N(0,1)|] = int_0^inf (1 - erf(x/sqrt2)^n) dx.

    Jitter is drawn as |N(0,sigma)| (negative delay is unphysical), so the last
    useful byte slips by sigma * E[max of N half-normals]. 
    The integral is evaluated numerically and cached for reuse.
    """
    # return math.sqrt(2.0 * math.log(max(1, n)))   # normal asymptote, overstates by 10-14%
    n = int(max(1, n))
    if n not in _EMAX_CACHE:
        _EMAX_CACHE[n] = float(np.trapz(1.0 - _F ** n, _X))
    return _EMAX_CACHE[n]


@dataclass
class Evasion:
    name: str = "none"
    schedule: bool = False     # mask cross-layer RTT discrepancy
    pad_frac: float = 0.0      # chaff bytes added, as a multiple of the payload (may exceed 1)
    reshape_mult: float = 1.0  # packet split factor (>= 1)
    jitter_sigma: float = 0.0  # IPD jitter std (seconds); MEAN sigma if jitter_variable
    jitter_variable: bool = False  # per-connection sigma ~ U(0, 2*jitter_sigma)
    shift_s: float = 0.0       # CONSTANT delay (seconds) applied to the whole relayed series


def effective_sigma(ev: Evasion, cid=0):
    """The jitter sigma actually applied to connection `cid`.

    With `jitter_variable` the operator randomises intensity per flow. 
    The draw is keyed on `cid` so the feature side and the cost side see the same sigma, and E[sigma] is unchanged.
    """
    if not ev.jitter_sigma or ev.jitter_sigma <= 0:
        return 0.0
    if not ev.jitter_variable:
        return float(ev.jitter_sigma)
    rng = np.random.default_rng(VAR_JITTER_SEED + int(cid))
    return float(rng.uniform(0.0, 2.0 * ev.jitter_sigma))


def apply_evasion(t, b, d, start_s, end_s, ev: Evasion, rng, cid=0):
    """Transform a relay flow's (times, bytes, dirs); returns sorted arrays."""
    t = np.asarray(t, float).copy()
    b = np.asarray(b, float).copy()
    d = np.asarray(d, int).copy()
    if rng is None:
        rng = np.random.default_rng(0)

    if ev.reshape_mult and ev.reshape_mult > 1.0:          # split packets (+ header overhead each)
        m = int(round(ev.reshape_mult))
        if m > 1:
            t = np.repeat(t, m) + rng.normal(0, 1e-3, size=len(t) * m)
            d = np.repeat(d, m)
            b = np.repeat(b / m, m) + HEADER_BYTES

    sigma = effective_sigma(ev, cid)
    if sigma > 0:                                          # IPD jitter -> latency, decorrelates bins
        t = t + np.abs(rng.normal(0, sigma, size=t.shape))

    if ev.shift_s:                                         # constant lag: shifts, does not smear
        t = t + float(ev.shift_s)

    if ev.pad_frac and ev.pad_frac > 0:                    # chaff at independent times -> bw, decorrelates
        chaff_bytes = ev.pad_frac * float(b.sum())
        n_ch = max(1, int(chaff_bytes / CHAFF_PKT))
        ct = rng.uniform(start_s, max(end_s, start_s + 1e-3), size=n_ch)
        t = np.concatenate([t, ct])
        b = np.concatenate([b, np.full(n_ch, chaff_bytes / n_ch)])
        d = np.concatenate([d, np.full(n_ch, -1, dtype=int)])

    order = np.argsort(t)
    return t[order], b[order], d[order]


def qos_cost(c, link, ev: Evasion):
    """
    Per-FETCH QoS cost of an evasion config, from the knobs and the link.

    A relayed fetch's latency is handshake setup (TCP+TLS, ~2 end-to-end RTTs
    through the backhaul) plus transfer, which for a short response is the larger
    of uplink serialization S/mu_up and a few TCP slow-start rounds. 
    
    Each primitive maps to a cost:
      jitter: delays the last byte by sigma*E[max of N half-normals].
      pad: chaff bytes over the uplink: a BANDWIDTH cost (and latency only
                 when the uplink is the bottleneck, e.g. on DSL).
      reshape: (m-1)*N header bytes: a (smaller) bandwidth cost.

    """
    
    b0 = np.asarray(c.pkt_bytes, float)
    S = float(b0.sum())
    N = max(2, len(b0))
    mu_up = link.up_mbps * 1e6 / 8.0                       # uplink bytes/s (relay bottleneck)
    rtt = 2.0 * (c.last_hop_ms + c.backhaul_ms) / 1000.0   # end-to-end RTT through the backhaul
    T_setup = 2.0 * rtt                                    # TCP + TLS handshakes (~2 RTT)
    n_ss = 3.0                                             # TCP slow-start rounds for a short fetch
    # n_ss = math.ceil(math.log2(1 + S / (10 * 1460)))     # exact rounds; 3 is the median anyway

    extra_header = (round(ev.reshape_mult) - 1) * N * HEADER_BYTES if ev.reshape_mult > 1 else 0.0
    chaff = ev.pad_frac * S if ev.pad_frac > 0 else 0.0
    Be = S + extra_header + chaff
    sigma = effective_sigma(ev, getattr(c, "cid", 0))
    # A constant shift delays every packet equally, so it moves the completion
    # time by exactly tau -- it costs tau, not sigma*E[max].
    dspan = sigma * emax_halfnormal(N) + float(ev.shift_s)

    T0 = T_setup + max(S / mu_up, n_ss * rtt)
    Te = T_setup + max(Be / mu_up, n_ss * rtt) + dspan
    return dict(
        base_latency=T0,
        added_latency=max(0.0, Te - T0),
        total_latency=Te,                                  # per-FETCH latency implying SLA violations
        bw_overhead=(Be - S) / S if S > 0 else 0.0,
        goodput_loss=(1.0 - T0 / Te) if Te > 0 else 0.0,
        added_jitter=sigma,
    )
