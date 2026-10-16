
# RESIP_WF-IoT_26 Repository
Code to reproduce "Fast or Stealthy, Not Both: The QoS Cost of Evading Residential-Proxy Detection" published in WF-IoT 2026

# Intended use

This repository contains a simulator. It does not implement a proxy, relay traffic, or interact with any network. The evasion primitives modelled here with the goal to *measure their cost* and **not to enable them.** The work is defensive in aim. Quantifying what evasion costs a proxy operator lets detector designers choose parameters that maximise that cost.


# Install

```
pip install -r requirements.txt
```

`xgboost` is required to exactly replicate the numbers. Otherwise, (`detect.py`) falls back to scikit-learn's `HistGradientBoostingClassifier`, which shifts the numbers a little.


# Reproduction

Data are written in  CSV  files in `results/` folder.

| paper artifact                             | command                        |
| ------------------------------------------ | ------------------------------ |
| Table I (simulator settings)               | `python run.py --exp settings` |
| Table II (combined attacks)                | `python run.py --exp combined` |
| Table III + Eq. (6) (bin-width scaling) | `python run.py --exp scaling` |
| Figs. 2 and 3                              | `python run.py --exp figures`  |
| Sec. V-B surcharge numbers | `python run.py --exp jitter` |

`--seeds` changes the Monte-Carlo draw (the default is the six seeds used for the paper), 
and `--delta` sets the bin widths of the scaling sweep in ms. 


# Simulator files

| file             | role                                                            |
| ---------------- | --------------------------------------------------------------- |
| `simlib.py`      | four-entity relay simulator, link model, workload generation    |
| `evasion.py`     | the evasion primitives and the per-fetch QoS cost model         |
| `features.py`    | RTT, traffic-analysis and correlation features |
| `detect.py`      | gradient-boosted detector and threshold pinning at a target FPR |
| `experiments.py` | the sweeps behind Tables I–III and Figs. 2–3                    |
| `tables.py`      | renders Tables I–III and the Eq. (6) fits                       |
| `figures.py`     | Figs. 2 and 3                                                   |
| `run.py`         | command-line entry point   |


# Scope and limitations

The simulator's two detection signals emerge from an explicit topology rather than being assigned by label, and the QoS model is analytic and single-connection. Results have not been validated against real relay captures. The paper's `TA` and `Corr` families are reimplemented from the reference work's description rather than from its code.
