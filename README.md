# AVERT — Adversary-aware VErification with Risk-controlled Trust

Research code for **"How Much Can an Adaptive Adversary Extract? Randomized Verification with Provable
Loss Bounds for Device-Initiated Payments"** (manuscript in preparation).

Connected assets increasingly trigger payments on their own reports: leased machines bill by metered hours,
smart meters by consumption. Pre-settlement defences that combine an anomaly detector, device trust and
approve / hold / reject thresholds are usually evaluated against attacks drawn from a fixed distribution.
AVERT studies the **rational, adaptive adversary** that learns how the defence behaves and chooses how much
to over-claim (or under-report). It measures the money such an adversary can extract and bounds it with
randomized verification, clawback, quarantine, aggregate audits and conformal thresholds.

## Status

| Component | State |
|---|---|
| AdvLease simulator with financial ground truth, honest pre-attack telemetry and benign shift scenarios | done |
| Deterministic trust-aware authorization (DTA) baseline; streaming re-implementation verified against the batch pipeline | done |
| Attacker library: oblivious, constant, salami, on-off, K1 feedback, K3 white-box, K3* risk-aware white-box | done |
| AVERT defence (`avert/verification.py`) | done |
| Theory: deterrence condition, extraction bound under quarantine, aggregate audits, conformal hold rate (`docs/theory.md`) | done |
| Real-data case study on SGCC smart-meter data (`avert/sgcc.py`) | done |
| Pre-registered confirmatory evaluation (`docs/confirmatory_plan.md`) | running |

## Layout

| Path | Content |
|---|---|
| `avert/generator.py` | AdvLease simulator |
| `avert/features.py`, `avert/windows.py`, `avert/detectors.py` | observable features, causal windows, detectors |
| `avert/trust.py`, `avert/authorisation.py` | trust models; risk, decisions, settlement |
| `avert/online.py` | streaming feature engine, detector and DTA decision layer |
| `avert/verification.py` | AVERT: randomized audits, clawback and fines, quarantine, aggregate test |
| `avert/adaptive.py` | attacker library and the attacker-vs-defence runner (simulator) |
| `avert/sgcc.py` | SGCC case study: data loading, detector, under-reporting attackers |
| `scripts/dev_defence.py` | defences × attacker library (development and confirmatory roles, shift scenarios) |
| `scripts/sgcc_case.py` | the same on SGCC data |
| `scripts/theory_check.py` | Monte Carlo and simulator checks of the theory |
| `scripts/analyse.py` | hypothesis tests, tables and figures |
| `scripts/pilot_adaptive.py` | RQ1 pilot |
| `docs/` | research plan, theory, confirmatory plan |

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
python scripts/dev_defence.py --seeds 310 --which main --out results/dev
```

**SGCC data.** Download the three split archives (`data.zip`, `data.z01`, `data.z02`) from
<https://github.com/henryRDlab/ElectricityTheftDetection> into `data/sgcc_raw/` and extract `data.csv`
(e.g., `7z x data.zip`). The dataset was released by State Grid Corporation of China with
Zheng et al., *IEEE TII* 14(4), 2018; please cite it when you use it.

The confirmatory commands are listed in `docs/confirmatory_plan.md`. Every run is seeded and deterministic
when launched through the provided scripts, which fix the number of BLAS threads. CPU only.
