# AVERT — Adversary-aware VErification with Risk-controlled Trust

Research code for **"How Much Can an Adaptive Adversary Extract? Game-Theoretic Verification and Guaranteed
Risk Control for IoT-Initiated Payments"** (work in progress).

Connected assets increasingly trigger payments automatically: leased machines report operating hours, meters
report consumption, devices claim pay-per-use services. Pre-settlement defences that combine an anomaly
detector, device trust and approve / hold / reject thresholds are usually evaluated against attackers drawn
from a fixed distribution. AVERT studies the **rational, adaptive attacker** that learns how the defence
behaves and optimises how much it over-claims, and designs verification policies whose **worst-case**
financial loss is provably bounded.

## Status

| Component | State |
|---|---|
| AdvLease simulator (usage-based leasing telemetry with financial ground truth and honest pre-attack telemetry) | done |
| Deterministic trust-aware authorisation (DTA) baseline: TCN+XGBoost detector, adaptive Beta trust, calibrated thresholds | done |
| Streaming pipeline (day-by-day features, detector and decisions), verified bit-exact against the batch pipeline | done |
| Adaptive attackers: oracle constant over-claim, K1 feedback (AIMD), K3 white-box greedy | done |
| Pilot: worst-case loss of the deterministic defence (RQ1) | done (`results/pilot/`) |
| Randomised verification game, aggregate audits, conformal risk control, RL attacker | in progress |
| Real-data study (SGCC electricity theft) | planned |

**Pilot finding (seeds 300–302).** Against the same deterministic defence, a feedback attacker extracts
**5.1×** and a white-box attacker **9.1×** more value than oblivious attacks, while honest devices see no
change in hold rate. See `docs/research_plan.md`; theory in `docs/theory.md`.

## Layout

| Path | Content |
|---|---|
| `avert/generator.py` | AdvLease simulator |
| `avert/features.py`, `avert/windows.py` | platform-observable features, hard rules, causal windows |
| `avert/detectors.py` | TCN, XGBoost and baseline detectors; calibration and stacking |
| `avert/trust.py`, `avert/authorisation.py` | trust models; risk, decisions, settlement and verification simulation |
| `avert/online.py` | streaming feature engine, detector and defender |
| `avert/adaptive.py` | adaptive attackers and the attacker-vs-defence runner |
| `scripts/pilot_adaptive.py` | RQ1 pilot |
| `scripts/generate_dataset.py` | export simulated fleets as CSV |
| `configs/default.yaml` | all parameters |
| `tests/` | leakage guard, invariants, streaming-engine exactness |

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
python scripts/pilot_adaptive.py --seeds 300,301,302 --level 0.20 --out results/pilot
```

Everything is seeded and deterministic when run through the provided scripts (they fix the number of BLAS
threads). CPU only.
