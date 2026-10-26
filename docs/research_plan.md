# AVERT — Research plan

**Working title:** *How Much Can an Adaptive Adversary Extract? Game-Theoretic Verification and Guaranteed
Risk Control for IoT-Initiated Payments*
**Target venue:** IEEE Transactions on Information Forensics and Security (alternative: IEEE TDSC)

**Problem.** Connected assets increasingly trigger payments automatically (usage-based leasing, metering,
pay-per-use services). Pre-settlement defences that combine an anomaly detector, device trust and
calibrated approve / hold / reject thresholds are evaluated against attackers drawn from a fixed
distribution and report *average-case* protection. A rational attacker instead *adapts* to the defence.
AVERT asks how much money an adaptive attacker can extract from such a deterministic trust-aware
authorisation (DTA) defence, and how to **bound** that amount with provable guarantees.

## Research questions

| RQ | Question |
|---|---|
| RQ1 | How much value can an adaptive attacker extract from a deterministic trust-aware authorisation defence when it observes settlement outcomes, or knows the defence (white-box)? |
| RQ2 | Can randomised, risk- and value-weighted verification (a Stackelberg inspection game) with clawback and trust slashing make over-claiming unprofitable, with a provable bound on extractable value? |
| RQ3 | Can conformal risk control give finite-sample guarantees on the legitimate-claim delay rate, also under drift? |
| RQ4 | Do the findings hold on real usage-billing fraud (SGCC electricity theft, under-reporting) as well as on synthetic over-claiming? |

## Threat model

The attacker controls compromised devices and chooses, every reporting period, how much to over-claim
(or under-report) and how consistently to fake correlated channels. Knowledge levels:

| Level | Attacker knows / observes |
|---|---|
| K0 oblivious | nothing (the simulator's fixed attack families) |
| K1 feedback | its own settlement outcomes (paid / held / rejected), with one period delay |
| K2 surrogate | K1 + a surrogate detector trained on data it can collect |
| K3 white-box | the full defender (features, detector, trust, thresholds), but not the verification randomness |

The attacker's objective is the **extracted value** Σ (paid − true amount), including amounts below the
fraud-labelling tolerance (salami over-claiming is real money).

## Hypotheses (to be fixed before the confirmatory run)

- **H1** Against K1–K3 attackers, a deterministic-threshold defence loses substantially more value than
  against K0 attackers.
- **H2** Randomised value-weighted verification with clawback reduces worst-case extracted value relative to
  deterministic thresholds at equal verification budget.
- **H3** The empirical deterrence threshold (verification probability at which attacker profit becomes
  ≤ 0) agrees with the theoretical condition within the confidence interval.
- **H4** Conformal calibration keeps the legitimate-hold rate at or below its target in ≥ 95 % of fleets,
  including drift scenarios.

## Seed allocation (never reused across roles)

| Role | Seeds |
|---|---|
| Pilot (RQ1 gap measurement) | 300–302 |
| Development (method design, tuning) | 310–319 |
| Confirmatory evaluation | 400–419 (in-distribution), 420–424 per shift scenario |

No other seeds are used for any reported claim.

## Pilot result (RQ1, seeds 300–302, 20 % of assets compromised; `results/pilot/`)

The streaming defence reproduces the batch pipeline exactly (feature difference ≤ 4·10⁻⁶, score difference
≤ 5·10⁻⁶, identical settled value). Mean extracted value over the test period:

| Attacker | Extracted value | × oblivious | Share of attempted over-claim paid |
|---|---|---|---|
| K0 oblivious | 14.3k | 1.0 | 5.5 % |
| Oracle constant 5 % over-claim | 53.5k | 3.7 | 82 % |
| Oracle constant 3 % (salami; 99.5 % below tolerance) | 36.8k | 2.6 | 94 % |
| K1 feedback (AIMD) | 73.6k | 5.1 | 66 % |
| K3 white-box greedy | 130.4k | 9.1 | 76 % |

Honest devices' legitimate-hold rate stays ≈ 0.5 % under every attacker, so the defender receives no signal.
This supports H1 and motivates the randomised-verification design (docs/theory.md).

## Phases

| Phase | Work | Status |
|---|---|---|
| 1 | Streaming feature engine; adaptive attackers K1/K3; pilot measurement (RQ1) | done |
| 2 | Theory (deterrence condition, extraction bound); randomised verification game; aggregate audits; conformal layer; RL attacker and best-response training — development seeds only | next |
| 3 | SGCC real-data pipeline (under-reporting) | |
| 4 | Pre-registration (`docs/confirmatory_plan.md`), confirmatory run on fresh seeds | |
| 5 | Manuscript and reproducibility package | |
