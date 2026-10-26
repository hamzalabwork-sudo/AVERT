# AVERT — Confirmatory plan (pre-registration)

This plan is committed to the repository **before any confirmatory seed (400–424) is run**. The commit
that adds it fixes the code, the configuration (`configs/default.yaml`) and the analysis
(`scripts/analyse.py`). Deviations, if any, are reported in the paper as deviations.

## Frozen design

All choices below were made on development seeds 310–319 only (`results/dev`, `results/dev2`,
`results/dev_grid`, `results/sgcc_dev`, `results/theory_dev`).

| Component | Setting |
|---|---|
| Baseline (DTA) | TCN + XGBoost stacked detector, adaptive Beta trust, validation-calibrated thresholds, hard rules, low-trust gate (`baseline_design`) |
| AVERT | `q_floor` 0.05, `kappa_q` 0.5, `q_max` 1, fine `penalty` 1.0, trust `slash` 100 (20 evidence units), aggregate test on sub-tolerance discrepancies with `agg_window` 30, `agg_min` 4, `agg_z` 4.265 (α = 10⁻⁵ per test), scale max(σ_commissioning, σ_fleet over the last 60 days, σ_device), each σ a robust 1.4826·MAD, conformal approve threshold at `alpha_hold` 0.05 (reject at `alpha_reject` 0.01) |
| Comparators | DTA; uniform audits q ∈ {0.05, 0.10, 0.15} without consequences; uniform audits with AVERT's fine and quarantine at q ∈ {0.05, 0.075, 0.10} |
| Ablations | AVERT without: aggregate test, fine, quarantine (slash = one verification), value weighting (κ_q = 0), conformal threshold |
| Attacker library (12) | K0 oblivious; constant over-claim 3, 5, 8, 12, 20, 35 %; salami +0.40 h (SGCC: 0.15 units); on-off 30 % 5/15 days; K1 AIMD feedback; K3 white-box greedy; K3* risk-aware white-box |
| Compromised share | 20 % (S3: 40 %) |

**Development history.** Development changed AVERT in three ways:

1. *Masking in the aggregate test.* Large, already-established over-claims inflated a sample standard
   deviation, so the buffer now keeps only sub-tolerance discrepancies and uses robust scales.
2. *False accusations when honest metering noise tripled (S2, seed 316).* A device scale estimated from a
   few claims underestimates the new noise, so a fleet-wide robust scale over the last 60 days was added.
3. *Tuning.* A grid over κ_q ∈ {0.5, 1, 2} × slash ∈ {50, 100} on seeds 310–313, plus S2 on seed 316 and
   SGCC on seed 313. The rule was to minimise the mean worst-case payer loss subject to the honest
   verification rate ≤ 10 % (simulator, in distribution and S2), the legitimate-hold rate ≤ 5 % and no
   falsely flagged honest device. It selected κ_q = 0.5 and slash = 100. Larger κ_q lowers the loss
   further but exceeds the budget under S2 and on SGCC, and on SGCC it flagged honest customers.

## Runs

| ID | Data | Seeds | Command |
|---|---|---|---|
| C1 | Simulator, in distribution | 400–419 | `python scripts/dev_defence.py --role confirmatory --seeds 400-419 --which all --tag main --out results/confirmatory` |
| C2–C5 | Simulator, shifts S1–S4 | 420–424 each | `python scripts/dev_defence.py --role confirmatory --seeds 420-424 --shift <S> --which shift --tag <S> --out results/confirmatory` |
| C6 | SGCC smart meters | 400–419 | `python scripts/sgcc_case.py --confirmatory --seeds 400-419 --which all --tag sgcc --out results/sgcc` |
| C7 | Theory checks (simulator) | 400–404 | `python scripts/theory_check.py --part ABC --role confirmatory --seeds 400,401,402,403,404 --out results/theory` |

Shift scenarios start on the first test day (day 584), after all calibration:
S1 usage surge (true usage × 1.25); S2 noisy meters (metering noise × 3); S3 more compromised assets
(40 %); S4 unreliable links (delay probability 0.15, channel dropout 0.05).

## Hypotheses and tests

The primary metric is the **worst-case payer loss** per fleet: the maximum over the 12 strategies of the
over-payment to compromised devices not clawed back (fine income excluded). Tests are paired by fleet,
one-sided Wilcoxon signed-rank. The p-values of H1, H2a–c and H5a–b are Holm-corrected together at
α = 0.05. Effect sizes are means with 95 % bootstrap confidence intervals (10,000 resamples).

- **H1 (RQ1).** Under DTA, the extracted value of the best adaptive attacker (max of K1, K3, K3*) exceeds
  that of the oblivious attacker (C1).
- **H2a (RQ2).** Worst-case payer loss is lower under AVERT than under DTA (C1).
- **H2b.** It is lower under AVERT than under uniform audits at q = 0.15 without consequences. That
  comparator verifies more honest claims than AVERT (C1).
- **H2c.** It is lower under AVERT than under uniform audits with fine and quarantine at q = 0.10. That
  comparator verifies at least as many honest claims as AVERT (C1).
- **H3 (RQ3, theory).**
  (a) In every cell of the C7 deterrence sweep (fleet × q × λ × attacker × quarantine variant), the
  measured payer loss and attacker profit do not exceed the fleet-level Theorem 2 bound. K is the shortest
  quarantine observed in that cell, and v̄ is each device's largest over-claim value.
  (b) Without quarantine, attacker profit is ≤ 0 in at least 95 % of cells with q ≥ q* = 1/(p_d(1+λ)).
  (c) The aggregate test flags 0 honest devices and detects salami over-claiming of +0.30 h and +0.40 h in
  at least 90 % of compromised devices.
- **H4 (RQ3, conformal).** AVERT's legitimate-hold rate of honest devices is ≤ 5 % in at least 95 % of all
  C1–C5 fleets (maximum over attackers per fleet).
- **H5 (RQ4, real data).** On SGCC (C6), (a) H1 and (b) H2a replicate.

Secondary, descriptive (no test): per-attacker losses, ablations, honest-side costs (honest verification
rate, legitimate-hold rate, falsely flagged honest devices, honest value clawed), quarantine lengths and
aggregate-test delays against Proposition 3.

## Reporting

`python scripts/analyse.py --paper paper` produces every number, table and figure in the paper from the
result files. All hypotheses are reported whatever their outcome.
