# Addendum to the confirmatory plan: learning adversary

This addendum was written **after** the main confirmatory evaluation (`docs/confirmatory_plan.md`, results
in `results/confirmatory`). It was committed together with the frozen policies **before** any evaluation
seed listed below was run. It adds one adversary and three hypotheses; nothing in the main plan changes.

## Motivation

The registered library contains hand-designed adaptive adversaries (K1, K3, K3*), which plan one step ahead.
A learning adversary can exploit long-horizon structure that they cannot. For example, it can wait out a
quarantine and strike when trust has recovered. Theorem 2 bounds any strategy. This addendum tests that
claim empirically against the strongest learner we can train cheaply.

## Adversary (`avert/rl.py`)

- **Learner:** tabular Q-learning with a policy shared by all compromised devices.
- **State:** 64 cells, observed before the report is committed:
  - trust bucket (quarantined, < 0.75, < 0.90, ≥ 0.90);
  - periods since the device's last hold, rejection or recovery (≤ 3, ≤ 10, ≤ 30, > 30);
  - risk of the honest report relative to the approve threshold (< 0.5, < 0.8, < 1, ≥ 1).
- **Actions:** over-claim fraction in {0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.40, 0.60}. The small fractions
  make sub-tolerance (salami) strategies learnable.
- **Reward:** realized profit between consecutive claim decisions (paid − true amount − clawbacks − fines),
  in units of one typical day's pay. Discount 0.95 per claim, learning rate 0.1, ε-greedy exploration from
  0.30 down to 0.02.
- **Training:** 1,500 episodes per target defence on development fleets 310–312 (simulator, 20 %
  compromised). The defender's verification randomness is re-drawn every episode, so audits cannot be
  memorized. One policy is trained against AVERT and one against DTA (`results/rl/q_AVERT.npy`,
  `results/rl/q_DTA.npy`, training curves in `results/rl/train_*.csv`).
- **Evaluation:** frozen, greedy policies.

## Evaluation

- **Fleets:** fresh simulator fleets, seeds **430–439** (never used before), 20 % compromised, frozen
  configuration.
- **Command:** `python scripts/rl_attacker.py eval --seeds <s> --out results/rl`, one process per seed.
- **Comparisons:** in each fleet the RL policy trained against a defence attacks that defence. K0, K1, K3 and
  K3* are run as references.

## Hypotheses

- **H6a (bound).** Against AVERT, the RL adversary's payer loss does not exceed the fleet-level Theorem 2
  bound in any of the 10 fleets. The bound uses q = q_floor = 0.05, λ = 0 (payer loss), K = the shortest
  quarantine observed in the run, and v̄ = each device's largest over-claim value.
- **H6b (robustness).** The RL adversary's payer loss is lower under AVERT than under DTA.
- **H6c (learner competence).** Under DTA, the RL adversary's payer loss exceeds that of the K1 feedback
  adversary. This checks that the learner learned something; if it fails, H6a and H6b are weak evidence and
  will be reported as such.

H6b and H6c use paired one-sided Wilcoxon signed-rank tests over the 10 fleets, Holm-corrected together
(α = 0.05). H6a is a per-fleet check.

Descriptive results: the RL adversary's payer loss against K3 and K3*, and AVERT's worst case over the
extended library (registered library plus RL).

## Reporting

The paper reports all three hypotheses, whatever their outcome, together with the training curves. The
results are labelled as an addendum that was registered after the main confirmatory evaluation.
