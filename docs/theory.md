# AVERT — Theory draft (to be refined; full proofs in the manuscript)

## Model

A compromised device submits claims t = 1, 2, … . Its honest usage is h_t; it reports h_t + δ_t with
δ_t ≥ 0 and is paid r(h_t + δ_t) unless the claim is rejected. The over-claimed value is v_t = r δ_t ≤ v̄.

**Verification.** The defender verifies claim t with probability q_t (independent randomisation that the
attacker cannot observe in advance). A verification reveals the true usage; an over-claim is *established*
with probability p_d if δ_t > τ (single-claim tolerance protecting honest metering noise) and never if δ_t ≤ τ.

**Consequences of detection.** (i) the claim is settled at the true amount; (ii) a clawback penalty λ v_t
(e.g., forfeited escrow); (iii) trust slashing places the device in quarantine, where every claim is
verified, until K consecutive verified-clean claims restore its trust (K follows from the Beta update:
the smallest K with T ≥ T_min after a detection, given ω, γ and the prior).

A **deterministic** defence is the special case q_t ∈ {0, 1}, with q_t = 0 on the auto-approval region
{x : R(x) < θ_a}. The DTA baseline is deterministic in this sense.

## Results

**Proposition 1 (deterministic defences are exploitable).** If the auto-approval region of a deterministic
defence contains claims with over-claim value v > 0, a white-box attacker (K3) that only submits claims in
that region extracts at least v per claim indefinitely, irrespective of detector accuracy on
oblivious attacks. *(Trivial; it frames the empirical gap measured in RQ1.)*

**Theorem 1 (deterrence condition).** For a single claim with δ > τ verified with probability q,
the attacker's expected gain is v [1 − q p_d (1 + λ)]. Hence every over-claim above the tolerance has
non-positive expected value if and only if

    q · p_d ≥ 1 / (1 + λ).

*Proof.* Undetected with probability 1 − q p_d (gain v); detected with probability q p_d (gain −λ v). ∎

**Theorem 2 (bounded extraction under randomised verification and quarantine).** Let every claim outside
quarantine be verified with probability at least q, and let the attacker's per-claim over-claim value be at
most v̄. Over any strategy for which each over-claim above τ is detected independently with probability at
least q p_d, the long-run expected extracted value per claim satisfies

    g ≤ r τ  +  v̄ · max(0, (1 − q p_d) − λ q p_d) / (1 + K q p_d).

*Proof sketch.* Renewal–reward. A cycle consists of a fraud phase, ending at the first detection
(N ~ Geometric(q p_d) over-claims, gain v̄ (N − 1) − λ v̄), followed by K quarantine claims with no gain
above the tolerance. E[N] = 1/(q p_d). Interleaving honest claims within a fraud phase adds cycle length
without gain, so it cannot increase the average. Claims with δ ≤ τ are never established and contribute at
most r τ each. ∎

*Reading.* Without penalty (λ = 0): g ≤ rτ + v̄ (1 − q p_d)/(1 + K q p_d), which decreases in both the
verification rate q and the quarantine length K. With the deterrence condition of Theorem 1 the second term
vanishes, leaving only the **tolerance leakage** r τ.

**Proposition 3 (aggregate audits close the tolerance gap).** Honest metering error per claim has standard
deviation σ. Testing the mean discrepancy of m verified claims of the same device with threshold
z_{1−α} σ / √m keeps the false-accusation probability at α, and detects a systematic over-claim δ̄ with
power ≥ 1 − β once

    m ≥ ((z_{1−α} + z_{1−β}) σ / δ̄)².

*Example.* σ = 0.05 h, α = 10⁻³, β = 0.05, δ̄ = 0.1 h (1.25 % of an 8 h day): m ≥ 6 verified claims.
Sub-tolerance ("salami") over-claiming is therefore detectable after a handful of audits, whereas a
per-claim tolerance of τ = 0.5 h never detects it.

**Proposition 4 (distribution-free hold-rate guarantee).** Choosing the hold threshold as the
⌈(1 − α)(n + 1)⌉-th smallest risk among n exchangeable legitimate calibration claims guarantees
P(hold | legitimate) ≤ α + 1/(n + 1) (split conformal). Under drift, adaptive conformal inference keeps the
long-run hold rate at α.

## Defence design implied by the theory (RQ2)

1. **Verification floor.** Every claim, including auto-approved ones, is verified with probability
   q ≥ q_floor, removing the safe region that Proposition 1 exploits.
2. **Risk- and value-weighted allocation.** Above the floor, verification probability increases with risk R
   and claim value relative to expected usage, subject to a verification budget B (Stackelberg inspection
   game; optimised against best-response attackers on development seeds).
3. **Penalty and quarantine.** Clawback λ and trust slashing; choose (q_floor, λ) to satisfy Theorem 1.
4. **Aggregate audits.** Per-device cumulative discrepancy test across verified claims (Proposition 3).
5. **Conformal calibration** of the deterministic hold threshold (Proposition 4).

## Empirical checks (H3)

- Measured attacker profit vs verification probability q should cross zero near q* = 1/(p_d(1 + λ)).
- Measured extraction per claim should stay below the Theorem 2 bound for every attacker, including the
  RL and white-box attackers.
