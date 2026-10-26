import copy

import numpy as np
import pytest

from avert.authorisation import HOLD, REJECT, SimData, SystemSpec, risk_score, simulate
from avert.config import load_config
from avert.features import FEATURES, OBSERVABLE, build_features, hard_violations, observable_view, to_tensor
from avert.generator import EVALUATOR_ONLY, generate_world
from avert.trust import AdaptiveTrust, BetaTrust
from avert.windows import WindowStore

CFG = load_config("configs/default.yaml")


@pytest.fixture(scope="module")
def world():
    g = copy.deepcopy(CFG["generator"])
    g.update(n_assets=20, n_sites=2, compromise_fraction=0.3)
    rec, assets = generate_world(g, 3)
    return g, rec.sort_values(["asset_id", "day"]).reset_index(drop=True), assets


# ---------------------------------------------------------------- generator
def test_deterministic():
    g = copy.deepcopy(CFG["generator"])
    g.update(n_assets=5, n_sites=1)
    a, _ = generate_world(g, 11)
    b, _ = generate_world(g, 11)
    assert a.equals(b)


def test_no_attacks_during_commissioning(world):
    g, rec, _ = world
    assert rec.loc[rec.day < g["commissioning_days"], "anomaly_label"].sum() == 0


def test_fraud_implies_manipulation(world):
    _, rec, _ = world
    assert ((rec.fraud_label == 1) & (rec.anomaly_label == 0)).sum() == 0
    honest = rec[rec.anomaly_label == 0]
    assert (honest.reported_hours - honest.reference_hours).abs().max() < 0.5


def test_only_compromised_assets_attacked(world):
    _, rec, assets = world
    attacked = set(rec.loc[rec.anomaly_label == 1, "asset_id"])
    assert attacked <= set(assets.loc[assets.compromised, "asset_id"])


def test_overclaim_matches_labels(world):
    _, rec, _ = world
    assert (rec.overclaim_amount[rec.fraud_label == 0] == 0).all()
    assert (rec.overclaim_amount[rec.fraud_label == 1] > 0).all()


# ---------------------------------------------------------------- leakage guard
def test_observable_excludes_ground_truth():
    assert not set(OBSERVABLE) & set(EVALUATOR_ONLY)


def test_features_invariant_to_ground_truth(world):
    """Scrambling every evaluator-only column must not change a single feature or hard-rule value."""
    g, rec, assets = world
    f1 = build_features(observable_view(rec), assets)
    h1 = hard_violations(observable_view(rec), g["contract"])
    rng = np.random.default_rng(0)
    poisoned = rec.copy()
    for c in EVALUATOR_ONLY:
        poisoned[c] = rng.permutation(poisoned[c].to_numpy())
    f2 = build_features(observable_view(poisoned), assets)
    h2 = hard_violations(observable_view(poisoned), g["contract"])
    assert f1.equals(f2) and h1.equals(h2)


def test_hard_rules_catch_protocol_attacks(world):
    g, rec, _ = world
    hard = hard_violations(observable_view(rec), g["contract"]).to_numpy()
    naive_replay = ((rec.attack_family == "replay") & (rec.knowledge == "naive")).to_numpy()
    naive_forged = ((rec.attack_family == "forged_identity") & (rec.knowledge == "naive")).to_numpy()
    assert hard[naive_replay | naive_forged].all()


# ---------------------------------------------------------------- windows
def test_windows_are_causal_and_per_asset(world):
    g, rec, assets = world
    A, D = g["n_assets"], g["n_days"]
    X = to_tensor(build_features(observable_view(rec), assets), A, D)
    W = 10
    st = WindowStore(X, W, np.zeros(len(FEATURES)), np.ones(len(FEATURES)))
    a, t = np.array([3, 3]), np.array([50, 2])
    w = st.windows(a, t).numpy()
    np.testing.assert_allclose(w[0].T, X[3, 41:51], rtol=1e-6)
    np.testing.assert_allclose(w[1].T[-3:], X[3, 0:3], rtol=1e-6)   # left padding at sequence start
    s = st.summary(a[:1], t[:1])
    F = len(FEATURES)
    np.testing.assert_allclose(s[0, F:2 * F], X[3, 41:51].mean(0), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- trust and risk
def test_risk_monotone():
    s, F = np.array([0.1]), np.array([0.2])
    assert risk_score("noisy_or", np.array([0.9]), s, F, None) < risk_score("noisy_or", np.array([0.5]), s, F, None)
    assert risk_score("noisy_or", np.array([1.0]), np.array([0.0]), np.array([0.0]), None) == 0.0


def test_adaptive_trust_drops_faster_and_recovers():
    tc = CFG["trust"]
    beta, adaptive = BetaTrust(1, tc), AdaptiveTrust(1, tc)
    one, z = np.ones(1, bool), np.zeros(1)
    for _ in range(5):
        for m in (beta, adaptive):
            m.update(np.array([0.9]), z, np.ones(1), z, z, one)
    assert adaptive.value()[0] < beta.value()[0]
    low = adaptive.value()[0]
    for _ in range(200):
        adaptive.update(np.array([0.01]), z, np.ones(1), z, z, one)
    assert adaptive.value()[0] > max(low, tc["trusted"])


def _sim_data(n=4, D=30, hours=8.0):
    h = np.full((n, D), hours)
    z = np.zeros((n, D))
    return SimData(hours=h, claimed=h * 100, rate=np.full(n, 100.0), nominal=np.full(n, 8.0),
                   peer_med=h.copy(), hard=z.copy(), scores={"d": z.copy()}, ref_hours=h.copy(),
                   overclaim=z.copy(), fraud=z.copy(), compromised=np.zeros(n, bool))


def test_hard_violation_overrides_trust():
    data = _sim_data()
    data.hard[0, 10] = 1
    m = simulate(SystemSpec("x", detector="d", trust="none"), data, CFG, (0.3, 0.6), trace=True)
    assert m["_trace"]["dec"][0, 10] == REJECT


def test_low_trust_gate_blocks_auto_approval():
    data = _sim_data()
    data.scores["d"][0, :15] = 1.0          # sustained anomalies drive trust down
    data.scores["d"][0, 15:] = 0.0          # then a clean-looking report
    m = simulate(SystemSpec("x", detector="d", use_F=False), data, CFG, (0.99, 0.999), trace=True)
    T, dec = m["_trace"]["T"], m["_trace"]["dec"]
    assert T[0, 15] < CFG["trust"]["low"] and dec[0, 15] >= HOLD


def test_exposure_alone_is_bounded():
    data = _sim_data()
    data.hours[:, 20] = 30.0                # huge claim, zero anomaly score, full trust
    data.claimed[:, 20] = 3000.0
    m = simulate(SystemSpec("x", detector="d", trust="none"), data, CFG, (0.3, 0.6), trace=True)
    assert m["_trace"]["R"][0, 20] == pytest.approx(CFG["authorisation"]["f_max"])
    assert m["_trace"]["dec"][0, 20] == HOLD
