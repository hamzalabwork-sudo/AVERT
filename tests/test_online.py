"""The streaming feature engine must reproduce the batch features exactly."""
import copy

import numpy as np

from avert.adaptive import apply_overclaim
from avert.config import load_config
from avert.features import build_features, observable_view, to_tensor
from avert.generator import generate_world
from avert.online import REPORT_FIELDS, OnlineFeatureEngine

CFG = load_config("configs/default.yaml")


def _world(seed=7, n=20):
    g = copy.deepcopy(CFG["generator"])
    g.update(n_assets=n, n_sites=2, compromise_fraction=0.3)
    rec, assets = generate_world(g, seed)
    return g, rec.sort_values(["asset_id", "day"]).reset_index(drop=True), assets.sort_values("asset_id").reset_index(drop=True)


def test_engine_matches_batch_features():
    g, rec, assets = _world()
    A, D, t0 = g["n_assets"], g["n_days"], 438
    obs = observable_view(rec)
    X = to_tensor(build_features(obs, assets, g["commissioning_days"]), A, D)
    eng = OnlineFeatureEngine(obs, assets, g["commissioning_days"], t0)
    grid = {f: rec[f].to_numpy(float).reshape(A, D) for f in REPORT_FIELDS}
    for day in range(t0, D):
        rep = {f: grid[f][:, day].copy() for f in REPORT_FIELDS}
        rep["day"] = day
        f, p = eng.step(rep)
        assert np.abs(f.astype(np.float32) - X[:, day]).max() < 1e-4, day
        eng.commit(p)


def test_step_without_commit_is_side_effect_free():
    g, rec, assets = _world(seed=8)
    A, D, t0 = g["n_assets"], g["n_days"], 500
    eng = OnlineFeatureEngine(observable_view(rec), assets, g["commissioning_days"], t0)
    rep = {f: rec[f].to_numpy(float).reshape(A, D)[:, t0].copy() for f in REPORT_FIELDS}
    rep["day"] = t0
    f1, _ = eng.step(rep)
    eng.step(apply_overclaim(rep, np.arange(5), np.full(5, 0.5)))      # hypothetical evaluation
    f2, _ = eng.step(rep)
    assert np.array_equal(f1, f2)


def test_overclaim_is_channel_consistent():
    rep = {f: np.array([8.0, 0.0]) for f in REPORT_FIELDS}
    rep["fuel_l"] = np.array([80.0, 0.0])
    rep["idle_hours"] = np.array([2.0, 0.0])
    rep["day"] = 600
    out = apply_overclaim(rep, np.array([0, 1]), np.array([0.25, 0.25]))
    assert out["reported_hours"][0] == 10.0 and out["meter_delta"][0] == 10.0
    assert np.isclose(out["fuel_l"][0] / out["reported_hours"][0], 80.0 / 8.0)   # fuel per hour preserved
    assert out["reported_hours"][1] == 0.0                                       # no phantom claims on idle days
