"""Platform-side feature engineering and hard security rules.

Only columns a real settlement platform could observe are used: the device message, static
asset/contract metadata and same-day reports from peer devices at the same site. Evaluator-only
columns (avert.generator.EVALUATOR_ONLY) are never read here; tests/test_core.py enforces it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .generator import DROPOUT_CHANNELS, EVALUATOR_ONLY, REPORT_HOUR

FEATURES = (
    "hours", "worked", "meter_mismatch", "fuel_rate_ratio", "fuel_norm", "rpm_norm", "temp_rise",
    "idle_frac", "vib_norm", "ambient_peer_dev", "hours_peer_dev", "log_delay", "missing_count",
    "dow_sin", "dow_cos",
    # long-horizon features (v2): usage relative to the asset's own commissioning-period baseline
    "usage_dev", "usage_drift28", "usage_cusum", "fuel_drift28",
)
CUSUM_SLACK = 0.05   # half the smallest shift of interest (~0.1 relative over-claim)
DRIFT_DAYS = 28
OBSERVABLE = (
    "asset_id", "device_id", "site", "day", "seq", "ts_sent_h", "ts_recv_h", "delay_h",
    "signature_valid", "contract_active", "reported_hours", "meter_delta", "rate", "claimed_amount",
    *DROPOUT_CHANNELS,
)
ASSET_META = ("asset_id", "site", "asset_type", "nominal_hours", "rate", "fuel_lph", "idle_rpm",
              "rated_rpm", "vib", "base_temp")


def observable_view(records: pd.DataFrame) -> pd.DataFrame:
    """The only view of the records the detection/authorisation pipeline is given."""
    assert not set(OBSERVABLE) & set(EVALUATOR_ONLY)
    return records.loc[:, list(OBSERVABLE)].copy()


def expected_usage(obs: pd.DataFrame, commissioning_days: int = 120) -> pd.DataFrame:
    """Peer-adjusted, commissioning-anchored expected daily hours for every (asset, day).

    r_i = median over the asset's commissioning days of (own hours / same-site peer median hours);
    baseline_{i,t} = r_i * peer_median_t. Anchoring to the attack-free commissioning period (instead
    of a rolling median of recent paid claims) prevents gradual inflation from shifting the baseline
    it is measured against. Uses observable reports only. Rows in (asset, day) order.
    """
    df = obs[["asset_id", "site", "day", "reported_hours"]].sort_values(["asset_id", "day"]).reset_index(drop=True)
    h = df.reported_hours
    hw = h.where(h > 0.2)
    peer = hw.groupby([df.site, df.day]).transform("median").fillna(0.0)
    comm = (df.day < commissioning_days) & (h > 0.2) & (peer > 0.2)
    r = (h / peer.where(peer > 0.2)).where(comm).groupby(df.asset_id).transform("median").fillna(1.0)
    baseline = np.where(peer > 0.2, r * peer, 0.0)
    return pd.DataFrame({"asset_id": df.asset_id, "day": df.day, "peer_med": peer.to_numpy(),
                         "baseline": baseline})


def _cusum(dev: np.ndarray, k: float) -> np.ndarray:
    s, out = 0.0, np.empty_like(dev)
    for t, x in enumerate(dev):
        s = max(0.0, s + x - k)
        out[t] = s
    return out


def build_features(obs: pd.DataFrame, assets: pd.DataFrame, commissioning_days: int = 120) -> pd.DataFrame:
    meta = assets.loc[:, list(ASSET_META)].drop(columns=["rate", "site"])
    df = obs.merge(meta, on="asset_id", how="left").sort_values(["asset_id", "day"]).reset_index(drop=True)
    missing = df[list(DROPOUT_CHANNELS)].isna().sum(axis=1)
    df[list(DROPOUT_CHANNELS)] = df.groupby("asset_id")[list(DROPOUT_CHANNELS)].ffill().bfill()

    h = df.reported_hours
    worked = h > 0.2
    f = pd.DataFrame(index=df.index)
    f["hours"] = h
    f["worked"] = worked.astype(float)
    f["meter_mismatch"] = h - df.meter_delta
    f["fuel_rate_ratio"] = np.where(worked, df.fuel_l / (h.clip(lower=0.2) * df.fuel_lph), 0.0).clip(0, 3)
    f["fuel_norm"] = df.fuel_l / df.fuel_lph
    f["rpm_norm"] = np.where(worked, (df.avg_rpm - df.idle_rpm) / (df.rated_rpm - df.idle_rpm), 0.0).clip(-1, 2)
    f["temp_rise"] = df.engine_temp_c - df.ambient_c
    f["idle_frac"] = np.where(worked, df.idle_hours / h.clip(lower=0.2), 0.0).clip(0, 1)
    f["vib_norm"] = df.vibration_rms / df.vib
    f["ambient_peer_dev"] = df.ambient_c - df.groupby(["site", "day"]).ambient_c.transform("median")
    f["hours_peer_dev"] = h - df.groupby(["site", "day"]).reported_hours.transform("median")
    f["log_delay"] = np.log1p(df.delay_h.clip(lower=0))
    f["missing_count"] = missing.values
    dow = df.day % 7
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    eu = expected_usage(obs, commissioning_days)          # same (asset, day) order as df
    base = eu.baseline.to_numpy()
    active = worked.to_numpy() | (eu.peer_med.to_numpy() > 0.2)
    dev = np.where(active, (h.to_numpy() - base) / np.maximum(base, 1.0), 0.0).clip(-1, 3)
    f["usage_dev"] = dev
    # drift statistics accumulate over worked days only (absences are not evidence of over-claiming)
    # and are carried forward over idle days
    wk = worked.to_numpy()
    drift, cus = np.zeros(len(df)), np.zeros(len(df))
    for _, idx in df.groupby("asset_id").indices.items():
        idx = np.asarray(idx)
        w_idx = idx[wk[idx]]
        if len(w_idx):
            d = dev[w_idx]
            drift[w_idx] = pd.Series(d).rolling(DRIFT_DAYS, min_periods=1).mean().to_numpy()
            cus[w_idx] = _cusum(d, CUSUM_SLACK)
        drift[idx] = pd.Series(np.where(wk[idx], drift[idx], np.nan)).ffill().fillna(0.0).to_numpy()
        cus[idx] = pd.Series(np.where(wk[idx], cus[idx], np.nan)).ffill().fillna(0.0).to_numpy()
    f["usage_drift28"] = drift
    f["usage_cusum"] = cus.clip(0, 20)
    frr = f["fuel_rate_ratio"].where(worked)
    roll = frr.groupby(df.asset_id).transform(lambda x: x.rolling(DRIFT_DAYS, min_periods=1).mean())
    comm_frr = frr.where(df.day < commissioning_days).groupby(df.asset_id).transform("mean")
    f["fuel_drift28"] = (roll / comm_frr - 1).clip(-2, 2)
    out = pd.concat([df[["asset_id", "day"]], f], axis=1)
    return out.fillna(0.0)


def hard_violations(obs: pd.DataFrame, contract: dict) -> pd.Series:
    """Hard security rules that no trust level can override. Indexed like `obs` (asset, day order)."""
    df = obs.sort_values(["asset_id", "ts_recv_h"])
    dup_seq = df.duplicated(["asset_id", "seq"], keep="first")
    df = df.assign(dup_seq=dup_seq).sort_values(["asset_id", "day"])
    expected_ts = df.day * 24.0 + REPORT_HOUR
    viol = (
        (df.signature_valid == 0)
        | df.dup_seq
        | ((df.ts_sent_h - expected_ts).abs() > contract["ts_window_hours"])
        | (df.reported_hours > contract["max_daily_hours"])
        | (df.contract_active == 0)
    )
    return viol.reset_index(drop=True)


def to_tensor(feat: pd.DataFrame, n_assets: int, n_days: int) -> np.ndarray:
    """[asset, day, feature] float32 array."""
    feat = feat.sort_values(["asset_id", "day"])
    return feat[list(FEATURES)].to_numpy(np.float32).reshape(n_assets, n_days, len(FEATURES))


def to_grid(series_by_row: np.ndarray, n_assets: int, n_days: int) -> np.ndarray:
    return np.asarray(series_by_row).reshape(n_assets, n_days)
