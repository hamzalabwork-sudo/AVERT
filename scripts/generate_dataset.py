"""Export AdvLease worlds as CSV files (for inspection or use outside this code base).

    python scripts/generate_dataset.py --seed 20 --level 0.20 --out data/advlease
    python scripts/generate_dataset.py --seeds 20-39 --levels 0.05,0.10,0.20,0.30 --out data/advlease

For every (seed, level) it writes three files:
  <name>_observable.csv   what the platform sees: one row per asset-day (device report + claim)
  <name>_groundtruth.csv  evaluator-only labels (true hours, fraud/anomaly labels, attack family, ...)
  <name>_assets.csv       static asset / contract metadata (type, site, rate, rated specifications, ...)
Rows of the two record files align on (asset_id, day). A data dictionary is written once.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avert.config import load_config  # noqa: E402
from avert.features import OBSERVABLE  # noqa: E402
from avert.generator import EVALUATOR_ONLY, generate_world  # noqa: E402

DICTIONARY = """# AdvLease data dictionary

One row = one asset-day. A row with reported_hours > 0 is a payment claim (claimed_amount = reported_hours x rate).

## <name>_observable.csv (inputs available to the platform)
| Column | Meaning |
|---|---|
| asset_id, device_id, site | asset index, device identifier, site index |
| day | day index (day 0 = Monday); 730 days |
| seq | per-device message sequence number (duplicates indicate stale replay) |
| ts_sent_h, ts_recv_h, delay_h | send / receive timestamps (hours since day 0) and delivery delay |
| signature_valid | 1 if the device signature verifies |
| contract_active | 1 if the leasing contract is active |
| reported_hours | operating hours claimed by the device |
| meter_delta | engine-hour meter increment (independent channel) |
| rate | contractual rate per operating hour |
| claimed_amount | reported_hours x rate |
| fuel_l, avg_rpm, engine_temp_c, vibration_rms, ambient_c, idle_hours | correlated telemetry (NaN = transmission dropout) |

## <name>_groundtruth.csv (evaluator only; never used as model input)
| Column | Meaning |
|---|---|
| reference_hours | true operating hours |
| reference_amount | reference_hours x rate (correct payment) |
| overclaim_amount | (reported - true hours) x rate for fraudulent claims, else 0 |
| anomaly_label | 1 if the report was manipulated by an attack |
| fraud_label | 1 if reported hours exceed true hours by more than 0.5 h due to an attack |
| attack_family | none / inflation / replay / forged_identity / gradual / on_off |
| knowledge | none / naive / consistent / adaptive |
| benign_event | none / dropout / stuck_sensor / delay / overtime / maintenance |
| compromised | True if the asset is compromised (attacks start after the 120-day commissioning period) |

## <name>_assets.csv
asset_id, device_id, site, asset_type, nominal_hours, mu_hours, sd_hours, load_mean, drift_min_per_day,
rate, fuel_lph, idle_rpm, rated_rpm, vib, base_temp, compromised, family, knowledge.
(compromised/family/knowledge are evaluator-only.)

Generation parameters: configs/default.yaml (section `generator`); method: docs/methodology.md, Section 2.
"""


def _expand(spec: str, cast):
    out = []
    for part in spec.split(","):
        if "-" in part and cast is int:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(cast(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", help="e.g. 20-39 or 20,21,22")
    ap.add_argument("--level", type=float)
    ap.add_argument("--levels", help="e.g. 0.05,0.10,0.20,0.30")
    ap.add_argument("--out", default="data/advlease")
    args = ap.parse_args()
    seeds = _expand(args.seeds, int) if args.seeds else [args.seed if args.seed is not None else 20]
    levels = _expand(args.levels, float) if args.levels else [args.level if args.level is not None else 0.20]
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DATA_DICTIONARY.md").write_text(DICTIONARY, encoding="utf-8")
    for level in levels:
        for seed in seeds:
            g = copy.deepcopy(cfg["generator"])
            g["compromise_fraction"] = level
            rec, assets = generate_world(g, seed)
            rec = rec.sort_values(["asset_id", "day"]).reset_index(drop=True)
            name = f"advlease_L{int(round(level * 100)):02d}_s{seed}"
            rec[list(OBSERVABLE)].to_csv(out / f"{name}_observable.csv", index=False)
            rec[["asset_id", "day", *EVALUATOR_ONLY]].to_csv(out / f"{name}_groundtruth.csv", index=False)
            assets.to_csv(out / f"{name}_assets.csv", index=False)
            claims = rec.reported_hours > 0
            print(f"{name}: {len(rec)} reports, {int(claims.sum())} claims, "
                  f"fraud claims {rec.fraud_label[claims].mean():.3%}, manipulated {rec.anomaly_label.mean():.3%}")


if __name__ == "__main__":
    main()
