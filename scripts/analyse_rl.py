"""Analysis of the learning-adversary addendum (docs/confirmatory_addendum_rl.md).

    python scripts/analyse_rl.py --paper paper

Writes results/rl/analysis.md, paper/tables/rl.tex, paper/figs/rl_training.pdf and appends macros to
paper/numbers_rl.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RL = "RL (Q-learning)"


def holm(p):
    keys = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default="results/rl")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    rl, paper = Path(args.rl), Path(args.paper)
    ev = pd.concat([pd.read_csv(f) for f in sorted(rl.glob("eval_*.csv"))], ignore_index=True)
    piv = ev.pivot_table(index="seed", columns=["defence", "attacker"], values="payer_net_loss")
    md = [f"fleets: {sorted(ev.seed.unique())}"]

    av = ev[(ev.defence == "AVERT") & (ev.attacker == RL)].set_index("seed")
    ok = (av.payer_net_loss <= av.bound_payer_loss)
    md.append(f"**H6a** RL payer loss <= Theorem 2 bound under AVERT in {ok.sum()}/{len(ok)} fleets "
              f"(median loss/bound = {np.median(av.payer_net_loss / av.bound_payer_loss):.3f})")
    p = {}
    d = piv[("DTA", RL)] - piv[("AVERT", RL)]
    p["H6b"] = wilcoxon(d, alternative="greater").pvalue
    d2 = piv[("DTA", RL)] - piv[("DTA", "K1 feedback")]
    p["H6c"] = wilcoxon(d2, alternative="greater").pvalue
    adj = holm(p)
    md.append(f"**H6b** RL loss DTA {piv[('DTA', RL)].mean():.0f} vs AVERT {piv[('AVERT', RL)].mean():.0f}; "
              f"p_Holm={adj['H6b']:.3g}")
    md.append(f"**H6c** RL loss under DTA {piv[('DTA', RL)].mean():.0f} vs K1 {piv[('DTA', 'K1 feedback')].mean():.0f}; "
              f"p_Holm={adj['H6c']:.3g}")
    worst_lib = ev[(ev.defence == "AVERT") & (ev.attacker != RL)].groupby("seed").payer_net_loss.max()
    worst_ext = ev[ev.defence == "AVERT"].groupby("seed").payer_net_loss.max()
    md.append(f"AVERT worst case over references {worst_lib.mean():.0f} -> with RL {worst_ext.mean():.0f}")
    means = ev.groupby(["defence", "attacker"])[["payer_net_loss", "extracted_value"]].mean().round(0)
    md.append("\n" + means.to_markdown())
    (rl / "analysis.md").write_text("\n\n".join(md), encoding="utf-8")
    print("\n\n".join(md))

    # table
    order = ["K0 oblivious", "K1 feedback", "K3 white-box", "K3* risk-aware", RL]
    lines = [r"\begin{table}[t]\centering\caption{Learning adversary (addendum; 10 fresh fleets, seeds 430--439): "
             r"mean payer loss per strategy}\label{tab:rl}",
             r"\footnotesize\begin{tabular}{lrr}\toprule Strategy & DTA & AVERT \\\midrule"]
    for a in order:
        name = r"\textbf{RL (Q-learning)}" if a == RL else a
        lines.append(f"{name} & {means.loc[('DTA', a), 'payer_net_loss'] / 1000:,.1f}k & "
                     f"{means.loc[('AVERT', a), 'payer_net_loss'] / 1000:,.1f}k \\\\")
    lines.append(r"\midrule Theorem~\ref{thm:bound} bound (AVERT, RL run) & -- & "
                 f"{av.bound_payer_loss.mean() / 1000:,.0f}k \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (paper / "tables" / "rl.tex").write_text("\n".join(lines), encoding="utf-8")

    # training curves
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for dname, col in (("DTA", "#eb6834"), ("AVERT", "#2a78d6")):
        tr = pd.read_csv(rl / f"train_{dname}.csv")
        y = tr.payer_net_loss.rolling(50, min_periods=10).mean() / 1000
        ax.plot(tr.episode, y, color=col, lw=1.5, label=f"against {dname}")
    ax.set_xlabel("training episode (development fleets)")
    ax.set_ylabel("payer loss per episode (k)")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(paper / "figs" / "rl_training.pdf")
    plt.close(fig)

    nums = dict(RlLossAVERT=f"{piv[('AVERT', RL)].mean() / 1000:,.1f}k", RlLossDTA=f"{piv[('DTA', RL)].mean() / 1000:,.0f}k",
                RlBoundOK=f"{ok.sum()}/{len(ok)}", RlWorstExt=f"{worst_ext.mean() / 1000:,.0f}k",
                RlWorstLib=f"{worst_lib.mean() / 1000:,.0f}k",
                RlHsixb=("supported" if adj["H6b"] < 0.05 else "not supported") + f", $p_{{\\mathrm{{Holm}}}}={adj['H6b']:.3f}$",
                RlHsixc=("supported" if adj["H6c"] < 0.05 else "not supported") + f", $p_{{\\mathrm{{Holm}}}}={adj['H6c']:.3f}$")
    with open(paper / "numbers_rl.tex", "w", encoding="utf-8") as f:
        for k, v in nums.items():
            f.write("\\newcommand{\\" + k + "}{" + v + "}\n")


if __name__ == "__main__":
    main()
