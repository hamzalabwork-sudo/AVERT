"""Confirmatory analysis (docs/confirmatory_plan.md): hypothesis tests, paper numbers, tables and figures.

    python scripts/analyse.py --paper paper

Reads results/confirmatory/*.csv, results/sgcc/*.csv and results/theory/theory_sim.csv; writes
results/confirmatory/analysis.md, paper/numbers.tex, paper/tables/*.tex and paper/figs/*.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ADAPTIVE = ["K1 feedback", "K3 white-box", "K3* risk-aware"]
AVERT, DTA = "AVERT", "DTA"
UNI_MAX = "uniform audits q=0.15"
UNI_PEN = "uniform audits q=0.100 + penalty/slash"
# fixed categorical order (validated palette, slots 1-4) + markers for grayscale print
STYLE = {AVERT: ("#2a78d6", "o"), DTA: ("#eb6834", "s"), "uniform": ("#1baf7a", "^"), "uniform+": ("#eda100", "D")}


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, x.size)).mean(1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * pvals[k]))
        out[k] = run
    return out


def worst(df, metric="payer_net_loss"):
    return df.groupby(["shift", "seed", "defence"])[metric].max().unstack("defence")


def paired(a, b):
    """One-sided Wilcoxon signed-rank test that a > b (paired by fleet), plus the mean difference CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    p = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    return d, p


def k(x, unit=""):
    """Money in thousands for the text."""
    return f"{x / 1000:,.0f}k" if abs(x) >= 1000 else f"{x:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/confirmatory")
    ap.add_argument("--sgcc", default="results/sgcc")
    ap.add_argument("--theory", default="results/theory/theory_sim.csv")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()
    paper = Path(args.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figs").mkdir(parents=True, exist_ok=True)
    md, nums = [], {}
    av_cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))["avert"]
    nums.update(Qfloor=av_cfg["q_floor"], Kappaq=av_cfg["kappa_q"], Penalty=av_cfg["penalty"],
                Slash=f"{av_cfg['slash'] / 5.0:.0f}", AggWindow=av_cfg["agg_window"], AggMin=av_cfg["agg_min"],
                AggZ=av_cfg["agg_z"], AlphaHold=av_cfg["alpha_hold"], SgccCustomers="12,568")

    df = pd.concat([pd.read_csv(f) for f in sorted(Path(args.main).glob("*.csv"))], ignore_index=True)
    ind = df[df["shift"] == "none"]
    sg = pd.concat([pd.read_csv(f) for f in sorted(Path(args.sgcc).glob("*.csv"))], ignore_index=True)
    sg["shift"] = "none"

    # ------------------------------------------------------------------ H1 (RQ1) and H5a
    pvals = {}

    def h1(d, name):
        dta = d[d.defence == DTA]
        ext = dta.pivot_table(index="seed", columns="attacker", values="extracted_value")
        adapt = ext[ADAPTIVE].max(1)
        diff, p = paired(adapt, ext["K0 oblivious"])
        ratio = adapt.mean() / ext["K0 oblivious"].mean()
        pvals[name] = p
        md.append(f"**{name}** DTA extracted, adaptive (max K1/K3/K3*) vs K0: {k(adapt.mean())} vs "
                  f"{k(ext['K0 oblivious'].mean())} (x{ratio:.1f}), Wilcoxon p={p:.2g}")
        return ext.mean(), ratio

    ext_sim, r_sim = h1(ind, "H1")
    ext_sg, r_sg = h1(sg, "H5a")
    nums.update(KoneGain=f"{ext_sim['K1 feedback'] / ext_sim['K0 oblivious']:.1f}",
                KthreeGain=f"{max(ext_sim['K3 white-box'], ext_sim['K3* risk-aware']) / ext_sim['K0 oblivious']:.1f}",
                SgccKthreeGain=f"{max(ext_sg['K3 white-box'], ext_sg['K3* risk-aware']) / ext_sg['K0 oblivious']:.1f}")

    # ------------------------------------------------------------------ H2 (RQ2) and H5b
    W = worst(ind)
    for name, comp in (("H2a", DTA), ("H2b", UNI_MAX), ("H2c", UNI_PEN)):
        diff, p = paired(W[comp], W[AVERT])
        m, lo, hi = boot_ci(diff)
        pvals[name] = p
        md.append(f"**{name}** worst-case payer loss {comp} - AVERT: {k(m)} [{k(lo)}, {k(hi)}], p={p:.2g}")
    Wsg = worst(sg)
    diff, p = paired(Wsg[DTA], Wsg[AVERT])
    pvals["H5b"] = p
    m, lo, hi = boot_ci(diff)
    md.append(f"**H5b** SGCC worst-case payer loss DTA - AVERT: {k(m)} [{k(lo)}, {k(hi)}], p={p:.2g}")
    adj = holm(pvals)
    md.append("Holm-adjusted p: " + ", ".join(f"{h}={adj[h]:.2g}" for h in sorted(adj)))
    nums.update(WorstDTA=k(W[DTA].mean()), WorstAVERT=k(W[AVERT].mean()), WorstUniform=k(W[UNI_MAX].mean()),
                SgccWorstDTA=k(Wsg[DTA].mean()), SgccWorstAVERT=k(Wsg[AVERT].mean()))

    # main table: worst case and honest-side costs per defence (simulator and SGCC)
    def table(d, Wd, label, caption):
        ops = d.groupby("defence")[["honest_verification_rate", "honest_legit_hold_rate", "honest_devices_flagged"]].mean()
        ext = worst(d, "extracted_value").mean()
        lines = [r"\begin{table}[t]\centering\caption{" + caption + r"}\label{" + label + r"}",
                 r"\setlength{\tabcolsep}{3pt}\footnotesize",
                 r"\begin{tabular}{lrrrrr}\toprule",
                 r"Defence & \multicolumn{2}{c}{Worst-case loss} & Honest & Legit. & Honest \\",
                 r" & payer & extracted & verified & held & flagged \\\midrule"]
        order = [c for c in Wd.columns if not c.startswith("AVERT -")] + [c for c in Wd.columns if c.startswith("AVERT -")]
        for c in order:
            m, lo, hi = boot_ci(Wd[c])
            name = c.replace("uniform audits", "Uniform").replace(" + penalty/slash", " +fine/quar.")
            lines.append(f"{name} & {m / 1000:,.0f}k {{\\scriptsize[{lo / 1000:,.0f}, {hi / 1000:,.0f}]}} & {ext[c] / 1000:,.0f}k & "
                         f"{100 * ops.loc[c, 'honest_verification_rate']:.1f}\\% & {100 * ops.loc[c, 'honest_legit_hold_rate']:.1f}\\% & "
                         f"{ops.loc[c, 'honest_devices_flagged']:.2f} \\\\")
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        return "\n".join(lines)

    (paper / "tables" / "main.tex").write_text(table(ind, W, "tab:main",
        "Worst case over the twelve-strategy library (20 confirmatory fleets, mean with 95\\% bootstrap CI) "
        "and honest-side costs"), encoding="utf-8")
    (paper / "tables" / "sgcc.tex").write_text(table(sg, Wsg, "tab:sgcc",
        "SGCC smart-meter case study (20 customer draws): worst case over the library and honest-side costs"),
        encoding="utf-8")

    # per-attacker table under DTA and AVERT
    piv = ind.pivot_table(index="attacker", columns="defence", values="payer_net_loss", aggfunc="mean", sort=False)
    lines = [r"\begin{table}[t]\centering\caption{Mean payer loss per strategy (simulator, 20 fleets)}\label{tab:attackers}",
             r"\footnotesize\begin{tabular}{lrrr}\toprule Strategy & DTA & Uniform $q{=}0.15$ & AVERT \\\midrule"]
    for a, row in piv.iterrows():
        lines.append(f"{a} & {row[DTA] / 1000:,.1f}k & {row[UNI_MAX] / 1000:,.1f}k & {row[AVERT] / 1000:,.1f}k \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (paper / "tables" / "attackers.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ H4 shifts + conformal
    av = df[df.defence == AVERT]
    hold = av.groupby(["shift", "seed"]).honest_legit_hold_rate.max()
    share = float((hold <= 0.05).mean())
    md.append(f"**H4** AVERT legitimate-hold rate <= 5% in {share:.0%} of {hold.size} fleets (all shifts; max over attackers)")
    rows = []
    for sh, d in df.groupby("shift"):
        Ws = worst(d)
        ops = d.groupby("defence")[["honest_legit_hold_rate", "honest_devices_flagged"]].mean()
        for c in Ws.columns:
            rows.append(dict(shift=sh, defence=c, worst=Ws[c].mean(), hold=ops.loc[c, "honest_legit_hold_rate"],
                             flagged=ops.loc[c, "honest_devices_flagged"]))
    sh = pd.DataFrame(rows)
    md.append("\n" + sh.round(3).to_markdown(index=False))
    lines = [r"\begin{table}[t]\centering\caption{Distribution shift (5 fleets each): worst-case payer loss, "
             r"legitimate-hold rate and falsely flagged honest devices}\label{tab:shift}",
             r"\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{llrrr}\toprule Scenario & Defence & Worst loss & Held & Flagged \\\midrule"]
    for _, r in sh[sh.defence.isin([DTA, "uniform audits q=0.075 + penalty/slash", AVERT])].iterrows():
        name = r.defence.replace("uniform audits", "Uniform").replace(" + penalty/slash", " +fine/quar.")
        lines.append(f"{r['shift']} & {name} & {r.worst / 1000:,.0f}k & {100 * r.hold:.1f}\\% & {r.flagged:.1f} \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (paper / "tables" / "shift.tex").write_text("\n".join(lines), encoding="utf-8")
    nums.update(HoldShare=f"{100 * share:.0f}")

    # ------------------------------------------------------------------ H3 theory
    th = pd.read_csv(args.theory)
    B = th[th.part == "B"]
    ok_loss = float((B.payer_net_loss <= B.bound_payer_loss).mean())
    ok_prof = float((B.attacker_profit <= B.bound_profit).mean())
    p = 0.95
    det = B[(B.variant == "no quarantine")]
    qstar = 1 / (p * (1 + det.lam))
    above = det[det.q >= qstar]
    ok_det = float((above.attacker_profit <= 0).mean()) if len(above) else float("nan")
    md.append(f"**H3** payer loss <= Theorem 2 bound in {ok_loss:.0%} of {len(B)} cells; profit <= bound in {ok_prof:.0%}; "
              f"profit <= 0 at q >= q* in {ok_det:.0%} of {len(above)} cells")
    C = th[th.part == "C"]
    big = C[C.attacker.isin(["salami +0.30 h", "salami +0.40 h"])]
    md.append(f"**H3c** honest devices flagged (max over runs): {C.honest_devices_flagged.max()}; salami +0.30/+0.40 h "
              f"detected share (min over fleets): {big.detected_share.min():.0%}")
    md.append("\n" + C.groupby("attacker")[["detected_share", "median_verified_before_flag", "prop3_m_median",
                                             "prop3_m_95", "honest_devices_flagged"]].mean().round(2).to_markdown())
    nums.update(BoundShare=f"{100 * min(ok_loss, ok_prof):.0f}")

    # figure: deterrence sweep
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for (lam, var), d in B[B.attacker == "constant 0.20"].groupby(["lam", "variant"]):
        g = d.groupby("q")
        y = (g.attacker_profit.mean() / g.attempted_overclaim.mean())
        ls = "-" if var == "no quarantine" else "--"
        ax.plot(y.index, y.values, ls, marker="o", ms=4, lw=1.5, label=f"$\\lambda$={lam:.0f}, {var}")
        if var == "no quarantine":
            qq = np.linspace(0, 0.7, 50)
            ax.plot(qq, 1 - qq * p * (1 + lam), ":", lw=1, color="0.5")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("audit probability $q$")
    ax.set_ylabel("profit / attempted over-claim")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(paper / "figs" / "deterrence.pdf")
    plt.close(fig)

    # figure: worst-case loss vs honest verification budget
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ops = ind.groupby("defence").honest_verification_rate.mean()
    for c in W.columns:
        if c.startswith("AVERT -"):
            continue
        key = AVERT if c == AVERT else DTA if c == DTA else "uniform+" if "penalty" in c else "uniform"
        col, mk = STYLE[key]
        m, lo, hi = boot_ci(W[c])
        ax.errorbar(100 * ops[c], m / 1000, yerr=[[(m - lo) / 1000], [(hi - m) / 1000]], fmt=mk, color=col, ms=6, capsize=2,
                    label={AVERT: "AVERT", DTA: "DTA", "uniform": "uniform audits", "uniform+": "uniform + fine/quarantine"}[key])
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=6, frameon=False)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("honest claims verified (%)")
    ax.set_ylabel("worst-case payer loss (k)")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(paper / "figs" / "frontier.pdf")
    plt.close(fig)

    # figure: RQ1 extracted value by strategy under DTA (simulator and SGCC)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    for ax, (d, title) in zip(axes, ((ind, "Simulated fleets"), (sg, "SGCC smart meters"))):
        e = d[d.defence == DTA].groupby("attacker", sort=False).extracted_value.mean() / 1000
        cols = ["#eb6834" if a in ADAPTIVE else "#86b6ef" for a in e.index]
        ax.barh(range(len(e)), e.values, color=cols, height=0.7)
        ax.set_yticks(range(len(e)), e.index, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("extracted value under DTA (k)")
        ax.set_title(title, fontsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(paper / "figs" / "rq1.pdf")
    plt.close(fig)

    with open(paper / "numbers.tex", "w", encoding="utf-8") as f:
        f.write("% results (scripts/analyse.py)\n")
        for key, v in nums.items():
            f.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")
    out = Path(args.main) / "analysis.md"
    out.write_text("\n\n".join(md), encoding="utf-8")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
