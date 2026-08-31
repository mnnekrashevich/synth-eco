#!/usr/bin/env python3
"""Generate publication-quality figures from evaluation results.

Reads results/evaluation.json and produces figures in paper/figures/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EDU_ORDER = ["high_school", "some_college", "bachelors", "masters", "phd"]
EDU_LABELS = ["HS", "Some\ncoll.", "Bachelors", "Masters", "PhD"]

# Consistent colour palette
COLORS = {
    "priors":     "#2166AC",
    "structural": "#67A9CF",
    "minimal":    "#D1E5F0",
    "marginal":   "#F4A582",
    "uniform":    "#FDDBC7",
    "reference":  "#B2182B",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load():
    with open(os.path.join(RESULTS_DIR, "evaluation.json")) as f:
        return json.load(f)


def _label(key):
    """Turn 'model / condition' into a short label."""
    parts = key.split(" / ")
    cond = parts[-1] if len(parts) > 1 else key
    return cond


def _color(key):
    cond = _label(key)
    return COLORS.get(cond, "#999999")


def fig_mare_comparison(results):
    """Bar chart: MARE across all conditions + baselines, with bootstrap CIs."""
    entries = []
    # Synthetic
    for key, metrics in results.get("synthetic", {}).items():
        mare = metrics.get("median_income_mare")
        if mare is None:
            continue
        ci = results.get("bootstrap_ci", {}).get(key, {})
        ci_mare = ci.get("mare", [mare, mare, mare])
        entries.append((_label(key), mare, ci_mare, _color(key)))
    # Baselines
    for name, metrics in results.get("baselines", {}).items():
        mare = metrics.get("median_income_mare")
        if mare is None:
            continue
        ci = results.get("bootstrap_ci", {}).get(f"baseline_{name}", {})
        ci_mare = ci.get("mare", [mare, mare, mare])
        entries.append((f"BL/{name}", mare, ci_mare, COLORS.get(name, "#999")))

    if not entries:
        print("  [SKIP] fig_mare: no data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    # Error bars: [low, median, high]
    yerr_lo = [max(e[1] - e[2][0], 0) for e in entries]
    yerr_hi = [max(e[2][2] - e[1], 0) for e in entries]
    colors = [e[3] for e in entries]
    x = np.arange(len(entries))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.65)
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MARE (median income by education)")
    ax.set_title("Income fidelity: MARE across prompt conditions & baselines")
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="10% threshold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mare_comparison.png"))
    plt.close(fig)
    print("  [OK] mare_comparison.png")


def fig_education_distribution(results):
    """Grouped bar chart: education distribution per condition vs reference."""
    # Pick one representative synthetic key (priors for the main model)
    synth_keys = [k for k in results.get("synthetic", {})
                  if "priors" in k]
    if not synth_keys:
        synth_keys = list(results.get("synthetic", {}).keys())
    if not synth_keys:
        print("  [SKIP] fig_education: no data")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_edu = len(EDU_ORDER)
    n_groups = len(synth_keys) + 2  # + baselines + reference
    width = 0.7 / max(n_groups, 1)
    x = np.arange(n_edu)

    offset = 0
    all_bars = []

    # Baselines
    for bname in ["marginal", "uniform"]:
        bmetrics = results.get("baselines", {}).get(bname, {})
        edist = bmetrics.get("education_distribution", {})
        if not edist:
            continue
        vals = [edist.get(e, 0) for e in EDU_ORDER]
        bars = ax.bar(x + offset * width, vals, width,
                      color=COLORS.get(bname, "#999"), label=f"BL/{bname}")
        all_bars.append(bars)
        offset += 1

    # Synthetic conditions
    for key in synth_keys:
        metrics = results["synthetic"][key]
        edist = metrics.get("education_distribution", {})
        vals = [edist.get(e, 0) for e in EDU_ORDER]
        bars = ax.bar(x + offset * width, vals, width,
                      color=_color(key), label=_label(key))
        all_bars.append(bars)
        offset += 1

    # Reference
    with open(os.path.join(RESULTS_DIR, "reference.json")) as f:
        ref = json.load(f)
    ref_vals = [ref["education_distribution"][e] for e in EDU_ORDER]
    ax.bar(x + offset * width, ref_vals, width,
           color=COLORS["reference"], label="Reference (Census)", hatch="//")

    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(EDU_LABELS)
    ax.set_ylabel("Share of population")
    ax.set_title("Educational attainment: synthetic vs reference")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "education_distribution.png"))
    plt.close(fig)
    print("  [OK] education_distribution.png")


def fig_mincer_comparison(results):
    """Grouped bar chart: Mincer R^2 and beta_1 across conditions + baselines."""
    entries = []
    for key, metrics in results.get("synthetic", {}).items():
        m = metrics.get("mincer", {})
        if "r_squared" not in m:
            continue
        ci = results.get("bootstrap_ci", {}).get(key, {})
        entries.append({
            "label": _label(key),
            "r2": m["r_squared"],
            "r2_ci": ci.get("mincer_r2", [m["r_squared"]]*3),
            "beta1": m["return_to_education"],
            "beta1_ci": ci.get("mincer_beta1", [m["return_to_education"]]*3),
            "color": _color(key),
        })
    for bname, metrics in results.get("baselines", {}).items():
        m = metrics.get("mincer", {})
        if "r_squared" not in m:
            continue
        ci = results.get("bootstrap_ci", {}).get(f"baseline_{bname}", {})
        entries.append({
            "label": f"BL/{bname}",
            "r2": m["r_squared"],
            "r2_ci": ci.get("mincer_r2", [m["r_squared"]]*3),
            "beta1": m["return_to_education"],
            "beta1_ci": ci.get("mincer_beta1", [m["return_to_education"]]*3),
            "color": COLORS.get(bname, "#999"),
        })

    if not entries:
        print("  [SKIP] fig_mincer: no data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(entries))
    labels = [e["label"] for e in entries]
    colors = [e["color"] for e in entries]

    # R^2
    r2s = [e["r2"] for e in entries]
    r2_lo = [max(e["r2"] - e["r2_ci"][0], 0) for e in entries]
    r2_hi = [max(e["r2_ci"][2] - e["r2"], 0) for e in entries]
    ax1.bar(x, r2s, color=colors, width=0.6, edgecolor="white")
    ax1.errorbar(x, r2s, yerr=[r2_lo, r2_hi], fmt="none", ecolor="black",
                 capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("$R^2$")
    ax1.set_title("Mincer regression $R^2$")
    ax1.set_ylim(0, 1.05)

    # beta_1
    b1s = [e["beta1"] for e in entries]
    b1_lo = [max(e["beta1"] - e["beta1_ci"][0], 0) for e in entries]
    b1_hi = [max(e["beta1_ci"][2] - e["beta1"], 0) for e in entries]
    ax2.bar(x, b1s, color=colors, width=0.6, edgecolor="white")
    ax2.errorbar(x, b1s, yerr=[b1_lo, b1_hi], fmt="none", ecolor="black",
                 capsize=3)
    ax2.axhline(0.09, color=COLORS["reference"], linestyle="--", linewidth=1.2,
                label="Canonical ~9%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Return to education ($\\hat{\\beta}_1$)")
    ax2.set_title("Mincer return to schooling")
    ax2.legend(loc="upper right")

    fig.suptitle("Downstream econometric utility across conditions", y=1.02,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mincer_comparison.png"))
    plt.close(fig)
    print("  [OK] mincer_comparison.png")


def fig_consistency_heatmap(results):
    """Heatmap: consistency metrics across conditions."""
    cons_keys = [
        ("consistency_hours_positive_given_employed",
         "hours>0|employed"),
        ("consistency_hours_zero_given_unemployed",
         "hours=0|unemployed"),
        ("consistency_income_positive_given_employed",
         "income>0|employed"),
        ("consistency_income_zero_given_unemployed",
         "income=0|unemployed"),
    ]
    # Collect all conditions
    all_keys = list(results.get("synthetic", {}).keys())
    for bname in results.get("baselines", {}):
        all_keys.append(f"baseline_{bname}")

    if not all_keys:
        print("  [SKIP] fig_consistency: no data")
        return

    data = np.zeros((len(cons_keys), len(all_keys)))
    labels_x = []
    for j, key in enumerate(all_keys):
        if key.startswith("baseline_"):
            metrics = results["baselines"].get(key.replace("baseline_", ""), {})
            labels_x.append(f"BL/{key.split('_',1)[1]}")
        else:
            metrics = results["synthetic"].get(key, {})
            labels_x.append(_label(key))
        for i, (ck, _) in enumerate(cons_keys):
            data[i, j] = metrics.get(ck, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(all_keys) * 1.2), 3.5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right")
    ax.set_yticks(range(len(cons_keys)))
    ax.set_yticklabels([label for _, label in cons_keys])
    # Annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=8, color="black" if val > 0.5 else "white")
    plt.colorbar(im, ax=ax, label="Agreement rate", shrink=0.8)
    ax.set_title("Internal consistency across conditions")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "consistency_heatmap.png"))
    plt.close(fig)
    print("  [OK] consistency_heatmap.png")


def fig_income_by_education(results):
    """Bar chart: synthetic vs reference median income by education.

    Shows one representative 'priors' condition and the marginal baseline.
    """
    # Find priors key
    priors_keys = [k for k in results.get("synthetic", {}) if "priors" in k]
    if not priors_keys:
        print("  [SKIP] fig_income_by_edu: no priors data")
        return
    key = priors_keys[0]
    metrics = results["synthetic"][key]
    syn_med = metrics.get("median_income_by_education", {})

    with open(os.path.join(RESULTS_DIR, "reference.json")) as f:
        ref = json.load(f)
    ref_med = ref["median_income_by_education"]

    marginal = results.get("baselines", {}).get("marginal", {})
    marg_med = marginal.get("median_income_by_education", {})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(EDU_ORDER))
    w = 0.25

    vals_syn = [syn_med.get(e, 0) for e in EDU_ORDER]
    vals_ref = [ref_med.get(e, 0) for e in EDU_ORDER]
    vals_marg = [marg_med.get(e, 0) for e in EDU_ORDER]

    ax.bar(x - w, vals_syn, w, label=f"Synthetic ({_label(key)})",
           color=COLORS["priors"])
    ax.bar(x, vals_ref, w, label="Reference (BLS 2024)",
           color=COLORS["reference"])
    ax.bar(x + w, vals_marg, w, label="Marginal baseline",
           color=COLORS["marginal"])

    ax.set_xticks(x)
    ax.set_xticklabels(EDU_LABELS)
    ax.set_ylabel("Median annual income (USD)")
    ax.set_title("Median income by education: synthetic vs reference vs baseline")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "median_income.png"))
    plt.close(fig)
    print("  [OK] median_income.png")


def fig_age_income(results):
    """Age-income lifecycle: priors condition vs marginal baseline."""
    # We need the raw data, not just metrics. Load from data/ for the priors key.
    import glob as globmod
    import pandas as pd

    all_dfs = []
    for fname in sorted(globmod.glob(os.path.join(ROOT, "data", "*.json"))):
        base = os.path.basename(fname).replace(".json", "")
        # Accept files with "priors" in name OR old-format files (no condition suffix)
        if "priors" not in base and "_seed" in base:
            # Check if it's an old-format file (no condition between model and seed)
            parts = base.rsplit("_seed", 1)
            if len(parts) == 2 and parts[0] not in ("minimal", "structural", "priors"):
                # Old naming: treat as priors
                pass
            else:
                continue
        with open(fname) as f:
            all_dfs.append(pd.DataFrame(json.load(f)))
    if not all_dfs:
        print("  [SKIP] fig_age_income: no raw priors data")
        return
    df = pd.concat(all_dfs, ignore_index=True)

    employed = df[df["employed"] & (df["income"] > 0)]
    bins = np.arange(18, 86, 5)
    idx = np.digitize(employed["age"], bins)
    grp = employed.groupby(idx)["income"]
    means = grp.mean()
    stderrs = grp.std() / np.sqrt(grp.count())
    centers = bins[:-1] + 2.5

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers[:len(means)], means.values, marker="o",
            color=COLORS["priors"], label="Synthetic (priors)")
    ax.fill_between(centers[:len(means)],
                    (means.values - 1.96 * stderrs.values[:len(means)]),
                    (means.values + 1.96 * stderrs.values[:len(means)]),
                    alpha=0.2, color=COLORS["priors"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean income (USD)")
    ax.set_title("Age-income lifecycle profile (employed)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "age_income.png"))
    plt.close(fig)
    print("  [OK] age_income.png")


def fig_unemployment_by_education(results):
    """Bar chart: unemployment by education - synthetic vs reference.

    This is a KEY failure mode that the paper must address.
    """
    priors_keys = [k for k in results.get("synthetic", {}) if "priors" in k]
    if not priors_keys:
        print("  [SKIP] fig_unemp: no priors data")
        return
    key = priors_keys[0]
    metrics = results["synthetic"][key]
    unemp_syn = metrics.get("unemployment_by_education", {})

    with open(os.path.join(RESULTS_DIR, "reference.json")) as f:
        ref = json.load(f)
    unemp_ref = ref["unemployment_by_education"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(EDU_ORDER))
    w = 0.35
    vals_syn = [unemp_syn.get(e, 0) * 100 for e in EDU_ORDER]
    vals_ref = [unemp_ref.get(e, 0) * 100 for e in EDU_ORDER]

    ax.bar(x - w / 2, vals_syn, w, label="Synthetic", color=COLORS["priors"])
    ax.bar(x + w / 2, vals_ref, w, label="Reference (BLS)", color=COLORS["reference"])
    ax.set_xticks(x)
    ax.set_xticklabels(EDU_LABELS)
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_title("Unemployment by education: synthetic vs BLS reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "unemployment_by_education.png"))
    plt.close(fig)
    print("  [OK] unemployment_by_education.png")


def fig_radar(results):
    """Radar chart: multi-dimensional comparison of models (priors condition).

    Dimensions (higher = better, normalized to 0-1):
      - Income fidelity (1 - MARE)
      - Education fidelity (1 - TVD)
      - Econometric utility (R^2)
      - Internal consistency (mean of 4 consistency metrics)
      - Diversity (unique ratio)
    """
    import numpy as np

    priors_keys = [k for k in results.get("synthetic", {}) if "priors" in k]
    if not priors_keys:
        print("  [SKIP] fig_radar: no priors data")
        return

    # Short model names
    def short(key):
        model = key.split(" / ")[0]
        mapping = {
            "inclusionai/ling-3.0-flash": "Ling",
            "~deepseek/deepseek-v4-flash-latest": "DeepSeek",
            "qwen/qwen3.7-flash": "Qwen",
            "upstage/solar-pro4": "Solar",
        }
        return mapping.get(model, model)

    dims = ["Income\nfidelity", "Education\nfidelity", "Econometric\nutility",
            "Consistency", "Diversity"]
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for key in priors_keys:
        m = results["synthetic"][key]
        mare = m.get("median_income_mare", 1.0) or 1.0
        tvd = m.get("education_tvd", 1.0) or 1.0
        r2 = m.get("mincer", {}).get("r_squared", 0.0)
        cons = np.mean([
            m.get("consistency_hours_positive_given_employed", 0) or 0,
            m.get("consistency_hours_zero_given_unemployed", 0) or 0,
            m.get("consistency_income_positive_given_employed", 0) or 0,
            m.get("consistency_income_zero_given_unemployed", 0) or 0,
        ])
        div = m.get("unique_records_ratio", 0.0) or 0.0

        values = [1 - min(mare, 1.0), 1 - min(tvd, 1.0), r2, cons, div]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=short(key))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-dimensional model comparison (priors condition)",
                 pad=20, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "radar_comparison.png"))
    plt.close(fig)
    print("  [OK] radar_comparison.png")


def fig_seed_stability():
    """Bar chart: coefficient of variation of MARE and R^2 across seeds."""
    ext_path = os.path.join(RESULTS_DIR, "extended_analysis.json")
    if not os.path.exists(ext_path):
        print("  [SKIP] fig_seed_stability: no extended_analysis.json")
        return
    with open(ext_path) as f:
        ext = json.load(f)
    stability = ext.get("seed_stability", {})
    if not stability:
        print("  [SKIP] fig_seed_stability: no seed stability data")
        return

    def short(key):
        model = key.split(" / ")[0]
        mapping = {
            "inclusionai/ling-3.0-flash": "Ling",
            "~deepseek/deepseek-v4-flash-latest": "DeepSeek",
            "qwen/qwen3.7-flash": "Qwen",
            "upstage/solar-pro4": "Solar",
        }
        cond = key.split(" / ")[1]
        return f"{mapping.get(model, model)}/{cond}"

    labels = []
    mare_cvs = []
    r2_cvs = []
    for key, v in stability.items():
        if "mare_cv" in v and "r2_cv" in v:
            labels.append(short(key))
            mare_cvs.append(v["mare_cv"] * 100)
            r2_cvs.append(v["r2_cv"] * 100)

    if not labels:
        print("  [SKIP] fig_seed_stability: no CV data")
        return

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, mare_cvs, w, label="MARE CV", color=COLORS["priors"])
    ax.bar(x + w / 2, r2_cvs, w, label="$R^2$ CV", color=COLORS["marginal"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Coefficient of variation (%)")
    ax.set_title("Seed stability: coefficient of variation across two seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "seed_stability.png"))
    plt.close(fig)
    print("  [OK] seed_stability.png")


def fig_gender_gap():
    """Bar chart: gender pay gap across models vs reference."""
    ext_path = os.path.join(RESULTS_DIR, "extended_analysis.json")
    if not os.path.exists(ext_path):
        print("  [SKIP] fig_gender_gap: no extended_analysis.json")
        return
    with open(ext_path) as f:
        ext = json.load(f)
    gender = ext.get("gender_pay_gap", {})
    if not gender:
        print("  [SKIP] fig_gender_gap: no gender data")
        return

    def short(key):
        model = key.split(" / ")[0]
        mapping = {
            "inclusionai/ling-3.0-flash": "Ling",
            "~deepseek/deepseek-v4-flash-latest": "DeepSeek",
            "qwen/qwen3.7-flash": "Qwen",
            "upstage/solar-pro4": "Solar",
        }
        return mapping.get(model, model)

    labels = []
    gaps = []
    for key, v in gender.items():
        if key == "reference":
            continue
        g = v.get("median_gap_pct")
        if g is not None:
            labels.append(short(key))
            gaps.append(g)

    if not labels:
        print("  [SKIP] fig_gender_gap: no gap data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x, gaps, color=COLORS["priors"], width=0.6)
    ax.axhline(16.0, color=COLORS["reference"], linestyle="--", linewidth=1.2,
               label="Reference ~16% (BLS 2024)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gender pay gap (%)")
    ax.set_title("Gender pay gap: synthetic vs BLS reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gender_gap.png"))
    plt.close(fig)
    print("  [OK] gender_gap.png")


def main():
    results = _load()
    print("Generating figures...\n")
    fig_mare_comparison(results)
    fig_education_distribution(results)
    fig_mincer_comparison(results)
    fig_consistency_heatmap(results)
    fig_income_by_education(results)
    fig_age_income(results)
    fig_unemployment_by_education(results)
    fig_radar(results)
    fig_seed_stability()
    fig_gender_gap()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
