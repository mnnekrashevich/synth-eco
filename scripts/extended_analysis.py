#!/usr/bin/env python3
"""Extended analysis: additional metrics for the paper.

Computes:
  1. Seed stability (within-model variance across seeds)
  2. Gender pay gap analysis
  3. Logistic regression for employment prediction
  4. Cross-model agreement (Jensen-Shannon divergence)
  5. Cost-effectiveness metrics
  6. Wasserstein distances for income distributions

Reads from data/ and results/reference.json.
Outputs: results/extended_analysis.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_all_data, load_reference, compute_mincer, EDU_ORDER, EDU_YEARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")


# ======================================================================
# Seed stability
# ======================================================================

def seed_stability(data):
    """Compare metrics across seeds for each (model, condition)."""
    # Group by (model, condition), then split by seed
    seed_groups = {}
    for (model, condition), df in data.items():
        # Extract seed from dataframe (we need to reload per-seed files)
        pass

    # Reload per-seed files
    seed_data = {}  # (model, condition, seed) -> DataFrame
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, fname)
        base = fname[:-5]
        parts = base.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        prefix = parts[0]
        seed = int(parts[1])
        found = False
        for cond in ["priors", "structural", "minimal"]:
            if prefix.endswith("_" + cond):
                model_part = prefix[: -(len(cond) + 1)]
                condition = cond
                found = True
                break
        if not found:
            model_part = prefix
            condition = "priors"
        model = model_part.replace("_", "/", 1)
        with open(path) as f:
            records = json.load(f)
        if records:
            seed_data[(model, condition, seed)] = pd.DataFrame(records)

    # Compute metrics per seed
    ref = load_reference()
    seed_metrics = {}  # (model, condition, seed) -> metrics dict
    for key, df in seed_data.items():
        employed = df[df["employed"]]
        # MARE
        ref_med = ref["median_income_by_education"]
        errs = []
        for edu in EDU_ORDER:
            med = employed[employed["education"] == edu]["income"].median()
            if med and ref_med.get(edu):
                errs.append(abs(med - ref_med[edu]) / ref_med[edu])
        mare = float(np.mean(errs)) if errs else None

        # TVD
        edu_counts = df["education"].value_counts(normalize=True).to_dict()
        tvd = 0.5 * sum(abs(edu_counts.get(k, 0) - ref["education_distribution"].get(k, 0))
                         for k in set(edu_counts) | set(ref["education_distribution"]))

        # Mincer
        m = compute_mincer(employed)
        r2 = m.get("r_squared") if m else None
        beta1 = m.get("return_to_education") if m else None

        # Employment rate
        emp_rate = float(df["employed"].mean())

        seed_metrics[key] = {
            "mare": mare, "tvd": tvd, "r2": r2, "beta1": beta1,
            "emp_rate": emp_rate, "n": len(df)
        }

    # Aggregate by (model, condition): mean and std across seeds
    stability = {}
    groups = {}
    for (model, condition, seed), metrics in seed_metrics.items():
        groups.setdefault((model, condition), []).append(metrics)

    for (model, condition), mlist in groups.items():
        if len(mlist) < 2:
            continue
        key = f"{model} / {condition}"
        agg = {}
        for metric in ["mare", "tvd", "r2", "beta1", "emp_rate"]:
            vals = [m[metric] for m in mlist if m[metric] is not None]
            if len(vals) >= 2:
                agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{metric}_std"] = round(float(np.std(vals, ddof=1)), 4)
                agg[f"{metric}_cv"] = round(float(np.std(vals, ddof=1) / np.mean(vals)), 4) if np.mean(vals) != 0 else None
        agg["n_seeds"] = len(mlist)
        stability[key] = agg

    return stability


# ======================================================================
# Gender pay gap
# ======================================================================

def gender_pay_gap(data):
    """Compute gender pay gap for each (model, condition) in priors condition."""
    results = {}
    for (model, condition), df in data.items():
        if condition != "priors":
            continue
        employed = df[df["employed"]]
        male_income = employed[employed["gender"] == "male"]["income"]
        female_income = employed[employed["gender"] == "female"]["income"]

        if len(male_income) < 10 or len(female_income) < 10:
            continue

        male_median = float(male_income.median())
        female_median = float(female_income.median())
        if male_median <= 0:
            continue
        gap = (male_median - female_median) / male_median

        # Also compute mean
        male_mean = float(male_income.mean())
        female_mean = float(female_income.mean())
        mean_gap = (male_mean - female_mean) / male_mean if male_mean > 0 else None

        key = f"{model} / {condition}"
        results[key] = {
            "male_median": round(male_median, 0),
            "female_median": round(female_median, 0),
            "median_gap_pct": round(gap * 100, 2) if gap else None,
            "male_mean": round(male_mean, 0),
            "female_mean": round(female_mean, 0),
            "mean_gap_pct": round(mean_gap * 100, 2) if mean_gap else None,
            "n_male": len(male_income),
            "n_female": len(female_income),
        }

    # Also compute for baselines
    from evaluate import _generate_marginal, _generate_uniform
    rng = np.random.default_rng(42)
    ref = load_reference()

    for name, gen_fn in [("uniform", lambda n: _generate_uniform(n, rng)),
                          ("marginal", lambda n: _generate_marginal(n, ref, rng))]:
        bdf = gen_fn(1800)
        employed = bdf[bdf["employed"]]
        male_income = employed[employed["gender"] == "male"]["income"]
        female_income = employed[employed["gender"] == "female"]["income"]
        male_median = float(male_income.median())
        female_median = float(female_income.median())
        gap = (male_median - female_median) / male_median if male_median > 0 else None
        results[f"BL/{name}"] = {
            "male_median": round(male_median, 0),
            "female_median": round(female_median, 0),
            "median_gap_pct": round(gap * 100, 2) if gap else None,
            "n_male": len(male_income),
            "n_female": len(female_income),
        }

    # US reference: ~16% gender pay gap (BLS 2024)
    results["reference"] = {"median_gap_pct": 16.0, "note": "BLS 2024 approximate"}

    return results


# ======================================================================
# Logistic regression for employment prediction
# ======================================================================

def employment_logistic(data):
    """Fit a logistic regression predicting employment from education and age.

    This tests whether the synthetic data captures the well-documented
    positive relationship between education and employment probability.
    """
    results = {}
    ref = load_reference()

    for (model, condition), df in data.items():
        if condition != "priors":
            continue

        # Encode education as ordinal
        edu_map = {e: i for i, e in enumerate(EDU_ORDER)}
        df = df.copy()
        df["edu_ord"] = df["education"].map(edu_map)
        df["age_c"] = df["age"] - 40  # center at 40
        df["age_c2"] = df["age_c"] ** 2

        X = df[["edu_ord", "age_c", "age_c2"]].values
        y = df["employed"].astype(int).values
        Xc = np.column_stack([np.ones(len(X)), X])

        try:
            # Newton-Raphson for logistic regression with L2 regularization
            lam = 0.01  # regularization to handle near-singularity
            beta = np.zeros(Xc.shape[1])
            for _ in range(100):
                p = 1 / (1 + np.exp(-Xc @ beta))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                W = np.diag(p * (1 - p))
                grad = Xc.T @ (y - p) - lam * beta
                H = -(Xc.T @ W @ Xc + lam * np.eye(Xc.shape[1]))
                try:
                    delta = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    break
                beta += delta
                if np.max(np.abs(delta)) < 1e-8:
                    break

            p = 1 / (1 + np.exp(-Xc @ beta))
            ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            ll_null = np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean()))
            pseudo_r2 = 1 - ll / ll_null if ll_null != 0 else None

            # SE from Hessian (with regularization)
            W = np.diag(p * (1 - p))
            H = -(Xc.T @ W @ Xc + lam * np.eye(Xc.shape[1]))
            try:
                cov = np.linalg.inv(H)
                se = np.sqrt(np.maximum(np.diag(cov), 0))
            except np.linalg.LinAlgError:
                se = np.full(len(beta), np.nan)

            # Odds ratio for education
            edu_or = np.exp(beta[1])
            edu_or_ci_lo = np.exp(beta[1] - 1.96 * se[1])
            edu_or_ci_hi = np.exp(beta[1] + 1.96 * se[1])

            key = f"{model} / {condition}"
            results[key] = {
                "n": len(df),
                "employment_rate": round(float(y.mean()), 4),
                "edu_coef": round(float(beta[1]), 4),
                "edu_se": round(float(se[1]), 4),
                "edu_odds_ratio": round(float(edu_or), 4),
                "edu_or_ci": [round(float(edu_or_ci_lo), 4), round(float(edu_or_ci_hi), 4)],
                "pseudo_r2": round(float(pseudo_r2), 4) if pseudo_r2 else None,
                "converged": True,
            }
        except Exception as e:
            results[f"{model} / {condition}"] = {"error": str(e), "converged": False}

    # Reference: BLS data shows strong education-employment relationship
    # Approximate: odds ratio of ~1.3-1.5 per education level
    results["reference"] = {
        "note": "BLS 2024: education strongly predicts employment; OR ~1.3-1.5 per level",
        "employment_rate": 0.63,
    }

    return results


# ======================================================================
# Jensen-Shannon divergence between models
# ======================================================================

def cross_model_jsd(data):
    """Compute JSD between each pair of models on key distributions."""
    from scipy.spatial.distance import jensenshannon

    # Collect distributions per (model, condition=priors)
    distributions = {}
    for (model, condition), df in data.items():
        if condition != "priors":
            continue
        # Education distribution
        edu = df["education"].value_counts(normalize=True)
        edu_vec = np.array([edu.get(e, 0) for e in EDU_ORDER])
        edu_vec = edu_vec / edu_vec.sum()

        # Income distribution (binned)
        employed = df[df["employed"] & (df["income"] > 0)]
        bins = np.logspace(np.log10(5000), np.log10(300000), 20)
        hist, _ = np.histogram(employed["income"], bins=bins, density=True)
        hist = hist / hist.sum() if hist.sum() > 0 else np.ones(19) / 19

        distributions[model] = {"education": edu_vec, "income": hist}

    # Compute pairwise JSD
    models = sorted(distributions.keys())
    jsd_edu = {}
    jsd_income = {}
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            pair = f"{m1} vs {m2}"
            jsd_edu[pair] = round(float(jensenshannon(
                distributions[m1]["education"],
                distributions[m2]["education"]
            ) ** 2), 4)  # JSD (squared, since jensenshannon returns sqrt(JSD))
            jsd_income[pair] = round(float(jensenshannon(
                distributions[m1]["income"],
                distributions[m2]["income"]
            ) ** 2), 4)

    return {"education_jsd": jsd_edu, "income_jsd": jsd_income}


# ======================================================================
# Wasserstein distance
# ======================================================================

def wasserstein_analysis(data):
    """Compute Wasserstein distance for income distributions."""
    ref = load_reference()
    results = {}

    for (model, condition), df in data.items():
        if condition != "priors":
            continue
        employed = df[df["employed"] & (df["income"] > 0)]
        if len(employed) < 20:
            continue

        # Generate reference income sample using marginal baseline parameters
        ref_medians = ref["median_income_by_education"]
        ref_edu = ref["education_distribution"]
        ref_unemp = ref["unemployment_by_education"]

        ref_samples = []
        for edu in EDU_ORDER:
            n_ref = int(ref_edu[edu] * 10000)
            log_med = np.log(ref_medians[edu])
            samples = np.random.lognormal(log_med, 0.5, n_ref)
            ref_samples.extend(samples)
        ref_samples = np.array(ref_samples)

        # Wasserstein-1 distance
        w1 = float(sp_stats.wasserstein_distance(employed["income"].values, ref_samples))
        # Normalized by median income
        w1_norm = w1 / np.median(ref_samples)

        key = f"{model} / {condition}"
        results[key] = {
            "wasserstein_1": round(w1, 2),
            "wasserstein_1_normalized": round(w1_norm, 4),
            "synthetic_median": round(float(employed["income"].median()), 0),
            "reference_median_approx": round(float(np.median(ref_samples)), 0),
        }

    return results


# ======================================================================
# Summary: practical recipe metrics
# ======================================================================

def practical_recipe_metrics(data, ref):
    """Compute metrics showing what a hybrid approach could achieve."""
    results = {}

    for (model, condition), df in data.items():
        if condition != "priors":
            continue

        employed = df[df["employed"]]

        # What if we reweight to match education marginals?
        # Compute importance weights
        edu_counts = df["education"].value_counts(normalize=True)
        weights = {}
        for edu in EDU_ORDER:
            synth_share = edu_counts.get(edu, 0.01)
            ref_share = ref["education_distribution"].get(edu, 0.01)
            weights[edu] = ref_share / synth_share if synth_share > 0 else 1.0

        # Apply weights to income
        weighted_incomes = []
        for _, row in employed.iterrows():
            w = weights.get(row["education"], 1.0)
            weighted_incomes.append(row["income"] * w)

        if weighted_incomes:
            # Reweighted median income by education
            reweighted_medians = {}
            for edu in EDU_ORDER:
                edu_mask = employed["education"] == edu
                if edu_mask.sum() > 0:
                    incomes = employed[edu_mask]["income"].values
                    wts = np.array([weights.get(edu, 1.0)] * len(incomes))
                    # Weighted median approximation
                    sorted_idx = np.argsort(incomes)
                    sorted_incomes = incomes[sorted_idx]
                    sorted_wts = wts[sorted_idx]
                    cum_wts = np.cumsum(sorted_wts) / sorted_wts.sum()
                    median_idx = np.searchsorted(cum_wts, 0.5)
                    reweighted_medians[edu] = float(sorted_incomes[min(median_idx, len(sorted_incomes)-1)])

            ref_med = ref["median_income_by_education"]
            re_mare = np.mean([abs(reweighted_medians.get(e, 0) - ref_med.get(e, 0)) / ref_med.get(e, 1)
                               for e in EDU_ORDER if e in reweighted_medians])
            results[f"{model} / {condition}"] = {
                "original_mare": None,  # filled from main eval
                "reweighted_mare": round(float(re_mare), 4),
            }

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    warnings.filterwarnings("ignore")
    ref = load_reference()
    data = load_all_data()

    print("Running extended analysis...\n")

    # 1. Seed stability
    print("1. Seed stability...")
    stability = seed_stability(data)

    # 2. Gender pay gap
    print("2. Gender pay gap...")
    gender = gender_pay_gap(data)

    # 3. Employment logistic regression
    print("3. Employment logistic regression...")
    logistic = employment_logistic(data)

    # 4. Cross-model JSD
    print("4. Cross-model JSD...")
    jsd = cross_model_jsd(data)

    # 5. Wasserstein distance
    print("5. Wasserstein distance...")
    wasserstein = wasserstein_analysis(data)

    # 6. Practical recipe
    print("6. Practical recipe metrics...")
    recipe = practical_recipe_metrics(data, ref)

    # Assemble
    results = {
        "seed_stability": stability,
        "gender_pay_gap": gender,
        "employment_logistic": logistic,
        "cross_model_jsd": jsd,
        "wasserstein": wasserstein,
        "practical_recipe": recipe,
    }

    out_path = os.path.join(RESULTS_DIR, "extended_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExtended analysis written to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SEED STABILITY (CV across seeds)")
    print("=" * 60)
    for k, v in stability.items():
        if "mare_cv" in v:
            print(f"  {k}: MARE CV={v['mare_cv']:.3f}, R² CV={v.get('r2_cv', 'n/a')}")

    print("\n" + "=" * 60)
    print("GENDER PAY GAP")
    print("=" * 60)
    for k, v in gender.items():
        g = v.get("median_gap_pct")
        if g is not None:
            print(f"  {k}: {g:.1f}% (ref: ~16%)")
        else:
            print(f"  {k}: n/a")

    print("\n" + "=" * 60)
    print("EMPLOYMENT LOGISTIC (edu odds ratio)")
    print("=" * 60)
    for k, v in logistic.items():
        if "edu_odds_ratio" in v:
            ci = v.get("edu_or_ci", [None, None])
            pr2 = v.get("pseudo_r2")
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a"
            pr2_str = f"{pr2:.4f}" if pr2 is not None else "n/a"
            print(f"  {k}: OR={v['edu_odds_ratio']:.3f} {ci_str}, Pseudo-R²={pr2_str}")
        elif "note" in v:
            print(f"  {k}: {v['note']}")

    print("\n" + "=" * 60)
    print("WASSERSTEIN-1 (income, normalized)")
    print("=" * 60)
    for k, v in wasserstein.items():
        print(f"  {k}: W1_norm={v['wasserstein_1_normalized']:.4f}")


if __name__ == "__main__":
    main()
