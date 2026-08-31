#!/usr/bin/env python3
"""Evaluate synthetic data quality against public reference statistics.

Evaluates every (model, condition) pair AND two baseline generators.

Metrics:
  1. Fidelity:   MARE (median income by education), TVD (education dist),
                 KS-test (age, log-income), gender split.
  2. Utility:    Mincer wage regression (R^2, beta_1), coefficients.
  3. Consistency: employed<->hours>0, income=0 iff unemployed,
                  monotone income by education.
  4. Diversity:  unique-record ratio, Shannon entropy of categorical fields.

Uncertainty: 1000-round bootstrap 95% CIs for MARE, TVD, R^2, beta_1.

Baselines (no LLM):
  uniform   - each field sampled independently from uniform distribution
  marginal  - education from Census, employment from BLS-by-edu,
              income from log-normal centred on BLS medians

Output: results/evaluation.json  (full metrics + CIs + baselines)
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EDU_ORDER = ["high_school", "some_college", "bachelors", "masters", "phd"]
EDU_YEARS = {"high_school": 12, "some_college": 14, "bachelors": 16,
             "masters": 18, "phd": 21}

N_BOOT = 1000
CI_LEVEL = 0.95


# ======================================================================
# Data loading
# ======================================================================

def load_all_data():
    """Load all JSON files, grouping by (model, condition).

    New naming:  {model}_{condition}_seed{N}.json
    Old naming:  {model}_seed{N}.json  (treated as 'priors' for compat)
    """
    groups = {}  # (model, condition) -> list[DataFrame]
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, fname)
        base = fname[:-5]  # strip .json
        # Try new naming first
        parts = base.rsplit("_seed", 1)
        if len(parts) == 2:
            prefix = parts[0]
            # Check if prefix ends with a known condition
            found = False
            for cond in ["priors", "structural", "minimal"]:
                if prefix.endswith("_" + cond):
                    model_part = prefix[: -(len(cond) + 1)]
                    condition = cond
                    found = True
                    break
            if not found:
                # Old naming: no condition suffix -> treat as 'priors'
                model_part = prefix
                condition = "priors"
            model = model_part.replace("_", "/", 1)  # restore first slash
        else:
            continue
        with open(path) as f:
            records = json.load(f)
        if not records:
            continue
        df = pd.DataFrame(records)
        df["_model"] = model
        df["_condition"] = condition
        groups.setdefault((model, condition), []).append(df)

    result = {}
    for key, dfs in groups.items():
        result[key] = pd.concat(dfs, ignore_index=True)
    return result


def load_reference():
    with open(os.path.join(RESULTS_DIR, "reference.json")) as f:
        return json.load(f)


# ======================================================================
# Baseline generators
# ======================================================================

def _generate_uniform(n, rng):
    """Uniform random baseline (no economic structure)."""
    records = []
    for _ in range(n):
        age = int(rng.integers(18, 86))
        gender = rng.choice(["male", "female"])
        education = rng.choice(EDU_ORDER)
        employed = bool(rng.random() < 0.5)
        if employed:
            income = int(rng.integers(15000, 250000))
            hours = int(rng.integers(20, 81))
        else:
            income = 0
            hours = 0
        records.append({
            "age": age, "gender": gender, "education": education,
            "income": income, "employed": employed, "hours_worked": hours,
            "marital_status": rng.choice(["single", "married", "divorced", "widowed"]),
            "children": int(rng.integers(0, 7)),
            "state": rng.choice([
                "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
                "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
                "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
                "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
                "WI","WY",
            ]),
        })
    return pd.DataFrame(records)


def _generate_marginal(n, ref, rng):
    """Marginal-sampling baseline (uses reference marginals, no correlations)."""
    edu_probs = np.array([ref["education_distribution"][e] for e in EDU_ORDER])
    edu_probs /= edu_probs.sum()

    unemp_rates = np.array([ref["unemployment_by_education"][e] for e in EDU_ORDER])
    medians = np.array([ref["median_income_by_education"][e] for e in EDU_ORDER])
    # Log-normal sigma ~ 0.5 (typical US income dispersion)
    LOG_SIGMA = 0.5

    records = []
    for _ in range(n):
        age = int(rng.integers(18, 86))
        gender = rng.choice(["male", "female"])
        edu_idx = rng.choice(len(EDU_ORDER), p=edu_probs)
        education = EDU_ORDER[edu_idx]
        employed = bool(rng.random() >= unemp_rates[edu_idx])
        if employed:
            median = medians[edu_idx]
            log_income = np.log(max(median, 1)) + rng.normal(0, LOG_SIGMA)
            income = max(int(np.exp(log_income)), 5000)
            hours = int(rng.integers(35, 51))  # full-time workers
        else:
            income = 0
            hours = 0
        records.append({
            "age": age, "gender": gender, "education": education,
            "income": income, "employed": employed, "hours_worked": hours,
            "marital_status": rng.choice(
                ["single", "married", "divorced", "widowed"],
                p=[0.30, 0.52, 0.13, 0.05],
            ),
            "children": min(int(rng.poisson(1.0)), 6),
            "state": rng.choice([
                "CA","TX","FL","NY","PA","IL","OH","GA","NC","MI",
                "NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI",
                "CO","MN","SC","AL","LA","KY","OR","OK","CT","UT",
                "IA","NV","AR","MS","KS","NM","NE","ID","WV","HI",
                "NH","ME","MT","RI","DE","SD","ND","AK","VT","WY",
            ]),
        })
    return pd.DataFrame(records)


# ======================================================================
# Core metrics
# ======================================================================

def total_variation_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def compute_mincer(df_emp):
    """Fit Mincer wage regression. Returns dict or None."""
    if len(df_emp) < 20:
        return None
    emp = df_emp.copy()
    emp["log_income"] = np.log(emp["income"].clip(lower=1))
    emp["educ_years"] = emp["education"].map(EDU_YEARS)
    emp["experience"] = (emp["age"] - emp["educ_years"] - 6).clip(lower=0)
    emp["exp_sq"] = emp["experience"] ** 2
    X = emp[["educ_years", "experience", "exp_sq"]].values
    y = emp["log_income"].values
    Xc = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        ss_res = resid @ resid
        ss_tot = (y - y.mean()) @ (y - y.mean())
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n_obs, k = Xc.shape
        sigma2 = ss_res / (n_obs - k) if n_obs > k else 1e-10
        cov = sigma2 * np.linalg.inv(Xc.T @ Xc)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        return {
            "n_employed": int(n_obs),
            "intercept": round(float(beta[0]), 4),
            "return_to_education": round(float(beta[1]), 4),
            "return_se": round(float(se[1]), 4),
            "experience": round(float(beta[2]), 4),
            "experience_sq": round(float(beta[3]), 6),
            "r_squared": round(float(r_sq), 4),
        }
    except Exception:
        return None


def compute_all_metrics(df, ref):
    """Compute the full metric suite for one DataFrame."""
    res = {}
    n = len(df)
    res["n_records"] = n

    # --- Fidelity: education distribution ---
    edu_counts = df["education"].value_counts(normalize=True).to_dict()
    ref_edu = ref["education_distribution"]
    res["education_distribution"] = {
        k: round(edu_counts.get(k, 0.0), 4) for k in EDU_ORDER
    }
    res["education_tvd"] = round(
        total_variation_distance(edu_counts, ref_edu), 4
    )

    # --- Fidelity: median income by education (employed only) ---
    employed = df[df["employed"]]
    med_income = {}
    for edu in EDU_ORDER:
        sub = employed[employed["education"] == edu]["income"]
        med_income[edu] = float(sub.median()) if len(sub) else None
    res["median_income_by_education"] = med_income
    ref_med = ref["median_income_by_education"]
    rel_err = {}
    for edu in EDU_ORDER:
        if med_income.get(edu) and ref_med.get(edu):
            rel_err[edu] = round(
                abs(med_income[edu] - ref_med[edu]) / ref_med[edu], 4
            )
    res["median_income_rel_error"] = rel_err
    res["median_income_mare"] = round(
        float(np.mean(list(rel_err.values()))), 4
    ) if rel_err else None

    # --- Fidelity: gender split ---
    g = df["gender"].value_counts(normalize=True).to_dict()
    res["gender_split"] = {k: round(g.get(k, 0.0), 4) for k in ("male", "female")}

    # --- Fidelity: unemployment by education ---
    unemp_by_edu = {}
    for edu in EDU_ORDER:
        sub = df[df["education"] == edu]
        if len(sub):
            unemp_by_edu[edu] = round(float((~sub["employed"]).mean()), 4)
    res["unemployment_by_education"] = unemp_by_edu
    ref_unemp = ref["unemployment_by_education"]
    unemp_rel = {}
    for edu in EDU_ORDER:
        if unemp_by_edu.get(edu) is not None and ref_unemp.get(edu):
            unemp_rel[edu] = round(
                abs(unemp_by_edu[edu] - ref_unemp[edu]) / ref_unemp[edu], 4
            )
    res["unemployment_rel_error"] = unemp_rel
    res["unemployment_mare"] = round(
        float(np.mean(list(unemp_rel.values()))), 4
    ) if unemp_rel else None

    # --- Fidelity: age distribution ---
    res["age_mean"] = round(float(df["age"].mean()), 2)
    res["age_std"] = round(float(df["age"].std()), 2)

    # KS test for age (vs uniform 18-85)
    ks_age = sp_stats.kstest(
        df["age"].values, "uniform", args=(18, 67)
    )
    res["age_ks_stat"] = round(float(ks_age.statistic), 4)
    res["age_ks_pvalue"] = round(float(ks_age.pvalue), 6)

    # KS test for log income of employed (vs normal)
    if len(employed) > 10:
        log_inc = np.log(employed["income"].clip(lower=1).values)
        ks_inc = sp_stats.kstest(log_inc, "norm",
                                 args=(log_inc.mean(), log_inc.std()))
        res["income_ks_stat"] = round(float(ks_inc.statistic), 4)
        res["income_ks_pvalue"] = round(float(ks_inc.pvalue), 6)

    # --- Downstream utility: Mincer regression ---
    res["mincer"] = compute_mincer(employed) or {"error": "insufficient data"}
    if "r_squared" in res["mincer"]:
        res["mincer_return_vs_canonical"] = round(
            abs(res["mincer"]["return_to_education"] - ref["mincer_return"])
            / ref["mincer_return"], 4
        )

    # --- Internal consistency ---
    unemp = df[~df["employed"]]
    res["employment_rate"] = round(float(df["employed"].mean()), 4)
    res["employment_rate_rel_error"] = round(
        abs(df["employed"].mean() - ref["employment_rate"])
        / ref["employment_rate"], 4
    )
    res["consistency_hours_positive_given_employed"] = (
        round(float((employed["hours_worked"] > 0).mean()), 4)
        if len(employed) else None
    )
    res["consistency_hours_zero_given_unemployed"] = (
        round(float((unemp["hours_worked"] == 0).mean()), 4)
        if len(unemp) else None
    )
    res["consistency_income_positive_given_employed"] = (
        round(float((employed["income"] > 0).mean()), 4)
        if len(employed) else None
    )
    res["consistency_income_zero_given_unemployed"] = (
        round(float((unemp["income"] == 0).mean()), 4)
        if len(unemp) else None
    )
    meds = [med_income.get(e) for e in EDU_ORDER]
    monotone = all(
        meds[i] is not None and meds[i + 1] is not None and meds[i] <= meds[i + 1]
        for i in range(len(meds) - 1)
    )
    res["consistency_income_monotone_in_education"] = bool(monotone)

    # --- Diversity ---
    res["unique_records_ratio"] = round(float(df.drop_duplicates().shape[0] / n), 4)
    def entropy(col):
        vc = df[col].value_counts(normalize=True)
        return float(-(vc * np.log(vc)).sum())
    res["entropy"] = {
        "gender": round(entropy("gender"), 3),
        "education": round(entropy("education"), 3),
        "marital_status": round(entropy("marital_status"), 3),
        "state": round(entropy("state"), 3),
    }

    return res


# ======================================================================
# Bootstrap confidence intervals
# ======================================================================

def _bootstrap_mare(df, ref, rng):
    """Single bootstrap resample -> MARE."""
    idx = rng.choice(len(df), size=len(df), replace=True)
    sub = df.iloc[idx]
    employed = sub[sub["employed"]]
    ref_med = ref["median_income_by_education"]
    errs = []
    for edu in EDU_ORDER:
        med = employed[employed["education"] == edu]["income"].median()
        if med and ref_med.get(edu):
            errs.append(abs(med - ref_med[edu]) / ref_med[edu])
    return float(np.mean(errs)) if errs else np.nan


def _bootstrap_tvd(df, ref, rng):
    """Single bootstrap -> TVD of education distribution."""
    idx = rng.choice(len(df), size=len(df), replace=True)
    sub = df.iloc[idx]
    edu_counts = sub["education"].value_counts(normalize=True).to_dict()
    return total_variation_distance(edu_counts, ref["education_distribution"])


def _bootstrap_mincer(df, ref, rng):
    """Single bootstrap -> (R^2, return_to_education)."""
    idx = rng.choice(len(df), size=len(df), replace=True)
    sub = df.iloc[idx]
    employed = sub[sub["employed"]]
    m = compute_mincer(employed)
    if m and "r_squared" in m:
        return m["r_squared"], m["return_to_education"]
    return np.nan, np.nan


def bootstrap_cis(df, ref, n_boot=N_BOOT):
    """Compute bootstrap 95% CIs for key metrics."""
    rng = np.random.default_rng(42)
    ci = {}
    # MARE
    maress = [_bootstrap_mare(df, ref, rng) for _ in range(n_boot)]
    maress = [x for x in maress if not np.isnan(x)]
    if maress:
        lo = (1 - CI_LEVEL) / 2 * 100
        hi = (1 + CI_LEVEL) / 2 * 100
        ci["mare"] = [round(float(np.percentile(maress, lo)), 4),
                       round(float(np.percentile(maress, 50)), 4),
                       round(float(np.percentile(maress, hi)), 4)]
    # TVD
    tvds = [_bootstrap_tvd(df, ref, rng) for _ in range(n_boot)]
    if tvds:
        lo = (1 - CI_LEVEL) / 2 * 100
        hi = (1 + CI_LEVEL) / 2 * 100
        ci["tvd"] = [round(float(np.percentile(tvds, lo)), 4),
                      round(float(np.percentile(tvds, 50)), 4),
                      round(float(np.percentile(tvds, hi)), 4)]
    # Mincer R^2 and beta_1
    mc = [_bootstrap_mincer(df, ref, rng) for _ in range(n_boot)]
    r2s = [x[0] for x in mc if not np.isnan(x[0])]
    b1s = [x[1] for x in mc if not np.isnan(x[1])]
    lo = (1 - CI_LEVEL) / 2 * 100
    hi = (1 + CI_LEVEL) / 2 * 100
    if r2s:
        ci["mincer_r2"] = [round(float(np.percentile(r2s, lo)), 4),
                            round(float(np.percentile(r2s, 50)), 4),
                            round(float(np.percentile(r2s, hi)), 4)]
    if b1s:
        ci["mincer_beta1"] = [round(float(np.percentile(b1s, lo)), 4),
                               round(float(np.percentile(b1s, 50)), 4),
                               round(float(np.percentile(b1s, hi)), 4)]
    return ci


# ======================================================================
# Main
# ======================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ref = load_reference()
    data = load_all_data()
    rng = np.random.default_rng(0)

    all_results = {"synthetic": {}, "baselines": {}, "bootstrap_ci": {}}

    # --- Evaluate all (model, condition) pairs ---
    for (model, condition), df in sorted(data.items()):
        key = f"{model} / {condition}"
        print(f"=== {key} (n={len(df)}) ===")
        metrics = compute_all_metrics(df, ref)
        all_results["synthetic"][key] = metrics
        # Bootstrap CIs
        ci = bootstrap_cis(df, ref)
        all_results["bootstrap_ci"][key] = ci
        print(json.dumps(metrics, indent=2))
        if ci:
            print(f"  CIs: {json.dumps(ci)}")
        print()

    # --- Baselines ---
    N_BASE = 1800
    baselines = {
        "uniform": _generate_uniform(N_BASE, rng),
        "marginal": _generate_marginal(N_BASE, ref, rng),
    }
    for name, bdf in baselines.items():
        print(f"=== BASELINE: {name} (n={len(bdf)}) ===")
        metrics = compute_all_metrics(bdf, ref)
        all_results["baselines"][name] = metrics
        ci = bootstrap_cis(bdf, ref)
        all_results["bootstrap_ci"][f"baseline_{name}"] = ci
        print(json.dumps(metrics, indent=2))
        print()

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Name':<40} {'MARE':>7} {'TVD':>7} {'Emp.R':>7} {'R^2':>7} {'beta1':>7}"
    print(header)
    print("-" * 80)
    for name, metrics in {**all_results["synthetic"],
                          **{f"BL/{k}": v for k, v in all_results["baselines"].items()}}.items():
        mare = metrics.get("median_income_mare", "n/a")
        tvd = metrics.get("education_tvd", "n/a")
        emp = metrics.get("employment_rate", "n/a")
        m = metrics.get("mincer", {})
        r2 = m.get("r_squared", "n/a")
        b1 = m.get("return_to_education", "n/a")
        mare_s = f"{mare:.3f}" if isinstance(mare, float) else str(mare)
        tvd_s = f"{tvd:.3f}" if isinstance(tvd, float) else str(tvd)
        emp_s = f"{emp:.3f}" if isinstance(emp, float) else str(emp)
        r2_s = f"{r2:.3f}" if isinstance(r2, float) else str(r2)
        b1_s = f"{b1:.3f}" if isinstance(b1, float) else str(b1)
        print(f"{name:<40} {mare_s:>7} {tvd_s:>7} {emp_s:>7} {r2_s:>7} {b1_s:>7}")

    # --- Save ---
    out_path = os.path.join(RESULTS_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
