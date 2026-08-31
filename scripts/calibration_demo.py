#!/usr/bin/env python3
"""Calibration demonstration: does the proposed 4-step recipe work?

The paper's central recommendation is a hybrid pipeline:
  1. Generate  : small LLM with the 'priors' prompt
  2. Validate  : compute MARE against known BLS medians
  3. Calibrate : reweight records to match known population marginals
                 (education distribution, employment rate) via post-stratification
  4. Verify    : refit the intended econometric model on the corrected data

This script implements step 3 (post-stratification) and step 4 (verify) for
every (model, priors) cell, and reports:
  - the marginal error BEFORE vs AFTER calibration (education TVD, employment rate)
  - the Mincer R^2 and return-to-schooling BEFORE vs AFTER calibration
    (to confirm the joint structure survives reweighting)

Output: results/calibration_demo.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_all_data, load_reference, compute_mincer, EDU_ORDER, EDU_YEARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")


def total_variation_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def post_stratify(df, ref):
    """Compute post-stratification weights matching the reference education distribution.

    We target the education distribution -- the marginal the LLMs distort most
    (over-representing advanced degrees). A raking-style weight is built as the
    ratio of the reference education share to the synthetic share, then
    normalized to sum to n. This is the minimal, defensible calibration step:
    it uses only publicly available aggregate shares and requires no real records.
    """
    n = len(df)
    edu_counts = df["education"].value_counts(normalize=True)
    weights = np.empty(n)
    edu_arr = df["education"].values
    for e in EDU_ORDER:
        mask = edu_arr == e
        synth = edu_counts.get(e, 1e-6)
        w = ref["education_distribution"].get(e, 0.0) / synth
        weights[mask] = w
    weights = weights / weights.sum() * n  # normalize to sum to n
    return weights


def weighted_median(values, weights):
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cum = np.cumsum(sw) / sw.sum()
    idx = np.searchsorted(cum, 0.5)
    return float(sv[min(idx, len(sv) - 1)])


def evaluate(df, ref, weights=None):
    """Compute the headline metrics, optionally weighted."""
    n = len(df)
    if weights is None:
        weights = np.ones(n)

    # Education distribution (weighted)
    edu_dist = {}
    for e in EDU_ORDER:
        mask = (df["education"] == e).values
        edu_dist[e] = float(weights[mask].sum() / weights.sum())
    tvd = total_variation_distance(edu_dist, ref["education_distribution"])

    # Employment rate (weighted)
    emp_rate = float(weights[df["employed"].values].sum() / weights.sum())

    # Median income by education (weighted, employed only)
    employed_mask = df["employed"].values
    med_income = {}
    for e in EDU_ORDER:
        mask = employed_mask & (df["education"] == e).values
        if mask.sum() > 0:
            med_income[e] = weighted_median(
                df["income"].values[mask], weights[mask]
            )
    ref_med = ref["median_income_by_education"]
    errs = []
    for e in EDU_ORDER:
        if med_income.get(e) and ref_med.get(e):
            errs.append(abs(med_income[e] - ref_med[e]) / ref_med[e])
    mare = float(np.mean(errs)) if errs else None

    # Mincer regression (weighted least squares)
    emp = df[employed_mask].copy()
    w_emp = weights[employed_mask]
    emp["log_income"] = np.log(emp["income"].clip(lower=1))
    emp["educ_years"] = emp["education"].map(EDU_YEARS)
    emp["experience"] = (emp["age"] - emp["educ_years"] - 6).clip(lower=0)
    emp["exp_sq"] = emp["experience"] ** 2
    X = emp[["educ_years", "experience", "exp_sq"]].values
    y = emp["log_income"].values
    Xc = np.column_stack([np.ones(len(X)), X])
    W = np.diag(w_emp)
    try:
        beta, _, _, _ = np.linalg.lstsq(Xc.T @ W @ Xc, Xc.T @ W @ y, rcond=None)
        resid = y - Xc @ beta
        ss_res = (resid ** 2 * w_emp).sum()
        ss_tot = ((y - np.average(y, weights=w_emp)) ** 2 * w_emp).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mincer = {"r_squared": round(float(r2), 4),
                  "return_to_education": round(float(beta[1]), 4)}
    except Exception:
        mincer = {"r_squared": None, "return_to_education": None}

    return {
        "education_tvd": round(tvd, 4),
        "employment_rate": round(emp_rate, 4),
        "median_income_mare": round(mare, 4) if mare is not None else None,
        "mincer_r2": mincer["r_squared"],
        "mincer_beta1": mincer["return_to_education"],
    }


def main():
    warnings.filterwarnings("ignore")
    ref = load_reference()
    data = load_all_data()

    results = {}
    for (model, condition), df in data.items():
        if condition != "priors":
            continue
        key = f"{model} / {condition}"
        before = evaluate(df, ref)
        weights = post_stratify(df, ref)
        after = evaluate(df, ref, weights)
        results[key] = {
            "before": before,
            "after": after,
            "improvement": {
                "tvd_delta": round(before["education_tvd"] - after["education_tvd"], 4),
                "emp_rate_delta": round(abs(before["employment_rate"] - ref["employment_rate"])
                                        - abs(after["employment_rate"] - ref["employment_rate"]), 4),
                "r2_delta": round(after["mincer_r2"] - before["mincer_r2"], 4) if before["mincer_r2"] and after["mincer_r2"] else None,
            },
        }
        print(f"{key}:")
        print(f"  before: TVD={before['education_tvd']}, emp={before['employment_rate']:.3f}, "
              f"MARE={before['median_income_mare']}, R2={before['mincer_r2']}")
        print(f"  after : TVD={after['education_tvd']}, emp={after['employment_rate']:.3f}, "
              f"MARE={after['median_income_mare']}, R2={after['mincer_r2']}")

    out_path = os.path.join(RESULTS_DIR, "calibration_demo.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibration demonstration written to {out_path}")


if __name__ == "__main__":
    main()
