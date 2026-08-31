"""Build reference (ground-truth) statistics from public CPS/BLS/Census aggregates.

These are the public aggregated statistics we compare synthetic data against.
Sources: BLS median weekly earnings by education (2024), Census educational
attainment distribution (2021/2024), BLS unemployment by education (2024).
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Median annual income by education (BLS 2024, full-time workers 25+).
# Weekly medians converted to annual (x52).
MEDIAN_INCOME_BY_EDU = {
    "high_school": 48360,     # $930/week
    "some_college": 53040,    # $1020/week (some college, no degree)
    "bachelors": 80236,       # $1543/week
    "masters": 95680,         # $1840/week
    "phd": 118456,            # $2278/week (professional/doctoral)
}

# Educational attainment distribution, adults 25+ (Census 2021/2024).
EDU_DISTRIBUTION = {
    "high_school": 0.368,     # HS diploma or less (incl. <HS ~8.9%)
    "some_college": 0.254,    # some college + associate (~14.9% + ~10.5%)
    "bachelors": 0.235,       # bachelor's degree
    "masters": 0.110,         # master's degree
    "phd": 0.033,             # doctoral/professional
}

# Unemployment rate by education (BLS 2024).
UNEMPLOYMENT_BY_EDU = {
    "high_school": 0.042,
    "some_college": 0.033,
    "bachelors": 0.025,
    "masters": 0.022,
    "phd": 0.012,
}

# Mincer return to education (canonical ~8-10% per year of schooling).
MINCER_RETURN = 0.09

# Gender split (approx).
GENDER_SPLIT = {"male": 0.5, "female": 0.5}

# Employment-population ratio, adults 25+ (BLS 2024, ~62-63%).
EMPLOYMENT_RATE = 0.63


def main():
    reference = {
        "median_income_by_education": MEDIAN_INCOME_BY_EDU,
        "education_distribution": EDU_DISTRIBUTION,
        "unemployment_by_education": UNEMPLOYMENT_BY_EDU,
        "mincer_return": MINCER_RETURN,
        "gender_split": GENDER_SPLIT,
        "employment_rate": EMPLOYMENT_RATE,
        "sources": {
            "median_income": "BLS 2024 median weekly earnings by education (annualized x52)",
            "education_distribution": "US Census educational attainment 2021/2024 (25+)",
            "unemployment": "BLS 2024 unemployment rate by education",
            "mincer_return": "Canonical Mincer (1974) return to schooling ~8-10%",
        },
    }
    out_path = os.path.join(RESULTS_DIR, "reference.json")
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference written to {out_path}")


if __name__ == "__main__":
    main()
