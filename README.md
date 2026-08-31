# The Synthetic Economy: Can Small Language Models Generate Research-Grade Economic Microdata?

This repository contains the code, data, and paper for the study:

> **The Synthetic Economy: Can Small Language Models Generate Research-Grade Economic Microdata?**
> Mikhail N. Nekrashevich, Orel State University named after I.S. Turgenev
> arXiv preprint, 2026

We investigate whether **small, low-cost language models** (Ling 3.0 Flash, DeepSeek V4 Flash, Solar Pro 4, Qwen 3.7 Flash — each under \$0.03) can generate **research-grade synthetic economic microdata** from a text prompt alone, with no fine-tuning and no access to real records.

**Total API cost of the entire study: \$0.08.**

## Key Findings

- Small LLMs reproduce **conditional** economic structure with high fidelity: education→income gradients within 4.7–11.6% MARE, Mincer wage regressions with R² = 0.86–0.90, and near-perfect employment–hours–income consistency.
- They systematically distort **unconditional** marginals: overstating the employment rate (62–83% vs. 63% reference) and the share of advanced degrees.
- A three-condition prompt ablation (minimal / structural / priors) shows explicit priors improve income fidelity (MARE reduction 40–53%) but have limited effect on econometric utility (R² change < 0.04).
- Compared with non-LLM baselines, the LLMs' unique value is their **joint structure** (R² ≥ 0.86), not their marginals — a marginal sampler achieves MARE of 2.9% but R² of only 0.23.
- Cross-seed analysis demonstrates high reproducibility (CV < 3.4% for R²).

## Repository Structure

```
synthetic_economy/
├── paper/
│   ├── paper.tex          # LaTeX source
│   ├── paper.pdf          # Compiled PDF
│   ├── references.bib     # Bibliography
│   └── figures/           # Publication-quality figures
├── scripts/
│   ├── build_reference.py # Build ground-truth statistics from public CPS/BLS/Census aggregates
│   ├── generate_data.py   # Generate synthetic microdata (prompt ablation, parallel)
│   ├── evaluate.py        # Evaluate synthetic data (fidelity, utility, consistency, diversity)
│   ├── extended_analysis.py # Extended metrics (seed stability, gender gap, JSD, Wasserstein)
│   ├── make_figures.py    # Generate publication-quality figures
│   └── llm_client.py      # OpenRouter LLM client with cost tracking
├── data/                  # Synthetic microdata (JSON, 9,600 records)
├── results/               # Evaluation results (JSON)
├── run_all.sh             # Full pipeline: generate → evaluate → figures
├── requirements.txt       # Python dependencies
└── LICENSE
```

## Reproducing the Results

### Prerequisites

1. Python 3.9+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `requests`
2. An [OpenRouter](https://openrouter.ai) API key (for data generation)
3. LaTeX (for compiling the paper)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full pipeline

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
bash run_all.sh
```

To re-evaluate existing data without regenerating (no API cost):

```bash
bash run_all.sh --skip-generate
```

### Individual steps

```bash
# 1. Build reference statistics
python3 scripts/build_reference.py

# 2. Generate synthetic data (requires API key, ~20-35 min, ~$0.04)
python3 scripts/generate_data.py

# 3. Evaluate + bootstrap CIs
python3 scripts/evaluate.py

# 4. Extended analysis (seed stability, gender gap, JSD, Wasserstein)
python3 scripts/extended_analysis.py

# 5. Generate figures
python3 scripts/make_figures.py
```

## Data

The `data/` directory contains 9,600 synthetic household records:
- **4 models** × **3 prompt conditions** × **2 seeds** × **400 records** = 9,600 records
- Each record has 9 fields: age, gender, education, income, employed, hours_worked, marital_status, children, state

## Citation

If you use this work, please cite:

```bibtex
@misc{nekrashevich2026synthetic,
  title={The Synthetic Economy: Can Small Language Models Generate Research-Grade Economic Microdata?},
  author={Nekrashevich, Mikhail N.},
  year={2026}
}
```

## License

This work is licensed under the MIT License. See [LICENSE](LICENSE).

## Contact

Mikhail N. Nekrashevich — mn.nekrashevich@yandex.ru
ORCID: 0009-0005-1808-8906
