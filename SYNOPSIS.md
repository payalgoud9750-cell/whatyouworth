# Synopsis — Whatyouworth (Salary Prediction)

## Overview

**Whatyouworth** is a machine-learning salary estimation tool that predicts annual compensation in USD from a professional profile. Users describe their background—experience, education, role, industry, company size, location tier, remote-work level, and employment type—and receive an instant salary estimate. The project includes a command-line workflow for data generation, training, and prediction, plus a Netflix-inspired web app for interactive use.

## Problem

Salary expectations vary widely by role, market, education, and work arrangement. Professionals and students often lack a quick, structured way to explore how these factors combine into a plausible pay range. Whatyouworth offers a lightweight, data-driven estimate as a starting point for research and negotiation—not a substitute for official offers or market surveys.

## Approach

1. **Synthetic dataset** — A generator (`src/generate_data.py`) builds ~2,000 labeled rows with realistic relationships between features and salary (base pay by role, experience and education premiums, location and industry adjustments, company-size multipliers, employment-type scaling, and noise).

2. **Model** — A scikit-learn pipeline (`src/preprocess.py`, `src/train.py`) scales numeric inputs, one-hot encodes categories, and fits a **Gradient Boosting** regressor. The trained model is saved to `models/salary_model.joblib`.

3. **Inference** — `src/predict.py` loads the pipeline and returns a single USD prediction for any valid profile.

4. **Web UI** — `src/web.py` (Flask) serves a dark, Netflix-style interface where users fill dropdowns and sliders and submit a form to see the predicted salary.

## Features

| Area | Details |
|------|---------|
| **Inputs** | Years of experience, 9 education levels, 26 job roles, 8 company sizes, 10 location tiers, 15 industries, 6 employment types, remote work % (0–100) |
| **Output** | Predicted annual salary (USD) |
| **Interfaces** | CLI (`python -m src.*`) and browser (`python -m src.web` → http://127.0.0.1:5000) |
| **Stack** | Python 3, pandas, scikit-learn, joblib, Flask |

## Typical workflow

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m src.generate_data --rows 2000
python -m src.train
python -m src.web
```

## Model performance (reference)

On a held-out 20% test split of the synthetic data, the model typically achieves roughly **R² ≈ 0.96**, **MAE ≈ $7k**, and **RMSE ≈ $9k**. Metrics depend on the generated dataset and should be re-evaluated after retraining.

## Limitations

- Training data is **synthetic**, not real market surveys; predictions are illustrative.
- New dropdown values not seen during training are handled with `handle_unknown="ignore"` but may be less reliable until data and model are refreshed.
- The tool does not account for equity, bonuses, benefits, or cost-of-living beyond coarse city-tier labels.

## Project layout

```
project/
├── data/salaries.csv      # Generated training data
├── models/                # Saved pipeline (after train)
├── src/                   # generate_data, preprocess, train, predict, web, config
├── templates/index.html   # Web form
├── static/styles.css      # Netflix-themed UI
└── requirements.txt
```

## Summary

Whatyouworth demonstrates an end-to-end ML product: synthetic data → trained regressor → CLI and web delivery—with a polished UI for non-technical users. It is suitable as a portfolio demo, classroom exercise, or baseline to extend with real salary datasets and additional features.
