# Whatyouworth — Salary Prediction (Python + ML)

Predict annual salary (USD) from experience, education, role, industry, company size, location tier, remote work, and employment type using a **Gradient Boosting** regressor. Includes a Netflix-themed web app.

## Synopsis

**Whatyouworth** estimates annual compensation from a professional profile (26 roles, 15 industries, 10 location tiers, and more). Synthetic data is generated, a Gradient Boosting model is trained, and results are available via CLI or Flask web UI at `http://127.0.0.1:5000`. See [SYNOPSIS.md](SYNOPSIS.md) for the full project overview, workflow, limitations, and layout.

## Setup

```bash
cd project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.generate_data --rows 2000
python -m src.train
python -m src.predict --years 5 --education Master --role "Data Scientist" --company-size "Enterprise (1000+)" --remote 80 --city-tier "Tier 1 - SF / NYC / Seattle" --industry Technology --employment-type Full-time
```

### Web app

After training the model, start the site:

```bash
python -m src.web
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

See src/ for modules: generate_data, preprocess, train, predict, web.
