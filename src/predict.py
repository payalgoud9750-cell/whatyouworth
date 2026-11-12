"""Predict salary from profile features."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    CITY_TIER,
    COMPANY_SIZE,
    EDUCATION,
    EMPLOYMENT_TYPE,
    FEATURE_COLUMNS,
    INDUSTRY,
    JOB_ROLES,
    MODEL_PATH,
)


def predict_one(
    years_experience: float,
    education: str,
    job_role: str,
    company_size: str,
    remote_ratio: int,
    city_tier: str,
    industry: str = "Technology",
    employment_type: str = "Full-time",
    model_path: Path = MODEL_PATH,
) -> float:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.train"
        )

    pipeline = joblib.load(model_path)
    row = pd.DataFrame(
        [
            {
                "years_experience": years_experience,
                "education": education,
                "job_role": job_role,
                "company_size": company_size,
                "remote_ratio": remote_ratio,
                "city_tier": city_tier,
                "industry": industry,
                "employment_type": employment_type,
            }
        ]
    )
    return float(pipeline.predict(row[FEATURE_COLUMNS])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict annual salary (USD)")
    parser.add_argument("--years", type=float, required=True, help="Years of experience")
    parser.add_argument("--education", required=True, choices=EDUCATION)
    parser.add_argument("--role", required=True, choices=JOB_ROLES)
    parser.add_argument("--company-size", required=True, choices=COMPANY_SIZE)
    parser.add_argument("--remote", type=int, required=True, help="Remote work % (0-100)")
    parser.add_argument("--city-tier", required=True, choices=CITY_TIER)
    parser.add_argument("--industry", default="Technology", choices=INDUSTRY)
    parser.add_argument("--employment-type", default="Full-time", choices=EMPLOYMENT_TYPE)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    salary = predict_one(
        years_experience=args.years,
        education=args.education,
        job_role=args.role,
        company_size=args.company_size,
        remote_ratio=args.remote,
        city_tier=args.city_tier,
        industry=args.industry,
        employment_type=args.employment_type,
        model_path=args.model,
    )

    print(f"Predicted annual salary: ${salary:,.2f} USD")


if __name__ == "__main__":
    main()
