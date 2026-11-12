"""Generate a synthetic salary dataset for training and demos."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CITY_TIER,
    COMPANY_SIZE,
    DATA_PATH,
    EDUCATION,
    EMPLOYMENT_TYPE,
    INDUSTRY,
    JOB_ROLES,
)

RNG = np.random.default_rng(42)

BASE_SALARY = {
    "Software Engineer": 75_000,
    "Frontend Engineer": 72_000,
    "Backend Engineer": 76_000,
    "Full Stack Engineer": 78_000,
    "Mobile Developer": 74_000,
    "Data Scientist": 85_000,
    "Data Engineer": 82_000,
    "ML Engineer": 88_000,
    "AI Researcher": 92_000,
    "Product Manager": 90_000,
    "Product Designer": 78_000,
    "UI/UX Designer": 76_000,
    "DevOps Engineer": 80_000,
    "Cloud Architect": 95_000,
    "Site Reliability Engineer": 86_000,
    "Security Engineer": 84_000,
    "QA Engineer": 60_000,
    "Automation Engineer": 70_000,
    "Business Analyst": 65_000,
    "Project Manager": 72_000,
    "Scrum Master": 68_000,
    "Technical Writer": 62_000,
    "Solutions Architect": 98_000,
    "Database Administrator": 74_000,
    "Game Developer": 70_000,
    "Blockchain Developer": 90_000,
}

EDUCATION_BONUS = {
    "High School": 0,
    "Associate": 4_000,
    "Bootcamp": 6_000,
    "Bachelor": 8_000,
    "Professional Certificate": 10_000,
    "Master": 18_000,
    "MBA": 22_000,
    "PhD": 28_000,
    "Postdoctoral": 32_000,
}

SIZE_MULTIPLIER = {
    "Solo / Freelance": 0.75,
    "Startup (1-10)": 0.85,
    "Small (11-50)": 0.92,
    "SMB (51-200)": 1.0,
    "Mid-Market (201-1000)": 1.08,
    "Enterprise (1000+)": 1.15,
    "Fortune 500": 1.22,
    "Government / NGO": 0.95,
}

TIER_ADJUSTMENT = {
    "Tier 1 - SF / NYC / Seattle": 15_000,
    "Tier 1 - London / Paris": 14_000,
    "Tier 2 - Austin / Denver / Boston": 6_000,
    "Tier 2 - Toronto / Berlin": 5_000,
    "Tier 3 - Midwest / South US": 0,
    "Tier 4 - Rural / Low COL": -10_000,
    "Remote - US National": 3_000,
    "Remote - Global Hub": 4_000,
    "International - Asia Pacific": 2_000,
    "International - LATAM": -2_000,
}

INDUSTRY_BONUS = {
    "Technology": 8_000,
    "Finance & Banking": 12_000,
    "Healthcare": 6_000,
    "Retail & E-commerce": 4_000,
    "Media & Entertainment": 5_000,
    "Education": 2_000,
    "Government": 3_000,
    "Manufacturing": 4_000,
    "Consulting": 10_000,
    "Energy": 7_000,
    "Real Estate": 5_000,
    "Telecommunications": 6_000,
    "Automotive": 5_000,
    "Aerospace": 9_000,
    "Non-profit": 0,
}

EMPLOYMENT_MULTIPLIER = {
    "Full-time": 1.0,
    "Part-time": 0.55,
    "Contract": 0.95,
    "Freelance": 0.88,
    "Internship": 0.35,
    "Co-op": 0.40,
}


def _sample(rows: int) -> pd.DataFrame:
    years = RNG.integers(0, 21, size=rows).astype(float)
    education = RNG.choice(
        EDUCATION,
        size=rows,
        p=[0.05, 0.06, 0.04, 0.32, 0.05, 0.28, 0.08, 0.10, 0.02],
    )
    job_role = RNG.choice(JOB_ROLES, size=rows)
    company_size = RNG.choice(
        COMPANY_SIZE,
        size=rows,
        p=[0.05, 0.12, 0.15, 0.18, 0.20, 0.18, 0.08, 0.04],
    )
    remote_ratio = RNG.integers(0, 101, size=rows)
    city_tier = RNG.choice(
        CITY_TIER,
        size=rows,
        p=[0.18, 0.08, 0.15, 0.07, 0.12, 0.06, 0.14, 0.08, 0.07, 0.05],
    )
    industry = RNG.choice(
        INDUSTRY,
        size=rows,
        p=[0.26, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.06, 0.07, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01],
    )
    employment_type = RNG.choice(
        EMPLOYMENT_TYPE,
        size=rows,
        p=[0.72, 0.06, 0.10, 0.05, 0.04, 0.03],
    )

    salary = np.zeros(rows, dtype=float)
    for i in range(rows):
        base = BASE_SALARY[job_role[i]]
        exp_bonus = years[i] * 4_500
        edu_bonus = EDUCATION_BONUS[education[i]]
        size_mult = SIZE_MULTIPLIER[company_size[i]]
        tier_bonus = TIER_ADJUSTMENT[city_tier[i]]
        industry_bonus = INDUSTRY_BONUS[industry[i]]
        emp_mult = EMPLOYMENT_MULTIPLIER[employment_type[i]]
        remote_bonus = (remote_ratio[i] / 100) * 6_000
        noise = RNG.normal(0, 5_000)

        salary[i] = (
            (base + exp_bonus + edu_bonus + tier_bonus + industry_bonus + remote_bonus)
            * size_mult
            * emp_mult
            + noise
        )

    salary = np.clip(salary, 25_000, 280_000).round(2)

    return pd.DataFrame(
        {
            "years_experience": years,
            "education": education,
            "job_role": job_role,
            "company_size": company_size,
            "remote_ratio": remote_ratio,
            "city_tier": city_tier,
            "industry": industry,
            "employment_type": employment_type,
            "salary_usd": salary,
        }
    )


def generate_dataset(rows: int = 2000, output: Path | None = None) -> Path:
    output = output or DATA_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    df = _sample(rows)
    df.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic salary data")
    parser.add_argument("--rows", type=int, default=2000, help="Number of samples")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PATH,
        help="Output CSV path",
    )
    args = parser.parse_args()
    path = generate_dataset(rows=args.rows, output=args.output)
    print(f"Wrote {args.rows} rows to {path}")


if __name__ == "__main__":
    main()
