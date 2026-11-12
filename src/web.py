"""Flask web app for salary prediction."""

from flask import Flask, render_template, request

from src.config import (
    CITY_TIER,
    COMPANY_SIZE,
    EDUCATION,
    EMPLOYMENT_TYPE,
    INDUSTRY,
    JOB_ROLES,
    MODEL_PATH,
    ROOT,
)
from src.predict import predict_one

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

DEFAULT_FORM = {
    "years_experience": "5",
    "education": "Master",
    "job_role": "Data Scientist",
    "company_size": "Enterprise (1000+)",
    "remote_ratio": "80",
    "city_tier": "Tier 1 - SF / NYC / Seattle",
    "industry": "Technology",
    "employment_type": "Full-time",
}


@app.route("/", methods=["GET", "POST"])
def index():
    form = dict(DEFAULT_FORM)
    prediction = None
    error = None

    if request.method == "POST":
        form = {
            "years_experience": request.form.get("years_experience", ""),
            "education": request.form.get("education", ""),
            "job_role": request.form.get("job_role", ""),
            "company_size": request.form.get("company_size", ""),
            "remote_ratio": request.form.get("remote_ratio", ""),
            "city_tier": request.form.get("city_tier", ""),
            "industry": request.form.get("industry", ""),
            "employment_type": request.form.get("employment_type", ""),
        }
        try:
            years = float(form["years_experience"])
            remote = int(form["remote_ratio"])
            if years < 0 or years > 40:
                raise ValueError("Years of experience must be between 0 and 40.")
            if remote < 0 or remote > 100:
                raise ValueError("Remote work must be between 0 and 100 percent.")

            prediction = predict_one(
                years_experience=years,
                education=form["education"],
                job_role=form["job_role"],
                company_size=form["company_size"],
                remote_ratio=remote,
                city_tier=form["city_tier"],
                industry=form["industry"],
                employment_type=form["employment_type"],
            )
        except FileNotFoundError:
            error = "Model not found. Train it first: python -m src.train"
        except (ValueError, TypeError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        form=form,
        prediction=prediction,
        error=error,
        model_ready=MODEL_PATH.exists(),
        educations=EDUCATION,
        job_roles=JOB_ROLES,
        company_sizes=COMPANY_SIZE,
        city_tiers=CITY_TIER,
        industries=INDUSTRY,
        employment_types=EMPLOYMENT_TYPE,
    )


def main() -> None:
    if not MODEL_PATH.exists():
        print("Warning: model not found. Run: python -m src.train")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
