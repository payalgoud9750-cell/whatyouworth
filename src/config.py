from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "salaries.csv"
MODEL_PATH = MODELS_DIR / "salary_model.joblib"

EDUCATION = [
    "High School",
    "Associate",
    "Bootcamp",
    "Bachelor",
    "Professional Certificate",
    "Master",
    "MBA",
    "PhD",
    "Postdoctoral",
]

JOB_ROLES = [
    "Software Engineer",
    "Frontend Engineer",
    "Backend Engineer",
    "Full Stack Engineer",
    "Mobile Developer",
    "Data Scientist",
    "Data Engineer",
    "ML Engineer",
    "AI Researcher",
    "Product Manager",
    "Product Designer",
    "UI/UX Designer",
    "DevOps Engineer",
    "Cloud Architect",
    "Site Reliability Engineer",
    "Security Engineer",
    "QA Engineer",
    "Automation Engineer",
    "Business Analyst",
    "Project Manager",
    "Scrum Master",
    "Technical Writer",
    "Solutions Architect",
    "Database Administrator",
    "Game Developer",
    "Blockchain Developer",
]

COMPANY_SIZE = [
    "Solo / Freelance",
    "Startup (1-10)",
    "Small (11-50)",
    "SMB (51-200)",
    "Mid-Market (201-1000)",
    "Enterprise (1000+)",
    "Fortune 500",
    "Government / NGO",
]

CITY_TIER = [
    "Tier 1 - SF / NYC / Seattle",
    "Tier 1 - London / Paris",
    "Tier 2 - Austin / Denver / Boston",
    "Tier 2 - Toronto / Berlin",
    "Tier 3 - Midwest / South US",
    "Tier 4 - Rural / Low COL",
    "Remote - US National",
    "Remote - Global Hub",
    "International - Asia Pacific",
    "International - LATAM",
]

INDUSTRY = [
    "Technology",
    "Finance & Banking",
    "Healthcare",
    "Retail & E-commerce",
    "Media & Entertainment",
    "Education",
    "Government",
    "Manufacturing",
    "Consulting",
    "Energy",
    "Real Estate",
    "Telecommunications",
    "Automotive",
    "Aerospace",
    "Non-profit",
]

EMPLOYMENT_TYPE = [
    "Full-time",
    "Part-time",
    "Contract",
    "Freelance",
    "Internship",
    "Co-op",
]

FEATURE_COLUMNS = [
    "years_experience",
    "education",
    "job_role",
    "company_size",
    "remote_ratio",
    "city_tier",
    "industry",
    "employment_type",
]

TARGET_COLUMN = "salary_usd"
