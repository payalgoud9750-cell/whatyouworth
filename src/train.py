"""Train the salary prediction model and save metrics."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, MODELS_DIR, TARGET_COLUMN
from src.generate_data import generate_dataset
from src.preprocess import build_pipeline


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Dataset not found at {path}. Generating synthetic data...")
        generate_dataset()
    return pd.read_csv(path)


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    df = load_data(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train salary prediction model")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    metrics = train(data_path=args.data, model_path=args.model)

    print("Training complete.")
    print(f"  Model saved to: {args.model}")
    print(f"  MAE:  ${metrics['mae']:,.2f}")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
