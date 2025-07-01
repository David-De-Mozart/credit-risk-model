import os
import mlflow
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# Initialize MLflow
mlflow.set_tracking_uri("file:///" + os.path.abspath("../mlruns"))
mlflow.set_experiment("credit-risk-modeling")


def load_data():
    """Load processed data"""
    # Get project root directory
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / 'data' / 'processed'

    # Debug: List files in processed directory
    print(f"Checking directory: {processed_dir}")
    if processed_dir.exists():
        print("Files in directory:")
        for f in processed_dir.iterdir():
            print(f" - {f.name}")
    else:
        print("Directory does not exist!")

    # Load features and target
    X_path = processed_dir / 'X_processed.joblib'
    y_path = processed_dir / 'y.joblib'

    print(f"Loading X from: {X_path}")
    print(f"Loading y from: {y_path}")

    X = joblib.load(X_path)
    y = joblib.load(y_path)

    return X, y


def train_models():
    """Train and evaluate multiple models"""
    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42), "RandomForest": RandomForestClassifier(
            random_state=42), "GradientBoosting": GradientBoostingClassifier(
                random_state=42), "XGBoost": XGBClassifier(
                    random_state=42)}

    best_score = 0
    best_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            # Log parameters
            mlflow.log_params(model.get_params())

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Print results
            print(f"\n{name} Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model

    # Save best model

    project_dir = Path(__file__).resolve().parents[1]
    models_dir = project_dir / 'models'
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = models_dir / 'best_model.joblib'
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    train_models()
