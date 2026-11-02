"""
ml/train_regression.py
========================
Trains and compares three regression models for next-day AQI prediction:
  1. Linear Regression        (baseline)
  2. Random Forest Regressor
  3. Gradient Boosting Regressor

Model selection is based on TEST SET MAE (lower is better).
The winning model is registered in MLflow Model Registry.

Usage:
    python -m ml.train_regression

Output:
  - MLflow experiment: "AQI_Regression"
  - Best model saved to models/best_regressor.joblib
  - Comparison printed to console
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.logger import get_logger
from config.settings import get_mlflow_tracking_uri, get_project_root, settings
from ml.evaluate import evaluate_regression, print_regression_summary
from ml.features import get_X_y, load_data, prepare_features

logger = get_logger(__name__, log_file="ml.log")

_ML_CFG = settings["ml"]
_MODELS_DIR = get_project_root() / "models"
_MODELS_DIR.mkdir(exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────
# Each entry: (display_name, sklearn_estimator, hyperparams_to_log)
REGRESSION_MODELS = [
    (
        "LinearRegression",
        LinearRegression(),
        {"fit_intercept": True},
    ),
    (
        "RandomForestRegressor",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=_ML_CFG["random_state"],
            n_jobs=-1,
        ),
        {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "random_state": _ML_CFG["random_state"],
        },
    ),
    (
        "GradientBoostingRegressor",
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=_ML_CFG["random_state"],
        ),
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "random_state": _ML_CFG["random_state"],
        },
    ),
]


# ── MLflow setup ──────────────────────────────────────────────────────────────

def _setup_mlflow() -> str:
    """Configure MLflow tracking URI and create/get the experiment."""
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = settings["mlflow"]["experiment_regression"]

    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        mlflow.create_experiment(exp_name)
        logger.info("Created MLflow experiment: %s", exp_name)
    else:
        logger.info("Using MLflow experiment: %s (id=%s)", exp_name, experiment.experiment_id)

    mlflow.set_experiment(exp_name)
    return exp_name


# ── Single model training + evaluation ───────────────────────────────────────

def train_and_evaluate_model(
    model_name: str,
    estimator,
    params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    dataset_version: str = "v1",
) -> dict:
    """
    Train one regression model, log everything to MLflow, and return metrics.

    Parameters
    ----------
    model_name    : Display name for MLflow
    estimator     : Sklearn estimator instance
    params        : Hyperparameters to log
    train_df      : Training split DataFrame
    val_df        : Validation split DataFrame
    test_df       : Test split DataFrame
    feature_cols  : List of feature column names
    dataset_version : Version tag for the dataset used

    Returns
    -------
    dict with: model_name, run_id, train/val/test MAE/RMSE/R², training_time_s
    """
    X_train, y_train = get_X_y(train_df, task="regression")
    X_val,   y_val   = get_X_y(val_df,   task="regression")
    X_test,  y_test  = get_X_y(test_df,  task="regression")

    # Wrap in a pipeline with StandardScaler (helps Linear Regression most)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator),
    ])

    with mlflow.start_run(run_name=model_name) as run:
        # ── Log parameters ──────────────────────────────────────────────────
        mlflow.log_params({
            "model_name":       model_name,
            "feature_count":    len(feature_cols),
            "train_samples":    len(train_df),
            "val_samples":      len(val_df),
            "test_samples":     len(test_df),
            "dataset_version":  dataset_version,
            **params,
        })

        # ── Train ───────────────────────────────────────────────────────────
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        training_time = round(time.time() - t0, 2)
        mlflow.log_metric("training_time_s", training_time)

        # ── Evaluate on all splits ──────────────────────────────────────────
        results = {"model_name": model_name, "run_id": run.info.run_id,
                   "training_time_s": training_time}

        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
        ]:
            y_pred = pipeline.predict(X_split)
            metrics = evaluate_regression(y_split, y_pred, model_name, split_name)
            for k, v in metrics.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
                results[f"{split_name}_{k}"] = v

        # ── Feature importance (for tree models) ────────────────────────────
        inner = pipeline.named_steps["model"]
        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
            fi_df = pd.DataFrame({
                "feature": feature_cols[:len(importances)],
                "importance": importances,
            }).sort_values("importance", ascending=False)
            fi_path = _MODELS_DIR / f"feature_importance_{model_name}.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path), "feature_importance")

        # ── Log model artifact ───────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=None,  # Register only the winner later
        )

        # ── Save locally too ─────────────────────────────────────────────────
        local_path = _MODELS_DIR / f"regressor_{model_name}.joblib"
        joblib.dump(pipeline, local_path)
        mlflow.log_artifact(str(local_path), "local_model")

        logger.info(
            "✓ %s trained in %.1fs | test MAE=%.4f | test R²=%.4f",
            model_name, training_time,
            results.get("test_mae", 0), results.get("test_r2", 0),
        )

    return results


# ── Main training loop ────────────────────────────────────────────────────────

def train_all_regression_models(use_mock: bool = True) -> dict:
    """
    Train all regression models, compare results, register the best one.

    Parameters
    ----------
    use_mock : Fall back to mock data if PostgreSQL is unavailable.

    Returns
    -------
    dict mapping model_name → metrics dict
    """
    logger.info("=" * 60)
    logger.info("REGRESSION TRAINING STARTED")
    logger.info("=" * 60)

    # ── Load and prepare data ────────────────────────────────────────────────
    df = load_data(use_mock_on_failure=use_mock)
    train_df, val_df, test_df, _ = prepare_features(df)

    from transformation.feature_engineering import get_feature_columns
    feature_cols = [c for c in get_feature_columns() if c in train_df.columns]
    logger.info("Features used (%d): %s", len(feature_cols), feature_cols)

    # ── Setup MLflow ─────────────────────────────────────────────────────────
    try:
        exp_name = _setup_mlflow()
        mlflow_available = True
    except Exception as exc:
        logger.warning("MLflow not available (%s) – training without tracking", exc)
        mlflow_available = False

    # ── Train each model ─────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    for model_name, estimator, params in REGRESSION_MODELS:
        logger.info("Training: %s", model_name)
        try:
            if mlflow_available:
                result = train_and_evaluate_model(
                    model_name, estimator, params,
                    train_df, val_df, test_df, feature_cols,
                )
            else:
                # Train without MLflow
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                X_train, y_train = get_X_y(train_df, task="regression")
                X_test,  y_test  = get_X_y(test_df,  task="regression")
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
                t0 = time.time()
                pipeline.fit(X_train, y_train)
                training_time = round(time.time() - t0, 2)
                y_pred = pipeline.predict(X_test)
                metrics = evaluate_regression(y_test, y_pred, model_name, "test")
                result = {"model_name": model_name, "training_time_s": training_time, **{f"test_{k}": v for k, v in metrics.items()}}
                joblib.dump(pipeline, _MODELS_DIR / f"regressor_{model_name}.joblib")
            all_results[model_name] = result
        except Exception as exc:
            logger.error("Training failed for %s: %s", model_name, exc)

    # ── Select best model (lowest test MAE) ──────────────────────────────────
    if not all_results:
        logger.error("No models trained successfully.")
        return {}

    best_name = min(all_results, key=lambda k: all_results[k].get("test_mae", 9999))
    best_result = all_results[best_name]

    logger.info(
        "\n🏆 BEST REGRESSION MODEL: %s\n"
        "   Test MAE  = %.4f\n"
        "   Test RMSE = %.4f\n"
        "   Test R²   = %.4f\n"
        "   Selected because it achieved the lowest Test MAE "
        "among all trained models.",
        best_name,
        best_result.get("test_mae", 0),
        best_result.get("test_rmse", 0),
        best_result.get("test_r2", 0),
    )

    # ── Save best model as canonical file ────────────────────────────────────
    best_src = _MODELS_DIR / f"regressor_{best_name}.joblib"
    best_dst = _MODELS_DIR / "best_regressor.joblib"
    if best_src.exists():
        import shutil
        shutil.copy2(best_src, best_dst)
        logger.info("Best regressor saved → %s", best_dst)

    # ── Register best model in MLflow Model Registry ─────────────────────────
    if mlflow_available and "run_id" in best_result:
        try:
            model_uri = f"runs:/{best_result['run_id']}/model"
            reg_name = settings["mlflow"]["model_name_regression"]
            mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
            logger.info(
                "Registered model '%s' version %s in MLflow registry",
                reg_name, mv.version,
            )
            # Add selection rationale as a tag
            mlflow.MlflowClient().set_model_version_tag(
                name=reg_name,
                version=mv.version,
                key="selection_reason",
                value=f"Lowest test MAE ({best_result.get('test_mae', 'N/A'):.4f}) "
                      f"among LinearRegression, RandomForestRegressor, GradientBoostingRegressor",
            )
        except Exception as exc:
            logger.warning("MLflow registration failed: %s", exc)

    # ── Print summary table ───────────────────────────────────────────────────
    print_regression_summary(all_results)

    return all_results


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = train_all_regression_models(use_mock=True)
    if results:
        best = min(results, key=lambda k: results[k].get("test_mae", 9999))
        print(f"\n✅ Best model: {best}")
        print(f"   MAE  = {results[best].get('test_mae', 'N/A')}")
        print(f"   RMSE = {results[best].get('test_rmse', 'N/A')}")
        print(f"   R²   = {results[best].get('test_r2', 'N/A')}")
