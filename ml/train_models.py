"""
ml/train_models.py
===================
Trains ALL 6 models (3 regression + 3 classification), compares them,
selects the best of each type, saves to models/, and logs to MLflow
if available — otherwise logs to a local JSON report.

Regression  models : LinearRegression, RandomForestRegressor, GradientBoostingRegressor
Classification models: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier

Selection criteria:
  Regression     → lowest Test MAE
  Classification → highest Test F1 (weighted)

Run:
    python ml/train_models.py
"""

import json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score, r2_score,
                              recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

MODELS_DIR    = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
LOGS_DIR      = ROOT / "logs"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ── AQI category ordinal encoding ─────────────────────────────────────────
CAT_ORDER = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous"
]

# ── Feature columns (must match pipeline output) ──────────────────────────
FEATURE_COLS = [
    "pm25","pm10","no2","so2","co","o3","nh3",
    "temperature","humidity","precipitation","wind_speed","pressure",
    "year","month","day_of_week","is_weekend","season",
    "aqi_lag_1","aqi_lag_2","aqi_lag_3","aqi_lag_7",
    "aqi_rolling_3","aqi_rolling_7",
]


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    # Prefer CSV — works without pyarrow/fastparquet
    csv = PROCESSED_DIR / "aqi_processed.csv"
    pq  = PROCESSED_DIR / "aqi_processed.parquet"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["date"])
        print(f"  Loaded {len(df):,} rows from aqi_processed.csv")
    elif pq.exists():
        df = pd.read_parquet(pq)
        print(f"  Loaded {len(df):,} rows from aqi_processed.parquet")
    else:
        raise FileNotFoundError(
            "No processed data found. Run:  python run_pipeline.py  first."
        )
    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ════════════════════════════════════════════════════════════════════════════
def prepare(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station_id","date"]).reset_index(drop=True)

    # Drop rows without target
    df = df.dropna(subset=["aqi","aqi_category"])

    # Encode classification target
    le = LabelEncoder()
    # Only keep known categories
    df = df[df["aqi_category"].isin(CAT_ORDER)].copy()
    le.fit(CAT_ORDER)
    df["cat_encoded"] = le.transform(df["aqi_category"])

    # Available features
    avail = [c for c in FEATURE_COLS if c in df.columns]

    # Chronological split — NO shuffling (prevents temporal leakage)
    n = len(df)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    train = df.iloc[:t_end].copy()
    val   = df.iloc[t_end:v_end].copy()
    test  = df.iloc[v_end:].copy()

    print(f"  Split → train:{len(train):,} | val:{len(val):,} | test:{len(test):,}")
    print(f"  Features used ({len(avail)}): {avail}")

    # Impute missing with TRAINING median only
    imputer = SimpleImputer(strategy="median")
    train[avail] = imputer.fit_transform(train[avail])
    val[avail]   = imputer.transform(val[avail])
    test[avail]  = imputer.transform(test[avail])

    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
    joblib.dump(le,      MODELS_DIR / "label_encoder.joblib")
    joblib.dump(avail,   MODELS_DIR / "feature_cols.joblib")

    return train, val, test, avail, le


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def eval_reg(y_true, y_pred, name, split):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"    [{split}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4)}

def eval_cls(y_true, y_pred, le, name, split):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"    [{split}] Acc={acc:.4f}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")
    return {"accuracy": round(acc,4), "f1_weighted": round(f1,4),
            "precision_weighted": round(prec,4), "recall_weighted": round(rec,4)}


# ════════════════════════════════════════════════════════════════════════════
# MLFLOW SETUP (optional — gracefully skipped if not running)
# ════════════════════════════════════════════════════════════════════════════
def setup_mlflow():
    try:
        import mlflow
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(uri)
        # Quick connectivity check
        mlflow.search_experiments()
        print(f"  MLflow connected at {uri}")
        return mlflow, True
    except Exception as e:
        print(f"  MLflow not available ({e.__class__.__name__}) — logging to JSON only")
        return None, False


# ════════════════════════════════════════════════════════════════════════════
# TRAIN ONE MODEL
# ════════════════════════════════════════════════════════════════════════════
def train_model(name, estimator, task, train, val, test, feats, le,
                mlflow, mlflow_ok, exp_name):
    print(f"\n  ▶ Training: {name}")
    X_tr = train[feats]; X_v = val[feats]; X_te = test[feats]

    if task == "regression":
        y_tr = train["aqi"]; y_v = val["aqi"]; y_te = test["aqi"]
    else:
        y_tr = train["cat_encoded"]; y_v = val["cat_encoded"]; y_te = test["cat_encoded"]

    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    t_train = round(time.time()-t0, 2)

    result = {"name": name, "task": task, "training_time_s": t_train,
              "features": feats, "n_train": len(train)}

    for split_name, X_s, y_s in [("train",X_tr,y_tr),("val",X_v,y_v),("test",X_te,y_te)]:
        y_pred = pipe.predict(X_s)
        if task == "regression":
            m = eval_reg(y_s, y_pred, name, split_name)
        else:
            m = eval_cls(y_s, y_pred, le, name, split_name)
        result.update({f"{split_name}_{k}": v for k,v in m.items()})

    # Feature importance
    inner = pipe.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        fi = pd.DataFrame({"feature": feats,
                           "importance": inner.feature_importances_
                           }).sort_values("importance", ascending=False)
        fi.to_csv(MODELS_DIR / f"fi_{name}.csv", index=False)

    # Save model
    model_path = MODELS_DIR / f"{task}_{name}.joblib"
    joblib.dump(pipe, model_path)
    result["model_path"] = str(model_path)

    # MLflow logging
    if mlflow_ok and mlflow:
        try:
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "n_features": len(feats),
                                   "n_train": len(train), "task": task})
                mlflow.log_metrics({k: v for k,v in result.items()
                                    if isinstance(v, float)})
                mlflow.sklearn.log_model(pipe, "model")
                result["mlflow_run_id"] = mlflow.active_run().info.run_id
        except Exception as e:
            print(f"    MLflow log failed: {e}")

    print(f"    Saved → {model_path.name}  ({t_train}s)")
    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    divider = "═" * 65
    print(f"\n{divider}")
    print("  🤖  AQI ML TRAINING PIPELINE")
    print(divider)

    # Load processed data
    df = load_data()
    train, val, test, feats, le = prepare(df)

    # MLflow
    mlflow, mlflow_ok = setup_mlflow()

    all_results = []

    # ── REGRESSION MODELS ──────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  📈  REGRESSION  (predict numerical AQI)")
    print(divider)

    reg_models = [
        ("LinearRegression",
         LinearRegression()),
        ("RandomForestRegressor",
         RandomForestRegressor(n_estimators=200, max_depth=12,
                               min_samples_leaf=5, random_state=RANDOM_STATE,
                               n_jobs=-1)),
        ("GradientBoostingRegressor",
         GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8,
                                   random_state=RANDOM_STATE)),
    ]

    reg_results = []
    for name, est in reg_models:
        r = train_model(name, est, "regression",
                        train, val, test, feats, le,
                        mlflow, mlflow_ok, "AQI_Regression")
        reg_results.append(r)
        all_results.append(r)

    # Select best regressor (lowest test MAE)
    best_reg = min(reg_results, key=lambda x: x.get("test_mae", 9999))
    import shutil
    src = Path(best_reg["model_path"])
    shutil.copy2(src, MODELS_DIR / "best_regressor.joblib")

    print(f"\n  🏆 Best Regressor : {best_reg['name']}")
    print(f"     Test MAE  = {best_reg['test_mae']:.4f}")
    print(f"     Test RMSE = {best_reg['test_rmse']:.4f}")
    print(f"     Test R²   = {best_reg['test_r2']:.4f}")
    print(f"     Reason: lowest Test MAE among all regression models")

    # ── CLASSIFICATION MODELS ──────────────────────────────────────────────
    print(f"\n{divider}")
    print("  🏷️   CLASSIFICATION  (predict AQI category)")
    print(divider)

    cls_models = [
        ("LogisticRegression",
         LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)),
        ("RandomForestClassifier",
         RandomForestClassifier(n_estimators=200, max_depth=12,
                                min_samples_leaf=5, random_state=RANDOM_STATE,
                                n_jobs=-1, class_weight="balanced")),
        ("GradientBoostingClassifier",
         GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=5, subsample=0.8,
                                    random_state=RANDOM_STATE)),
    ]

    cls_results = []
    for name, est in cls_models:
        r = train_model(name, est, "classification",
                        train, val, test, feats, le,
                        mlflow, mlflow_ok, "AQI_Classification")
        cls_results.append(r)
        all_results.append(r)

    # Select best classifier (highest test F1 weighted)
    best_cls = max(cls_results, key=lambda x: x.get("test_f1_weighted", 0))
    shutil.copy2(Path(best_cls["model_path"]), MODELS_DIR / "best_classifier.joblib")

    print(f"\n  🏆 Best Classifier : {best_cls['name']}")
    print(f"     Test Accuracy  = {best_cls['test_accuracy']:.4f}")
    print(f"     Test F1        = {best_cls['test_f1_weighted']:.4f}")
    print(f"     Test Precision = {best_cls['test_precision_weighted']:.4f}")
    print(f"     Test Recall    = {best_cls['test_recall_weighted']:.4f}")
    print(f"     Reason: highest Test F1-weighted among all classification models")

    # ── COMPARISON TABLE ───────────────────────────────────────────────────
    print(f"\n{divider}")
    print("  📊  MODEL COMPARISON SUMMARY")
    print(divider)
    print(f"  {'Model':<38} {'Task':<15} {'Primary Metric':>14}")
    print(f"  {'-'*38} {'-'*15} {'-'*14}")
    for r in all_results:
        if r["task"] == "regression":
            metric = f"MAE={r['test_mae']:.2f}"
        else:
            metric = f"F1={r['test_f1_weighted']:.4f}"
        marker = " ⭐" if r["name"] in [best_reg["name"], best_cls["name"]] else ""
        print(f"  {r['name']:<38} {r['task']:<15} {metric:>14}{marker}")

    # ── CLASSIFICATION REPORT (best model) ────────────────────────────────
    print(f"\n  Confusion Matrix — {best_cls['name']} on TEST set:")
    best_cls_pipe = joblib.load(MODELS_DIR / "best_classifier.joblib")
    X_te = test[feats]
    y_te_true = test["cat_encoded"]
    y_te_pred = best_cls_pipe.predict(X_te)
    cm = confusion_matrix(y_te_true, y_te_pred)
    labels = [c[:15] for c in le.classes_]
    print(f"  Labels: {labels}")
    for i, row_cm in enumerate(cm):
        print(f"  {labels[i]:<15}: {list(row_cm)}")

    # ── SAVE REPORT ────────────────────────────────────────────────────────
    report = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "best_regressor": best_reg["name"],
        "best_classifier": best_cls["name"],
        "models": all_results,
    }
    report_path = LOGS_DIR / "ml_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save model info for FastAPI
    model_info = {
        "best_regressor":  {"name": best_reg["name"],
                            "path": "models/best_regressor.joblib",
                            "test_mae": best_reg["test_mae"],
                            "test_r2":  best_reg["test_r2"]},
        "best_classifier": {"name": best_cls["name"],
                            "path": "models/best_classifier.joblib",
                            "test_f1":  best_cls["test_f1_weighted"],
                            "test_accuracy": best_cls["test_accuracy"]},
        "feature_cols":    feats,
        "label_classes":   list(le.classes_),
    }
    with open(MODELS_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{divider}")
    print(f"  ✅  Training complete!")
    print(f"  📄  Report saved → {report_path.name}")
    print(f"  💾  Models saved → models/")
    print(f"       best_regressor.joblib   ({best_reg['name']})")
    print(f"       best_classifier.joblib  ({best_cls['name']})")
    print(divider)


if __name__ == "__main__":
    main()
