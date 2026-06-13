import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import FEATURE_COLUMNS, HORIZON_DAYS, RANDOM_STATE
from .utils import directional_accuracy, safe_mape


def _catboost_model():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=350, learning_rate=0.04, depth=5, loss_function="RMSE",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=RANDOM_STATE)


def _catboost_classifier():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=350, learning_rate=0.04, depth=5, loss_function="Logloss",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=RANDOM_STATE)


def _rf_model():
    return RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def evaluate_prediction(ticker: str, model_name: str, y_true, y_pred) -> dict:
    return {
        "ticker": ticker,
        "model": model_name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": safe_mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def evaluate_classifier(ticker: str, y_true, prob_positive) -> dict:
    pred_label = np.asarray(prob_positive) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[False, True]).ravel()
    out = {
        "ticker": ticker,
        "model": "CatBoostClassifier",
        "accuracy": accuracy_score(y_true, pred_label),
        "precision_buy": precision_score(y_true, pred_label, zero_division=0),
        "recall_buy": recall_score(y_true, pred_label, zero_division=0),
        "f1_buy": f1_score(y_true, pred_label, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "positive_rate_actual": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(pred_label)),
        "avg_probability_positive": float(np.mean(prob_positive)),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob_positive)
    except ValueError:
        out["roc_auc"] = np.nan
    return out


def train_models_for_ticker(df: pd.DataFrame, ticker: str):
    one = df[df["Ticker"] == ticker].sort_values("Date").copy()
    split = max(int(len(one) * 0.8), len(one) - 252)
    train = one.iloc[:split]
    valid = one.iloc[split:]
    X_train = train[FEATURE_COLUMNS]
    X_valid = valid[FEATURE_COLUMNS]
    y_train = train["target_return_120d"]
    y_valid = valid["target_return_120d"]
    y_train_label = y_train > 0
    y_valid_label = y_valid > 0

    horizon_models = {}
    horizon_predictions = {}
    for horizon, target_col in [(20, "target_return_20d"), (60, "target_return_60d"), (120, "target_return_120d")]:
        rf_h = _rf_model()
        cat_h = _catboost_model()
        rf_h.fit(X_train, train[target_col])
        cat_h.fit(X_train, train[target_col])
        horizon_models[horizon] = {"rf": rf_h, "cat": cat_h}
        horizon_predictions[horizon] = {
            "rf": rf_h.predict(X_valid),
            "cat": cat_h.predict(X_valid),
        }

    rf = horizon_models[120]["rf"]
    cat = horizon_models[120]["cat"]
    cat_cls = _catboost_classifier()
    cat_cls.fit(X_train, y_train_label)
    rf_pred = horizon_predictions[120]["rf"]
    cat_pred = horizon_predictions[120]["cat"]
    cat_cls_prob = cat_cls.predict_proba(X_valid)[:, 1]
    ens_pred = 0.45 * cat_pred + 0.35 * rf_pred + 0.20 * np.mean([rf_pred, cat_pred], axis=0)
    metrics = [
        evaluate_prediction(ticker, "RandomForest", y_valid, rf_pred),
        evaluate_prediction(ticker, "CatBoostRegressor", y_valid, cat_pred),
        evaluate_prediction(ticker, "Ensemble", y_valid, ens_pred),
    ]
    classifier_metrics = evaluate_classifier(ticker, y_valid_label, cat_cls_prob)
    test_predictions = valid[["Date", "Ticker", "Close"]].copy()
    test_predictions["Input_Date"] = pd.to_datetime(test_predictions["Date"])
    test_predictions["Date"] = test_predictions["Input_Date"] + pd.offsets.BDay(HORIZON_DAYS)
    test_predictions["Actual"] = test_predictions["Close"] * (1 + y_valid.to_numpy())
    test_predictions["RF_Predicted"] = test_predictions["Close"] * (1 + rf_pred)
    test_predictions["CatBoost_Predicted"] = test_predictions["Close"] * (1 + cat_pred)
    test_predictions["Ensemble_Predicted"] = test_predictions["Close"] * (1 + ens_pred)
    test_predictions["Predicted"] = test_predictions["Ensemble_Predicted"]
    test_predictions["Model"] = "Ensemble"
    test_predictions = test_predictions.drop(columns=["Close"])

    importance_rows = []
    if hasattr(rf, "feature_importances_"):
        importance_rows.extend(
            {"ticker": ticker, "model": "RandomForest", "feature": feature, "importance": importance}
            for feature, importance in zip(FEATURE_COLUMNS, rf.feature_importances_)
        )
    if hasattr(cat, "get_feature_importance"):
        importance_rows.extend(
            {"ticker": ticker, "model": "CatBoostRegressor", "feature": feature, "importance": importance}
            for feature, importance in zip(FEATURE_COLUMNS, cat.get_feature_importance())
        )
    if hasattr(cat_cls, "get_feature_importance"):
        importance_rows.extend(
            {"ticker": ticker, "model": "CatBoostClassifier", "feature": feature, "importance": importance}
            for feature, importance in zip(FEATURE_COLUMNS, cat_cls.get_feature_importance())
        )

    return rf, cat, cat_cls, horizon_models, metrics, classifier_metrics, test_predictions, importance_rows


def predict_latest_return(model, latest_features: pd.DataFrame) -> float:
    return float(model.predict(latest_features[FEATURE_COLUMNS])[0])


def predict_latest_positive_probability(model, latest_features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(latest_features[FEATURE_COLUMNS])[:, 1][0])
    return float(model.predict(latest_features[FEATURE_COLUMNS])[0])
