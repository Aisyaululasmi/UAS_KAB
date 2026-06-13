import numpy as np
import pandas as pd
import os

from .config import HORIZON_DAYS

_TIMESFM_MODEL = None
_TIMESFM_ERROR = None


def _fallback_forecast(values: np.ndarray, horizon: int, reason: str) -> tuple[np.ndarray, str]:
    recent = values[-252:] if len(values) >= 252 else values
    log_ret = np.diff(np.log(recent))
    drift = np.nanmedian(log_ret[-60:]) if len(log_ret) >= 60 else np.nanmedian(log_ret)
    drift = float(np.clip(np.nan_to_num(drift, nan=0.0), -0.002, 0.002))
    steps = np.arange(1, horizon + 1)
    forecast = values[-1] * np.exp(drift * steps)
    return forecast, f"TimesFM 2.5 Fallback ({reason})"


def _get_timesfm_model(horizon: int):
    global _TIMESFM_MODEL, _TIMESFM_ERROR

    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL
    if _TIMESFM_ERROR is not None:
        raise RuntimeError(_TIMESFM_ERROR)

    try:
        import timesfm
        from huggingface_hub import hf_hub_download

        local_files_only = os.getenv("TIMESFM_LOCAL_FILES_ONLY", "1") != "0"
        checkpoint_path = hf_hub_download(
            repo_id="google/timesfm-2.5-200m-pytorch",
            filename="model.safetensors",
            local_files_only=local_files_only,
        )
        model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        model.model.load_checkpoint(checkpoint_path, torch_compile=False)
        model.compile(
            timesfm.ForecastConfig(
                max_context=2048,
                max_horizon=horizon,
                normalize_inputs=True,
                per_core_batch_size=1,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        _TIMESFM_MODEL = model
        return _TIMESFM_MODEL
    except Exception as exc:
        _TIMESFM_ERROR = str(exc)
        raise


def timesfm_forecast(close: pd.Series, horizon: int = HORIZON_DAYS) -> tuple[np.ndarray, str]:
    """Forecast with real TimesFM 2.5 when available.

    If package loading, checkpoint loading, or inference fails, use a conservative
    drift forecast and return a label that documents the failure reason.
    """
    values = close.dropna().astype(float).to_numpy()
    if len(values) < 64:
        return _fallback_forecast(values, horizon, "insufficient history")

    try:
        model = _get_timesfm_model(horizon)
        point_forecast, _ = model.forecast(horizon, [values])
        forecast = np.asarray(point_forecast[0], dtype=float)[:horizon]
        if len(forecast) != horizon or not np.isfinite(forecast).all():
            raise RuntimeError("TimesFM returned invalid forecast values")
        return forecast, "TimesFM 2.5 200M PyTorch"
    except Exception as exc:
        reason = str(exc).splitlines()[0][:160] or exc.__class__.__name__
        return _fallback_forecast(values, horizon, reason)
