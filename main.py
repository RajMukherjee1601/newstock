# app.py
"""
Flask API: 7-day BSE close-price forecast for multiple tickers (UPL, ONGC, etc.)
More-accurate version vs plain normal model:
- Uses Adjusted prices (auto_adjust=True) from yfinance
- Fits a market factor model: r_stock = a + b * r_mkt + residual
- Simulates future returns via bootstrap sampling (market returns + residuals)
- Outputs Median/Mean + 68% and 95% empirical bands (percentiles from simulations)

Endpoints:
- GET  /health
- POST /forecast           (custom payload)
- GET  /forecast/default   (runs with defaults)

Run:
  pip install -r requirements.txt
  python app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, render_template

# -------------------- DEFAULT CONFIG --------------------
DEFAULT_BSE_TICKERS = {
    "UPL": "UPL.BO",
    "ONGC": "ONGC.BO",
    "HDFCBANK": "HDFCBANK.BO",
    "ICICIBANK": "ICICIBANK.BO",
    "BRIGADE": "BRIGADE.BO",
    "SUZLON": "SUZLON.BO",
    "SBIN": "SBIN.BO",
}

DEFAULT_MARKET_TICKER = "^BSESN"   # Sensex on Yahoo
DEFAULT_FORECAST_DAYS = 7
DEFAULT_WINDOW_VOL = 60
DEFAULT_WINDOW_BETA = 252
DEFAULT_HISTORY_PERIOD = "3y"
DEFAULT_N_SIMS = 20000
DEFAULT_SEED = 42
DEFAULT_AUTO_ADJUST = True

# Cache yfinance downloads to reduce repeated calls
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes


# -------------------- SIMPLE CACHE --------------------
@dataclass
class CacheEntry:
    ts: float
    data: pd.DataFrame

_YF_CACHE: Dict[Tuple, CacheEntry] = {}


def _cache_get(key: Tuple) -> pd.DataFrame | None:
    ent = _YF_CACHE.get(key)
    if not ent:
        return None
    if time.time() - ent.ts > CACHE_TTL_SECONDS:
        _YF_CACHE.pop(key, None)
        return None
    return ent.data


def _cache_set(key: Tuple, df: pd.DataFrame) -> None:
    _YF_CACHE[key] = CacheEntry(ts=time.time(), data=df)


# -------------------- CORE MATH --------------------
def log_returns_from_prices(px: pd.Series) -> pd.Series:
    px = px.dropna().sort_index()
    return np.log(px / px.shift(1)).dropna()


def fit_market_model(stock_r: pd.Series, mkt_r: pd.Series, beta_window: int) -> Tuple[float, float, pd.Series]:
    """
    OLS fit over last beta_window overlapping returns:
      r_stock = a + b * r_mkt + e
    Returns (a, b, residuals_series)
    """
    df = pd.concat([stock_r, mkt_r], axis=1, join="inner")
    df.columns = ["rs", "rm"]
    df = df.dropna()
    if len(df) < max(50, beta_window // 3):
        raise ValueError(f"Not enough overlapping return history (rows={len(df)}).")

    dfw = df.tail(beta_window)
    x = dfw["rm"].values
    y = dfw["rs"].values

    x_mean = x.mean()
    y_mean = y.mean()
    var = np.mean((x - x_mean) ** 2)
    cov = np.mean((x - x_mean) * (y - y_mean))

    b = cov / var if var > 0 else 0.0
    a = y_mean - b * x_mean

    resid = dfw["rs"] - (a + b * dfw["rm"])
    return float(a), float(b), resid


def next_trading_days(last_dt: pd.Timestamp, n: int, extra_holidays: set) -> List[pd.Timestamp]:
    """
    Simple Mon–Fri calendar with optional extra holidays (YYYY-MM-DD dates).
    Note: BSE has specific holidays—add them via extra_holidays if desired.
    """
    days = []
    cur = pd.Timestamp(last_dt).normalize()
    while len(days) < n:
        cur += pd.Timedelta(days=1)
        if cur.weekday() >= 5:
            continue
        if cur.date().isoformat() in extra_holidays:
            continue
        days.append(cur)
    return days


def bootstrap_factor_paths(
    P0: float,
    a: float,
    b: float,
    mkt_returns: pd.Series,
    resid_returns: pd.Series,
    vol_window: int,
    horizon: int,
    n_sims: int,
    seed: int
) -> np.ndarray:
    """
    Simulate prices via bootstrapped factor model:
      r_stock = a + b * r_mkt + e
    where r_mkt sampled from last vol_window market returns,
          e sampled from residuals (beta_window residuals).
    Returns: prices shape (horizon, n_sims)
    """
    rng = np.random.default_rng(seed)

    mkt_pool = mkt_returns.dropna().tail(vol_window).values
    resid_pool = resid_returns.dropna().values

    if len(mkt_pool) < 10:
        raise ValueError("Not enough market returns in vol window.")
    if len(resid_pool) < 50:
        raise ValueError("Not enough residuals to sample from.")

    mkt_draws = rng.choice(mkt_pool, size=(horizon, n_sims), replace=True)
    resid_draws = rng.choice(resid_pool, size=(horizon, n_sims), replace=True)

    r_stock = a + b * mkt_draws + resid_draws
    factors = np.exp(r_stock)
    paths = P0 * np.cumprod(factors, axis=0)
    return paths


def bootstrap_direct_paths(
    P0: float,
    stock_returns: pd.Series,
    vol_window: int,
    horizon: int,
    n_sims: int,
    seed: int
) -> np.ndarray:
    """
    Fallback: directly bootstrap stock returns from last vol_window returns.
    """
    rng = np.random.default_rng(seed)
    pool = stock_returns.dropna().tail(vol_window).values
    if len(pool) < 10:
        raise ValueError("Not enough stock returns in vol window.")
    draws = rng.choice(pool, size=(horizon, n_sims), replace=True)
    paths = P0 * np.cumprod(np.exp(draws), axis=0)
    return paths


def summarize_paths(paths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Empirical summary from simulations (no normal assumption):
    - median/mean
    - 68% band: 16th–84th percentile
    - 95% band: 2.5th–97.5th percentile
    """
    return {
        "median": np.median(paths, axis=1),
        "mean": paths.mean(axis=1),
        "low68": np.percentile(paths, 16, axis=1),
        "high68": np.percentile(paths, 84, axis=1),
        "low95": np.percentile(paths, 2.5, axis=1),
        "high95": np.percentile(paths, 97.5, axis=1),
    }


# -------------------- DATA FETCH --------------------
def fetch_closes(tickers: list[str], period: str, auto_adjust: bool) -> pd.DataFrame:
    """
    Fetch Close series for multiple tickers.
    If auto_adjust=True, Yahoo returns adjusted prices as 'Close'.
    Returns DataFrame with columns = tickers.
    """
    key = (tuple(sorted(tickers)), period, auto_adjust)
    cached = _cache_get(key)
    if cached is not None:
        return cached.copy()

    df = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",   # keep this
        threads=True
    )

    if df.empty:
        raise ValueError("yfinance returned empty dataframe (no data).")

    # ✅ Correct parsing:
    # With group_by="column", for multi-ticker downloads the structure is:
    # df["Close"] is a DataFrame whose columns are tickers.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise ValueError(f"'Close' not found in downloaded data. Columns: {df.columns.levels[0].tolist()}")
        closes = df["Close"].copy()
    else:
        # Single ticker case
        closes = df[["Close"]].rename(columns={"Close": tickers[0]})

    closes = closes.dropna(how="all")

    # Optional: keep only requested tickers in correct order
    closes = closes[[t for t in tickers if t in closes.columns]]

    _cache_set(key, closes)
    return closes.copy()



# -------------------- FORECAST ENGINE --------------------
def forecast_multi(
    ticker_map: Dict[str, str],
    market_ticker: str,
    forecast_days: int,
    window_vol: int,
    window_beta: int,
    history_period: str,
    n_sims: int,
    seed: int,
    auto_adjust: bool,
    extra_holidays: set
) -> Dict[str, Any]:

    all_tickers = list(ticker_map.values()) + [market_ticker]
    closes = fetch_closes(all_tickers, history_period, auto_adjust=auto_adjust)

    out_all = {}
    warnings = []

    # Try market returns (but DO NOT crash if missing)
    use_market = market_ticker in closes.columns and closes[market_ticker].dropna().shape[0] > 50
    if use_market:
        mkt_px = closes[market_ticker].dropna()
        mkt_r = log_returns_from_prices(mkt_px)
    else:
        mkt_r = None
        warnings.append(f"Market ticker '{market_ticker}' unavailable. Using direct bootstrap for all stocks.")

    for name, tkr in ticker_map.items():
        if tkr not in closes.columns:
            warnings.append(f"{name} ({tkr}): missing price series.")
            continue

        px = closes[tkr].dropna()
        if len(px) < window_vol + 5:
            warnings.append(f"{name} ({tkr}): not enough closes ({len(px)}) for window_vol={window_vol}.")
            continue

        stock_r = log_returns_from_prices(px)
        P0 = float(px.iloc[-1])
        last_dt = px.index[-1]

        used_method = "direct_bootstrap"
        alpha = beta = None

        # If market exists and we have enough history, try factor model
        if use_market and len(px) >= (window_beta + 10):
            try:
                alpha, beta, resid = fit_market_model(stock_r, mkt_r, beta_window=window_beta)
                paths = bootstrap_factor_paths(
                    P0=P0,
                    a=alpha,
                    b=beta,
                    mkt_returns=mkt_r,
                    resid_returns=resid,
                    vol_window=window_vol,
                    horizon=forecast_days,
                    n_sims=n_sims,
                    seed=seed
                )
                used_method = "factor_bootstrap"
            except Exception as e:
                warnings.append(f"{name} ({tkr}): factor model failed -> fallback ({str(e)})")
                paths = bootstrap_direct_paths(
                    P0=P0,
                    stock_returns=stock_r,
                    vol_window=window_vol,
                    horizon=forecast_days,
                    n_sims=n_sims,
                    seed=seed
                )
        else:
            # Direct bootstrap fallback
            paths = bootstrap_direct_paths(
                P0=P0,
                stock_returns=stock_r,
                vol_window=window_vol,
                horizon=forecast_days,
                n_sims=n_sims,
                seed=seed
            )

        summ = summarize_paths(paths)
        f_days = next_trading_days(pd.Timestamp(last_dt), forecast_days, extra_holidays)

        rows = []
        for i, dt in enumerate(f_days):
            rows.append({
                "date": dt.date().isoformat(),
                "pred_close_median": round(float(summ["median"][i]), 2),
                "pred_close_mean": round(float(summ["mean"][i]), 2),
                "low_68": round(float(summ["low68"][i]), 2),
                "high_68": round(float(summ["high68"][i]), 2),
                "low_95": round(float(summ["low95"][i]), 2),
                "high_95": round(float(summ["high95"][i]), 2),
            })

        out_all[name] = {
            "ticker": tkr,
            "method": used_method,
            "last_close": round(P0, 2),
            "last_close_date": pd.Timestamp(last_dt).date().isoformat(),
            "window_vol": window_vol,
            "window_beta": window_beta,
            "market_ticker": market_ticker,
            "alpha": None if alpha is None else round(alpha, 8),
            "beta": None if beta is None else round(beta, 4),
            "n_sims": n_sims,
            "auto_adjust": auto_adjust,
            "forecast_days": forecast_days,
            "forecast": rows
        }

    return {"results": out_all, "warnings": warnings}



# -------------------- FLASK APP --------------------
app = Flask(__name__)
from flask_cors import CORS
CORS(app)
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/forecast/default")
def forecast_default():
    try:
        payload = {
            "tickers": DEFAULT_BSE_TICKERS,
            "market_ticker": DEFAULT_MARKET_TICKER,
            "forecast_days": DEFAULT_FORECAST_DAYS,
            "window_vol": DEFAULT_WINDOW_VOL,
            "window_beta": DEFAULT_WINDOW_BETA,
            "history_period": DEFAULT_HISTORY_PERIOD,
            "n_sims": DEFAULT_N_SIMS,
            "seed": DEFAULT_SEED,
            "auto_adjust": DEFAULT_AUTO_ADJUST,
            "extra_holidays": []  # add "YYYY-MM-DD" strings if needed
        }
        result = forecast_multi(
            ticker_map=payload["tickers"],
            market_ticker=payload["market_ticker"],
            forecast_days=int(payload["forecast_days"]),
            window_vol=int(payload["window_vol"]),
            window_beta=int(payload["window_beta"]),
            history_period=str(payload["history_period"]),
            n_sims=int(payload["n_sims"]),
            seed=int(payload["seed"]),
            auto_adjust=bool(payload["auto_adjust"]),
            extra_holidays=set(payload["extra_holidays"]),
        )
        return jsonify({"config": payload, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/forecast")
def forecast_custom():
    """
    JSON body example:
    {
      "tickers": {
        "UPL":"UPL.BO",
        "ONGC":"ONGC.BO"
      },
      "market_ticker":"^BSESN",
      "forecast_days":7,
      "window_vol":60,
      "window_beta":252,
      "history_period":"3y",
      "n_sims":20000,
      "seed":42,
      "auto_adjust":true,
      "extra_holidays":["2025-12-25"]
    }
    """
    try:
        payload = request.get_json(force=True) or {}

        ticker_map = payload.get("tickers", DEFAULT_BSE_TICKERS)
        if not isinstance(ticker_map, dict) or not ticker_map:
            raise ValueError("Field 'tickers' must be a non-empty object/dict.")

        market_ticker = payload.get("market_ticker", DEFAULT_MARKET_TICKER)
        forecast_days = int(payload.get("forecast_days", DEFAULT_FORECAST_DAYS))
        window_vol = int(payload.get("window_vol", DEFAULT_WINDOW_VOL))
        window_beta = int(payload.get("window_beta", DEFAULT_WINDOW_BETA))
        history_period = str(payload.get("history_period", DEFAULT_HISTORY_PERIOD))
        n_sims = int(payload.get("n_sims", DEFAULT_N_SIMS))
        seed = int(payload.get("seed", DEFAULT_SEED))
        auto_adjust = bool(payload.get("auto_adjust", DEFAULT_AUTO_ADJUST))
        extra_holidays = set(payload.get("extra_holidays", []))

        if forecast_days < 1 or forecast_days > 30:
            raise ValueError("forecast_days must be between 1 and 30.")
        if window_vol < 10 or window_vol > 500:
            raise ValueError("window_vol must be between 10 and 500.")
        if window_beta < 50 or window_beta > 2000:
            raise ValueError("window_beta must be between 50 and 2000.")
        if n_sims < 1000 or n_sims > 200000:
            raise ValueError("n_sims must be between 1000 and 200000.")

        result = forecast_multi(
            ticker_map=ticker_map,
            market_ticker=market_ticker,
            forecast_days=forecast_days,
            window_vol=window_vol,
            window_beta=window_beta,
            history_period=history_period,
            n_sims=n_sims,
            seed=seed,
            auto_adjust=auto_adjust,
            extra_holidays=extra_holidays,
        )

        return jsonify({
            "config": {
                "tickers": ticker_map,
                "market_ticker": market_ticker,
                "forecast_days": forecast_days,
                "window_vol": window_vol,
                "window_beta": window_beta,
                "history_period": history_period,
                "n_sims": n_sims,
                "seed": seed,
                "auto_adjust": auto_adjust,
                "extra_holidays": sorted(list(extra_holidays)),
            },
            **result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    # For production, use gunicorn: gunicorn -w 2 -b 0.0.0.0:5000 app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
