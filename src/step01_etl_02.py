# src/step01_etl_02.py

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import polars as pl
import yfinance as yf

# 讓你用「python src/step01_etl_02.py」或「streamlit run ...」都能 import src.*
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.step00_model import MertonSolver  # noqa: E402


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ETL] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------
TRADING_DAYS = 252

# 主 DD：維持 20D，但把 20D vol 做平滑化
VOL20_WINDOW = 20
SMOOTH_LAMBDA = 0.94  # level EWMA smoothing on volatility series

# 疊圖版本
VOL63_WINDOW = 63
VOL252_WINDOW = 252

# 用報酬算的 EWMA vol（RiskMetrics）
EWMA_LAMBDA_RET = 0.94
EWMA_INIT_WINDOW = 63

# sigma 保護（避免極端值導致求解器不穩）
SIGMA_FLOOR = 0.05
SIGMA_CAP = 3.00

# Merton maturity
MATURITY_YEARS = 1.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class RateLimiter:
    """API 請求守門員：避免請求太快被鎖。"""

    @staticmethod
    def sleep_random(min_seconds: float = 1.2, max_seconds: float = 2.8) -> None:
        time.sleep(random.uniform(min_seconds, max_seconds))


class MarketDataFetcher:
    """負責與外部 API 通訊，包含 Cache 機制。"""

    def __init__(self) -> None:
        self.raw_path = Path("data/01_raw")
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def _check_cache(self, filename: str, max_age_hours: int = 24) -> Optional[pl.DataFrame]:
        fp = self.raw_path / filename
        if not fp.exists():
            return None
        mtime = datetime.fromtimestamp(fp.stat().st_mtime)
        age = datetime.now() - mtime
        if age < timedelta(hours=max_age_hours):
            logger.info(f"📦 Cache hit for {filename} (Age: {age.total_seconds()/3600:.1f}h)")
            return pl.read_parquet(fp)
        return None

    def _save_cache(self, df: pl.DataFrame, filename: str) -> None:
        fp = self.raw_path / filename
        df.write_parquet(fp)
        logger.info(f"💾 Saved cache to {fp}")

    def fetch_stock_history(self, ticker: str, period: str = "5y") -> pl.DataFrame:
        filename = f"{ticker}_history.parquet"
        cached = self._check_cache(filename)
        if cached is not None:
            return cached

        RateLimiter.sleep_random()
        logger.info(f"🌐 Fetching stock data for {ticker} ({period}) from yfinance...")

        stock = yf.Ticker(ticker)
        pdf = stock.history(period=period)
        if pdf is None or pdf.empty:
            raise ValueError(f"No data found for {ticker}")

        pdf = pdf.reset_index()
        df = (
            pl.from_pandas(pdf)
            .select(
                pl.col("Date").cast(pl.Date).alias("date"),
                pl.col("Close").cast(pl.Float64).alias("close"),
                pl.col("Volume").cast(pl.Int64).alias("volume"),
            )
            .sort("date")
        )
        self._save_cache(df, filename)
        return df

    def fetch_financials(self, ticker: str) -> Dict[str, float]:
        """sharesOutstanding / totalDebt 若缺，用 fallback。"""
        RateLimiter.sleep_random(0.8, 1.8)
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            shares = info.get("sharesOutstanding")
            if not shares:
                mcap = info.get("marketCap")
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if mcap and price:
                    shares = mcap / price

            debt = info.get("totalDebt")
            if not debt:
                logger.warning(f"⚠️ Debt info missing for {ticker}, using fallback=1000.0")
                debt = 1000.0

            return {
                "shares": float(shares) if shares else 1e9,
                "total_debt": float(debt),
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch financials: {e}")
            return {"shares": 1e9, "total_debt": 1.0}

    def fetch_risk_free_rate_us(self) -> pl.DataFrame:
        """抓取美股無風險利率 (^IRX)"""
        filename = "risk_free_irx.parquet"
        cached = self._check_cache(filename, max_age_hours=24)
        if cached is not None:
            return cached

        RateLimiter.sleep_random()
        logger.info("🌐 Fetching US Risk-Free Rate (^IRX) from yfinance...")

        try:
            irx = yf.Ticker("^IRX")
            pdf = irx.history(period="2y")
            if pdf is None or pdf.empty:
                raise ValueError("Empty data for ^IRX")

            pdf = pdf.reset_index()
            df = (
                pl.from_pandas(pdf)
                .select(
                    pl.col("Date").cast(pl.Date).alias("date"),
                    (pl.col("Close") / 100).cast(pl.Float64).alias("risk_free_rate"),
                )
                .sort("date")
                .with_columns(pl.col("risk_free_rate").fill_null(strategy="forward"))
            )
            self._save_cache(df, filename)
            return df
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch ^IRX ({e})")
            return pl.DataFrame({"date": [], "risk_free_rate": []})


def _collect_streaming(q: pl.LazyFrame) -> pl.DataFrame:
    """相容 polars 新舊版本：streaming -> engine='streaming'"""
    try:
        return q.collect(engine="streaming")
    except TypeError:
        return q.collect(streaming=True)


def clip_sigma_np(x: np.ndarray, low: float = SIGMA_FLOOR, high: float = SIGMA_CAP) -> np.ndarray:
    y = x.astype(float, copy=True)
    y[~np.isfinite(y)] = np.nan
    y = np.where(np.isfinite(y), np.clip(y, low, high), np.nan)
    return y


def ewma_smooth_level(x: np.ndarray, lam: float = SMOOTH_LAMBDA) -> np.ndarray:
    """對波動率序列做 EWMA level smoothing：y_t = lam*y_{t-1} + (1-lam)*x_t"""
    y = np.full(len(x), np.nan, dtype=float)
    idx = np.where(np.isfinite(x))[0]
    if len(idx) == 0:
        return y
    t0 = idx[0]
    y[t0] = float(x[t0])
    for t in range(t0 + 1, len(x)):
        if np.isfinite(x[t]):
            y[t] = lam * y[t - 1] + (1.0 - lam) * float(x[t])
        else:
            y[t] = y[t - 1]
    return y


def ewma_vol_from_returns(
    r: np.ndarray,
    lam: float = EWMA_LAMBDA_RET,
    init_window: int = EWMA_INIT_WINDOW,
    trading_days: int = TRADING_DAYS,
) -> np.ndarray:
    """
    RiskMetrics: var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2
    回傳年化 vol: sqrt(var_t * trading_days)
    """
    n = len(r)
    var = np.full(n, np.nan, dtype=float)

    valid = np.where(np.isfinite(r))[0]
    if len(valid) < init_window + 2:
        return np.full(n, np.nan, dtype=float)

    start = valid[0]
    init_end = min(n, start + init_window + 1)
    init_slice = r[start:init_end]
    init_slice = init_slice[np.isfinite(init_slice)]
    if len(init_slice) < 2:
        return np.full(n, np.nan, dtype=float)

    init_var = float(np.var(init_slice, ddof=1))
    t0 = init_end - 1
    var[t0] = init_var

    for t in range(t0 + 1, n):
        vprev = var[t - 1]
        if not np.isfinite(vprev):
            vprev = init_var

        rprev = r[t - 1]
        if np.isfinite(rprev):
            var[t] = lam * vprev + (1.0 - lam) * (rprev * rprev)
        else:
            var[t] = vprev

    return np.sqrt(var * trading_days)


def solve_merton_compat(equity: float, debt: float, vol: float, rf: float, maturity: float = MATURITY_YEARS):
    """
    相容不同版本的 MertonSolver.solve 參數命名：
    - 優先嘗試你的原本命名：vol_equity / risk_free [file:4]
    - 其次嘗試 volequity / riskfree
    - 最後嘗試位置參數
    """
    try:
        return MertonSolver.solve(equity=equity, debt=debt, vol_equity=vol, risk_free=rf, maturity=maturity)
    except TypeError:
        try:
            return MertonSolver.solve(equity=equity, debt=debt, volequity=vol, riskfree=rf, maturity=maturity)
        except TypeError:
            return MertonSolver.solve(equity, debt, vol, rf, maturity)


def pick_attr(res: Any, *names: str, default=np.nan):
    for n in names:
        if hasattr(res, n):
            return getattr(res, n)
    return default


def solve_row_full(s: dict) -> dict:
    """主輸出（欄位名保持原本，避免 app/viz 壞掉）"""
    try:
        E = float(s["equity_value"])
        D = float(s["total_debt"])
        vol = float(s["volatility_equity"])
        rf = float(s["risk_free_rate"])

        if (not np.isfinite(vol)) or vol <= 0:
            raise ValueError("bad vol")

        res = solve_merton_compat(E, D, vol, rf, MATURITY_YEARS)

        return {
            "solved_asset_value": float(pick_attr(res, "asset_value", "assetvalue")),
            "solved_asset_vol": float(pick_attr(res, "asset_volatility", "assetvolatility")),
            "distance_to_default": float(pick_attr(res, "distance_to_default", "distancetodefault")),
            "default_prob": float(pick_attr(res, "default_prob", "defaultprob")),
            "credit_spread_bps": float(pick_attr(res, "credit_spread", "creditspread")),
            "leverage_ratio": float(pick_attr(res, "leverage_ratio", "leverageratio")),
            "solver_converged": bool(pick_attr(res, "converged", default=False)),
        }
    except Exception:
        return {
            "solved_asset_value": np.nan,
            "solved_asset_vol": np.nan,
            "distance_to_default": np.nan,
            "default_prob": np.nan,
            "credit_spread_bps": np.nan,
            "leverage_ratio": np.nan,
            "solver_converged": False,
        }


def make_dd_solver(vol_col: str):
    """只回傳 DD（給疊圖用）"""
    def _f(s: dict) -> float:
        try:
            E = float(s["equity_value"])
            D = float(s["total_debt"])
            rf = float(s["risk_free_rate"])
            vol = float(s[vol_col])

            if (not np.isfinite(vol)) or vol <= 0:
                return np.nan

            res = solve_merton_compat(E, D, vol, rf, MATURITY_YEARS)
            return float(pick_attr(res, "distance_to_default", "distancetodefault"))
        except Exception:
            return np.nan
    return _f


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class MertonPipeline:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.fetcher = MarketDataFetcher()
        self.output_path = Path("data/03_primary")
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run(self, save_parquet: bool = True) -> pl.DataFrame:
        # 1) Stock + Financials
        df_stock = self.fetcher.fetch_stock_history(self.ticker)
        fin = self.fetcher.fetch_financials(self.ticker)
        shares = fin["shares"]
        debt_scalar = fin["total_debt"]

        # 2) Risk-free selector
        is_tw_stock = (".TW" in self.ticker) or (".TWO" in self.ticker)
        if is_tw_stock:
            logger.info(f"🇹🇼 Detected Taiwan Stock ({self.ticker}). Using constant Risk-Free Rate: 1.5%")
            q = df_stock.lazy().with_columns(pl.lit(0.015).alias("risk_free_rate"))
        else:
            df_rf = self.fetcher.fetch_risk_free_rate_us()
            if df_rf.is_empty():
                logger.warning("⚠️ ^IRX fetch failed, fallback to constant 4.0%")
                q = df_stock.lazy().with_columns(pl.lit(0.04).alias("risk_free_rate"))
            else:
                logger.info(f"🇺🇸 Applied Market Risk-Free Rate (^IRX) for {self.ticker}")
                q = (
                    df_stock.lazy()
                    .sort("date")
                    .join_asof(df_rf.lazy().sort("date"), on="date", strategy="backward")
                    .with_columns(pl.col("risk_free_rate").fill_null(0.04))
                )

        # 3) Feature engineering
        q = q.with_columns(
            (pl.col("close") * pl.lit(shares)).alias("equity_value"),
            pl.lit(debt_scalar).alias("total_debt"),
            pl.col("close").log().diff().alias("log_return"),
        )

        # 4) Rolling vols (annualized)
        q = q.with_columns(
            (pl.col("log_return").rolling_std(window_size=VOL20_WINDOW) * np.sqrt(TRADING_DAYS)).alias("volatility_equity_20_raw"),
            (pl.col("log_return").rolling_std(window_size=VOL63_WINDOW) * np.sqrt(TRADING_DAYS)).alias("volatility_equity_63"),
            (pl.col("log_return").rolling_std(window_size=VOL252_WINDOW) * np.sqrt(TRADING_DAYS)).alias("volatility_equity_252"),
        ).filter(pl.col("volatility_equity_20_raw").is_not_null())

        df = _collect_streaming(q).sort("date")

        # 5) EWMA vol from returns (annualized)
        rets = df.get_column("log_return").to_numpy()
        vol_ewma = ewma_vol_from_returns(rets)

        # 6) 主 volatility_equity：平滑 20D raw
        vol20_raw = df.get_column("volatility_equity_20_raw").to_numpy()
        vol20_smooth = ewma_smooth_level(vol20_raw, lam=SMOOTH_LAMBDA)

        # 7) clip/floor/cap
        vol20_raw = clip_sigma_np(vol20_raw)
        vol20_smooth = clip_sigma_np(vol20_smooth)
        vol63 = clip_sigma_np(df.get_column("volatility_equity_63").to_numpy())
        vol252 = clip_sigma_np(df.get_column("volatility_equity_252").to_numpy())
        vol_ewma = clip_sigma_np(vol_ewma)

        df = df.with_columns(
            pl.Series("volatility_equity_20_raw", vol20_raw),
            pl.Series("volatility_equity_20", vol20_smooth),
            pl.Series("volatility_equity_63", vol63),
            pl.Series("volatility_equity_252", vol252),
            pl.Series("volatility_equity_ewma", vol_ewma),
            # 主欄位：保留欄位名，讓你原本 app/viz 不用改
            pl.Series("volatility_equity", vol20_smooth),
        )

        # 8) Solve Merton main（輸出欄位名保持原本）
        out_dtype = pl.Struct(
            [
                pl.Field("solved_asset_value", pl.Float64),
                pl.Field("solved_asset_vol", pl.Float64),
                pl.Field("distance_to_default", pl.Float64),
                pl.Field("default_prob", pl.Float64),
                pl.Field("credit_spread_bps", pl.Float64),
                pl.Field("leverage_ratio", pl.Float64),
                pl.Field("solver_converged", pl.Boolean),
            ]
        )

        logger.info("🧮 Solving Merton main (smoothed 20D as volatility_equity) ...")
        df = df.with_columns(
            pl.struct(["equity_value", "total_debt", "volatility_equity", "risk_free_rate"])
            .map_elements(solve_row_full, return_dtype=out_dtype)
            .alias("merton_main")
        ).unnest("merton_main")

        # 9) Solve DD overlays（只做 DD，給疊圖用）
        logger.info("🧮 Solving DD overlays (20_raw / 63 / 252 / ewma) ...")
        df = df.with_columns(
            pl.struct(["equity_value", "total_debt", "volatility_equity_20_raw", "risk_free_rate"])
            .map_elements(make_dd_solver("volatility_equity_20_raw"), return_dtype=pl.Float64)
            .alias("distance_to_default_20_raw"),

            pl.struct(["equity_value", "total_debt", "volatility_equity_63", "risk_free_rate"])
            .map_elements(make_dd_solver("volatility_equity_63"), return_dtype=pl.Float64)
            .alias("distance_to_default_63"),

            pl.struct(["equity_value", "total_debt", "volatility_equity_252", "risk_free_rate"])
            .map_elements(make_dd_solver("volatility_equity_252"), return_dtype=pl.Float64)
            .alias("distance_to_default_252"),

            pl.struct(["equity_value", "total_debt", "volatility_equity_ewma", "risk_free_rate"])
            .map_elements(make_dd_solver("volatility_equity_ewma"), return_dtype=pl.Float64)
            .alias("distance_to_default_ewma"),
        )

        # 10) Save
        if save_parquet:
            save_path = self.output_path / f"{self.ticker}_merton_history.parquet"
            df.write_parquet(save_path)
            logger.info(f"✅ Data saved to {save_path}")

        return df


if __name__ == "__main__":
    print("🚀 Running ETL ...")
    MertonPipeline("2330.TW").run(save_parquet=True)
