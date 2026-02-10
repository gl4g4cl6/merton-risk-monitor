# src/step02_viz_10.py

import polars as pl
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pathlib import Path

from step00_model import MertonSolver


class MertonDashboard:
    def __init__(self, ticker: str, data_dir: str = "data/03_primary"):
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.df = None
        self.latest = None

    def load_data(self):
        file_path = self.data_dir / f"{self.ticker}_merton_history.parquet"
        if not file_path.exists():
            return False
        self.df = pl.read_parquet(file_path).sort("date")
        self.latest = self.df.tail(1).to_dicts()[0]
        return True

    def _get_scale(self, value):
        if value >= 1e12:
            return 1e12, "T"
        if value >= 1e9:
            return 1e9, "B"
        if value >= 1e6:
            return 1e6, "M"
        return 1, ""

    def _col_exists(self, col: str) -> bool:
        return (self.df is not None) and (col in self.df.columns)

    # -------------------------------------------------------------------------
    # 1. 信用結構（保留原本 Equity/Implied Debt；DD 改疊圖）
    # -------------------------------------------------------------------------
    def plot_credit_structure_history(self):
        """【1. 信用結構】"""
        d = self.df
        scale, unit = self._get_scale(d["solved_asset_value"].max())

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 堆疊面積：資產構成（維持原本：使用主 solved_asset_value）
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["equity_value"] / scale,
                name="權益價值 (Equity)",
                stackgroup="one",
                fillcolor="rgba(46, 204, 113, 0.5)",
                line=dict(width=0),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=(d["solved_asset_value"] - d["equity_value"]) / scale,
                name="隱含債務 (Implied Debt)",
                stackgroup="one",
                fillcolor="rgba(231, 76, 60, 0.5)",
                line=dict(width=0),
            ),
            secondary_y=False,
        )

        # DD 疊圖（若欄位不存在就自動略過）
        dd_specs = [
            ("distance_to_default", "DD (主：平滑20D)", dict(color="#f1c40f", width=3)),
            ("distance_to_default_20_raw", "DD (20D raw)", dict(color="#95a5a6", width=1.6, dash="dot")),
            ("distance_to_default_63", "DD (63D)", dict(color="#9b59b6", width=2)),
            ("distance_to_default_252", "DD (252D)", dict(color="#3498db", width=2)),
            ("distance_to_default_ewma", "DD (EWMA)", dict(color="#e67e22", width=2)),
        ]

        for col, name, style in dd_specs:
            if col in d.columns:
                fig.add_trace(
                    go.Scatter(x=d["date"], y=d[col], name=name, line=style),
                    secondary_y=True,
                )

        fig.update_layout(
            title="1. 信用結構歷史走勢 (DD 疊圖：20/63/252/EWMA)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
            height=470,
            template="plotly_dark",
        )
        fig.update_yaxes(title_text=f"價值 ({unit})", secondary_y=False)
        fig.update_yaxes(title_text="DD (Sigma)", secondary_y=True)
        return fig

    # -------------------------------------------------------------------------
    # 2. 資產分佈（維持原本：主 solved_asset_value / solved_asset_vol）
    # -------------------------------------------------------------------------
    def plot_asset_distribution(self):
        """【2. 資產分佈】"""
        d = self.latest
        A, D, sigma_A = d["solved_asset_value"], d["total_debt"], d["solved_asset_vol"]
        r, T = d["risk_free_rate"], 1.0

        mu = np.log(A) + (r - 0.5 * sigma_A**2) * T
        sigma = sigma_A * np.sqrt(T)

        x = np.linspace(max(0.01, A * 0.1), A * 3.0, 1000)
        valid_mask = (x > 0) & np.isfinite(np.log(x))
        x = x[valid_mask]

        pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)

        scale, unit = self._get_scale(A)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x / scale,
                y=pdf,
                name="資產價值分佈",
                fill="tozeroy",
                fillcolor="rgba(52, 152, 219, 0.3)",
                line=dict(color="#3498db"),
            )
        )
        fig.add_vline(x=D / scale, line_dash="dash", line_color="#e74c3c", annotation_text="債務門檻", annotation_position="top left")
        fig.add_vline(x=A / scale, line_color="#2ecc71", annotation_text="當前資產", annotation_position="top right")

        fig.update_layout(
            title="2. 資產價值分佈 (1Y Horizon)",
            xaxis_title=f"資產價值 ({unit})",
            yaxis_title="機率密度",
            template="plotly_dark",
            height=400,
        )
        return fig

    # -------------------------------------------------------------------------
    # 3. DD 指針（維持原本：主 distance_to_default）
    # -------------------------------------------------------------------------
    def plot_gauge_dd(self):
        """【3. 違約距離】指針"""
        dd = self.latest["distance_to_default"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=dd,
                delta={"reference": 3, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e74c3c"}},
                title={"text": "3. 違約距離 (DD)"},
                gauge={
                    "axis": {"range": [0, 10], "tickwidth": 1},
                    "bar": {"color": "white"},
                    "steps": [
                        {"range": [0, 1.5], "color": "#e74c3c"},
                        {"range": [1.5, 3], "color": "#f1c40f"},
                        {"range": [3, 10], "color": "#2ecc71"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": dd},
                },
            )
        )
        fig.update_layout(height=350, template="plotly_dark")
        return fig

    # -------------------------------------------------------------------------
    # 4. 資本結構（維持原本）
    # -------------------------------------------------------------------------
    def plot_capital_structure(self):
        """【4. 資本結構】"""
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["權益 (Equity)", "負債 (Debt)"],
                    values=[self.latest["equity_value"], self.latest["total_debt"]],
                    hole=0.5,
                    marker_colors=["#2ecc71", "#e74c3c"],
                )
            ]
        )
        fig.update_layout(title="4. 資本結構", template="plotly_dark")
        return fig

    # -------------------------------------------------------------------------
    # 5. 股價（維持原本）
    # -------------------------------------------------------------------------
    def plot_stock_history(self):
        """【5. 股價】"""
        fig = px.line(self.df, x="date", y="close", title="5. 股價歷史走勢")
        fig.update_layout(template="plotly_dark")
        return fig

    # -------------------------------------------------------------------------
    # 6. 波動率（改成疊圖）
    # -------------------------------------------------------------------------
    def plot_rolling_volatility(self):
        """【6. 波動率】"""
        d = self.df
        fig = go.Figure()

        vol_specs = [
            ("volatility_equity", "主：平滑20D (volatility_equity)", dict(color="#f1c40f", width=3)),
            ("volatility_equity_20_raw", "20D raw", dict(color="#95a5a6", width=1.6, dash="dot")),
            ("volatility_equity_63", "63D", dict(color="#9b59b6", width=2)),
            ("volatility_equity_252", "252D", dict(color="#3498db", width=2)),
            ("volatility_equity_ewma", "EWMA (from returns)", dict(color="#e67e22", width=2)),
        ]

        for col, name, style in vol_specs:
            if col in d.columns:
                fig.add_trace(go.Scatter(x=d["date"], y=d[col], name=name, line=style))

        fig.update_layout(
            title="6. 權益波動率疊圖 (annualized)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
            height=380,
            template="plotly_dark",
        )
        return fig

    # -------------------------------------------------------------------------
    # 7. 風險熱力圖（維持原本：solve_grid + 主 volatility_equity）
    # -------------------------------------------------------------------------
    def plot_risk_heatmap(self):
        """【7. 壓力測試熱力圖】修正：解鎖寬度 + Log Scale"""
        grid = MertonSolver.solve_grid(
            self.latest["equity_value"],
            self.latest["total_debt"],
            self.latest["volatility_equity"],
            self.latest["risk_free_rate"],
        )

        fig = px.imshow(
            grid["z_pd"] * 100,
            x=grid["x_debt_mults"],
            y=grid["y_vol_mults"],
            title="7. 風險熱力圖 (PD %)",
            color_continuous_scale="RdYlGn_r",
            labels={"x": "債務倍數 (Debt x)", "y": "波動倍數 (Vol x)"},
        )
        fig.update_layout(template="plotly_dark", height=450)
        return fig

    # -------------------------------------------------------------------------
    # 8. 債務敏感度（維持原本：主 volatility_equity；用位置參數避免命名差異）
    # -------------------------------------------------------------------------
    def plot_sens_debt(self):
        """【8. 債務敏感度】"""
        debt_mults = np.linspace(0.5, 2.0, 50)
        pds = [
            MertonSolver.solve(
                self.latest["equity_value"],
                self.latest["total_debt"] * m,
                self.latest["volatility_equity"],
                self.latest["risk_free_rate"],
            ).default_prob
            * 100
            for m in debt_mults
        ]
        fig = px.line(x=debt_mults, y=pds, title="8. 債務敏感度")
        fig.update_layout(template="plotly_dark", xaxis_title="債務倍數", yaxis_title="PD (%)")
        return fig

    # -------------------------------------------------------------------------
    # 9. 波動敏感度（維持原本：主 volatility_equity）
    # -------------------------------------------------------------------------
    def plot_sens_vol(self):
        """【9. 波動敏感度】"""
        vol_mults = np.linspace(0.5, 2.5, 50)
        pds = [
            MertonSolver.solve(
                self.latest["equity_value"],
                self.latest["total_debt"],
                self.latest["volatility_equity"] * m,
                self.latest["risk_free_rate"],
            ).default_prob
            * 100
            for m in vol_mults
        ]
        fig = px.line(x=vol_mults, y=pds, title="9. 波動敏感度")
        fig.update_layout(template="plotly_dark", xaxis_title="波動倍數", yaxis_title="PD (%)")
        return fig

    # -------------------------------------------------------------------------
    # 10. 摘要（維持原本欄位）
    # -------------------------------------------------------------------------
    def get_summary_metrics(self):
        """【10. 摘要】"""
        return {
            "股價 (Price)": f"{self.latest['close']:.1f}",
            "權益價值 (Equity)": f"{self.latest['equity_value']:,.0f}",
            "資產價值 (Asset)": f"{self.latest['solved_asset_value']:,.0f}",
            "總債務 (Debt)": f"{self.latest['total_debt']:,.0f}",
            "資產波動率 (Asset Vol)": f"{self.latest['solved_asset_vol']:.1%}",
            "權益波動率 (Equity Vol)": f"{self.latest['volatility_equity']:.1%}",
            "違約距離 (DD)": f"{self.latest['distance_to_default']:.2f}",
            "違約機率 (PD)": f"{self.latest['default_prob']:.4%}",
            "信用價差 (Spread bps)": f"{self.latest['credit_spread_bps']:.0f}",
            "槓桿比率 (Leverage)": f"{self.latest['leverage_ratio']:.1%}",
        }
