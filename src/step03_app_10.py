# src/step03_app_10.py

from pathlib import Path
import sys

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Fix import path for: streamlit run src/step03_app_10.py
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from step02_viz_10 import MertonDashboard
from step01_etl_02 import MertonPipeline


def _trace_names(fig):
    return [t.name for t in fig.data if getattr(t, "name", None)]


def _set_trace_visible(fig, visible_names: set[str], predicate=None):
    for t in fig.data:
        name = getattr(t, "name", "") or ""
        if predicate is not None and not predicate(name):
            continue
        t.visible = (name in visible_names)
    return fig


def _checkbox_grid(title: str, names: list[str], key_prefix: str, default_on=None, expanded: bool = False) -> set[str]:
    if default_on is None:
        default_on = set(names)

    with st.expander(title, expanded=expanded):
        if len(names) == 0:
            st.caption("沒有可勾選的線。")
            return set()

        cols = st.columns(min(5, max(1, len(names))))
        chosen = set()
        for i, name in enumerate(names):
            with cols[i % len(cols)]:
                v0 = name in default_on
                if st.checkbox(name, value=v0, key=f"{key_prefix}_{name}"):
                    chosen.add(name)
        return chosen


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Merton Risk Monitor", layout="wide")
st.markdown("""""", unsafe_allow_html=True)

st.title("🛡️Merton Model: Credit Risk Dashboard ")
st.markdown("**Credit Risk System with Interpretations**")

# Top bar
col_input, col_status = st.columns([1, 3])
with col_input:
    input_ticker = st.text_input(
        "🔍 輸入股票代號 (Enter Ticker)",
        value="2330.TW",
        help="支援台股 (2330.TW) 與美股 (NVDA)",
    ).upper()

data_dir = Path("data/03_primary")
data_dir.mkdir(parents=True, exist_ok=True)

viz = MertonDashboard(input_ticker)
is_data_loaded = viz.load_data()

with col_status:
    if not is_data_loaded:
        with st.status(f"⚙️ 正在運算 {input_ticker} 的 Merton 模型...", expanded=True) as status:
            try:
                st.write("連線 Yahoo Finance 下載財報數據...")
                pipeline = MertonPipeline(input_ticker)
                pipeline.run(save_parquet=True)

                st.write("解算微分方程與資產路徑...")
                status.update(label="✅ 運算完成！", state="complete", expanded=False)

                if not viz.load_data():
                    st.error("❌ 數據載入失敗，請檢查代號是否正確。")
                    st.stop()

            except Exception as e:
                st.error(f"❌ 發生錯誤: {str(e)}")
                st.stop()
    else:
        st.success(f"✅ {input_ticker} 數據已載入 (Data Ready)")

st.divider()

latest = viz.latest

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("當前股價", f"{latest['close']:.1f}")

dd = latest["distance_to_default"]
k2.metric(
    "違約距離 (DD)",
    f"{dd:.2f}",
    delta="安全" if dd > 3 else "警戒",
    delta_color="normal" if dd > 3 else "inverse",
)

k3.metric("槓桿比率", f"{latest['leverage_ratio']:.1%}")
k4.metric("資產波動率", f"{latest['solved_asset_vol']:.1%}")

# -----------------------------------------------------------------------------
# Row 1: Credit structure + DD overlay (with checkboxes for DD lines)
# -----------------------------------------------------------------------------
fig_cs = viz.plot_credit_structure_history()

# 只控制 DD 線（不動 Equity/Implied Debt 面積）
dd_names = [n for n in _trace_names(fig_cs) if "DD" in str(n)]
default_dd_on = set(dd_names) - {n for n in dd_names if "20D raw" in str(n)}  # 20D raw 預設關掉

chosen_dd = _checkbox_grid(
    "1) 信用結構圖：勾選要顯示的 DD 線",
    dd_names,
    key_prefix="cs_dd",
    default_on=default_dd_on,
    expanded=False,
)

_set_trace_visible(fig_cs, chosen_dd, predicate=lambda name: "DD" in str(name))

st.plotly_chart(fig_cs, width="stretch")

st.caption(
    """
**💡 圖解說明：** 這張圖展示了公司的「隱藏資產負債表」。

* **綠色區域 (Equity)**：這是我們看得到的市值。
* **紅色區域 (Implied Debt)**：這是模型推算出的債務壓力。
* **DD 線**：代表「距離破產還有多遠」。如果 DD 向下跌破 0，代表資產價值已不足以償債。
"""
)

# -----------------------------------------------------------------------------
# Row 2: Structure / distribution
# -----------------------------------------------------------------------------
st.subheader("📊 風險結構分析")
c1, c2, c3 = st.columns([3, 2, 2])

with c1:
    st.plotly_chart(viz.plot_asset_distribution(), width="stretch")
    st.caption(
        """
**💡 資產分佈預測：** 這是預測「一年後」公司資產價值的機率分佈。

* **綠線**：現在的位置。
* **紅虛線**：債務違約點 (Default Point)。
* **藍色面積**：資產價值落在紅線左側的機率，就是違約率 (PD)。
"""
    )

with c2:
    st.plotly_chart(viz.plot_gauge_dd(), width="stretch")
    st.caption(
        """
**💡 違約距離 (Sigma)：** 簡單來說，這是公司的「安全氣囊」厚度。

* **> 3**：非常安全。
* **< 1.5**：高風險，隨時可能違約。
"""
    )

with c3:
    st.plotly_chart(viz.plot_capital_structure(), width="stretch")
    st.caption("**💡 資本結構：** 直接比較市值與負債的比例。綠色越多，體質越強健。")

# -----------------------------------------------------------------------------
# Row 3: Market data
# -----------------------------------------------------------------------------
st.divider()
c4, c5 = st.columns(2)

with c4:
    st.plotly_chart(viz.plot_stock_history(), width="stretch")
    st.caption("**💡 股價走勢：** 模型的原始輸入數據之一，反映市場情緒。")

with c5:
    fig_vol = viz.plot_rolling_volatility()
    vol_names = _trace_names(fig_vol)

    # 預設把 20D raw 關掉
    default_vol_on = set(vol_names) - {n for n in vol_names if "20D raw" in str(n)}
    chosen_vol = _checkbox_grid(
        "6) 權益波動率圖：勾選要顯示的線",
        vol_names,
        key_prefix="vol",
        default_on=default_vol_on,
        expanded=False,
    )

    _set_trace_visible(fig_vol, chosen_vol)

    st.plotly_chart(fig_vol, width="stretch")
    st.caption(
        """
**💡 波動率監控：** 波動率是 Merton 模型最敏感的參數。

* 波動率飆高 = 資產價值不確定性大增 = 違約風險上升。
"""
    )

# -----------------------------------------------------------------------------
# Row 4: Stress test
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🔥 壓力測試實驗室 (Stress Test)")

c6, c7, c8 = st.columns([3, 2, 2])

with c6:
    st.plotly_chart(viz.plot_risk_heatmap(), width="stretch")

with c7:
    st.plotly_chart(viz.plot_sens_debt(), width="stretch")

with c8:
    st.plotly_chart(viz.plot_sens_vol(), width="stretch")

# -----------------------------------------------------------------------------
# Row 5: Summary
# -----------------------------------------------------------------------------
with st.expander("📊 查看詳細數據摘要 (Summary Statistics)", expanded=False):
    metrics = viz.get_summary_metrics()
    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
