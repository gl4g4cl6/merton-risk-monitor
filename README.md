🛡️ Merton Risk Monitor：信用風險結構化監控系統

本專案開發於 Windows 11 (WSL 2 Ubuntu 24.04) 環境。
這是一個端到端的信用風險評估工具，旨在透過 Merton 結構型模型 (Merton Structural Model)，
將市場股價資訊轉化為企業的違約機率 (PD) 與違約距離 (DD)。

💡 建構基礎 (The Essence)

將公司股權視為對公司總資產的買權 (Call Option)，
藉此推導出隱含的資產價值與波動率，以衡量企業償債能力的邊際安全性。

📖 核心理論與公式 (The Theory)

根據 Merton (1974)，
公司股權價值 $E$ 可表示為：

$$E = A \cdot N(d_1) - D e^{-rT} \cdot N(d_2)$$

其中關鍵參數定義如下：

資產價值與風險路徑：

$$d_1 = \frac{\ln(A/D) + (r + 0.5 \sigma_A^2)T}{\sigma_A \sqrt{T}}$$

$$d_2 = d_1 - \sigma_A \sqrt{T}$$

違約距離 (Distance to Default)：

$$DD = \frac{\ln(A/D) + (r - 0.5 \sigma_A^2)T}{\sigma_A \sqrt{T}}$$

🛠️ 技術架構 (System Architecture)
專案採模組化設計，強制分離邏輯、數據與表現層：
模組
檔案名稱
技術重點
*** 核心模型 ***
step00_model.py
使用 scipy.optimize.fsolve 進行非線性聯立方程求解。
*** 數據工程 ***
step01_etl_02.py
Polars LazyFrame 驅動，支持高效能數據血統分析與多維波動率計算。
*** 視覺化 ***
step02_viz_10.py
基於 Plotly 的信用結構疊圖與壓力測試熱力圖。
*** 應用介面 ***
step03_app_10.py
Streamlit 互動式儀表板，支援即時 Ticker 數據抓取與風險監控。

🚀 執行指南 (Actionable Steps)
環境需求
Python 3.10+
WSL2 (建議) 或 Windows/Mac 環境
安裝步驟
複製專案：
Bash
git clone <your-repo-url>
cd MERTON_MODEL_V2


安裝依賴套件：
Bash
pip install -r requirements.txt


啟動監控儀表板：
Bash
streamlit run src/step03_app_10.py


🛡️ 開發者檢查哨 (Developer Checkpoints)
效能優化：所有數據處理均強制採用 collect(streaming=True) 以確保單機 HPC 效能 。
數據驅動：禁止使用 pl.Date.today()，所有邏輯嚴格遵循歷史數據路徑。
誤差分析：內建數據檢核機制，確保求解器收斂後之資產價值具備學術級誤差一致性。
