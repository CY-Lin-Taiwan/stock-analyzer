"""
Metrics - 進階指標計算
========================
- 夏普值 (Sharpe Ratio): 個股 / 組合
- 布林通道 (Bollinger Bands): 個股位階

設計原則:
  - 計算簡潔,失敗回 None 而非拋 exception
  - 不抓資料,只計算(資料由 caller 提供)
  - 提供「現價在布林帶哪個位置」給 AI 用
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


# === 參數 ===
RISK_FREE_RATE = 0.015         # 無風險利率: 1.5% (台灣定存)
TRADING_DAYS_PER_YEAR = 252    # 一年交易日
BOLLINGER_PERIOD = 20          # 布林通道週期
BOLLINGER_STD = 2              # 標準差倍數

# === KD (台股慣用 9,3,3) ===
# 注意:台股 KD 是「遞歸平滑」,跟國外標準 stochastic(%D = %K 的 3 日 SMA)不同。
#   RSV = (C − L9) / (H9 − L9) × 100
#   K   = (1 − 1/3) × 前K + (1/3) × RSV
#   D   = (1 − 1/3) × 前D + (1/3) × K
# 用國外公式算出來的數字會跟券商 App 對不上。
KD_PERIOD = 9                  # RSV 回看天數
KD_SMOOTH = 3                  # 平滑常數(1/3)
KD_SEED = 50.0                 # 遞歸起始值
KD_MIN_BARS = 30               # 遞歸暖機:少於這麼多筆不給結果

# 交叉「有效區」門檻:
# K/D 在中間區纏繞的交叉是雜訊,只有發生在極端區的交叉才有參考價值。
# 為什麼用 30/70 而不是 20/80:K 是平滑後的值,等它由下往上穿越 D 時,
# 通常已經從谷底反彈到 20~35 之間了。用 20 會把絕大多數真實的低檔黃金交叉濾掉。
KD_CROSS_LOW = 30              # 黃金交叉發生在此值以下才算有效
KD_CROSS_HIGH = 70             # 死亡交叉發生在此值以上才算有效
KD_CROSS_MAX_AGE = 15          # 超過這麼多天的交叉視為過期,不再回報

# 交叉確認門檻:K/D 幾乎重疊時會出現「擦邊而過」的假交叉
# (實測有 K−D 只差 0.076 就被判為交叉的情況,那是數值噪音不是訊號)。
# 要求交叉後在確認窗內,K−D 差距至少擴大到這個幅度才算成立。
KD_CROSS_MIN_GAP = 1.0         # K/D 差距最小幅度
KD_CROSS_CONFIRM_BARS = 3      # 確認窗(交叉後幾根內要達到上述幅度)

# 鈍化:連續 N 天 K 值停在極端區
KD_BLUNT_HIGH = 80
KD_BLUNT_LOW = 20
KD_BLUNT_DAYS = 3              # 連續幾天才算鈍化

# === 大盤基準 ===
# KD 由價格驅動,而個股價格高度受市場因子影響。多檔同時出現同一訊號時,
# 那多半是「大盤的故事」而非「個股的故事」——
# 實測 2026-08-19:某使用者 7 檔持股有 6 檔死亡交叉,日期集中在 08-14 / 08-18,
# 而 0050 同期也死亡交叉。若不做對照,AI 會對 6 檔各自說「出現轉弱訊號」,
# 把 beta 誤讀成 alpha。
#
# 母體必須是固定的大盤基準,不能用「使用者的持股」:
#   1. 個人持股會變、檔數少,母體不穩定
#   2. 同一天同一檔股票,不同使用者會得到不同結論 —— 指標不該隨觀察者改變
#   3. 跨使用者統計會侵犯隱私(此工具有分享給親友)
MARKET_BENCHMARK = "0050"
MARKET_SYNC_TOLERANCE_DAYS = 5   # 個股與大盤交叉相差幾天內視為同步


# === Regime(市場狀態濾網)===
REGIME_SLOPE_DAYS = 10         # 中軌斜率的回看天數
REGIME_SLOPE_THRESHOLD = 2.0   # 中軌 N 日變化超過 ±這個 % 才算有方向
REGIME_BW_LOOKBACK = 252       # 帶寬百分位的回看範圍(約 1 年)
REGIME_SQUEEZE_PCTL = 20       # 帶寬低於一年的這個百分位 → squeeze(無法判斷)


# =============================================================
# 夏普值
# =============================================================
def calculate_total_returns(
    prices: pd.Series,
    dividends: Optional[Dict] = None,
) -> pd.Series:
    """
    計算「含息(還原)」日報酬序列。

    為什麼需要這個:
      資料庫存的是「原始收盤價」(FinMind taiwan_stock_daily),
      除息當天股價會跳空往下掉一整個股利的幅度。若直接用 pct_change(),
      那天會被當成一根大跌,把高配息股(如長榮)的報酬硬拉低、波動撐大,
      導致夏普值嚴重失真甚至變負。這裡把除息日的股利「加回」報酬以還原。

    Args:
        prices: pd.Series of close prices, index = 日期 (sorted asc)
        dividends: {ex_date(str, 'YYYY-MM-DD'): cash}  或
                   {ex_date(str): {"cash": float, "stock": float}}
                   - cash : 現金股利(元/股),加法還原
                   - stock: 股票股利(元,面額10),配股率 = stock/10,乘法還原

    Returns:
        含息日報酬 pd.Series (已 dropna)
    """
    if prices is None or len(prices) < 2:
        return pd.Series(dtype=float)

    p = prices.astype(float)
    r = p.pct_change()

    if dividends:
        # 把價格序列 index 正規化成 'YYYY-MM-DD' 字串以對齊除息日
        idx_str = [str(x)[:10] for x in p.index]
        pos = {d: i for i, d in enumerate(idx_str)}

        for ex_date, info in dividends.items():
            key = str(ex_date)[:10]
            i = pos.get(key)
            # 除息日不在交易日序列(例如用公告日暫代的)或在序列第一天 -> 跳過,安全
            if i is None or i == 0:
                continue

            if isinstance(info, dict):
                cash = float(info.get("cash", 0) or 0)
                stock = float(info.get("stock", 0) or 0)
            else:
                cash = float(info or 0)
                stock = 0.0

            if cash <= 0 and stock <= 0:
                continue

            prev = p.iloc[i - 1]
            if prev <= 0:
                continue

            g = stock / 10.0  # 股票股利(元) -> 每股配股率
            # 含息報酬:除息後持有 (1+g) 股、每股值 P_t,外加現金 cash
            r.iloc[i] = (p.iloc[i] * (1.0 + g) + cash - prev) / prev

    return r.dropna()


def calculate_sharpe(
    prices: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    annualize: bool = True,
    dividends: Optional[Dict] = None,
) -> Optional[float]:
    """
    計算夏普值
    
    Args:
        prices: pd.Series of close prices (sorted by date asc)
        risk_free_rate: 年化無風險利率 (預設 1.5%)
        annualize: 是否年化(預設 True)
        dividends: 除息資料 {ex_date: cash 或 {"cash":..,"stock":..}}。
                   有給就用「含息(還原)報酬」計算,避免高配息股失真;
                   不給則沿用原始價差報酬(向後相容)。
    
    Returns:
        夏普值(float),失敗回 None
    """
    if prices is None or len(prices) < 30:
        return None  # 至少 30 日才有意義
    
    try:
        # 日報酬率:有股利就用含息還原報酬,否則用原始價差
        if dividends:
            daily_returns = calculate_total_returns(prices, dividends)
        else:
            daily_returns = prices.pct_change().dropna()
        
        if len(daily_returns) < 30:
            return None
        
        # 日報酬率平均 / 標準差
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0 or pd.isna(std_return):
            return None
        
        # 換成年化
        if annualize:
            annual_return = mean_return * TRADING_DAYS_PER_YEAR
            annual_std = std_return * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            annual_return = mean_return
            annual_std = std_return
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        
        return round(float(sharpe), 2)
        
    except Exception as e:
        print(f"[metrics] 夏普值計算失敗: {e}")
        return None


def calculate_portfolio_sharpe(
    daily_prices_by_symbol: Dict[str, pd.Series],
    weights: Dict[str, float],
    risk_free_rate: float = RISK_FREE_RATE,
) -> Optional[float]:
    """
    計算投資組合夏普值
    
    Args:
        daily_prices_by_symbol: {"2603": pd.Series, "2330": pd.Series, ...}
        weights: {"2603": 0.4, "2330": 0.3, ...} (加總 = 1)
        risk_free_rate: 年化無風險利率
    
    Returns:
        組合夏普值,失敗回 None
    """
    if not daily_prices_by_symbol or not weights:
        return None
    
    try:
        # 把所有股票日報酬合併成 DataFrame
        returns_df = pd.DataFrame()
        for symbol, prices in daily_prices_by_symbol.items():
            if prices is None or len(prices) < 30:
                continue
            if symbol not in weights:
                continue
            returns = prices.pct_change().dropna()
            returns_df[symbol] = returns
        
        if returns_df.empty or len(returns_df) < 30:
            return None
        
        # 對齊日期(只保留所有股票都有資料的日期)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            return None
        
        # 加權平均日報酬
        weight_array = np.array([weights.get(col, 0) for col in returns_df.columns])
        weight_sum = weight_array.sum()
        if weight_sum == 0:
            return None
        weight_array = weight_array / weight_sum  # normalize
        
        # 組合日報酬
        portfolio_returns = returns_df.values @ weight_array
        
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        if std_return == 0:
            return None
        
        annual_return = mean_return * TRADING_DAYS_PER_YEAR
        annual_std = std_return * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        
        return round(float(sharpe), 2)
        
    except Exception as e:
        print(f"[metrics] 組合夏普值計算失敗: {e}")
        return None


def interpret_sharpe(sharpe: Optional[float]) -> str:
    """夏普值的口語解讀(簡短版,給 caption 用)"""
    if sharpe is None:
        return "資料不足"
    if sharpe < 0:
        return "報酬不如無風險利率,風險調整後表現差"
    elif sharpe < 1.0:
        return "裸奔狀態:承受的風險大於報酬"
    elif sharpe < 2.0:
        return "標準裝甲:優秀基金經理人合格線"
    else:
        return "降維打擊:獲利曲線極度平滑"


def get_sharpe_grade(sharpe: Optional[float]) -> Dict:
    """
    回傳夏普值的詳細分級
    
    Returns:
        Dict 包含:
          - tier: 0/1/2/3 對應四個級距
          - label: 級距名稱
          - color: 顏色標籤
          - description: 詳細描述
          - position_text: 「目前在 X 區間」的標示
    """
    if sharpe is None:
        return {
            "tier": -1,
            "label": "資料不足",
            "color": "gray",
            "description": "需要至少 30 個交易日資料",
            "position_text": "—",
        }
    
    if sharpe < 0:
        return {
            "tier": 0,
            "label": "負值區",
            "color": "red",
            "description": "報酬不如定存,風險調整後表現差。每承擔一單位風險,反而虧損。",
            "position_text": "🔴 目前在「負值區」(<0)",
        }
    elif sharpe < 1.0:
        return {
            "tier": 1,
            "label": "裸奔狀態",
            "color": "orange",
            "description": "承受的風險大於賺到的報酬。賺著賣白菜的錢,操著賣白粉的心,隨時會被市場一波帶走。",
            "position_text": "🟠 目前在「裸奔狀態」(0 ~ 1.0)",
        }
    elif sharpe < 2.0:
        return {
            "tier": 2,
            "label": "標準裝甲",
            "color": "green",
            "description": "一般優秀基金經理人的合格線。承擔一單位風險,能換到一單位以上的報酬。",
            "position_text": "🟢 目前在「標準裝甲」(1.0 ~ 2.0)",
        }
    else:
        return {
            "tier": 3,
            "label": "降維打擊",
            "color": "blue",
            "description": "頂級量化模型或完美的零負債對沖陣地。獲利曲線極度平滑,幾乎沒有回撤。",
            "position_text": "🔵 目前在「降維打擊」(>2.0)",
        }


def get_bollinger_grade(bb: Optional[Dict]) -> Dict:
    """
    回傳布林通道位階的詳細分級
    
    Returns:
        Dict 包含:
          - tier: 0~5 對應位階區間
          - label: 區間名稱
          - color: 顏色
          - description: 詳細解讀
          - position_text: 「目前在 X 區間」標示
          - trading_implication: 對應的市場含義
    """
    if not bb or bb.get("percent_b") is None:
        return {
            "tier": -1,
            "label": "資料不足",
            "color": "gray",
            "description": "需要至少 20 個交易日資料",
            "position_text": "—",
            "trading_implication": "—",
        }
    
    pb = bb["percent_b"]
    current = bb.get("current")
    upper = bb.get("upper")
    lower = bb.get("lower")
    
    if current > upper:
        return {
            "tier": 5,
            "label": "突破上軌",
            "color": "red",
            "description": f"現價 {current} > 上軌 {upper},已穿越 2σ 範圍。",
            "position_text": "🔴 目前「突破上軌」(極強區,%B > 100%)",
            "trading_implication": "市場過熱訊號,短期可能回檔(但強趨勢中可能繼續走高)。建議警覺,不適合追高。",
        }
    elif pb > 80:
        return {
            "tier": 4,
            "label": "靠近上軌",
            "color": "orange",
            "description": f"%B = {pb}%,距離上軌很近(80% ~ 100%)。",
            "position_text": "🟠 目前「靠近上軌」(強勢區,%B 80~100%)",
            "trading_implication": "短期強勢,但接近常態波動上緣。觀察是否會回測中軌,或形成突破。",
        }
    elif pb > 50:
        return {
            "tier": 3,
            "label": "中軌之上",
            "color": "green",
            "description": f"%B = {pb}%,在中軌與上軌之間(50% ~ 80%)。",
            "position_text": "🟢 目前「中軌之上」(偏強區,%B 50~80%)",
            "trading_implication": "走勢偏多,但尚未過熱。持有者可繼續觀察,等待突破或回測訊號。",
        }
    elif pb > 20:
        return {
            "tier": 2,
            "label": "中軌之下",
            "color": "yellow",
            "description": f"%B = {pb}%,在中軌與下軌之間(20% ~ 50%)。",
            "position_text": "🟡 目前「中軌之下」(偏弱區,%B 20~50%)",
            "trading_implication": "走勢偏空,但尚未到極度恐慌。觀察是否會回測下軌,或重新站上中軌。",
        }
    elif pb > 0:
        return {
            "tier": 1,
            "label": "靠近下軌",
            "color": "orange",
            "description": f"%B = {pb}%,距離下軌很近(0% ~ 20%)。",
            "position_text": "🟠 目前「靠近下軌」(弱勢區,%B 0~20%)",
            "trading_implication": "短期弱勢,接近常態波動下緣。觀察是否會反彈中軌,或進一步跌破下軌。",
        }
    else:
        return {
            "tier": 0,
            "label": "跌破下軌",
            "color": "red",
            "description": f"現價 {current} < 下軌 {lower},已穿越 2σ 範圍。",
            "position_text": "🔴 目前「跌破下軌」(極弱區,%B < 0%)",
            "trading_implication": "市場恐慌訊號,短期可能反彈(但崩跌趨勢中可能繼續探底)。建議警覺,不適合追低。",
        }


def interpret_bollinger(bb: Optional[Dict]) -> str:
    """布林通道的口語解讀(簡短版,給 caption 用)"""
    if bb is None:
        return "資料不足"
    
    pb = bb.get("percent_b")
    pos = bb.get("position", "")
    
    if pb is None:
        return pos
    
    return f"{pos} (帶寬位階 %B = {pb}%)"


# =============================================================
# 布林通道
# =============================================================
def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = BOLLINGER_PERIOD,
    std_multiplier: float = BOLLINGER_STD,
) -> Optional[Dict]:
    """
    計算布林通道(只回傳最新一筆)
    
    Args:
        prices: pd.Series of close prices (sorted by date asc)
        period: 移動平均期數(預設 20 日)
        std_multiplier: 標準差倍數(預設 2)
    
    Returns:
        Dict 包含:
          - middle: 中軌(20 日均線)
          - upper: 上軌(中軌 + 2σ)
          - lower: 下軌(中軌 - 2σ)
          - bandwidth: 帶寬 %  ((upper - lower) / middle * 100)
          - percent_b: 現價位階  ((current - lower) / (upper - lower) * 100)
                       0 = 在下軌,50 = 在中軌,100 = 在上軌
          - current: 現價
          - position: 文字描述「在上軌之上 / 上軌附近 / 中軌之上 / 中軌之下 / 下軌附近 / 跌破下軌」
        失敗回 None
    """
    if prices is None or len(prices) < period:
        return None
    
    try:
        # 計算 N 日均線跟標準差
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + std_multiplier * std
        lower = middle - std_multiplier * std
        
        # 取最新一筆
        m = float(middle.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])
        current = float(prices.iloc[-1])
        
        if pd.isna(m) or pd.isna(u) or pd.isna(l):
            return None
        
        bandwidth = (u - l) / m * 100 if m else None

        # 中軌斜率:regime 判定用。
        # 用「N 日百分比變化」而非絕對值,因為 30 元的股票跟 1000 元的股票
        # 中軌斜率的「元/日」完全不能比。
        middle_slope_pct = None
        if len(middle) > period + REGIME_SLOPE_DAYS:
            m_past = middle.iloc[-1 - REGIME_SLOPE_DAYS]
            if pd.notna(m_past) and m_past > 0:
                middle_slope_pct = (m / float(m_past) - 1.0) * 100
        
        # percent_b: 0% = 下軌, 50% = 中軌, 100% = 上軌
        if u - l > 0:
            percent_b = (current - l) / (u - l) * 100
        else:
            percent_b = None
        
        # 位階文字
        if current > u:
            position = "突破上軌(極強)"
        elif percent_b is not None and percent_b > 80:
            position = "靠近上軌(強勢)"
        elif percent_b is not None and percent_b > 50:
            position = "中軌之上(偏強)"
        elif percent_b is not None and percent_b > 20:
            position = "中軌之下(偏弱)"
        elif percent_b is not None and percent_b > 0:
            position = "靠近下軌(弱勢)"
        elif current < l:
            position = "跌破下軌(極弱)"
        else:
            position = "中性"
        
        return {
            "middle": round(m, 2),
            "upper": round(u, 2),
            "lower": round(l, 2),
            "bandwidth": round(bandwidth, 2) if bandwidth else None,
            "middle_slope_pct": round(middle_slope_pct, 2) if middle_slope_pct is not None else None,
            "percent_b": round(percent_b, 1) if percent_b is not None else None,
            "current": round(current, 2),
            "position": position,
        }
        
    except Exception as e:
        print(f"[metrics] 布林通道計算失敗: {e}")
        return None

# =============================================================
# KD (台股 9,3,3)
# =============================================================
def _kd_series(
    df: pd.DataFrame,
    period: int = KD_PERIOD,
    smooth: int = KD_SMOOTH,
):
    """
    計算完整 K、D 序列(內部用)。

    台股遞歸公式:
        RSV = (C − L9) / (H9 − L9) × 100
        K   = (1 − 1/smooth) × 前K + (1/smooth) × RSV
        D   = (1 − 1/smooth) × 前D + (1/smooth) × K

    Returns:
        (k: pd.Series, d: pd.Series),失敗回 (None, None)
    """
    try:
        high = df["high"].rolling(window=period).max()
        low = df["low"].rolling(window=period).min()
        rng = high - low

        # H9 == L9(整段完全平盤,例如長期停牌)→ RSV 無意義,給中性值
        rsv = pd.Series(KD_SEED, index=df.index, dtype=float)
        valid = rng > 0
        rsv[valid] = (df["close"][valid] - low[valid]) / rng[valid] * 100
        rsv = rsv.fillna(KD_SEED).clip(0, 100)

        a = 1.0 / smooth
        k_vals, d_vals = [], []
        k = d = KD_SEED
        for v in rsv:
            k = (1 - a) * k + a * float(v)
            d = (1 - a) * d + a * k
            k_vals.append(k)
            d_vals.append(d)

        return (pd.Series(k_vals, index=df.index),
                pd.Series(d_vals, index=df.index))
    except Exception as e:
        print(f"[metrics] KD 序列計算失敗: {e}")
        return None, None


def _find_last_cross(k: pd.Series, d: pd.Series) -> Optional[Dict]:
    """
    找最近一次「確認過」的 K/D 交叉。

    會跳過擦邊交叉:K 與 D 幾乎重疊時,K−D 可能只差 0.0x 就翻正負號,
    那是數值噪音。要求交叉後在確認窗內差距擴大到 KD_CROSS_MIN_GAP 才算成立。

    Returns:
        {"type": "golden"/"death", "days_ago": int, "level": float} 或 None
    """
    diff = (k - d).values
    n = len(diff)

    def confirmed(i: int, sign: int) -> bool:
        """交叉後 KD_CROSS_CONFIRM_BARS 根內,差距是否朝新方向擴大到足夠幅度"""
        window = diff[i:min(n, i + KD_CROSS_CONFIRM_BARS + 1)]
        if len(window) == 0:
            return False
        best = max(window) if sign > 0 else min(window)
        return abs(best) >= KD_CROSS_MIN_GAP

    for i in range(n - 1, 0, -1):
        prev, cur = diff[i - 1], diff[i]
        if prev <= 0 < cur and confirmed(i, +1):
            return {"type": "golden", "days_ago": n - 1 - i,
                    "level": float(k.iloc[i])}
        if prev >= 0 > cur and confirmed(i, -1):
            return {"type": "death", "days_ago": n - 1 - i,
                    "level": float(k.iloc[i])}
    return None


def _find_blunting(k: pd.Series) -> Dict:
    """
    偵測鈍化:連續 N 天 K 值停在極端區。

    鈍化很重要,因為它跟交叉是相反的意義:
    高檔鈍化代表強趨勢延續(此時的死亡交叉多半是假訊號),
    低檔鈍化代表跌勢未止(此時的黃金交叉同樣不可信)。
    """
    vals = k.values
    n = len(vals)

    high_days = 0
    for i in range(n - 1, -1, -1):
        if vals[i] > KD_BLUNT_HIGH:
            high_days += 1
        else:
            break

    low_days = 0
    for i in range(n - 1, -1, -1):
        if vals[i] < KD_BLUNT_LOW:
            low_days += 1
        else:
            break

    if high_days >= KD_BLUNT_DAYS:
        return {"state": "high", "days": high_days}
    if low_days >= KD_BLUNT_DAYS:
        return {"state": "low", "days": low_days}
    return {"state": "none", "days": 0}


def calculate_kd(
    df: pd.DataFrame,
    period: int = KD_PERIOD,
    smooth: int = KD_SMOOTH,
) -> Optional[Dict]:
    """
    計算台股 KD(只回傳最新狀態)。

    ⚠️ 必須餵「還原股價」。原始收盤價在除權息當天會跳空,
       KD 只有 9 天窗口,對缺口極度敏感 —— 長榮 2023-06-30 配息 70 元,
       原始 K = 23.6(看似深度超賣),還原後真實值 80.2(其實超買),
       訊號完全反向。請用 price_adjust.build_adjusted_ohlc() 的輸出。

    Args:
        df: 需含 high / low / close 的 DataFrame(按日期升冪,已還原)
        period: RSV 回看天數
        smooth: 平滑常數

    Returns:
        Dict:
          - k, d            : 最新 K、D 值
          - k_prev, d_prev  : 前一日 K、D(判斷剛剛是否交叉用)
          - zone            : "oversold"/"weak"/"strong"/"overbought"
          - cross           : "golden"/"death"/"none"
          - cross_days_ago  : 交叉距今幾個交易日
          - cross_level     : 交叉發生時的 K 值
          - cross_valid     : 交叉是否落在有效區(中間區交叉 = 雜訊)
          - blunting        : "high"/"low"/"none"
          - blunting_days   : 鈍化連續天數
        資料不足或失敗回 None
    """
    if df is None or len(df) < max(KD_MIN_BARS, period):
        return None
    for c in ("high", "low", "close"):
        if c not in df.columns:
            return None

    try:
        k, d = _kd_series(df, period=period, smooth=smooth)
        if k is None or len(k) < 2:
            return None

        k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])

        if k_now < KD_BLUNT_LOW:
            zone = "oversold"
        elif k_now < 50:
            zone = "weak"
        elif k_now < KD_BLUNT_HIGH:
            zone = "strong"
        else:
            zone = "overbought"

        cross = _find_last_cross(k, d)
        blunt = _find_blunting(k)

        result = {
            "k": round(k_now, 1),
            "d": round(d_now, 1),
            "k_prev": round(float(k.iloc[-2]), 1),
            "d_prev": round(float(d.iloc[-2]), 1),
            "zone": zone,
            "cross": "none",
            "cross_days_ago": None,
            "cross_level": None,
            "cross_valid": False,
            "blunting": blunt["state"],
            "blunting_days": blunt["days"],
        }

        if cross and cross["days_ago"] <= KD_CROSS_MAX_AGE:
            lv = cross["level"]
            valid = (lv <= KD_CROSS_LOW) if cross["type"] == "golden" \
                else (lv >= KD_CROSS_HIGH)
            result.update({
                "cross": cross["type"],
                "cross_days_ago": cross["days_ago"],
                "cross_level": round(lv, 1),
                "cross_valid": bool(valid),
            })

        return result

    except Exception as e:
        print(f"[metrics] KD 計算失敗: {e}")
        return None


def get_kd_grade(kd: Optional[Dict]) -> Dict:
    """
    KD 的白話分級。

    刻意做成「雙軸」而不是像布林那樣單軸 0~5,因為
    「K=15 剛黃金交叉」和「K=15 低檔鈍化第 8 天」意思完全相反,
    用一個 tier 表達不了。

    Returns:
        Dict:
          - zone_tier / zone_label / color / zone_text
          - momentum_state / momentum_label / momentum_text
          - description / trading_implication
    """
    if not kd:
        return {
            "zone_tier": -1, "zone_label": "資料不足", "color": "gray",
            "zone_text": "—",
            "momentum_state": "none", "momentum_label": "—", "momentum_text": "—",
            "description": f"需要至少 {KD_MIN_BARS} 個交易日資料(遞歸平滑要暖機)",
            "trading_implication": "—",
        }

    k, d = kd["k"], kd["d"]
    zone = kd["zone"]

    zone_map = {
        "oversold": (0, "超賣區", "red",
                     f"🔴 目前在「超賣區」(K {k} < {KD_BLUNT_LOW})"),
        "weak":     (1, "偏弱區", "yellow",
                     f"🟡 目前在「偏弱區」(K {k},{KD_BLUNT_LOW}~50)"),
        "strong":   (2, "偏強區", "green",
                     f"🟢 目前在「偏強區」(K {k},50~{KD_BLUNT_HIGH})"),
        "overbought": (3, "超買區", "orange",
                       f"🟠 目前在「超買區」(K {k} > {KD_BLUNT_HIGH})"),
    }
    zone_tier, zone_label, color, zone_text = zone_map[zone]

    # --- 動能軸 ---
    blunt = kd["blunting"]
    cross = kd["cross"]
    ago = kd["cross_days_ago"]
    lv = kd["cross_level"]

    if blunt == "high":
        m_state, m_label = "blunt_high", "高檔鈍化"
        m_text = f"⚡ 高檔鈍化第 {kd['blunting_days']} 天(K 連續站在 {KD_BLUNT_HIGH} 以上)"
        implication = ("強趨勢延續的訊號,不是該賣的訊號。鈍化期間出現的死亡交叉"
                       "多半是假訊號,真正的轉折要等鈍化結束。")
    elif blunt == "low":
        m_state, m_label = "blunt_low", "低檔鈍化"
        m_text = f"⚡ 低檔鈍化第 {kd['blunting_days']} 天(K 連續趴在 {KD_BLUNT_LOW} 以下)"
        implication = ("跌勢未止,不是「超賣就會反彈」。鈍化期間的黃金交叉不可信,"
                       "等 K 站回 20 以上再看。")
    elif cross == "golden":
        m_state, m_label = "golden", "黃金交叉"
        age = "今天" if ago == 0 else f"{ago} 天前"
        m_text = f"📈 {age}黃金交叉(交叉時 K={lv})"
        if kd["cross_valid"]:
            implication = (f"低檔黃金交叉,K 在 {KD_CROSS_LOW} 以下穿越 D,"
                           "是較有參考價值的短線轉強訊號。")
        elif lv and lv > KD_CROSS_HIGH:
            implication = (f"黃金交叉發生在 K={lv} 的高檔區,不是低檔轉強 —— "
                           f"這是追高訊號,參考價值低。")
        else:
            implication = (f"交叉發生在 K={lv},落在中間區纏繞,"
                           "屬於雜訊等級的交叉,參考價值低。")
    elif cross == "death":
        m_state, m_label = "death", "死亡交叉"
        age = "今天" if ago == 0 else f"{ago} 天前"
        m_text = f"📉 {age}死亡交叉(交叉時 K={lv})"
        if kd["cross_valid"]:
            implication = (f"高檔死亡交叉,K 在 {KD_CROSS_HIGH} 以上跌破 D,"
                           "是較有參考價值的短線轉弱訊號。")
        elif lv and lv < KD_CROSS_LOW:
            implication = (f"死亡交叉發生在 K={lv} 的低檔區,多為恐慌殺低 —— "
                           f"不是高檔轉弱,參考價值低。")
        else:
            implication = (f"交叉發生在 K={lv},落在中間區纏繞,"
                           "屬於雜訊等級的交叉,參考價值低。")
    else:
        m_state, m_label = "none", "無明顯訊號"
        m_text = "➖ 近期無交叉,也未鈍化"
        implication = "K/D 平行移動中,短線沒有明確轉折訊號,看位階本身即可。"

    return {
        "zone_tier": zone_tier,
        "zone_label": zone_label,
        "color": color,
        "zone_text": zone_text,
        "momentum_state": m_state,
        "momentum_label": m_label,
        "momentum_text": m_text,
        "description": f"K {k} / D {d} —— {zone_label},{m_label}",
        "trading_implication": implication,
    }


# =============================================================
# Regime(市場狀態濾網)
# =============================================================
def calculate_bandwidth_percentile(
    prices: pd.Series,
    period: int = BOLLINGER_PERIOD,
    std_multiplier: float = BOLLINGER_STD,
    lookback: int = REGIME_BW_LOOKBACK,
) -> Optional[float]:
    """
    現在的布林帶寬,在過去一年裡排第幾百分位。

    用百分位而非絕對值,因為「帶寬 8%」對台積電是擴張、
    對某些小型股是收縮 —— 絕對值無法跨股比較。
    (作法沿用 app.py 的 calc_pe_percentile / calc_pb_percentile)
    """
    if prices is None or len(prices) < period + 20:
        return None
    try:
        mid = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        bw = ((mid + std_multiplier * std) - (mid - std_multiplier * std)) / mid * 100
        bw = bw.dropna()
        if len(bw) < 20:
            return None

        hist = bw.tail(lookback)
        current = float(hist.iloc[-1])
        pctl = (hist <= current).sum() / len(hist) * 100
        return round(float(pctl), 1)
    except Exception as e:
        print(f"[metrics] 帶寬百分位計算失敗: {e}")
        return None


def get_regime(bb: Optional[Dict], bandwidth_pctl: Optional[float]) -> Dict:
    """
    用「中軌斜率 + 帶寬位階」判定市場狀態,當 KD 交叉的濾網。

    為什麼需要:
      %B 只回答「價格在通道的哪個位置」,沒有方向資訊。
      靠近上軌 + 帶寬擴張 + 中軌上揚 = 強勢突破;
      靠近上軌 + 帶寬收縮 + 中軌走平 = 箱型上緣。
      兩者意思相反,但 %B 都是 85% —— 少了 regime 無法區分。

    誠實的限制:
      這個濾網是落後的。它從同一個 20 日窗口衍生,只能在趨勢
      成形之後才辨認出趨勢,因此會擋掉部分「最早的正確訊號」。
      換來的是假訊號變少、漏訊號變多 —— 這是結構性取捨,
      沒有指標組合能繞過。定位是「確認型」而非「預測型」。

    Returns:
        {"state", "label", "text", "slope_pct", "bandwidth_pctl", "description"}
    """
    if not bb:
        return {"state": "unknown", "label": "資料不足", "text": "—",
                "slope_pct": None, "bandwidth_pctl": bandwidth_pctl,
                "description": "需要布林通道資料"}

    slope = bb.get("middle_slope_pct")

    if slope is None:
        return {"state": "unknown", "label": "資料不足", "text": "—",
                "slope_pct": None, "bandwidth_pctl": bandwidth_pctl,
                "description": f"中軌斜率需要至少 {BOLLINGER_PERIOD + REGIME_SLOPE_DAYS} 筆資料"}

    bw_desc = (f"帶寬位於一年 {bandwidth_pctl:.0f} 百分位"
               if bandwidth_pctl is not None else "帶寬位階未知")
    slope_desc = f"中軌 {REGIME_SLOPE_DAYS} 日 {slope:+.1f}%"
    compressed = bandwidth_pctl is not None and bandwidth_pctl <= REGIME_SQUEEZE_PCTL

    # 先判方向,再判 squeeze。
    # 帶寬壓縮但中軌明確上揚 = 低波動的多頭(是好環境),
    # 不能因為波動小就說「測不準」—— squeeze 必須是「沒方向 + 沒波動」。
    if slope >= REGIME_SLOPE_THRESHOLD:
        extra = "低波動" if compressed else bw_desc
        return {
            "state": "trending_up", "label": "上升趨勢", "slope_pct": slope,
            "bandwidth_pctl": bandwidth_pctl,
            "text": f"📈 上升趨勢({slope_desc},{extra})",
            "description": ("中軌明確上揚。趨勢盤中股價可能沿上軌走、KD 高檔鈍化數週,"
                            "此時的死亡交叉大多是假訊號。"),
        }

    if slope <= -REGIME_SLOPE_THRESHOLD:
        extra = "低波動" if compressed else bw_desc
        return {
            "state": "trending_down", "label": "下降趨勢", "slope_pct": slope,
            "bandwidth_pctl": bandwidth_pctl,
            "text": f"📉 下降趨勢({slope_desc},{extra})",
            "description": ("中軌明確下彎。趨勢盤中股價可能沿下軌走、KD 低檔鈍化,"
                            "此時的黃金交叉大多是假訊號。"),
        }

    # 中軌走平 —— 此時才看波動大小分辨「盤整」與「壓縮待變」
    if compressed:
        return {
            "state": "squeeze", "label": "帶寬收縮", "slope_pct": slope,
            "bandwidth_pctl": bandwidth_pctl,
            "text": f"🔒 帶寬收縮待變({bw_desc},{slope_desc})",
            "description": ("中軌走平且波動壓縮到一年低檔。此時位階與 KD 都在無意義擺動,"
                            "突破方向無法從技術面判斷 —— 應明確視為「測不準」。"
                            "值得留意的是 squeeze 之後常出現較大行情。"),
        }

    return {
        "state": "ranging", "label": "區間震盪", "slope_pct": slope,
        "bandwidth_pctl": bandwidth_pctl,
        "text": f"↔️ 區間震盪({slope_desc},{bw_desc})",
        "description": ("中軌走平,價格在區間內來回。這是 KD 極端區反轉訊號"
                        "最可靠的環境。"),
    }


def assess_kd_cross(kd: Optional[Dict], regime: Optional[Dict]) -> Dict:
    """
    用 regime 評估 KD 交叉的可信度。

    設計原則:濾網不「吃掉」訊號,只做降級標註。
    因為這裡沒有回測基礎設施,門檻是憑經驗定的 —— 若濾網默默吞掉訊號,
    使用者永遠學不到門檻該不該調,AI 也拿不到「這裡有訊號但可疑」
    這個本身有價值的 context。

    Returns:
        {"confidence": "高"/"中高"/"低"/"不適用", "reason": str, "signal": str}

        「不適用」= 目前沒有可評估的交叉(鈍化中、或近期無交叉),
        不等於「不可信」—— 兩者意義完全不同,UI 必須分開呈現。
    """
    if not kd:
        return {"confidence": "不適用", "reason": "無 KD 資料", "signal": "none"}

    blunt = kd.get("blunting", "none")
    cross = kd.get("cross", "none")
    valid = kd.get("cross_valid", False)
    state = (regime or {}).get("state", "unknown")

    if blunt in ("high", "low"):
        return {"confidence": "不適用", "signal": f"blunt_{blunt}",
                "reason": f"目前處於{'高' if blunt == 'high' else '低'}檔鈍化,"
                          "此階段不以交叉判斷,而是看趨勢延續"}

    if cross == "none":
        return {"confidence": "不適用", "signal": "none", "reason": "近期沒有交叉訊號可評估"}

    if not valid:
        # 「無效」有兩種完全不同的成因,不可都寫成「中間區纏繞」——
        # 實測出現過黃金交叉發生在 K=81.7 卻被說成「中間區」,81.7 是高檔區。
        lv = kd.get("cross_level") or 0
        if cross == "golden" and lv > KD_CROSS_HIGH:
            why = (f"黃金交叉發生在 K={lv} 的高檔區,不是低檔轉強 —— "
                   f"這是追高訊號,參考價值低")
        elif cross == "death" and lv < KD_CROSS_LOW:
            why = (f"死亡交叉發生在 K={lv} 的低檔區,多為恐慌殺低 —— "
                   f"不是高檔轉弱,參考價值低")
        else:
            why = f"交叉發生在 K={lv},落在中間區纏繞,屬雜訊等級"
        return {"confidence": "低", "signal": cross, "reason": why}

    if state == "squeeze":
        return {"confidence": "低", "signal": cross,
                "reason": "帶寬收縮盤整中,方向無法判斷,交叉參考價值低"}

    if state == "ranging":
        return {"confidence": "高", "signal": cross,
                "reason": "區間震盪 + 極端區交叉,這是 KD 最可靠的環境"}

    # 趨勢盤:順勢交叉可信,逆勢交叉是最常爆的地方
    with_trend = ((state == "trending_up" and cross == "golden") or
                  (state == "trending_down" and cross == "death"))
    if with_trend:
        return {"confidence": "中高", "signal": cross,
                "reason": f"{(regime or {}).get('label', '趨勢盤')}中的順勢交叉"}

    return {"confidence": "低", "signal": cross,
            "reason": (f"{(regime or {}).get('label', '趨勢盤')}中的逆勢交叉 —— "
                       "這是 KD+布林組合最常失效的情況,趨勢盤可能連錯數次")}


def build_tech_state(
    bb: Optional[Dict],
    regime: Optional[Dict],
    kd: Optional[Dict],
    market_ctx: Optional[Dict] = None,
    is_benchmark: bool = False,
) -> Dict:
    """
    把「位階 + regime + KD」合成一句話 + 一個可信度標籤。

    為什麼要合成而不是把三組原始數字都丟給 AI:
      1. %B 和 K 都在回答「價格在近期區間的哪個位置」,高度重疊。
         分開餵,AI 會把同一件事當成兩個獨立證據,假性提高信心度。
      2. 三個技術指標各自強制引用,會讓 AI 變成清單朗讀,
         稀釋 qualitative_summary 的深度。
      3. 判斷邏輯留在 Python 裡可被檢查,不是丟給 AI 自由心證。

    同一份 text 也顯示給人看 —— 使用者看得到濾網怎麼判斷,
    幾個月後才知道門檻該不該調。

    Args:
        market_ctx: 大盤基準狀態(build_market_context 的輸出)。
                    給了就會判斷個股訊號是否只是跟著大盤動。
        is_benchmark: 此標的本身就是大盤基準時設 True,跳過自我比對。

    Returns:
        {"text", "confidence", "cross_reason", "has_data",
         "market_sync", "market_note"}
    """
    parts = []

    if bb and bb.get("percent_b") is not None:
        parts.append(f"布林位階 %B {bb['percent_b']:.0f}%({bb.get('position', '')})")
    if regime and regime.get("state") not in (None, "unknown"):
        parts.append(regime["text"].split(" ", 1)[-1] if " " in regime["text"]
                     else regime["text"])
    if kd:
        seg = f"KD K={kd['k']} D={kd['d']}"
        if kd.get("blunting") in ("high", "low"):
            seg += (f",{'高' if kd['blunting'] == 'high' else '低'}檔鈍化"
                    f"第 {kd['blunting_days']} 天")
        elif kd.get("cross") in ("golden", "death"):
            ago = kd["cross_days_ago"]
            when = "今天" if ago == 0 else f"{ago} 天前"
            name = "黃金交叉" if kd["cross"] == "golden" else "死亡交叉"
            seg += f",{when}於 K={kd['cross_level']} {name}"
        parts.append(seg)

    if not parts:
        return {"text": "(技術指標資料不足)", "confidence": "不適用",
                "cross_reason": "資料不足", "has_data": False,
                "market_sync": False, "market_note": ""}

    assess = assess_kd_cross(kd, regime)
    confidence = assess["confidence"]
    reason = assess["reason"]

    sync = {"is_sync": False, "note": ""}
    if not is_benchmark:
        sync = check_market_sync(kd, market_ctx)

    text = "技術狀態:" + ",".join(parts)

    downgraded = False
    if sync["is_sync"]:
        text += f"。註:{sync['note']}"
        # 同步訊號不含個股資訊,可信度一律降到「低」——
        # 否則「區間震盪 + 高檔交叉」會拿到「高」,讓 AI 特別看重一個純 beta 的訊號。
        if confidence in ("高", "中高"):
            downgraded = True          # 只有這種情況,大盤同步才「改變了結論」
            reason = (f"原評為「{confidence}」,但{sync['note']},"
                      "訊號不含個股資訊,已降級")
            confidence = "低"
        elif confidence == "低":
            # 交叉本來就是雜訊,再加一句「而且是大盤同步」不會改變任何判斷 ——
            # 對 UI 而言是重複資訊,故不標記為 downgraded。
            reason = f"{reason};且{sync['note']}"

    return {
        "text": text,
        "confidence": confidence,
        "cross_reason": reason,
        "has_data": True,
        "market_sync": sync["is_sync"],
        "market_note": sync["note"],
        # True 才代表「大盤同步這件事推翻了原本的結論」,
        # UI 只在此時才需要額外顯示 —— 其餘情況是重複資訊。
        "market_downgraded": downgraded,
    }


# =============================================================
# 大盤對照
# =============================================================
def build_market_context(
    market_kd: Optional[Dict],
    market_regime: Optional[Dict],
    symbol: str = MARKET_BENCHMARK,
) -> Optional[Dict]:
    """
    把大盤基準的狀態整理成可供個股比對的簡表。

    Args:
        market_kd / market_regime: 對大盤基準(預設 0050)算出來的結果
        symbol: 基準代號(顯示用)

    Returns:
        {"symbol", "cross", "cross_days_ago", "k", "regime_state", "regime_label"}
        或 None
    """
    if not market_kd:
        return None
    return {
        "symbol": symbol,
        "cross": market_kd.get("cross", "none"),
        "cross_days_ago": market_kd.get("cross_days_ago"),
        "k": market_kd.get("k"),
        "d": market_kd.get("d"),
        "blunting": market_kd.get("blunting", "none"),
        "regime_state": (market_regime or {}).get("state"),
        "regime_label": (market_regime or {}).get("label", "—"),
    }


def check_market_sync(kd: Optional[Dict], market_ctx: Optional[Dict]) -> Dict:
    """
    判斷個股的交叉訊號是否只是跟著大盤動。

    同步條件:方向相同,且交叉時間相差在容忍範圍內。

    Returns:
        {"is_sync": bool, "note": str}
    """
    none = {"is_sync": False, "note": ""}
    if not kd or not market_ctx:
        return none

    cross = kd.get("cross", "none")
    if cross not in ("golden", "death"):
        return none
    if market_ctx.get("cross") != cross:
        return none

    a, b = kd.get("cross_days_ago"), market_ctx.get("cross_days_ago")
    if a is None or b is None:
        return none
    if abs(a - b) > MARKET_SYNC_TOLERANCE_DAYS:
        return none

    name = "黃金交叉" if cross == "golden" else "死亡交叉"
    return {
        "is_sync": True,
        "note": (f"大盤 {market_ctx['symbol']} 同期({b} 天前)也是{name},"
                 f"此為全市場同步走勢,非個股特性"),
    }


# =============================================================
# 股息品質:填息率 + 含息報酬
# =============================================================
FILL_MAX_DAYS = 180            # 追蹤填息的最長天數

# 填息判定基準 —— 台股實務上有兩種慣例,兩種都通行:
#   "close" 收盤價:哪一天的收盤價漲回除息前收盤價。長期分析、ETF 填息率
#            統計多用此口徑(富果、多數 ETF 研究)。較嚴格、較少雜訊。
#   "high"  盤中最高:盤中最高價何時觸及除息前收盤價。新聞媒體幾乎都用這個 ——
#            「大立光 2 分鐘完成填息」只可能是盤中口徑。較寬鬆、數字好看。
# 預設用 close(此工具偏長線持有視角),但兩種結果都會一併回報,
# 想切換只需改這個常數。
FILL_BASIS = "close"
CAGR_MIN_BARS = 250            # 算年化至少要這麼多交易日


def calculate_dividend_fill(
    df_raw: pd.DataFrame,
    actions: List[Dict],
    max_days: int = FILL_MAX_DAYS,
) -> Optional[Dict]:
    """
    計算「填息」表現 —— 這是股息品質的核心檢驗。

    為什麼重要:
      除息當天股價就會掉掉配息的金額,所以「領到股息」本身不是獲利。
      真正的問題是「股價有沒有漲回來」:
        有填息 → 股息是真實的獲利分配
        沒填息 → 股息只是把你自己的本金還給你(而且還要課稅)
      對「領息為主」的策略,這比殖利率重要得多 —— 高殖利率但長期不填息
      的標的,實際上是在慢慢消耗本金。

    ⚠️ 必須餵「原始股價」(不是還原股價)。填息的定義就是原始價格
       漲回除息前的水準;還原序列已經把缺口抹平,無法判斷。

    Args:
        df_raw: 原始 OHLC,需含 date / close(升冪)
        actions: corporate_actions 列表(含 action_date/before_price/reference_price)
        max_days: 每次事件最多追蹤幾個交易日

    Returns:
        Dict:
          - events: 每次除息的明細
          - filled_count / total_count / fill_rate
          - avg_days_to_fill / median_days_to_fill(只計算有填息的)
        資料不足回 None
    """
    if df_raw is None or df_raw.empty or not actions:
        return None
    if "close" not in df_raw.columns or "date" not in df_raw.columns:
        return None

    try:
        dates = [str(d)[:10] for d in df_raw["date"]]
        idx = {d: i for i, d in enumerate(dates)}
        closes = df_raw["close"].astype(float).values
        highs = (df_raw["high"].astype(float).values
                 if "high" in df_raw.columns else None)

        # 只看除權息(減資不是配息,不適用填息概念),並按時間排序
        divs = sorted(
            [a for a in actions if a.get("kind") == "除權息"],
            key=lambda a: str(a.get("action_date"))[:10],
        )

        events = []
        for n, a in enumerate(divs):
            ds = str(a.get("action_date"))[:10]
            i = idx.get(ds)
            if i is None:
                continue
            try:
                before = float(a["before_price"])
                ref = float(a["reference_price"])
            except (KeyError, TypeError, ValueError):
                continue
            gap = before - ref
            if gap <= 0:
                continue

            # 追蹤窗口:遇到下一次除息就截止。
            # 否則季配/月配的 ETF 會在還沒填完就再掉一次,measurement 被污染。
            end = min(i + max_days, len(closes) - 1)
            if n + 1 < len(divs):
                nxt = idx.get(str(divs[n + 1].get("action_date"))[:10])
                if nxt is not None:
                    end = min(end, nxt - 1)
            if end <= i:
                continue

            window = closes[i:end + 1]
            peak = float(window.max())

            # 兩種口徑都算,主判定用 FILL_BASIS,另一種一併回報供對照
            filled_close = filled_high = None
            for j in range(i, end + 1):
                if filled_close is None and closes[j] >= before:
                    filled_close = j - i
                if (filled_high is None and highs is not None
                        and highs[j] >= before):
                    filled_high = j - i
                if filled_close is not None and filled_high is not None:
                    break
            filled_at = filled_high if FILL_BASIS == "high" else filled_close

            # --- 窗口之外的後續發展 ---
            # 只看 180 天窗口會漏掉「延遲填息」。而且比較基準要公平:
            # 若期間又除息,現在的股價已經是「再少配幾次」之後的價格,
            # 直接跟舊的 before_price 比並不對稱。
            # 例:長榮 2025-06-19 除息前收 246.50,今天 240.50 看似未填,
            #     但中間 2026-06-17 又配了 16 元 —— 加回去 256.50,其實已填回。
            # 「填息」以收盤價判定,因為 before_price 本身就是除息前收盤價。
            # 拿盤中最高去比一個收盤價是兩種度量混用;若真要用盤中,
            # 基準也該換成除息前的盤中最高,門檻反而更嚴。
            # 但二元判定資訊量太低(差 0.2% 和差 30% 都顯示 ❌),
            # 所以額外回報「還差多少」,以及盤中是否曾經觸及。
            best_close = float(closes[i:].max()) if i < len(closes) else None
            best_high = None
            if highs is not None and i < len(highs):
                best_high = float(highs[i:].max())

            late_at = late_div_at = None
            if filled_at is None and end < len(closes) - 1:
                cum_div = 0.0
                nxt_idx = n + 1
                for j in range(end + 1, len(closes)):
                    # 逐日累加「這之後已發放的配息」
                    while nxt_idx < len(divs):
                        nd = idx.get(str(divs[nxt_idx].get("action_date"))[:10])
                        if nd is not None and nd <= j:
                            try:
                                cum_div += (float(divs[nxt_idx]["before_price"])
                                            - float(divs[nxt_idx]["reference_price"]))
                            except (KeyError, TypeError, ValueError):
                                pass
                            nxt_idx += 1
                        else:
                            break
                    if late_at is None and closes[j] >= before:
                        late_at = j - i
                    if late_div_at is None and closes[j] + cum_div >= before:
                        late_div_at = j - i
                    if late_at is not None and late_div_at is not None:
                        break

            events.append({
                "ex_date": ds,
                "dividend": round(gap, 2),
                "before_price": round(before, 2),
                "reference_price": round(ref, 2),
                "filled": filled_at is not None,
                "days_to_fill": filled_at,
                "basis": FILL_BASIS,
                # 另一種口徑的結果,供對照(新聞多用盤中,長期分析多用收盤)
                "filled_by_close": filled_close is not None,
                "days_to_fill_close": filled_close,
                "filled_by_high": filled_high is not None,
                "days_to_fill_high": filled_high,
                # 窗口外才填回(純股價)
                "filled_late": late_at is not None,
                "days_to_fill_late": late_at,
                # 窗口外、且需計入後續配息才填回(總報酬角度)
                "filled_late_with_dividend": late_div_at is not None,
                "days_to_fill_late_div": late_div_at,
                # 還差多少才填息(以收盤價計)
                "best_close": round(best_close, 2) if best_close else None,
                "gap_to_fill": (round(before - best_close, 2)
                                if best_close is not None and best_close < before
                                else 0.0),
                "gap_to_fill_pct": (round((before - best_close) / before * 100, 2)
                                    if best_close is not None and best_close < before
                                    else 0.0),
                # 盤中是否曾觸及(僅供參考,不作為填息判定)
                "touched_intraday": (best_high is not None and best_high >= before
                                     and (best_close is None or best_close < before)),
                "best_high": round(best_high, 2) if best_high else None,
                # 回復率只在「未填息」時有意義:告訴你離填息還差多遠。
                # 填息後這個數字會因分母(配息)極小而爆表(實測出現過 981%),
                # 沒有解讀價值,故填息時不給值。
                "recovery_pct": (None if filled_at is not None
                                 else round((peak - ref) / gap * 100, 1) if gap else None),
                "window_days": end - i,
                "truncated": end < min(i + max_days, len(closes) - 1),
            })

        if not events:
            return None

        done = [e for e in events if e["filled"]]
        days = sorted(e["days_to_fill"] for e in done)
        late = [e for e in events
                if not e["filled"] and (e.get("filled_late")
                                        or e.get("filled_late_with_dividend"))]
        never = [e for e in events
                 if not e["filled"] and not e.get("filled_late")
                 and not e.get("filled_late_with_dividend")]

        # 配息頻率:影響填息的難度。年配一次配掉 10~45%(長榮 2023 配 70 元 = 45%),
        # 季配每次只有 2~4%,缺口小得多、填得快。兩者的填息率不可直接互比。
        span_days = 0
        if len(events) >= 2:
            span_days = (pd.Timestamp(events[-1]["ex_date"])
                         - pd.Timestamp(events[0]["ex_date"])).days
        per_year = (len(events) - 1) / (span_days / 365.0) if span_days > 0 else 1.0
        if per_year >= 8:
            freq = "月配"
        elif per_year >= 3:
            freq = "季配"
        elif per_year >= 1.5:
            freq = "半年配"
        else:
            freq = "年配"
        avg_yield = sum(e["dividend"] / e["before_price"] for e in events) / len(events) * 100

        return {
            "frequency": freq,
            "per_year": round(per_year, 1),
            "avg_gap_pct": round(avg_yield, 2),
            "events": events,
            "total_count": len(events),
            "filled_count": len(done),
            "fill_rate": round(len(done) / len(events) * 100, 1),
            "basis": FILL_BASIS,
            "fill_rate_close": round(
                sum(1 for e in events if e["filled_by_close"]) / len(events) * 100, 1),
            "fill_rate_high": round(
                sum(1 for e in events if e["filled_by_high"]) / len(events) * 100, 1),
            "late_count": len(late),
            "never_count": len(never),
            "avg_days_to_fill": round(sum(days) / len(days), 1) if days else None,
            "median_days_to_fill": days[len(days) // 2] if days else None,
        }
    except Exception as e:
        print(f"[metrics] 填息計算失敗: {e}")
        return None


def calculate_total_return(
    adj_close: pd.Series,
    raw_close: pd.Series,
    years: float = 3.0,
) -> Optional[Dict]:
    """
    含息總報酬 vs 純價格報酬。

    還原股價序列的漲幅 = 含息總報酬(股息已還原回去),
    原始股價序列的漲幅 = 純價格報酬。
    兩者的差 = 股息貢獻了多少。

    這回答一個關鍵問題:「我的股息是真獲利,還是本金返還?」
      含息 > 純價格,且兩者都為正 → 股息是真獲利
      含息為正但純價格為負       → 股價在跌,靠股息撐住(要警覺)
      兩者都為負                 → 本金與股息雙輸

    Returns:
        {"total_return_pct", "price_return_pct", "dividend_contribution_pct",
         "total_cagr", "price_cagr", "years", "interpretation"}
    """
    if adj_close is None or raw_close is None:
        return None
    if len(adj_close) < CAGR_MIN_BARS or len(raw_close) < CAGR_MIN_BARS:
        return None

    try:
        bars = int(min(years * TRADING_DAYS_PER_YEAR, len(adj_close), len(raw_close)))
        a = adj_close.tail(bars).astype(float)
        r = raw_close.tail(bars).astype(float)
        if a.iloc[0] <= 0 or r.iloc[0] <= 0:
            return None

        actual_years = bars / TRADING_DAYS_PER_YEAR
        tot = (a.iloc[-1] / a.iloc[0] - 1) * 100
        pri = (r.iloc[-1] / r.iloc[0] - 1) * 100
        tot_cagr = ((a.iloc[-1] / a.iloc[0]) ** (1 / actual_years) - 1) * 100
        pri_cagr = ((r.iloc[-1] / r.iloc[0]) ** (1 / actual_years) - 1) * 100

        if tot > 0 and pri > 0:
            interp = "股息與股價雙成長,配息是真實的獲利分配"
        elif tot > 0 >= pri:
            interp = ("股價下跌,總報酬靠股息撐住。要留意是否為"
                      "「賺股息、賠價差」——本金正在流失")
        elif tot <= 0:
            interp = "含息後仍為負報酬,本金與股息雙輸"
        else:
            interp = ""

        # 配息占總報酬的比重 —— 這才是有 finding 的數字。
        # 只報「含息 X% / 純價 Y%」等於把兩個數字重講一遍,使用者自己就看得出差距;
        # 真正要回答的是「這檔的報酬是股價在推,還是配息在推」,
        # 那決定了該用成長股還是存股的框架看它。
        div_share = None
        if tot > 0:
            div_share = round(float((tot - pri) / tot) * 100, 1)

        return {
            "years": round(actual_years, 1),
            "bars": bars,
            "dividend_share_of_total": div_share,
            "total_return_pct": round(float(tot), 1),
            "price_return_pct": round(float(pri), 1),
            "dividend_contribution_pct": round(float(tot - pri), 1),
            "total_cagr": round(float(tot_cagr), 2),
            "price_cagr": round(float(pri_cagr), 2),
            "interpretation": interp,
        }
    except Exception as e:
        print(f"[metrics] 總報酬計算失敗: {e}")
        return None


def get_dividend_quality_grade(
    fill: Optional[Dict],
    tr: Optional[Dict],
) -> Dict:
    """股息品質的白話分級(給 UI 與 AI 用)"""
    if not fill:
        return {"tier": -1, "label": "無配息紀錄", "color": "gray",
                "position_text": "—",
                "description": "此標的在資料範圍內沒有除權息紀錄",
                "implication": "—"}

    rate = fill["fill_rate"]
    med = fill.get("median_days_to_fill")
    days_txt = f",中位數 {med} 個交易日填息" if med is not None else ""

    if rate >= 90:
        tier, label, color = 3, "填息強勁", "blue"
        fill_txt = "配息幾乎都能在合理時間內填回。"
    elif rate >= 70:
        tier, label, color = 2, "填息良好", "green"
        fill_txt = "多數配息能填回,但有部分次數未填。"
    elif rate >= 40:
        tier, label, color = 1, "填息普通", "orange"
        fill_txt = "填息成功率一般,約有半數配息在窗口內沒填回。"
    else:
        tier, label, color = 0, "填息不佳", "red"
        fill_txt = "多數配息填不回來。"

    freq = fill.get("frequency", "")
    gap_pct = fill.get("avg_gap_pct")
    if freq and gap_pct:
        fill_txt += f"({freq},平均每次除息缺口 {gap_pct:.1f}%)"

    # 兩種口徑差異大時要說明,否則使用者對照新聞會覺得數字不一致
    rc, rh = fill.get("fill_rate_close"), fill.get("fill_rate_high")
    if rc is not None and rh is not None and rh != rc:
        basis_name = "收盤價" if fill.get("basis") == "close" else "盤中最高"
        fill_txt += (f" 本表以**{basis_name}**判定;"
                     f"若改用另一口徑,填息率為 "
                     f"{rh if fill.get('basis') == 'close' else rc:.0f}%"
                     f"(新聞媒體多採盤中口徑,數字通常較高)。")

    # 只講「這一檔」的具體未填事件,不寫通用訓誡。
    # 先前這裡硬寫死了「實測長榮與 00919…」這種來自開發測試的句子,
    # 結果任何人看任何一檔都會讀到別檔的代號 —— 那是 bug 不是說明。
    # 結構性結論(有沒有侵蝕本金)上方的報酬區塊已經講過,這裡不重複。
    events = fill.get("events", [])
    never = [e for e in events if not e.get("filled")
             and not e.get("filled_late") and not e.get("filled_late_with_dividend")]
    late = [e for e in events if not e.get("filled")
            and (e.get("filled_late") or e.get("filled_late_with_dividend"))]

    seg = [fill_txt]
    if late:
        parts_late = []
        for e in sorted(late, key=lambda x: x["ex_date"])[-2:]:
            if e.get("filled_late"):
                parts_late.append(
                    f"{e['ex_date']} 配 {e['dividend']:.2f} 元於第 "
                    f"{e['days_to_fill_late']} 個交易日填回(超出 180 天觀察窗)")
            else:
                near = ""
                if e.get("gap_to_fill"):
                    near = (f",收盤最高 {e['best_close']} 元,"
                            f"距填息僅差 {e['gap_to_fill']} 元"
                            f"({e['gap_to_fill_pct']}%)")
                    if e.get("touched_intraday"):
                        near += f";盤中曾觸及 {e['best_high']} 元,但收盤未站上"
                parts_late.append(
                    f"{e['ex_date']} 配 {e['dividend']:.2f} 元{near},"
                    f"計入其後配息已回本(總報酬角度)")
        seg.append("🕓 延遲填息:" + "、".join(parts_late) + "。")
    if never:
        desc = "、".join(
            f"{e['ex_date']} 配 {e['dividend']:.2f} 元"
            f"(距填息還差 {e.get('gap_to_fill', 0):.2f} 元 / "
            f"{e.get('gap_to_fill_pct', 0):.1f}%)"
            for e in sorted(never, key=lambda x: x["ex_date"])[-2:]
        )
        seg.append(f"❌ 至今未填回:{desc}。")
    imp = " ".join(seg)

    return {
        "tier": tier, "label": label, "color": color,
        "position_text": (f"{'🔵🟢🟠🔴'[3 - tier]} 填息成功率 "
                          f"{rate:.0f}%({fill['filled_count']}/{fill['total_count']} 次)"
                          f"{days_txt}"),
        "description": f"共 {fill['total_count']} 次除息,{fill['filled_count']} 次完成填息",
        "implication": imp,
    }


def build_dividend_signal(
    fill: Optional[Dict],
    tr: Optional[Dict],
    long_unfilled_months: int = 12,
    large_gap_pct: float = 10.0,
) -> Dict:
    """
    篩選出「值得送進 AI」的股息訊號。

    為什麼要篩,不整包送:
      填息率是個雜訊很大的指標 —— 年配股樣本數常常只有 3~4 次
      (長榮 75% 其實是 3/4,少一次變 50%),窗口 180 天也是人為設定,
      而且實測長榮與 00919 的失敗次數都集中在股價下跌期,
      代表它有很大一部分只是在複述「這段期間股價漲了沒」。

      把這種數字包裝成乾淨的百分比餵給 AI,它會當成硬證據下結論 ——
      比不給還糟。所以 UI 保留完整明細給人判讀,
      AI 端只收「結構性結論」與「真正該響的警報」。

    送進 AI 的只有兩類:
      1. 結構性:純價格報酬正負 —— 配息有沒有侵蝕本金(無任意參數)
      2. 警報型:大額配息長期未填 —— 只在觸發時出現

    Returns:
        {"lines": [str, ...], "has_warning": bool, "must_mention": bool}
        must_mention=True 時,prompt 會要求 AI 必須提及
        (僅限「殖利率看似不錯但實際侵蝕本金」這種會誤導人的情況)
    """
    lines, has_warning, must_mention = [], False, False

    if tr:
        if tr["price_return_pct"] > 0:
            lines.append(
                f"- 近 {tr['years']} 年純價格報酬 {tr['price_return_pct']:+.1f}%"
                f"(含息 {tr['total_return_pct']:+.1f}%),配息未侵蝕本金"
            )
        else:
            # 這是唯一強制要講的情況:只報殖利率而不提股價下跌,會誤導人
            has_warning = must_mention = True
            lines.append(
                f"- ⚠️ 近 {tr['years']} 年純價格報酬 {tr['price_return_pct']:+.1f}%"
                f"(含息 {tr['total_return_pct']:+.1f}%)——「賺股息、賠價差」,"
                f"本金正在流失。討論殖利率時必須一併說明此事"
            )

    # 警報:大額配息長期未填。小額未填是雜訊,不進 prompt。
    if fill:
        for e in fill.get("events", []):
            # 已填回(含延遲填回、或計入後續配息後回本)就不再示警 ——
            # 警報要留給「真的還沒回來」的情況,否則會一直誤報。
            if (e.get("filled") or e.get("filled_late")
                    or e.get("filled_late_with_dividend")):
                continue
            gap_pct = e["dividend"] / e["before_price"] * 100
            if gap_pct < large_gap_pct:
                continue
            try:
                months = (pd.Timestamp.now() - pd.Timestamp(e["ex_date"])).days / 30.4
            except Exception:
                continue
            if months >= long_unfilled_months:
                has_warning = True
                lines.append(
                    f"- ⚠️ {e['ex_date']} 除息 {e['dividend']:.2f} 元"
                    f"(缺口 {gap_pct:.1f}%),至今 {months:.0f} 個月仍未填息 —— "
                    f"該次高配息缺乏後續獲利支撐"
                )

    return {"lines": lines, "has_warning": has_warning,
            "must_mention": must_mention}


# =============================================================
# 股息持續性
# =============================================================
# 填息問的是「過去漲回來了嗎」,這裡問的是「未來還發得出來嗎」。
# 對「只進不出、領息為主」的策略,後者才是真正的死穴 ——
# 股價腰斬但股息照發,策略還活著;股息斷了,策略的前提就沒了。
PAYOUT_WARN = 100.0        # 配息率超過此值 = 配息大於當年盈餘
PAYOUT_DANGER = 150.0
# 變異係數只描述「可預測性」,不參與燈號 ——
# 它是雙邊統計量,對「配得越來越多」和「配得越來越少」一視同仁地扣分。
# 實測:股息 11→13→17→22 元(成長型)CV=0.27,會被評為「波動偏大」;
# 而股息風險是單邊的,只怕配得少。用雙邊統計量衡量單邊風險是方法學錯誤。
CV_UNPREDICTABLE = 0.4     # 高於此值 = 配息金額較難預估(中性描述)
FLOOR_YIELD_OK = 3.0       # 谷底年殖利率(對成本)高於此值 = 可接受
DECLINE_YEARS = 3          # 連續下滑幾年才算趨勢性衰退
MIN_YEARS_FOR_CV = 3       # 少於這麼多年不給變異係數(樣本太小沒意義)


def calculate_dividend_sustainability(
    dividends: List[Dict],
    quarterly_financials: List[Dict],
    actions: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    """
    股息持續性分析。

    台股慣例:N 年發放的股利來自 N−1 年度的盈餘
    (例如 2026-06 除息的 16 元,對應的是 2025 年 EPS)。
    `dividends` 表沒有「股利所屬年度」欄位,故用 ex_date 年份減 1 對齊。

    Args:
        dividends: [{ex_date, cash_dividend, stock_dividend}, ...]
        quarterly_financials: [{year_quarter: "2025-Q1", eps: ...}, ...]

    Returns:
        Dict:
          - years: 每年的配息 / 對應 EPS / 配息率
          - cv: 股利變異係數(標準差 ÷ 平均)
          - min_dividend / max_dividend / avg_dividend
          - consecutive_years: 連續配息年數
          - over_payout_years: 配息率 > 100% 的年份
          - loss_year_payout: 虧損年度仍配息的年份
          - sample_note: 樣本數的誠實說明
        資料不足回 None
    """
    if not dividends:
        return None

    try:
        # --- 依 ex_date 年份彙總配息 ---
        by_year: Dict[int, float] = {}
        for d in dividends:
            ex = str(d.get("ex_date") or "")[:10]
            if len(ex) < 4 or not ex[:4].isdigit():
                continue
            cash = float(d.get("cash_dividend") or 0)
            if cash <= 0:
                continue
            by_year[int(ex[:4])] = by_year.get(int(ex[:4]), 0.0) + cash

        if not by_year:
            return None

        # --- 年度 EPS(四季加總)---
        eps_by_year: Dict[int, float] = {}
        q_count: Dict[int, int] = {}
        for q in (quarterly_financials or []):
            yq = str(q.get("year_quarter") or "")
            if "-Q" not in yq:
                continue
            y = yq.split("-Q")[0]
            if not y.isdigit():
                continue
            e = q.get("eps")
            if e is None:
                continue
            try:
                eps_by_year[int(y)] = eps_by_year.get(int(y), 0.0) + float(e)
                q_count[int(y)] = q_count.get(int(y), 0) + 1
            except (TypeError, ValueError):
                continue

        # --- 股數變動年度:該年度的「四季 EPS 相加」無效 ---
        # 減資 / 配股會改變流通股數,各季 EPS 的分母因此不同,相加等於
        # 蘋果加橘子。實測長榮 2022 年 9 月現金減資 60%(股數 52.9 億 → 21.2 億):
        #   Q1 19.16 + Q2 19.33 = 38.49(減資前基礎,與當時財報一致)
        #   Q4 18.19(減資後基礎,分母只剩四成)
        # 四季相加得 93.93,據此算出 2023 年配息率 74.5% —— 是假的。
        # 長榮當年自己公告的配發率是 52.67%,跟近年的 50% 一致,
        # 所謂「配發比例逐步下調」完全是這個計算錯誤造出來的假象。
        share_change_years = set()
        for a in (actions or []):
            if a.get("kind") == "減資":
                ds = str(a.get("action_date") or "")[:4]
                if ds.isdigit():
                    share_change_years.add(int(ds))
        for d in dividends:
            if float(d.get("stock_dividend") or 0) > 0:
                ex = str(d.get("ex_date") or "")[:4]
                if ex.isdigit():
                    share_change_years.add(int(ex))

        rows, over, loss_pay = [], [], []
        for y in sorted(by_year):
            div = by_year[y]
            src = y - 1                      # 股利所屬年度
            eps = eps_by_year.get(src)
            full = q_count.get(src, 0) >= 4  # 四季齊全才算完整年度

            payout = None
            # 必須四季齊全才算配息率:季數不足會讓分母偏小、配息率被高估。
            # 實測長榮 2023 年只有 3 季(Q1 落在 3 年回看範圍外),
            # 導致 2024 年配息率被算成 70%,實際應更低。
            share_changed = src in share_change_years
            if eps is not None and eps > 0 and full and not share_changed:
                payout = round(div / eps * 100, 1)
                if payout > PAYOUT_WARN:
                    over.append((y, payout))
            elif eps is not None and eps <= 0:
                loss_pay.append((y, round(eps, 2), round(div, 2)))

            rows.append({
                "pay_year": y, "dividend": round(div, 2),
                "source_year": src,
                "share_changed": share_changed,
                "eps": round(eps, 2) if eps is not None else None,
                "eps_complete": full,
                "payout_ratio": payout,
            })

        divs = [r["dividend"] for r in rows]
        avg = sum(divs) / len(divs)
        _sorted = sorted(divs)
        _m = len(_sorted) // 2
        median_div = (_sorted[_m] if len(_sorted) % 2
                      else (_sorted[_m - 1] + _sorted[_m]) / 2)
        cv = None
        if len(divs) >= MIN_YEARS_FOR_CV and avg > 0:
            var = sum((x - avg) ** 2 for x in divs) / len(divs)
            cv = round((var ** 0.5) / avg, 2)

        # 連續配息年數(由最近往回數)
        yrs = sorted(by_year, reverse=True)
        consecutive = 1
        for i in range(1, len(yrs)):
            if yrs[i - 1] - yrs[i] == 1:
                consecutive += 1
            else:
                break

        # --- 單邊指標:谷底水準 + 趨勢方向 ---
        # 對「只進不出、領息為主」的人,真正的問題不是「穩不穩」,
        # 而是「最壞那年還剩多少」以及「是不是在持續縮水」。
        seq = [r["dividend"] for r in rows]
        declining = 0
        for i in range(len(seq) - 1, 0, -1):
            if seq[i] < seq[i - 1]:
                declining += 1
            else:
                break
        is_declining = declining >= DECLINE_YEARS - 1

        floor_year = min(rows, key=lambda r: r["dividend"])
        latest = rows[-1]

        # 成長判定用「前後半段中位數」,不用頭尾兩點。
        # 頭尾比較會被起點的極端值綁架 —— 長榮 2021 年只配 2.49 元(疫情前谷底),
        # 拿它當基準,2.49→18→70→9.97→32.5→16 這種循環波動也會被判成「成長」。
        def _median(xs):
            ys = sorted(xs)
            m = len(ys) // 2
            return ys[m] if len(ys) % 2 else (ys[m - 1] + ys[m]) / 2

        trend = "cyclical"
        if is_declining:
            trend = "declining"
        elif len(seq) >= 4:
            half = len(seq) // 2
            early, late = seq[:half], seq[half:]
            m_early, m_late = _median(early), _median(late)
            if m_early > 0 and m_late > m_early * 1.2 and min(late) >= m_early * 0.8:
                trend = "growing"

        have_eps = sum(1 for r in rows if r["payout_ratio"] is not None)
        note = (f"配息 {len(rows)} 個年度,其中 {have_eps} 個年度有對應的完整 EPS "
                f"可算配息率")
        if have_eps < 3:
            note += " —— 樣本過小,配息率僅供參考,不足以下結論"

        return {
            "years": rows,
            "cv": cv,
            "avg_dividend": round(avg, 2),
            "median_dividend": round(median_div, 2),
            "min_dividend": round(min(divs), 2),
            "max_dividend": round(max(divs), 2),
            "consecutive_years": consecutive,
            "payout_ratios": [(r["pay_year"], r["payout_ratio"]) for r in rows
                              if r["payout_ratio"] is not None],
            "share_change_years": sorted(share_change_years),
            "excluded_years": [r["pay_year"] for r in rows if r.get("share_changed")],
            "floor_year": floor_year["pay_year"],
            "floor_dividend": floor_year["dividend"],
            "latest_dividend": latest["dividend"],
            "trend": trend,
            "declining_years": declining,
            "over_payout_years": over,
            "loss_year_payout": loss_pay,
            "sample_note": note,
        }
    except Exception as e:
        print(f"[metrics] 股息持續性計算失敗: {e}")
        return None


def get_dividend_sustainability_grade(
    sus: Optional[Dict],
    avg_cost: Optional[float] = None,
) -> Dict:
    """
    股息持續性的白話分級。

    ⚠️ 燈號**不**由變異係數決定。
    CV 是雙邊統計量,會把「配息成長」和「配息衰退」一視同仁地扣分 ——
    實測股息 11→13→17→22 元的成長型會被評為「波動偏大」,顯然是錯的。
    股息風險是單邊的:只怕配得少。

    燈號由三個單邊指標決定:
      1. 配息來源 —— 有沒有吃老本(配息率 >100%、虧損年照配)
      2. 配息趨勢 —— 是不是連續衰退
      3. 谷底水準 —— 最壞那年還剩多少
    變異係數降級為中性描述「配息可預測性」,只說明好不好估,不判好壞。

    Args:
        avg_cost: 若提供,會用它算「谷底年殖利率」,讓谷底水準有客觀基準
    """
    if not sus:
        return {"tier": -1, "label": "資料不足", "color": "gray",
                "position_text": "—", "description": "沒有配息紀錄或財報資料",
                "implication": "—"}

    cv = sus.get("cv")
    over = sus.get("over_payout_years", [])
    loss = sus.get("loss_year_payout", [])
    trend = sus.get("trend", "cyclical")
    floor = sus.get("floor_dividend")
    floor_y = sus.get("floor_year")
    latest = sus.get("latest_dividend")
    avg = sus["avg_dividend"]

    floor_yield = (floor / avg_cost * 100) if (avg_cost and floor) else None

    # --- 燈號:單邊指標 ---
    if loss or len(over) >= 2:
        tier, label, color = 0, "配息靠老本", "red"
        core = "多個年度配息超過當年盈餘,或虧損年度仍配息 —— 配息靠吃老本。"
        if cv is not None and cv < 0.25:
            core += ("❗ 配息金額看似穩定,但那是靠保留盈餘硬撐 —— "
                     "盈餘惡化時仍維持配息,是典型的股息陷阱型態。")
    elif trend == "declining":
        tier, label, color = 0, "配息連續衰退", "red"
        core = (f"已連續 {sus.get('declining_years', 0)} 年下滑,最近一次 {latest} 元"
                f" —— 是趨勢性縮水,不是循環波動。")
    elif trend == "growing":
        tier, label, color = 2, "配息穩健成長", "green"
        core = f"最近一次 {latest} 元,呈成長趨勢,對長期持有有利。"
    else:
        tier, label, color = 1, "配息隨獲利循環波動", "orange"
        core = "沒有趨勢性衰退,但不宜用近期高配息推估未來。"

    # 主文只留「對決策有用」的兩件事:
    #   1. 配發規則是什麼(配息率)—— 決定未來大概能領多少
    #   2. 以中位數計對成本的殖利率 —— 決定值不值得繼續抱
    # 其餘(區間端點、政策曾經移動、排除年度的技術原因)全部移到 expander,
    # 先前塞在同一段共 257 字,實際上沒人讀得完。
    parts = [core]
    # 中位數已寫在標題,這裡只補「對你的成本是多少殖利率」
    med = sus.get("median_dividend")
    if med is not None and avg_cost:
        parts.append(f"以此計算,對你的成本約 {med / avg_cost * 100:.1f}% 殖利率。")

    ratios = sus.get("payout_ratios") or []
    if ratios:
        vals = [p for _, p in ratios]
        recent = vals[-3:]
        _avg = sum(recent) / len(recent)
        # 門檻 0.2:相對離散度 20% 以內才算「固定比例」。
        # 先前用 0.3,結果 62~83%(相對差 29%)也被說成固定配發 —— 那不叫固定。
        if len(vals) >= 3 and _avg and (max(recent) - min(recent)) / _avg <= 0.2:
            parts.append(f"公司近三年固定配發盈餘的 "
                         f"{min(recent):.0f}~{max(recent):.0f}% —— "
                         f"配息多寡取決於當年賺多少,配發規則本身很穩定。")
        else:
            parts.append(f"配息率 "
                         + "、".join(f"{y} 年 {p:.0f}%" for y, p in ratios[-3:])
                         + ",配發比例本身也在變動。")
    elif sus.get("years"):
        parts.append("配息率無法計算(ETF 無財報 EPS,屬正常)。")

    icon = {0: "🔴", 1: "🟡", 2: "🟢"}.get(tier, "⚪")
    return {
        "tier": tier, "label": label, "color": color,
        "position_text": (f"{icon} {label} — 近 {len(sus['years'])} 年中位數配息 "
                          f"{sus.get('median_dividend', avg)} 元,連續配息 "
                          f"{sus['consecutive_years']} 年"),
        "description": (
            (f"配息區間 {sus['min_dividend']} ~ {sus['max_dividend']} 元。"
             if sus.get("min_dividend") is not None else "")
            + (f"⚠️ {'、'.join(str(y) for y in sus['excluded_years'])} 年配息率無法計算:"
               f"對應盈餘年度有減資或配股,股數改變後四季 EPS 不能相加;"
               f"現金減資退還的股款也不計入配息,該年實際股東報酬高於帳面。"
               if sus.get("excluded_years") else "")
            + sus["sample_note"]
            + (f"。配息可預測性:變異係數 {cv}"
               + ("(金額較難預估)" if cv >= CV_UNPREDICTABLE else "(金額相對好估)")
               + "。此值只描述好不好預估,高不等於風險大 —— 配息成長也會拉高它"
               if cv is not None else "")
        ),
        "implication": " ".join(parts),
    }


def build_sustainability_signal(sus: Optional[Dict]) -> Dict:
    """
    篩選送進 AI 的股息持續性訊號。

    只送單邊風險(來源惡化、趨勢衰退、谷底過低),不送變異係數 ——
    CV 高可能只是配息成長,送進去 AI 會誤讀成風險。
    """
    lines, must = [], False
    if not sus:
        return {"lines": [], "must_mention": False}

    over = sus.get("over_payout_years", [])
    loss = sus.get("loss_year_payout", [])

    if loss:
        must = True
        yrs = "、".join(f"{y}年(EPS {e})" for y, e, _ in loss)
        lines.append(f"- ⚠️ 虧損年度仍配息:{yrs} —— 配息靠保留盈餘")
    if len(over) >= 2:
        must = True
        yrs = "、".join(f"{y}年 {p:.0f}%" for y, p in over)
        lines.append(f"- ⚠️ 多年配息率超過 100%:{yrs},配息大於盈餘")
        if sus.get("cv") is not None and sus["cv"] < 0.25:
            lines.append("- ❗ 配息金額穩定但多年超額配發 —— 靠保留盈餘硬撐,"
                         "此處「金額穩定」不可解讀為正面")
    if sus.get("trend") == "declining":
        must = True
        lines.append(
            f"- ⚠️ 配息已連續 {sus.get('declining_years', 0)} 年下滑,"
            f"最近一次 {sus.get('latest_dividend')} 元 —— 趨勢性縮水"
        )

    # 中性事實:用中位數當代表值,不送「谷底」。
    # 送單一極小值會被 AI 直接引用成「配息谷底僅有 X 元」——
    # 實測長榮 2021 年配 2.49 元被寫進分析,但那是減資前的公司
    # (股數為現在的 2.5 倍),與現在根本不可比,且落在使用者持股期間之外。
    # 極端值當代表值是錯的;要看下檔,中位數與區間才是合理錨點。
    med = sus.get("median_dividend", sus["avg_dividend"])
    lines.append(
        f"- 近 {len(sus['years'])} 年配息中位數 {med} 元,"
        f"區間 {sus['min_dividend']} ~ {sus['max_dividend']} 元,"
        f"連續配息 {sus['consecutive_years']} 年"
        f"(**不得以區間端點的單一年度代表未來配息水準**,"
        f"早期年度可能因減資、產業循環位置而不可比)"
    )
    ratios = sus.get("payout_ratios") or []
    if ratios:
        vals = [p for _, p in ratios]
        base = "- 配息率 " + "、".join(f"{y}年 {p:.0f}%" for y, p in ratios[-3:])
        recent = vals[-3:]
        avg = sum(recent) / len(recent)
        if len(vals) >= 3 and avg and (max(recent) - min(recent)) / avg <= 0.2:
            lines.append(base + f" → 近三年穩定在 {min(recent):.0f}~{max(recent):.0f}%,"
                                "屬固定比例配發,配息波動源自獲利而非政策")
        elif len(vals) == 2:
            lines.append(base + "(僅 2 年可算,不足以判斷配發政策)")
        else:
            lines.append(base)
    else:
        lines.append("- 配息率無法計算(無對應財報 EPS;ETF 屬正常)")

    if sus.get("excluded_years"):
        lines.append(
            "- 註:" + "、".join(str(y) for y in sus["excluded_years"])
            + " 年配息率無法計算(盈餘年度有減資/配股,股數改變致四季 EPS 不可相加);"
              "現金減資退還股款亦未計入配息"
        )

    if sus.get("trend") == "cyclical":
        # 原本這裡寫「請看谷底年配息」,與上面「不得用極端年度」互相矛盾。
        # 正確的錨點是中位數,不是最差的那一年。
        lines.append("- 配息隨獲利循環波動,無趨勢性衰退。"
                     "評估未來配息水準時以中位數為錨點,"
                     "不要以近期高配息外推,也不要以單一最差年度代表下檔")

    return {"lines": lines, "must_mention": must}


# =============================================================
# 交易輸入驗證
# =============================================================
# 背景:券商庫存頁的「平均成本」會隨除息調降,而 transactions.price
# 應該存的是「實際成交價」。抄錯欄位不會報錯,只會讓有效成本悄悄變好看。
# 實測踩過:某筆登錄 187.80,而該日實際成交區間是 195.50~200.50。
#
# 價格不可能低於當日最低成交價 —— 這是可以自動檢查的硬事實。
PRICE_CHECK_WINDOW = 45        # 日期為概估時,改用 ±N 天的區間比對


def validate_transaction_price(
    price: float,
    date_str: str,
    bars: List[Dict],
    window: int = PRICE_CHECK_WINDOW,
) -> Dict:
    """
    檢查登錄價格是否可能是真實成交價。

    Args:
        price: 登錄的價格
        date_str: 交易日期 'YYYY-MM-DD'
        bars: [{date, high, low}, ...] 該股的日線
        window: 非交易日時,改用 ±N 天區間比對

    Returns:
        {"level": "ok"/"warn"/"error"/"unknown", "message": str,
         "range": (low, high) 或 None}
    """
    if not bars or price is None or price <= 0:
        return {"level": "unknown", "message": "無股價資料可比對", "range": None}

    try:
        ds = str(date_str)[:10]
        bar_map = {str(b["date"])[:10]: b for b in bars}
        bar = bar_map.get(ds)

        if bar is not None:
            lo, hi = float(bar["low"]), float(bar["high"])
            if lo <= price <= hi:
                return {"level": "ok",
                        "message": f"✅ 落在當日成交區間 {lo:.2f}~{hi:.2f} 內",
                        "range": (lo, hi)}
            if price < lo:
                return {"level": "error",
                        "message": (f"🚩 低於當日最低成交價 {lo:.2f}(差 "
                                    f"{lo - price:.2f})。價格不可能低於當日最低 —— "
                                    f"這通常表示填的是券商「已扣息的平均成本」"
                                    f"而非成交價"),
                        "range": (lo, hi)}
            return {"level": "warn",
                    "message": f"⚠️ 高於當日最高成交價 {hi:.2f},請確認",
                    "range": (lo, hi)}

        # 非交易日 → 日期可能是概估,改比對區間
        target = pd.Timestamp(ds)
        near = [b for b in bars
                if abs((pd.Timestamp(str(b["date"])[:10]) - target).days) <= window]
        if not near:
            return {"level": "unknown",
                    "message": f"⚠️ {ds} 非交易日,且前後 {window} 天無股價資料",
                    "range": None}

        lo = min(float(b["low"]) for b in near)
        hi = max(float(b["high"]) for b in near)
        if lo <= price <= hi:
            return {"level": "warn",
                    "message": (f"⚠️ {ds} 非交易日(可能是概估日期或多筆彙總)。"
                                f"價格落在前後 {window} 天區間 {lo:.2f}~{hi:.2f} 內,"
                                f"數值本身合理"),
                    "range": (lo, hi)}
        if price < lo:
            return {"level": "error",
                    "message": (f"🚩 {ds} 非交易日,且價格低於前後 {window} 天"
                                f"最低價 {lo:.2f} —— 高機率是券商已扣息的平均成本"),
                    "range": (lo, hi)}
        return {"level": "warn",
                "message": f"⚠️ 價格高於前後 {window} 天最高價 {hi:.2f},請確認",
                "range": (lo, hi)}
    except Exception as e:
        return {"level": "unknown", "message": f"驗證失敗: {e}", "range": None}


def calculate_kd_series(
    df: pd.DataFrame,
    period: int = KD_PERIOD,
    smooth: int = KD_SMOOTH,
) -> Optional[pd.DataFrame]:
    """
    回傳完整的 K/D 序列(繪圖用),含 date 欄。

    calculate_kd() 只回最新一筆狀態,畫副圖需要整段序列。

    ⚠️ 一樣必須餵還原股價 —— 除權息缺口會讓 KD 嚴重失真。
    因此 K 線圖(原始價)與 KD 副圖(還原價)的資料來源不同,
    但兩者的日期軸一致,對照不受影響。
    """
    if df is None or len(df) < max(KD_MIN_BARS, period):
        return None
    for c in ("high", "low", "close", "date"):
        if c not in df.columns:
            return None
    try:
        k, d = _kd_series(df, period=period, smooth=smooth)
        if k is None:
            return None
        return pd.DataFrame({"date": df["date"].values,
                             "k": k.values, "d": d.values})
    except Exception as e:
        print(f"[metrics] KD 序列繪圖資料失敗: {e}")
        return None
