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
            "percent_b": round(percent_b, 1) if percent_b is not None else None,
            "current": round(current, 2),
            "position": position,
        }
        
    except Exception as e:
        print(f"[metrics] 布林通道計算失敗: {e}")
        return None