"""
台股資料管道:FinMind → Supabase
v3: 股價 + PE/PB/殖利率 + 月營收
"""
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from FinMind.data import DataLoader
from supabase import create_client, Client
import pandas as pd

load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

finmind = DataLoader()
finmind.login_by_token(api_token=os.getenv("FINMIND_TOKEN"))


# ============================================================
# 工具函式
# ============================================================
# 大盤基準:用來判斷個股的技術訊號是「個股特性」還是「全市場走勢」。
# 必須無條件同步 —— 不能依賴「剛好有人持有 0050」,
# 否則那個人一移除追蹤,大盤對照就會靜默失效而沒有任何提示。
# 需與 metrics.MARKET_BENCHMARK 一致。
MARKET_BENCHMARK = "0050"


def is_etf(symbol: str) -> bool:
    """
    判斷是否為 ETF(代號以 00 開頭:0050 / 00878 / 00953B / 009816 / 00403A)

    ETF 沒有 PE、月營收、財報、也不會減資,
    對這些 API 的呼叫必定回空,純浪費額度。
    """
    return symbol.startswith("00")


def print_api_usage(label: str = ""):
    """印出 FinMind API 用量(額度按小時重置)"""
    try:
        print(f"📊 API 用量{label}: {finmind.api_usage} / {finmind.api_usage_limit}")
    except Exception as e:
        print(f"📊 API 用量查詢失敗: {e}")


# ============================================================
# Fetch 函式:從 FinMind 抓資料
# ============================================================
def fetch_daily_prices(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓股價(OHLCV)"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    df = finmind.taiwan_stock_daily(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    df = df.rename(columns={
        "stock_id": "symbol",
        "Trading_Volume": "volume",
        "max": "high",
        "min": "low",
    })
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]


def fetch_per_pbr(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓 PE / PB / 殖利率"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    df = finmind.taiwan_stock_per_pbr(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    df = df.rename(columns={
        "stock_id": "symbol",
        "PER": "pe",
        "PBR": "pb",
        "dividend_yield": "dividend_yield",
    })
    return df[["symbol", "date", "pe", "pb", "dividend_yield"]]


def fetch_monthly_revenue(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓月營收 + 自行計算 YoY/MoM/累計"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = finmind.taiwan_stock_month_revenue(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    
    # 防禦性檢查:必要欄位存在
    required = ["stock_id", "revenue", "revenue_year", "revenue_month"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ FinMind 回傳缺欄位: {missing} | 實際: {list(df.columns)}")
        return pd.DataFrame()
    
    df = df.rename(columns={"stock_id": "symbol"})
    
    # 組 year_month 字串(例如 '2026-04')
    df["year_month"] = df.apply(
        lambda r: f"{int(r['revenue_year']):04d}-{int(r['revenue_month']):02d}",
        axis=1
    )
    
    # 按時間排序(很重要,後面 pct_change 才正確)
    df = df.sort_values("year_month").reset_index(drop=True)
    
    # 自己算 MoM (跟上個月比)
    df["revenue_mom"] = df["revenue"].pct_change() * 100
    
    # 自己算 YoY (跟去年同月比 = 往前 12 個月)
    df["revenue_yoy"] = df["revenue"].pct_change(periods=12) * 100
    
    # 累計營收 (同一年從 1 月累加)
    df["_year"] = df["year_month"].str[:4]
    df["cumulative_revenue"] = df.groupby("_year")["revenue"].cumsum()
    
    # 累計年增率 (同月份跟去年累計比)
    df["_month"] = df["year_month"].str[5:]
    df["cumulative_yoy"] = df.groupby("_month")["cumulative_revenue"].pct_change() * 100
    
    return df[["symbol", "year_month", "revenue", "revenue_yoy", "revenue_mom",
               "cumulative_revenue", "cumulative_yoy"]]

def fetch_quarterly_income(symbol: str, start_date: str) -> pd.DataFrame:
    """抓季度綜合損益表"""
    df = finmind.taiwan_stock_financial_statement(
        stock_id=symbol, start_date=start_date
    )
    if df.empty:
        return df
    
    # FinMind 回傳的是長表 (long format),要 pivot 成寬表 (wide format)
    # 欄位:date, stock_id, type, value, origin_name
    pivot = df.pivot_table(
        index=["stock_id", "date"],
        columns="type",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    return pivot


def fetch_quarterly_balance(symbol: str, start_date: str) -> pd.DataFrame:
    """抓季度資產負債表(主要用來算 ROE/ROA)"""
    df = finmind.taiwan_stock_balance_sheet(
        stock_id=symbol, start_date=start_date
    )
    if df.empty:
        return df
    
    pivot = df.pivot_table(
        index=["stock_id", "date"],
        columns="type",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    return pivot

# ============================================================
# 籌碼面 (Phase 2.3)
# ============================================================
def fetch_chips(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓三大法人買賣超"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = finmind.taiwan_stock_institutional_investors(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    
    # FinMind 回傳長表 (long format),要 pivot
    # 欄位:date, stock_id, name, buy, sell
    # name 會是 'Foreign_Investor', 'Investment_Trust', 'Dealer_self', 'Dealer_Hedging'
    
    # 計算 net (buy - sell)
    df["net"] = df["buy"] - df["sell"]
    
    # Dealer_self + Dealer_Hedging 合併為 dealer
    df["name"] = df["name"].replace({
        "Dealer_self": "Dealer",
        "Dealer_Hedging": "Dealer",
    })
    
    # pivot:每個 (symbol, date) 變一筆,法人別變欄位
    pivot_buy = df.pivot_table(index=["stock_id", "date"], columns="name", values="buy", aggfunc="sum").reset_index()
    pivot_sell = df.pivot_table(index=["stock_id", "date"], columns="name", values="sell", aggfunc="sum").reset_index()
    pivot_net = df.pivot_table(index=["stock_id", "date"], columns="name", values="net", aggfunc="sum").reset_index()
    
    # 合併三個 pivot
    result = pivot_buy.rename(columns={
        "stock_id": "symbol",
        "Foreign_Investor": "foreign_buy",
        "Investment_Trust": "trust_buy",
        "Dealer": "dealer_buy",
    })
    
    sell_cols = pivot_sell.rename(columns={
        "stock_id": "symbol",
        "Foreign_Investor": "foreign_sell",
        "Investment_Trust": "trust_sell",
        "Dealer": "dealer_sell",
    })
    net_cols = pivot_net.rename(columns={
        "stock_id": "symbol",
        "Foreign_Investor": "foreign_net",
        "Investment_Trust": "trust_net",
        "Dealer": "dealer_net",
    })
    
    result = result.merge(sell_cols, on=["symbol", "date"], how="outer")
    result = result.merge(net_cols, on=["symbol", "date"], how="outer")
    
    # 確保所有欄位都存在(萬一某天沒投信進出,欄位會缺)
    for col in ["foreign_buy", "foreign_sell", "foreign_net",
                "trust_buy", "trust_sell", "trust_net",
                "dealer_buy", "dealer_sell", "dealer_net"]:
        if col not in result.columns:
            result[col] = 0
    
    result = result.fillna(0)
    
    # 計算合計買賣超
    result["total_net"] = result["foreign_net"] + result["trust_net"] + result["dealer_net"]
    
    return result[["symbol", "date",
                   "foreign_buy", "foreign_sell", "foreign_net",
                   "trust_buy", "trust_sell", "trust_net",
                   "dealer_buy", "dealer_sell", "dealer_net",
                   "total_net"]]


def fetch_margin(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓融資融券餘額"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = finmind.taiwan_stock_margin_purchase_short_sale(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    
    df = df.rename(columns={
        "stock_id": "symbol",
        "MarginPurchaseTodayBalance": "margin_balance",
        "MarginPurchaseBuy": "_mp_buy",
        "MarginPurchaseSell": "_mp_sell",
        "ShortSaleTodayBalance": "short_balance",
        "ShortSaleBuy": "_ss_buy",
        "ShortSaleSell": "_ss_sell",
    })
    
    # 計算融資、融券每日增減
    df["margin_change"] = df["_mp_buy"] - df["_mp_sell"]
    df["short_change"] = df["_ss_sell"] - df["_ss_buy"]  # 注意:券是反過來
    
    return df[["symbol", "date", "margin_balance", "margin_change",
               "short_balance", "short_change"]]

def fetch_shareholding(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓外資/投信持股比例"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = finmind.taiwan_stock_shareholding(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    
    df = df.rename(columns={
        "stock_id": "symbol",
        "ForeignInvestmentRemainingShares": "_foreign_shares",
        "ForeignInvestmentSharesRatio": "foreign_holding_ratio",
        # 投信欄位視 FinMind 版本而異,可能不存在
    })
    
    # 確保欄位存在
    if "foreign_holding_ratio" not in df.columns:
        df["foreign_holding_ratio"] = None
    if "trust_holding_ratio" not in df.columns:
        df["trust_holding_ratio"] = None
    
    return df[["symbol", "date", "foreign_holding_ratio", "trust_holding_ratio"]]


def fetch_dividends(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """抓除權息紀錄"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = finmind.taiwan_stock_dividend(
        stock_id=symbol, start_date=start_date, end_date=end_date
    )
    if df.empty:
        return df
    
    # FinMind 欄位:
    # date(公告日), stock_id, year(股利所屬年度),
    # CashEarningsDistribution(現金股利), StockEarningsDistribution(股票股利),
    # CashExDividendTradingDate(現金除權息日), CashDividendPaymentDate(發放日)
    
    # 修改後：如果還沒公告除息日(空字串)，就先用「董事會公告日(date)」暫代，確保最新股息能進資料庫
    df = df.rename(columns={
        "stock_id": "symbol",
        "CashExDividendTradingDate": "ex_date",
        "CashEarningsDistribution": "cash_dividend",
        "StockEarningsDistribution": "stock_dividend",
    })
    
    # 填補空缺的除息日
    df.loc[df["ex_date"] == "", "ex_date"] = df["date"]
    df = df.dropna(subset=["ex_date"]) # 確保主鍵不為空
    
    df["cash_dividend"] = df["cash_dividend"].fillna(0)
    df["stock_dividend"] = df["stock_dividend"].fillna(0)
    df["total_dividend"] = df["cash_dividend"] + df["stock_dividend"]
    
    return df[["symbol", "ex_date", "cash_dividend", "stock_dividend", "total_dividend"]]


def fetch_corporate_actions(symbol: str, start_date: str) -> pd.DataFrame:
    """
    抓「會造成股價跳空」的公司行為參考價,供還原股價使用。

    為什麼需要:
      daily_prices 存的是原始收盤價,除權息當天會跳空。技術指標(KD / 布林)
      吃到這個缺口會嚴重失真 —— 例如長榮 2023-06-30 配息 70 元,
      原始 K 值 23.6(看似超賣),還原後真實值 80.2(其實超買)。

    調整因子取自交易所公告參考價,不自己算股利公式:
        r = reference_price / before_price
    (不可用 after_price,那是除息當天實際收盤,含當天市場波動)
    """
    rows = []

    # --- 除權除息結果表 ---
    try:
        dr = finmind.taiwan_stock_dividend_result(
            stock_id=symbol, start_date=start_date
        )
        if not dr.empty:
            for _, r in dr.iterrows():
                rows.append({
                    "symbol": symbol,
                    "action_date": str(r["date"])[:10],
                    "kind": "除權息",
                    "before_price": float(r["before_price"]),
                    "reference_price": float(r["reference_price"]),
                    "note": str(r.get("stock_or_cache_dividend", "") or ""),
                })
    except Exception as e:
        print(f"  [除權息參考價] ❌ {e}")

    # --- 減資恢復買賣參考價(ETF 不會減資,直接跳過省一次呼叫)---
    if not is_etf(symbol):
        time.sleep(0.5)
        try:
            cr = finmind.taiwan_stock_capital_reduction_reference_price(
                stock_id=symbol, start_date=start_date
            )
            if not cr.empty:
                for _, r in cr.iterrows():
                    rows.append({
                        "symbol": symbol,
                        "action_date": str(r["date"])[:10],
                        "kind": "減資",
                        "before_price": float(r["ClosingPriceonTheLastTradingDay"]),
                        "reference_price": float(r["PostReductionReferencePrice"]),
                        "note": str(r.get("ReasonforCapitalReduction", "") or ""),
                    })
        except Exception as e:
            # 多數個股從未減資,抓不到屬正常
            print(f"  [減資參考價] ⚠️ 無資料或失敗(未減資屬正常): {e}")

    return pd.DataFrame(rows)


def upsert_corporate_actions(df: pd.DataFrame) -> int:
    """寫入公司行為參考價"""
    if df.empty:
        return 0

    # FinMind 偶爾回重複列,同一批 upsert 撞主鍵會整批失敗
    df = df.drop_duplicates(subset=["symbol", "action_date", "kind"], keep="last")

    records = df.to_dict(orient="records")
    for r in records:
        r["action_date"] = str(r["action_date"])
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None

    batch_size = 200
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("corporate_actions").upsert(batch).execute()
        total += len(batch)
    return total


def upsert_dividends(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    # ETF 季配/月配 + 「除息日空白用公告日暫代」會把多筆壓到同一天,
    # 撞到主鍵 (symbol, ex_date) 導致整批 upsert 失敗:
    #   ON CONFLICT DO UPDATE command cannot affect row a second time
    df = df.drop_duplicates(subset=["symbol", "ex_date"], keep="last")

    records = df.to_dict(orient="records")
    for r in records:
        r["ex_date"] = str(r["ex_date"])
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("dividends").upsert(batch).execute()
        total += len(batch)
    return total


def upsert_chips(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    for r in records:
        r["date"] = str(r["date"])
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("daily_chips").upsert(batch).execute()
        total += len(batch)
    return total

def upsert_margin(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    for r in records:
        r["date"] = str(r["date"])
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("margin_balance").upsert(batch).execute()
        total += len(batch)
    return total

def upsert_shareholding(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    for r in records:
        r["date"] = str(r["date"])
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("shareholding").upsert(batch).execute()
        total += len(batch)
    return total

def build_quarterly_financials(income_df: pd.DataFrame, balance_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """整合損益表 + 資產負債表,計算各種比率"""
    if income_df.empty:
        return pd.DataFrame()
    
    # 季度日期 → year_quarter (例如 2025-03-31 → 2025-Q1)
    def date_to_quarter(d):
        ts = pd.to_datetime(d)
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{q}"
    
    rows = []
    for _, row in income_df.iterrows():
        try:
            ym = date_to_quarter(row["date"])
            revenue = row.get("Revenue", row.get("OperatingRevenue"))
            gross_profit = row.get("GrossProfit")
            op_income = row.get("OperatingIncome")
            net_income = row.get("IncomeAfterTaxes", row.get("ProfitAfterTax"))
            eps = row.get("EPS")
            
            # 計算比率(避免除以 0)
            gross_margin = (gross_profit / revenue * 100) if revenue and gross_profit else None
            op_margin = (op_income / revenue * 100) if revenue and op_income else None
            net_margin = (net_income / revenue * 100) if revenue and net_income else None
            
            # ROE / ROA 需要從資產負債表算
            roe = None
            roa = None
            if not balance_df.empty:
                bal = balance_df[balance_df["date"] == row["date"]]
                if not bal.empty:
                    bal_row = bal.iloc[0]
                    equity = bal_row.get("Equity", bal_row.get("TotalEquity"))
                    assets = bal_row.get("TotalAssets")
                    if equity and net_income:
                        roe = net_income / equity * 100
                    if assets and net_income:
                        roa = net_income / assets * 100
            
            rows.append({
                "symbol": symbol,
                "year_quarter": ym,
                "eps": eps,
                "revenue": revenue,
                "gross_margin": gross_margin,
                "operating_margin": op_margin,
                "net_margin": net_margin,
                "roe": roe,
                "roa": roa,
                "net_income": net_income,
            })
        except Exception as e:
            print(f"  ⚠️ 季度 {row.get('date')} 處理失敗: {e}")
            continue
    
    return pd.DataFrame(rows)


def upsert_quarterly_financials(df: pd.DataFrame) -> int:
    """寫入季報"""
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 100
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("quarterly_financials").upsert(batch).execute()
        total += len(batch)
    return total

# ============================================================
# 合併 / 寫入 函式
# ============================================================
def merge_price_and_valuation(prices: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """合併股價與估值資料"""
    if prices.empty:
        return prices
    if valuation.empty:
        return prices
    return prices.merge(valuation, on=["symbol", "date"], how="left")


def upsert_prices(df: pd.DataFrame) -> int:
    """寫入股價(含估值)"""
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    for r in records:
        r["date"] = str(r["date"])
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("daily_prices").upsert(batch).execute()
        total += len(batch)
    return total


def upsert_monthly_revenue(df: pd.DataFrame) -> int:
    """寫入月營收"""
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
    
    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("monthly_revenue").upsert(batch).execute()
        total += len(batch)
    return total


def sync_symbol(symbol: str, days_back: int = 1095):
    """同步單一個股(預設 3 年,籌碼只抓 1 年)"""
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    # 籌碼資料只抓近 1 年(避免資料量過大)
    chips_start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    print(f"[同步] {symbol} 從 {start_date} 開始...")
    
    # 1) 股價
    prices = fetch_daily_prices(symbol, start_date)
    if prices.empty:
        print(f"  ⚠️ {symbol} 無股價資料,跳過")
        return
    
    if is_etf(symbol):
        # ETF 沒有 PE / 月營收 / 財報,這三支 API 必定回空 → 跳過省 4 次呼叫
        count_p = upsert_prices(prices)
        print(f"  [股價] 寫入 {count_p} 筆 (ETF:跳過 PE / 月營收 / 季報)")
    else:
        # 2) PE/PB/殖利率
        time.sleep(0.5)
        valuation = fetch_per_pbr(symbol, start_date)

        merged = merge_price_and_valuation(prices, valuation)
        count_p = upsert_prices(merged)
        has_pe = "✅" if not valuation.empty else "❌"
        print(f"  [股價] 寫入 {count_p} 筆 (PE: {has_pe})")

        # 3) 月營收
        time.sleep(0.5)
        revenue = fetch_monthly_revenue(symbol, start_date)
        if not revenue.empty:
            count_r = upsert_monthly_revenue(revenue)
            print(f"  [月營收] 寫入 {count_r} 筆")
        else:
            print(f"  [月營收] ⚠️ 無資料")

        # 4) 季報財務(抓 5 年,與 dividends 對齊)
        #
        # 為什麼不跟著 days_back 的 3 年:
        #   配息率 = N 年配息 ÷ N−1 年度全年 EPS,需要「四季齊全」的年度。
        #   3 年只會蓋到約 12 季,最舊那年往往缺 Q1(被回看範圍切掉),
        #   實測長榮只剩 2 個年度可算配息率 —— 而 2 個樣本不足以判斷
        #   「是否為固定比例配發政策」,等於那個功能永遠觸發不了。
        #   循環股要看「谷底年賺多少」也需要更長的區間。
        #
        # 成本:FinMind 按「請求次數」計費,抓 3 年和 5 年都是同樣 2 次呼叫,
        #       只是 start_date 往前推 —— 額度零增加。
        fin_start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        time.sleep(0.5)
        try:
            income = fetch_quarterly_income(symbol, fin_start_date)
            time.sleep(0.5)
            balance = fetch_quarterly_balance(symbol, fin_start_date)

            qfin = build_quarterly_financials(income, balance, symbol)
            if not qfin.empty:
                count_q = upsert_quarterly_financials(qfin)
                print(f"  [季報] 寫入 {count_q} 筆")
            else:
                print(f"  [季報] ⚠️ 無資料")
        except Exception as e:
            print(f"  [季報] ❌ 抓取失敗: {e}")
    
    # 5) 籌碼:三大法人(只抓 1 年)
    time.sleep(0.5)
    try:
        chips = fetch_chips(symbol, chips_start_date)
        if not chips.empty:
            count_c = upsert_chips(chips)
            print(f"  [法人] 寫入 {count_c} 筆")
        else:
            print(f"  [法人] ⚠️ 無資料")
    except Exception as e:
        print(f"  [法人] ❌ 失敗: {e}")
    
    # 6) 籌碼:融資融券(只抓 1 年)
    time.sleep(0.5)
    try:
        margin = fetch_margin(symbol, chips_start_date)
        if not margin.empty:
            count_m = upsert_margin(margin)
            print(f"  [融資券] 寫入 {count_m} 筆")
        else:
            print(f"  [融資券] ⚠️ 無資料")
    except Exception as e:
        print(f"  [融資券] ❌ 失敗: {e}")
    
    # 7) 籌碼:外資持股比例(只抓 1 年)
    time.sleep(0.5)
    try:
        sh = fetch_shareholding(symbol, chips_start_date)
        if not sh.empty:
            count_s = upsert_shareholding(sh)
            print(f"  [持股比] 寫入 {count_s} 筆")
        else:
            print(f"  [持股比] ⚠️ 無資料")
    except Exception as e:
        print(f"  [持股比] ❌ 失敗: {e}")

    # 8) 除權息(抓 5 年,因為股息是長期數據)
    time.sleep(0.5)
    try:
        div_start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        divs = fetch_dividends(symbol, div_start)
        if not divs.empty:
            count_d = upsert_dividends(divs)
            print(f"  [除權息] 寫入 {count_d} 筆")
        else:
            print(f"  [除權息] ⚠️ 無資料")
    except Exception as e:
        print(f"  [除權息] ❌ 失敗: {e}")

    # 9) 公司行為參考價(還原股價用,抓 5 年)
    # 範圍必須比股價(3 年)更長,事件要涵蓋價格序列起點之前,
    # 否則最舊那段的累積調整因子會算錯。
    time.sleep(0.5)
    try:
        ca_start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        ca = fetch_corporate_actions(symbol, ca_start)
        if not ca.empty:
            count_a = upsert_corporate_actions(ca)
            print(f"  [公司行為] 寫入 {count_a} 筆")
        else:
            print(f"  [公司行為] ⚠️ 無資料")
    except Exception as e:
        print(f"  [公司行為] ❌ 失敗: {e}")


def sync_symbol_light(symbol: str, days_back: int = 30):
    """
    輕量同步:只更新「每天都會變」的資料。

    為什麼要分輕重:
      完整同步一輪約 296 次 API 呼叫,而 FinMind 免費額度是 600/小時 ——
      一小時只跑得動一輪。但每天真正會變的只有股價與籌碼,
      月營收一個月才一次、季報三個月才一次、股利一年才幾次。

    呼叫成本:個股 3 次(股價 + PE + 法人),ETF 2 次(無 PE)
      → 31 檔約 90 次,不到完整同步的三分之一。

    ⚠️ 刻意不碰月營收:
      fetch_monthly_revenue() 用 pct_change(12) 算 YoY,
      抓取範圍縮短會只拿到少數幾筆,YoY 變成 NaN 後 upsert 覆蓋掉
      原本正確的值 —— 這是縮短天數最容易踩的坑,故整支跳過。
    """
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    print(f"[輕量] {symbol} 從 {start_date}...")

    prices = fetch_daily_prices(symbol, start_date)
    if prices.empty:
        print(f"  ⚠️ 無股價資料,跳過")
        return

    if is_etf(symbol):
        print(f"  [股價] 寫入 {upsert_prices(prices)} 筆 (ETF)")
    else:
        time.sleep(0.4)
        valuation = fetch_per_pbr(symbol, start_date)
        merged = merge_price_and_valuation(prices, valuation)
        has_pe = "✅" if not valuation.empty else "❌"
        print(f"  [股價] 寫入 {upsert_prices(merged)} 筆 (PE: {has_pe})")

    time.sleep(0.4)
    try:
        chips = fetch_institutional(symbol, start_date)
        if not chips.empty:
            print(f"  [法人] 寫入 {upsert_chips(chips)} 筆")
        else:
            print(f"  [法人] ⚠️ 無資料")
    except Exception as e:
        print(f"  [法人] ❌ {e}")


def sync_all_light(days_back: int = 30):
    """輕量同步全部持股(平日日更用)"""
    result = supabase.table("stocks").select("symbol").execute()
    symbols = {row["symbol"] for row in result.data}
    symbols.add(MARKET_BENCHMARK)   # 大盤基準無條件納入
    symbols = sorted(symbols)
    cost = sum(2 if is_etf(s) else 3 for s in symbols)

    print(f"\n{'='*50}")
    print(f"輕量同步 {len(symbols)} 檔(只更新股價與籌碼,回看 {days_back} 天)")
    print(f"預估 API 呼叫:{cost} 次")
    print(f"{'='*50}")
    print_api_usage("(開始前)")
    print()

    start_time = time.time()
    for i, symbol in enumerate(symbols, 1):
        try:
            sync_symbol_light(symbol, days_back=days_back)
        except Exception as e:
            print(f"  ❌ {symbol} 失敗: {e}")
        if i < len(symbols):
            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"輕量同步完成 ✓ (耗時 {time.time() - start_time:.1f} 秒)")
    print_api_usage("(結束後)")
    print(f"{'='*50}\n")


def sync_all_holdings(days_back: int = 1095):
    """同步所有持股"""
    result = supabase.table("stocks").select("symbol").execute()
    # 去重:stocks 表有 user_id,同一檔被多筆持有會重複同步(純浪費額度)
    symbols = {row["symbol"] for row in result.data}
    # 大盤基準無條件納入,即使沒有任何使用者持有
    bench_added = MARKET_BENCHMARK not in symbols
    symbols.add(MARKET_BENCHMARK)
    symbols = sorted(symbols)
    n_etf = sum(1 for s in symbols if is_etf(s))

    print(f"\n{'='*50}")
    print(f"開始同步 {len(symbols)} 檔個股(回看 {days_back} 天 ≈ {days_back//365} 年)")
    print(f"其中 ETF {n_etf} 檔(跳過 PE / 月營收 / 季報 / 減資)")
    if bench_added:
        print(f"已自動納入大盤基準 {MARKET_BENCHMARK}(無人持有,但技術面對照需要)")
    print(f"{'='*50}")
    print_api_usage("(開始前)")
    print()

    start_time = time.time()
    for symbol in symbols:
        try:
            sync_symbol(symbol, days_back=days_back)
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ {symbol} 錯誤: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"全部完成 ✓ (耗時 {elapsed:.1f} 秒)")
    print_api_usage("(結束後)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode in ("light", "-l", "--light"):
        # 平日日更:只更新股價與籌碼,約 90 次呼叫
        sync_all_light()
    elif mode in ("full", "-f", "--full"):
        # 完整同步:所有資料,約 296 次呼叫(免費額度 600/小時)
        sync_all_holdings(days_back=1095)
    else:
        print("用法:")
        print("  python3 data_pipeline.py          # 完整同步(約 296 次呼叫)")
        print("  python3 data_pipeline.py light    # 輕量日更(約 90 次呼叫)")