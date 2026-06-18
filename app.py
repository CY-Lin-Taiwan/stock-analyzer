"""
持股分析 - MVP
Phase 2: 持股總覽 + 個股技術分析(K線/PE/月營收/季報) + 投資論點
"""
# ==========================================
# 1. 載入套件與環境變數
# ==========================================
import os
import time
from datetime import datetime
import subprocess
import sys
import ai_analyzer
import news_fetcher  # Phase 4.5: AI 觀察自動納入新聞
import metrics  # 進階指標(夏普 + 布林)
import streamlit as st
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

load_dotenv()

# Streamlit 頁面設定 (必須是第一個 st 指令)
st.set_page_config(page_title="持股分析", page_icon="📈", layout="wide")

# 初始化 Supabase 客戶端 (確保後續的 DB 查詢可使用)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# ==========================================
# 2. 身分識別與門禁邏輯
# ==========================================
def authenticate_user():
    """處理身分識別與密碼驗證
    
    設計:
      - 共用密碼 16888 (信任使用者不會自己改 URL)
      - URL ?user=alice 區分各自資料
      - 親友各自拿到專屬網址,進入後輸入密碼
    """
    # 從 URL 抓取 user_id
    user_id = st.query_params.get("user", "")
    
    # 沒帶 user 參數 → 拒絕
    if not user_id:
        st.title("🔒 投資者專屬通道")
        st.error("❌ 找不到使用者識別")
        st.caption("請使用提供給你的專屬網址,格式: `https://your-app.com/?user=你的代號`")
        st.stop()
    
    # 顯示名(可選,從 secrets 撈,沒設定就用 user_id)
    try:
        display_name = st.secrets.get("user_names", {}).get(user_id, user_id)
    except Exception:
        display_name = user_id
    
    # 寫入 session_state
    st.session_state["user_id"] = user_id
    st.session_state["user_name"] = display_name
    
    # 密碼門禁
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if st.session_state["password_correct"]:
        return True
    
    # 顯示登入介面
    st.title(f"🔒 {display_name} 專屬通道")
    password = st.text_input("請輸入訪問密碼", type="password")
    
    if st.button("登入", type="primary"):
        # 從 secrets 撈密碼(若沒設定就用預設 16888)
        try:
            valid_password = st.secrets.get("auth", {}).get("password", "16888")
        except Exception:
            valid_password = "16888"
        
        if password == valid_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ 密碼錯誤,請重新輸入")
    
    st.stop()


def get_user_id():
    """取得當前已認證的 user_id (供所有 query 使用)"""
    return st.session_state.get("user_id", "")


# 執行門禁驗證
authenticate_user()

# ==========================================
# 3. 共用資料撈取函式
# ==========================================
@st.cache_data(ttl=3600) # 快取 1 小時，完全不浪費資料庫額度
def get_latest_dividend(symbol: str):
    """從資料庫撈取該股票最新宣告的股利"""
    try:
        res = supabase.table("dividends") \
            .select("total_dividend") \
            .eq("symbol", symbol) \
            .order("ex_date", desc=True) \
            .limit(1) \
            .execute()
        if res.data and len(res.data) > 0:
            return float(res.data[0]["total_dividend"] or 0)
    except Exception as e:
        pass
    return 0.0


@st.cache_data(ttl=3600)
def get_all_dividends(symbol: str):
    """取得該股票所有除息紀錄(用於計算累積已領股息)"""
    try:
        res = supabase.table("dividends") \
            .select("ex_date, cash_dividend, stock_dividend, total_dividend") \
            .eq("symbol", symbol) \
            .order("ex_date", desc=False) \
            .execute()
        return res.data if res.data else []
    except Exception:
        return []


def calculate_cumulative_dividend_received(symbol: str, transactions: list) -> dict:
    """
    計算「累積已領股息」(從買進日起算,對齊每次除息日的當下持股)
    這個函式直接呼叫 ai_analyzer 的 calculate_accumulated_dividends,
    確保 UI 跟 AI 看到的數字一致
    """
    # 找該檔的初始建倉日
    sym_txns = [t for t in transactions if t.get("symbol") == symbol and t.get("action", "buy").lower() == "buy"]
    if not sym_txns:
        return {"total_dividend_received": 0.0, "per_share_avg": 0.0, "events": []}
    
    initial_date = min(t.get("date", "9999-99-99") for t in sym_txns)
    
    # 算當下持股
    current_shares = 0.0
    for t in transactions:
        if t.get("symbol") != symbol:
            continue
        action = t.get("action", "buy").lower()
        shares = float(t.get("shares", 0))
        if action == "buy":
            current_shares += shares
        elif action == "sell":
            current_shares -= shares
    
    if current_shares <= 0:
        return {"total_dividend_received": 0.0, "per_share_avg": 0.0, "events": []}
    
    # 呼叫 ai_analyzer 的精準計算
    div_info = ai_analyzer.calculate_accumulated_dividends(
        symbol, initial_date, int(current_shares), transactions=transactions
    )
    
    return {
        "total_dividend_received": div_info["total_received"],
        "per_share_avg": div_info["total_per_share"],
        "events": div_info["events"],
    }


# === 全域 CSS ===
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 16px; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 600; }
    [data-testid="stMetricDelta"] { font-size: 1.1rem !important; }
    .stDataFrame { font-size: 1rem; }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.3rem !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_supabase():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = get_supabase()


# ============================================================
# 資料存取
# ============================================================
@st.cache_data(ttl=300)
def load_stocks(user_id: str = None):
    """讀追蹤股票清單(個人資料,需 user_id 篩選)"""
    if user_id is None:
        user_id = get_user_id()
    return supabase.table("stocks").select("*").eq("user_id", user_id).order("symbol").execute().data

@st.cache_data(ttl=300)
def load_transactions(user_id: str = None):
    """讀交易紀錄(個人資料,需 user_id 篩選)"""
    if user_id is None:
        user_id = get_user_id()
    return supabase.table("transactions").select("*").eq("user_id", user_id).execute().data

@st.cache_data(ttl=300)
def load_latest_valuation(user_id: str = None):
    """取每檔最新一筆股價+PE+PB+殖利率 (只看當前 user 的追蹤清單)"""
    if user_id is None:
        user_id = get_user_id()
    stocks = supabase.table("stocks").select("symbol").eq("user_id", user_id).execute().data
    symbols = [s["symbol"] for s in stocks]
    
    result = {}
    for sym in symbols:
        rows = supabase.table("daily_prices") \
            .select("symbol,date,close,pe,pb,dividend_yield") \
            .eq("symbol", sym) \
            .order("date", desc=True) \
            .limit(1) \
            .execute().data
        if rows:
            result[sym] = rows[0]
    return result

@st.cache_data(ttl=300)
def load_pe_history(symbol: str):
    data = supabase.table("daily_prices") \
        .select("date,close,pe,pb,dividend_yield") \
        .eq("symbol", symbol) \
        .not_.is_("pe", "null") \
        .order("date") \
        .limit(2000) \
        .execute().data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_monthly_revenue(symbol: str):
    data = supabase.table("monthly_revenue") \
        .select("*") \
        .eq("symbol", symbol) \
        .order("year_month") \
        .limit(60) \
        .execute().data
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def load_data_freshness():
    """取每張表的最新日期,讓使用者知道資料新鮮度"""
    result = {}
    try:
        r = supabase.table("daily_prices").select("date").order("date", desc=True).limit(1).execute().data
        result["股價/PE"] = r[0]["date"] if r else None
    except Exception:
        result["股價/PE"] = None
    try:
        r = supabase.table("daily_chips").select("date").order("date", desc=True).limit(1).execute().data
        result["籌碼"] = r[0]["date"] if r else None
    except Exception:
        result["籌碼"] = None
    try:
        r = supabase.table("monthly_revenue").select("year_month").order("year_month", desc=True).limit(1).execute().data
        result["月營收"] = r[0]["year_month"] if r else None
    except Exception:
        result["月營收"] = None
    try:
        r = supabase.table("quarterly_financials").select("year_quarter").order("year_quarter", desc=True).limit(1).execute().data
        result["季報"] = r[0]["year_quarter"] if r else None
    except Exception:
        result["季報"] = None
    return result

@st.cache_data(ttl=300)
def load_quarterly_financials(symbol: str):
    data = supabase.table("quarterly_financials") \
        .select("*") \
        .eq("symbol", symbol) \
        .order("year_quarter") \
        .limit(20) \
        .execute().data
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def load_chips(symbol: str, days: int = 60):
    """取近 N 日的法人買賣超資料"""
    data = supabase.table("daily_chips") \
        .select("*") \
        .eq("symbol", symbol) \
        .order("date", desc=True) \
        .limit(days) \
        .execute().data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_margin(symbol: str, days: int = 60):
    """取近 N 日的融資融券資料"""
    data = supabase.table("margin_balance") \
        .select("*") \
        .eq("symbol", symbol) \
        .order("date", desc=True) \
        .limit(days) \
        .execute().data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=300)
def load_shareholding(symbol: str, days: int = 60):
    """取近 N 日的外資持股比例"""
    data = supabase.table("shareholding") \
        .select("*") \
        .eq("symbol", symbol) \
        .order("date", desc=True) \
        .limit(days) \
        .execute().data
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=300)
def load_prices(symbol):
    return supabase.table("daily_prices") \
        .select("*").eq("symbol", symbol) \
        .order("date", desc=False).execute().data


def calc_holdings():
    user_id = get_user_id()
    txns = load_transactions(user_id)
    if not txns:
        return pd.DataFrame()
    
    # 1. 抓取 stocks 表中的基本資料 (包含手動股息) - 只看當前 user 的
    try:
        stocks_data = supabase.table("stocks").select("symbol, name, industry, manual_dividend").eq("user_id", user_id).execute().data
        df_stocks = pd.DataFrame(stocks_data)
    except Exception as e:
        st.error(f"無法讀取股票基本資料: {e}")
        df_stocks = pd.DataFrame(columns=["symbol", "name", "industry", "manual_dividend"])

    df = pd.DataFrame(txns)
    holdings = []
    for symbol, group in df.groupby("symbol"):
        buy = group[group["action"] == "buy"]
        sell = group[group["action"] == "sell"]
        total_buy_shares = buy["shares"].sum()
        total_buy_cost = (buy["shares"] * buy["price"]).sum() + buy["fee"].sum()
        total_sell_shares = sell["shares"].sum()
        current_shares = total_buy_shares - total_sell_shares
        avg_cost = total_buy_cost / total_buy_shares if total_buy_shares > 0 else 0
        
        if current_shares > 0:
            holdings.append({
                "symbol": symbol,
                "shares": int(current_shares),
                "avg_cost": avg_cost,
                "total_cost": current_shares * avg_cost,
            })
            
    df_h = pd.DataFrame(holdings)
    
    # 2. 🌟 最關鍵的一步：將「計算出的持股」與「資料庫的基本資料」合併
    if not df_h.empty and not df_stocks.empty:
        # 透過 symbol (代號) 進行左合併
        df_h = pd.merge(df_h, df_stocks, on="symbol", how="left")
    
    # 確保 manual_dividend 如果是空值則補 0，避免後續計算報錯
    if "manual_dividend" in df_h.columns:
        df_h["manual_dividend"] = df_h["manual_dividend"].fillna(0)
    else:
        df_h["manual_dividend"] = 0
        
    return df_h


def calc_pe_percentile(symbol: str, current_pe: float):
    hist = load_pe_history(symbol)
    if hist.empty or current_pe is None:
        return None
    pes = hist["pe"].dropna()
    if len(pes) == 0:
        return None
    percentile = (pes < current_pe).sum() / len(pes) * 100
    return round(percentile, 0)


def calc_pb_percentile(symbol: str, current_pb: float):
    """計算 PB 歷史百分位"""
    hist = load_pe_history(symbol)  # 借用同一個資料讀取函式
    if hist.empty or current_pb is None:
        return None
    
    # 確保資料庫有撈出 pb 欄位，並剃除空值與極端異常值 (PB>20通常是防呆)
    if "pb" not in hist.columns:
        return None
        
    pbs = hist["pb"].dropna()
    pbs = pbs[(pbs > 0) & (pbs < 20)] # 防呆機制
    
    if len(pbs) == 0:
        return None
        
    percentile = (pbs < current_pb).sum() / len(pbs) * 100
    return round(percentile, 0)


def render_sticky_stock_header(label: str, latest, change: float, change_pct: float):
    """渲染 sticky header (用 JS hack 強制 sticky 在 Streamlit 環境生效)"""
    arrow = "🔺" if change >= 0 else "🔻"
    color = "#E74C3C" if change >= 0 else "#27AE60"
    
    st.markdown(f"""
    <div id="stock-sticky-wrapper" style="
        position: sticky;
        top: 0;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.97);
        backdrop-filter: blur(10px);
        padding: 14px 20px;
        margin: -1rem -1rem 1rem -1rem;
        border-bottom: 3px solid {color};
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    ">
        <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
            <span style="font-size:1.5rem; font-weight:700; color:#2C3E50;">📈 {label}</span>
            <span style="font-size:1.7rem; font-weight:700; color:{color};">{latest['close']:,.2f}</span>
            <span style="font-size:1.1rem; color:{color}; font-weight:600;">{arrow} {change:+.2f} ({change_pct:+.2f}%)</span>
            <span style="color:#888; font-size:0.95rem; margin-left:auto;">📅 {latest['date'].strftime('%Y-%m-%d')}</span>
        </div>
    </div>
    
    <script>
        // Streamlit 的容器有 overflow:auto,會破壞 position:sticky
        // 需要主動清除所有父容器的 overflow 限制
        (function() {{
            const fixSticky = () => {{
                const el = window.parent.document.getElementById('stock-sticky-wrapper') 
                          || document.getElementById('stock-sticky-wrapper');
                if (!el) return;
                let parent = el.parentElement;
                let depth = 0;
                while (parent && depth < 20) {{
                    try {{
                        const style = window.getComputedStyle(parent);
                        if (style.overflow === 'auto' || style.overflow === 'hidden' 
                            || style.overflowY === 'auto' || style.overflowY === 'hidden') {{
                            parent.style.overflow = 'visible';
                            parent.style.overflowY = 'visible';
                        }}
                    }} catch(e) {{}}
                    parent = parent.parentElement;
                    depth++;
                }}
            }};
            fixSticky();
            setTimeout(fixSticky, 200);
            setTimeout(fixSticky, 800);
            setTimeout(fixSticky, 2000);
        }})();
    </script>
    """, unsafe_allow_html=True)


# ============================================================
# 頁面 1: 持股總覽
# ============================================================
def page_portfolio_overview():
    st.title("📊 投資組合總覽")
    
    with st.expander("ℹ️ PE 是什麼?怎麼看?"):
        st.markdown("""
        **PE (本益比) = 股價 ÷ 每股盈餘 (EPS)**
        直觀理解:**用現在股價買,假設每年賺一樣多,要幾年才回本**。
        - **百分位** 比絕對值更有用:目前 PE 在過去 3 年的什麼位置
            - 0~30%:相對便宜 🟢 | 30~70%:中間 🟡 | 70~100%:相對昂貴 🔴
        """)
    
    holdings = calc_holdings()
    if holdings.empty:
        st.warning("尚無持股紀錄")
        return
    
    # 🌟 1. 載入 stocks 基本資料 (包含我們新增的 manual_dividend)
    stocks_raw = load_stocks(get_user_id())
    stocks = {s["symbol"]: s for s in stocks_raw}
    valuation = load_latest_valuation(get_user_id())
    
    # 🌟 2. 把資料對齊到 holdings DataFrame 中
    holdings["name"] = holdings["symbol"].map(lambda s: stocks.get(s, {}).get("name", s))
    holdings["industry"] = holdings["symbol"].map(lambda s: stocks.get(s, {}).get("industry", "-"))
    # 關鍵：把手動股息也對應進來
    holdings["manual_dividend"] = holdings["symbol"].map(lambda s: stocks.get(s, {}).get("manual_dividend", 0))
    
    holdings["current_price"] = holdings["symbol"].map(lambda s: valuation.get(s, {}).get("close", 0))
    holdings["pe"] = holdings["symbol"].map(lambda s: valuation.get(s, {}).get("pe"))
    holdings["pb"] = holdings["symbol"].map(lambda s: valuation.get(s, {}).get("pb"))
    
    holdings["market_value"] = holdings["shares"] * holdings["current_price"]
    holdings["pnl"] = holdings["market_value"] - holdings["total_cost"]
    holdings["pnl_pct"] = (holdings["pnl"] / holdings["total_cost"]) * 100
    
    # 處理估值百分位
    holdings["pe_percentile"] = holdings.apply(lambda row: calc_pe_percentile(row["symbol"], row["pe"]), axis=1)
    holdings["pb_percentile"] = holdings.apply(lambda row: calc_pb_percentile(row["symbol"], row["pb"]), axis=1)
    
    # === KPI ===
    total_cost = holdings["total_cost"].sum()
    total_value = holdings["market_value"].sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    # 🌟 3. 計算預估總股息 (修正後的邏輯)
    total_expected_dividend = 0
    for index, row in holdings.iterrows():
        sym = row["symbol"]
        shares = row["shares"] 
        man_div = row.get("manual_dividend", 0)
        
        # 判定：如果有手填且 > 0，就用手填的；否則才去抓系統自動的
        if pd.notna(man_div) and man_div > 0:
            latest_div_per_share = man_div
        else:
            latest_div_per_share = get_latest_dividend(sym)
            
        total_expected_dividend += (latest_div_per_share * shares)
        
    portfolio_cost_yield = (total_expected_dividend / total_cost * 100) if total_cost > 0 else 0
    
    # === Phase 4.7: 計算每檔股票的「累積已領股息」===
    all_txns = load_transactions(get_user_id())
    holdings["accumulated_dividend"] = holdings["symbol"].apply(
        lambda sym: calculate_cumulative_dividend_received(sym, all_txns)["total_dividend_received"]
    )
    total_accumulated_dividend = holdings["accumulated_dividend"].sum()
    
    # === Phase 4.7: 含息成本切換 ===
    include_div_in_cost = st.toggle(
        "💰 顯示含息成本 (扣除累積已領股息)",
        value=False,
        help=(
            "📖 **什麼是含息成本?**\n\n"
            "把「已經領到的股息」當作是「拿回部分本金」,從成本中扣除。\n\n"
            "**開啟後:**\n"
            "- 成本變低(扣已領股息)\n"
            "- 未實現損益變漂亮(因為基準下降)\n"
            "- 適合存股者觀察「真實風險暴露」\n\n"
            "**關閉時(預設):**\n"
            "- 成本不變(買進時的價錢)\n"
            "- 與券商 APP 口徑一致\n"
            "- 適合對帳、報稅\n\n"
            "**累積已領股息:** 從買進日起算,對齊每次除息日當下持股計算"
        ),
        key="include_div_in_cost"
    )
    
    if include_div_in_cost:
        # 含息口徑: 成本扣除累積已領股息
        holdings["display_cost"] = holdings["total_cost"] - holdings["accumulated_dividend"]
        holdings["display_avg_cost"] = holdings["display_cost"] / holdings["shares"]
        holdings["display_pnl"] = holdings["market_value"] - holdings["display_cost"]
        holdings["display_pnl_pct"] = (holdings["display_pnl"] / holdings["display_cost"]) * 100
        
        display_total_cost = total_cost - total_accumulated_dividend
        display_total_pnl = total_value - display_total_cost
        display_total_pnl_pct = (display_total_pnl / display_total_cost) * 100 if display_total_cost > 0 else 0
        
        cost_label_suffix = " (含息)"
    else:
        # 預設口徑: 成本 = 買進事實
        holdings["display_cost"] = holdings["total_cost"]
        holdings["display_avg_cost"] = holdings["avg_cost"]
        holdings["display_pnl"] = holdings["pnl"]
        holdings["display_pnl_pct"] = holdings["pnl_pct"]
        
        display_total_cost = total_cost
        display_total_pnl = total_pnl
        display_total_pnl_pct = total_pnl_pct
        
        cost_label_suffix = ""

    # 🌟 4. 顯示卡片 (採用「萬」單位避開被截斷)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"💰 總成本{cost_label_suffix}", f"NT$ {display_total_cost/10000:,.1f} 萬")
    c2.metric("📊 總市值", f"NT$ {total_value/10000:,.1f} 萬")
    c3.metric(
        f"📈 未實現損益{cost_label_suffix}", 
        f"NT$ {display_total_pnl/10000:+,.1f} 萬", 
        f"{display_total_pnl_pct:+.2f}%"
    )
    c4.metric("📦 持股檔數", f"{len(holdings)} 檔")
    c5.metric(
        "💸 預計年領股息", 
        f"NT$ {total_expected_dividend/10000:,.2f} 萬", # 🌟 改成 .2f 顯示精確位數
        f"成本殖利率: {portfolio_cost_yield:.2f}%"
    )
    
    # 累積已領股息資訊條 (永遠顯示,讓使用者知道「實際領了多少」)
    if total_accumulated_dividend > 0:
        st.info(
            f"💵 **累積已領股息: NT$ {total_accumulated_dividend:,.0f} 元** "
            f"({total_accumulated_dividend/10000:.2f} 萬) "
            f"｜ 從買進日起算,對齊各次除息日當下持股計算"
        )
    
    st.divider()
    
   # === 持股明細 ===
    st.subheader("📋 持股明細")

    # 1. 初始化列表並補齊 PE 與 PB 數據
    pe_values, pe_pcts = [], []
    pb_values, pb_pcts = [], []
    
    for index, row in holdings.iterrows():
        sym = row["symbol"]
        df_hist = load_pe_history(sym) # 這支函式裡面已經包含 pb 欄位
        if df_hist is not None and not df_hist.empty:
            last_val = df_hist.iloc[-1]
            
            # 處理 PE
            curr_pe = last_val.get("pe")
            pe_values.append(curr_pe)
            pe_pcts.append(calc_pe_percentile(sym, curr_pe))
            
            # 處理 PB
            curr_pb = last_val.get("pb")
            pb_values.append(curr_pb)
            pb_pcts.append(calc_pb_percentile(sym, curr_pb))
        else:
            pe_values.append(None); pe_pcts.append(None)
            pb_values.append(None); pb_pcts.append(None)

    # 將數據寫回原始 DataFrame
    holdings["pe"] = pe_values
    holdings["pe_percentile"] = pe_pcts
    holdings["pb"] = pb_values
    holdings["pb_percentile"] = pb_pcts

    # 2. 先進行重新命名與排序
    display = holdings.sort_values("market_value", ascending=False).copy()
    # Phase 4.7: 用 display_* 系列(會根據 toggle 切換含/不含息)
    display = display.rename(columns={
        "symbol": "代號", "name": "名稱", "industry": "產業",
        "shares": "股數", "display_avg_cost": "均價", "current_price": "現價",
        "display_cost": "成本", "market_value": "市值",
        "display_pnl": "損益", "display_pnl_pct": "報酬率",
    })

    # 3. 定義格式化函式 (PE 與 PB 共用邏輯)
    def format_valuation(val, pct):
        if pd.isna(val) or val is None:
            return "-"
        val_str = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
        if pd.notna(pct):
            p = int(pct)
            emoji = "🔴" if p >= 70 else ("🟡" if p >= 30 else "🟢")
            return f"{val_str} ({p}%) {emoji}"
        return val_str

    # 4. 套用格式化
    display["PE"] = display.apply(lambda r: format_valuation(r["pe"], r["pe_percentile"]), axis=1)
    display["PB"] = display.apply(lambda r: format_valuation(r["pb"], r["pb_percentile"]), axis=1)

    # 5. 欄位篩選 (加入 PB)
    display = display[[
        "代號", "名稱", "產業", "股數", "現價",
        "成本", "市值", "損益", "報酬率", "PE", "PB"
    ]]

    # 6. 渲染表格
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=(len(display) + 1) * 38 + 10,
        column_config={
            "代號":   st.column_config.TextColumn(width=70),
            "名稱":   st.column_config.TextColumn(width=90),
            "產業":   st.column_config.TextColumn(width=100),
            "股數":   st.column_config.NumberColumn(format="localized", width=80),
            "現價":   st.column_config.NumberColumn(format="%.2f", width=80),
            "成本":   st.column_config.NumberColumn(format="localized", width=110),
            "市值":   st.column_config.NumberColumn(format="localized", width=110),
            "損益":   st.column_config.NumberColumn(format="localized", width=110),
            "報酬率": st.column_config.NumberColumn(format="%+.2f%%", width=85),
            "PE":    st.column_config.TextColumn(width=120, help="本益比 (歷史位階)"),
            "PB":    st.column_config.TextColumn(width=120, help="股價淨值比 (歷史位階)"),
        }
    )
    
    st.caption("💡 括號內為過去 3 年歷史位階：🟢便宜(0-30%) 🟡合理(30-70%) 🔴偏貴(70-100%)")
    st.divider()
    
    # === 雙圓餅圖 ===
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🥧 個股配置")
        pie_df = holdings.copy()
        pie_df["label"] = pie_df["symbol"] + " " + pie_df["name"]
        fig = px.pie(pie_df, values="market_value", names="label", hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=14)
        fig.update_layout(height=400, legend=dict(orientation="v", x=1.05, y=0.5),
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🏭 產業分布")
        ind_df = holdings.groupby("industry")["market_value"].sum().reset_index()
        fig = px.pie(ind_df, values="market_value", names="industry", hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=14)
        fig.update_layout(height=400, legend=dict(orientation="v", x=1.05, y=0.5),
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # === 報酬率 ===
    st.subheader("📈 個股報酬率比較")
    bar_df = holdings.copy()
    bar_df["label"] = bar_df["symbol"] + " " + bar_df["name"]
    bar_df = bar_df.sort_values("pnl_pct", ascending=True)
    bar_df["color"] = bar_df["pnl_pct"].map(lambda x: "#E74C3C" if x >= 0 else "#27AE60")
    
    fig = go.Figure(go.Bar(
        x=bar_df["pnl_pct"], y=bar_df["label"],
        orientation="h", marker_color=bar_df["color"],
        text=bar_df["pnl_pct"].map(lambda x: f"{x:+.2f}%"),
        textposition="outside", textfont=dict(size=14),
    ))
    fig.update_layout(height=350, xaxis_title="報酬率 (%)", yaxis_title="",
                      showlegend=False, margin=dict(l=20, r=80, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 頁面 2: 個股技術分析
# ============================================================
def page_stock_detail():
    stocks = load_stocks(get_user_id())
    stock_options = {f"{s['symbol']} {s['name']}": s['symbol'] for s in stocks}
    
    # 個股選擇 + 圖表設定都放在側邊欄(scroll 也能切換)
    with st.sidebar:
        st.divider()
        st.subheader("📈 選擇個股")
        selected_label = st.selectbox(
            "個股",
            list(stock_options.keys()),
            label_visibility="collapsed"
        )
        selected_symbol = stock_options[selected_label]
        
        st.divider()
        st.subheader("圖表設定")
        show_ma5 = st.checkbox("5MA (週線)", value=True)
        show_ma20 = st.checkbox("20MA (月線)", value=True)
        show_ma60 = st.checkbox("60MA (季線)", value=False)
        days_range = st.slider("K線顯示天數", 30, 250, 90, step=10)
    
# 1. 讀取該檔股票的歷史資料，準備計算估值
    df = load_pe_history(selected_symbol) 
    
    if not df.empty:
        current_pe = df.iloc[-1]["pe"]
        current_pb = df.iloc[-1]["pb"] if "pb" in df.columns else None
        current_yield = df.iloc[-1]["dividend_yield"] if "dividend_yield" in df.columns else 0
        
        pe_percentile = calc_pe_percentile(selected_symbol, current_pe)
        pb_percentile = calc_pb_percentile(selected_symbol, current_pb)
        
        # 2. 渲染估值看板
        st.subheader("📊 估值看板")
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.metric(
                "PE (本益比)", 
                f"{current_pe:.2f}" if pd.notna(current_pe) else "N/A", 
                f"歷史位階: {pe_percentile:.0f}%" if pe_percentile is not None else "N/A",
                delta_color="inverse"
            )
        with vcol2:
            st.metric(
                "PB (股價淨值比)", 
                f"{current_pb:.2f}" if pd.notna(current_pb) else "N/A", 
                f"歷史位階: {pb_percentile:.0f}%" if pb_percentile is not None else "N/A",
                delta_color="inverse"
            )
        with vcol3:
            st.metric(
                "Yield (殖利率)", 
                f"{current_yield:.2f}%" if pd.notna(current_yield) else "N/A"
            )
        
        # === 進階指標: 夏普值 + 布林通道 ===
        # 用該股 1 年日線計算
        try:
            price_records = load_prices(selected_symbol)
            if price_records and len(price_records) >= 30:
                df_for_metrics = pd.DataFrame(price_records)
                df_for_metrics["date"] = pd.to_datetime(df_for_metrics["date"])
                df_for_metrics = df_for_metrics.sort_values("date").reset_index(drop=True)
                # 取最近 252 個交易日(約 1 年)算夏普
                last_year_prices = df_for_metrics["close"].tail(252)
                sharpe_val = metrics.calculate_sharpe(last_year_prices)
                # 布林通道用全部資料(會自動取最近 20 日)
                bb = metrics.calculate_bollinger_bands(df_for_metrics["close"])
            else:
                sharpe_val = None
                bb = None
        except Exception as e:
            print(f"[app] 進階指標計算失敗: {e}")
            sharpe_val = None
            bb = None
        
        st.markdown("### 📐 進階指標")
        
        # === 夏普值 ===
        st.markdown("##### 個股夏普值 (1 年)")
        sharpe_grade = metrics.get_sharpe_grade(sharpe_val)
        
        scol1, scol2 = st.columns([1, 3])
        with scol1:
            sharpe_display = f"{sharpe_val}" if sharpe_val is not None else "N/A"
            st.metric(
                "夏普值",
                sharpe_display,
                help=(
                    "📖 **夏普值 = (年化報酬 − 無風險利率) ÷ 年化波動度**\n\n"
                    "衡量「每承擔一單位風險換到多少超額報酬」。\n\n"
                    "**參數:** 過去 252 個交易日, 無風險利率 1.5%, 年化計算"
                ),
                label_visibility="collapsed"
            )
        with scol2:
            # 視覺化分級
            tier = sharpe_grade["tier"]
            if tier >= 0:
                bars = ["⬜", "⬜", "⬜", "⬜"]
                if tier == 0:
                    bars[0] = "🟥"
                elif tier == 1:
                    bars[1] = "🟧"
                elif tier == 2:
                    bars[2] = "🟩"
                elif tier == 3:
                    bars[3] = "🟦"
                
                st.markdown(
                    f"**{bars[0]} 負值區** | **{bars[1]} 裸奔狀態** | "
                    f"**{bars[2]} 標準裝甲** | **{bars[3]} 降維打擊**"
                )
                st.markdown(f"**{sharpe_grade['position_text']}**")
                st.caption(f"💬 {sharpe_grade['description']}")
            else:
                st.caption(sharpe_grade['description'])
        
        # === 布林通道位階 ===
        st.markdown("##### 布林通道位階")
        bb_grade = metrics.get_bollinger_grade(bb)
        
        bcol1, bcol2 = st.columns([1, 3])
        with bcol1:
            if bb and bb.get("percent_b") is not None:
                pb_display = f"{bb['percent_b']:.0f}%"
            else:
                pb_display = "N/A"
            
            st.metric(
                "%B 位階",
                pb_display,
                help=(
                    "📖 **布林通道 = 20 日均線 ± 2 倍標準差**\n\n"
                    "**%B 位階:** 現價在通道內的相對位置\n"
                    "- 0% = 跌到下軌 / 100% = 漲到上軌\n"
                    "- 50% = 在中軌(均線)\n\n"
                    "**參數:** 20 日, 2 標準差"
                ),
                label_visibility="collapsed"
            )
        with bcol2:
            tier = bb_grade["tier"]
            if tier >= 0:
                # 6 個區間的視覺化
                bars = ["⬜"] * 6
                labels = ["跌破\n下軌", "靠近\n下軌", "中軌\n之下", "中軌\n之上", "靠近\n上軌", "突破\n上軌"]
                colors = ["🟥", "🟧", "🟨", "🟩", "🟧", "🟥"]
                bars[tier] = colors[tier]
                
                bar_line = " | ".join([f"{bars[i]} {labels[i].replace(chr(10), '')}" for i in range(6)])
                st.markdown(bar_line)
                st.markdown(f"**{bb_grade['position_text']}**")
                if bb:
                    st.caption(
                        f"📊 現價 {bb.get('current')} / 上軌 {bb.get('upper')} / "
                        f"中軌 {bb.get('middle')} / 下軌 {bb.get('lower')}"
                    )
                st.caption(f"💬 {bb_grade['description']}")
                st.caption(f"📈 **市場含義:** {bb_grade['trading_implication']}")
            else:
                st.caption(bb_grade['description'])
        
        st.divider()

    else:
        st.warning("尚無此檔股票的歷史估值資料。")


    prices = load_prices(selected_symbol)
    if not prices:
        st.warning(f"{selected_symbol} 尚未有資料")
        return
    
    df = pd.DataFrame(prices)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()
    df_view = df.tail(days_range).reset_index(drop=True)
    
    latest = df_view.iloc[-1]
    prev = df_view.iloc[-2] if len(df_view) > 1 else latest
    change = latest["close"] - prev["close"]
    change_pct = (change / prev["close"]) * 100

    # 算 holding_context (給 AI 觀察跟結果顯示用)
    all_txns = load_transactions(get_user_id())
    valuation = load_latest_valuation(get_user_id())
    val_data = valuation.get(selected_symbol, {})
    holding_ctx = ai_analyzer.build_holding_context(
    symbol=selected_symbol,
    transactions=all_txns,
    current_price=val_data.get("close", 0),
    )
    
    # === Sticky Header(scroll 時固定在最上方) ===
    render_sticky_stock_header(selected_label, latest, change, change_pct)
    
    # === KPI 卡片 ===
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新收盤", f"{latest['close']:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    c2.metric("開盤", f"{latest['open']:,.2f}")
    c3.metric("最高", f"{latest['high']:,.2f}")
    c4.metric("最低", f"{latest['low']:,.2f}")
    c5.metric("成交量(張)", f"{latest['volume']/1000:,.0f}")
    
    # PE 訊息
    if pd.notna(latest.get("pe")):
        pe_pct = calc_pe_percentile(selected_symbol, latest["pe"])
        pe_msg = f"目前 PE: **{latest['pe']:.2f}**"
        if pe_pct is not None:
            pe_msg += f" | 在過去 3 年的第 **{int(pe_pct)}** 百分位"
            if pe_pct >= 70:
                pe_msg += " 🔴 偏貴"
            elif pe_pct >= 30:
                pe_msg += " 🟡 中間"
            else:
                pe_msg += " 🟢 偏便宜"
        st.info(pe_msg)
    
    st.caption(f"資料日期:{latest['date'].strftime('%Y-%m-%d')} | 顯示 {len(df_view)} 筆 (全部 {len(df)} 筆)")
    
    # === K 線圖 ===
    st.subheader("📊 K 線圖")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df_view["date"], open=df_view["open"], high=df_view["high"],
        low=df_view["low"], close=df_view["close"],
        increasing_line_color="red", decreasing_line_color="green",
        increasing_fillcolor="red", decreasing_fillcolor="green",
        name="K 線", showlegend=False,
    ), row=1, col=1)
    if show_ma5:
        fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["MA5"],
            name="5MA", line=dict(color="#FFA500", width=1.5)), row=1, col=1)
    if show_ma20:
        fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["MA20"],
            name="20MA", line=dict(color="#1E90FF", width=1.5)), row=1, col=1)
    if show_ma60:
        fig.add_trace(go.Scatter(x=df_view["date"], y=df_view["MA60"],
            name="60MA", line=dict(color="#9370DB", width=1.5)), row=1, col=1)
    volume_colors = ["red" if c >= o else "green"
                     for c, o in zip(df_view["close"], df_view["open"])]
    fig.add_trace(go.Bar(
        x=df_view["date"], y=df_view["volume"]/1000,
        marker_color=volume_colors, showlegend=False,
    ), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                      margin=dict(l=40, r=40, t=40, b=40))
    fig.update_yaxes(title_text="股價", row=1, col=1)
    fig.update_yaxes(title_text="張數", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    
    # === PE band ===
    pe_hist = load_pe_history(selected_symbol)
    if not pe_hist.empty:
        st.subheader("📐 PE 估值帶 (近 3 年)")
        
        pes = pe_hist["pe"].dropna()
        pe_min = pes.min()
        pe_max = pes.max()
        pe_median = pes.median()
        pe_p25 = pes.quantile(0.25)
        pe_p75 = pes.quantile(0.75)
        
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        cc1.metric("PE 最低", f"{pe_min:.2f}")
        cc2.metric("PE 25 分位", f"{pe_p25:.2f}")
        cc3.metric("PE 中位數", f"{pe_median:.2f}")
        cc4.metric("PE 75 分位", f"{pe_p75:.2f}")
        cc5.metric("PE 最高", f"{pe_max:.2f}")
        
        # =====================================================
    # 🌊 歷史估值河流圖 (PE & PB 雙頁籤)
    # =====================================================
    st.subheader("🌊 歷史估值河流圖")
    
    # 建立兩個頁籤
    tab_pe, tab_pb = st.tabs(["📊 PE (本益比) 歷史區間", "📉 PB (股價淨值比) 歷史區間"])
    
    # =========================
    # 第一頁籤：PE 圖表 (原本的邏輯)
    # =========================
    with tab_pe:
        pe_hist = df[(df["pe"] > 0) & (df["pe"] < 100)].copy()
        if not pe_hist.empty:
            pes = pe_hist["pe"].dropna()
            pe_min, pe_p25, pe_median, pe_p75, pe_max = pes.min(), pes.quantile(0.25), pes.median(), pes.quantile(0.75), pes.max()
            
            fig_pe = go.Figure()
            fig_pe.add_hrect(y0=pe_min, y1=pe_p25, fillcolor="#27AE60", opacity=0.1,
                             line_width=0, annotation_text="便宜區", annotation_position="left")
            fig_pe.add_hrect(y0=pe_p25, y1=pe_p75, fillcolor="#F39C12", opacity=0.1,
                             line_width=0, annotation_text="合理區", annotation_position="left")
            fig_pe.add_hrect(y0=pe_p75, y1=pe_max, fillcolor="#E74C3C", opacity=0.1,
                             line_width=0, annotation_text="偏貴區", annotation_position="left")
            fig_pe.add_trace(go.Scatter(
                x=pe_hist["date"], y=pe_hist["pe"],
                mode="lines", name="PE",
                line=dict(color="#2C3E50", width=1.5),
            ))
            fig_pe.add_hline(y=pe_median, line_dash="dash", line_color="gray",
                             annotation_text=f"中位數 {pe_median:.2f}",
                             annotation_position="right")
            fig_pe.update_layout(
                height=400, yaxis_title="PE (倍)", xaxis_title="日期",
                hovermode="x unified", showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_pe, use_container_width=True)
            # 將 datetime 轉為日期字串避免報錯
            st.caption(f"💡 PE 帶基於 {len(pes)} 筆歷史資料 ({pe_hist['date'].min().strftime('%Y-%m-%d')} ~ {pe_hist['date'].max().strftime('%Y-%m-%d')})")
        else:
            st.info("此檔尚無 PE 歷史資料")

    # =========================
    # 第二頁籤：PB 圖表 (依樣畫葫蘆)
    # =========================
    with tab_pb:
        pb_hist = df[(df["pb"] > 0) & (df["pb"] < 20)].copy() # 剃除異常值
        if not pb_hist.empty:
            pbs = pb_hist["pb"].dropna()
            pb_min, pb_p25, pb_median, pb_p75, pb_max = pbs.min(), pbs.quantile(0.25), pbs.median(), pbs.quantile(0.75), pbs.max()
            
            fig_pb = go.Figure()
            fig_pb.add_hrect(y0=pb_min, y1=pb_p25, fillcolor="#27AE60", opacity=0.1,
                             line_width=0, annotation_text="便宜區", annotation_position="left")
            fig_pb.add_hrect(y0=pb_p25, y1=pb_p75, fillcolor="#F39C12", opacity=0.1,
                             line_width=0, annotation_text="合理區", annotation_position="left")
            fig_pb.add_hrect(y0=pb_p75, y1=pb_max, fillcolor="#E74C3C", opacity=0.1,
                             line_width=0, annotation_text="偏貴區", annotation_position="left")
            fig_pb.add_trace(go.Scatter(
                x=pb_hist["date"], y=pb_hist["pb"],
                mode="lines", name="PB",
                line=dict(color="#2C3E50", width=1.5),
            ))
            fig_pb.add_hline(y=pb_median, line_dash="dash", line_color="gray",
                             annotation_text=f"中位數 {pb_median:.2f}",
                             annotation_position="right")
            fig_pb.update_layout(
                height=400, yaxis_title="PB (倍)", xaxis_title="日期",
                hovermode="x unified", showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_pb, use_container_width=True)
            st.caption(f"💡 PB 帶基於 {len(pbs)} 筆歷史資料 ({pb_hist['date'].min().strftime('%Y-%m-%d')} ~ {pb_hist['date'].max().strftime('%Y-%m-%d')})")
        else:
            st.info("此檔尚無 PB 歷史資料")
    
    # === 月營收區塊 ===
    revenue_df = load_monthly_revenue(selected_symbol)
    if not revenue_df.empty:
        st.divider()
        st.subheader("💰 月營收分析")
        
        latest_rev = revenue_df.iloc[-1]
        
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(
            f"📅 {latest_rev['year_month']} 月營收",
            f"{latest_rev['revenue']/100_000_000:,.2f} 億"   # ← 100_000_000 才是 1 億
        )
        if pd.notna(latest_rev.get("revenue_yoy")):
            yoy = latest_rev["revenue_yoy"]
            rc2.metric(
                "年增率 YoY",
                f"{yoy:+.2f}%",
                delta=f"{'成長' if yoy > 0 else '衰退'}",
                delta_color="normal" if yoy > 0 else "inverse"
            )
        if pd.notna(latest_rev.get("revenue_mom")):
            mom = latest_rev["revenue_mom"]
            rc3.metric("月增率 MoM", f"{mom:+.2f}%")
        if pd.notna(latest_rev.get("cumulative_yoy")):
            cum_yoy = latest_rev["cumulative_yoy"]
            rc4.metric("累計年增率", f"{cum_yoy:+.2f}%")
        
        recent = revenue_df.tail(24).copy()
        
        fig_rev = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rev.add_trace(
            go.Bar(
                x=recent["year_month"],
                y=recent["revenue"] / 100_000_000,
                name="月營收(億)",
                marker_color="#3498DB",
                opacity=0.7,
            ),
            secondary_y=False,
        )
        
        if "revenue_yoy" in recent.columns and recent["revenue_yoy"].notna().any():
            yoy_colors = ["#E74C3C" if v >= 0 else "#27AE60" 
                          for v in recent["revenue_yoy"].fillna(0)]
            fig_rev.add_trace(
                go.Scatter(
                    x=recent["year_month"],
                    y=recent["revenue_yoy"],
                    mode="lines+markers",
                    name="YoY (%)",
                    line=dict(color="#E67E22", width=2.5),
                    marker=dict(size=8, color=yoy_colors),
                ),
                secondary_y=True,
            )
            fig_rev.add_hline(y=0, line_dash="dot", line_color="gray",
                              secondary_y=True)
        
        fig_rev.update_layout(
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        fig_rev.update_xaxes(title_text="月份")
        fig_rev.update_yaxes(title_text="月營收 (億)", secondary_y=False)
        fig_rev.update_yaxes(title_text="YoY (%)", secondary_y=True)
        
        st.plotly_chart(fig_rev, use_container_width=True)
        st.caption(f"💡 共 {len(revenue_df)} 個月資料,顯示近 24 個月。月營收用藍色長條(左軸),YoY 用橘線(右軸,紅點=成長/綠點=衰退)")
    else:
        st.info("此檔尚無月營收資料")
    
    # === 季報財務指標區塊 ===
    qfin = load_quarterly_financials(selected_symbol)
    if not qfin.empty:
        st.divider()
        st.subheader("📑 季報財務指標")
        
        latest_q = qfin.iloc[-1]
        
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric(
            f"📅 {latest_q['year_quarter']} EPS",
            f"{latest_q['eps']:.2f}" if pd.notna(latest_q.get('eps')) else "-"
        )
        if pd.notna(latest_q.get("gross_margin")):
            qc2.metric("毛利率", f"{latest_q['gross_margin']:.2f}%")
        if pd.notna(latest_q.get("operating_margin")):
            qc3.metric("營業利益率", f"{latest_q['operating_margin']:.2f}%")
        if pd.notna(latest_q.get("roe")):
            qc4.metric("ROE", f"{latest_q['roe']:.2f}%")
        
        recent = qfin.tail(12).copy()
        
        fig_q = go.Figure()
        if "gross_margin" in recent.columns and recent["gross_margin"].notna().any():
            fig_q.add_trace(go.Scatter(
                x=recent["year_quarter"], y=recent["gross_margin"],
                mode="lines+markers", name="毛利率",
                line=dict(color="#3498DB", width=2.5),
                marker=dict(size=8),
            ))
        if "operating_margin" in recent.columns and recent["operating_margin"].notna().any():
            fig_q.add_trace(go.Scatter(
                x=recent["year_quarter"], y=recent["operating_margin"],
                mode="lines+markers", name="營業利益率",
                line=dict(color="#9B59B6", width=2.5),
                marker=dict(size=8),
            ))
        if "net_margin" in recent.columns and recent["net_margin"].notna().any():
            fig_q.add_trace(go.Scatter(
                x=recent["year_quarter"], y=recent["net_margin"],
                mode="lines+markers", name="淨利率",
                line=dict(color="#27AE60", width=2.5),
                marker=dict(size=8),
            ))
        if "roe" in recent.columns and recent["roe"].notna().any():
            fig_q.add_trace(go.Scatter(
                x=recent["year_quarter"], y=recent["roe"],
                mode="lines+markers", name="ROE",
                line=dict(color="#E74C3C", width=2.5, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            ))
        
        fig_q.update_layout(
            height=400,
            yaxis_title="%",
            xaxis_title="季度",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_q, use_container_width=True)
        
        # EPS 長條圖
        st.subheader("📊 季 EPS 走勢")
        eps_df = recent.dropna(subset=["eps"]).copy()
        if not eps_df.empty:
            eps_df["color"] = eps_df["eps"].map(lambda v: "#E74C3C" if v >= 0 else "#27AE60")
            fig_eps = go.Figure(go.Bar(
                x=eps_df["year_quarter"],
                y=eps_df["eps"],
                marker_color=eps_df["color"],
                text=eps_df["eps"].map(lambda v: f"{v:.2f}"),
                textposition="inside",
                textfont=dict(color="white", size=14, family="Arial Black"),
                insidetextanchor="end",
            ))
            y_max = eps_df["eps"].max()
            fig_eps.update_yaxes(range=[0, y_max * 1.15])
            fig_eps.update_layout(
                height=300,
                yaxis_title="EPS (元)",
                xaxis_title="季度",
                showlegend=False,
                margin=dict(l=40, r=40, t=20, b=40),
            )
            st.plotly_chart(fig_eps, use_container_width=True)
        
        with st.expander("📋 季報原始資料"):
            display_q = qfin.tail(12).sort_values("year_quarter", ascending=False)
            st.dataframe(
                display_q[[
                    "year_quarter", "eps", "gross_margin", "operating_margin",
                    "net_margin", "roe", "roa"
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "year_quarter": "季度",
                    "eps": st.column_config.NumberColumn("EPS", format="%.2f"),
                    "gross_margin": st.column_config.NumberColumn("毛利率%", format="%.2f"),
                    "operating_margin": st.column_config.NumberColumn("營業利益率%", format="%.2f"),
                    "net_margin": st.column_config.NumberColumn("淨利率%", format="%.2f"),
                    "roe": st.column_config.NumberColumn("ROE%", format="%.2f"),
                    "roa": st.column_config.NumberColumn("ROA%", format="%.2f"),
                }
            )
        
        st.caption(f"💡 共 {len(qfin)} 季資料,顯示近 12 季趨勢")
    else:
        st.info("此檔尚無季報資料")


# =====================================================
    # 籌碼面區塊
    # =====================================================
    chips = load_chips(selected_symbol, days=60)
    margin = load_margin(selected_symbol, days=60)
    sh_df = load_shareholding(selected_symbol, days=60)
    
    if not chips.empty:
        st.divider()
        st.subheader("🎯 籌碼面分析")
        
        # 取近 5 日累計
        recent_5 = chips.tail(5)
        foreign_5d = recent_5["foreign_net"].sum() / 1000  # 轉成張
        trust_5d = recent_5["trust_net"].sum() / 1000
        dealer_5d = recent_5["dealer_net"].sum() / 1000
        
        # 連續買賣超天數
        latest_dir = chips.iloc[-1]
        consecutive_days = 0
        for i in range(len(chips) - 1, -1, -1):
            if (chips.iloc[i]["foreign_net"] > 0) == (latest_dir["foreign_net"] > 0):
                consecutive_days += 1
            else:
                break
        
        # 融資變化
        margin_latest = margin.iloc[-1] if not margin.empty else None
        margin_30d_ago = margin.iloc[-30] if len(margin) >= 30 else (margin.iloc[0] if not margin.empty else None)
        
        # 外資持股比
        sh_latest = sh_df.iloc[-1] if not sh_df.empty else None
        sh_30d_ago = sh_df.iloc[-30] if len(sh_df) >= 30 else (sh_df.iloc[0] if not sh_df.empty else None)
        
        # KPI 卡片(4個)
        bc1, bc2, bc3, bc4 = st.columns(4)
        
        # 外資 5 日
        f_color = "🟢" if foreign_5d > 0 else "🔴" if foreign_5d < 0 else "⚪"
        bc1.metric(
            f"{f_color} 外資 5 日累計",
            f"{foreign_5d:+,.0f} 張"
        )
        
        # 投信 5 日
        t_color = "🟢" if trust_5d > 0 else "🔴" if trust_5d < 0 else "⚪"
        bc2.metric(
            f"{t_color} 投信 5 日累計",
            f"{trust_5d:+,.0f} 張"
        )
        
        # 融資餘額(張) + 30 日變化
        if margin_latest is not None and margin_30d_ago is not None:
            m_now = margin_latest["margin_balance"]
            m_old = margin_30d_ago["margin_balance"]
            m_chg = (m_now - m_old) / m_old * 100 if m_old > 0 else 0
            m_color = "🔴" if m_chg > 10 else "🟢" if m_chg < -10 else "🟡"
            bc3.metric(
                f"{m_color} 融資餘額",
                f"{m_now:,.0f} 張",
                f"30 日 {m_chg:+.1f}%"
            )
        
        # 外資持股比 + 30 日變化
        if sh_latest is not None and sh_30d_ago is not None:
            sh_now = sh_latest.get("foreign_holding_ratio")
            sh_old = sh_30d_ago.get("foreign_holding_ratio")
            if pd.notna(sh_now) and pd.notna(sh_old):
                sh_chg = sh_now - sh_old
                bc4.metric(
                    "🌐 外資持股比",
                    f"{sh_now:.2f}%",
                    f"30 日 {sh_chg:+.2f} 百分點"
                )
        
        # 連續買賣超提示
        direction = "買超" if latest_dir["foreign_net"] > 0 else "賣超"
        if consecutive_days >= 3:
            color_box = "success" if direction == "買超" else "error"
            getattr(st, color_box)(
                f"📊 外資已連續 **{consecutive_days}** 天{direction}"
            )
        
        # === 主圖:法人買賣超(堆疊柱)+股價(折線)疊圖 ===
        # 合併法人資料跟股價
        chips_with_price = chips.merge(
            pd.DataFrame(load_prices(selected_symbol))[["date", "close"]].assign(
                date=lambda d: pd.to_datetime(d["date"])
            ),
            on="date", how="left"
        )
        
        fig_chips = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 法人買賣超(堆疊長條,單位:張)
        fig_chips.add_trace(
            go.Bar(
                x=chips_with_price["date"],
                y=chips_with_price["foreign_net"] / 1000,
                name="外資",
                marker_color="#3498DB",
            ),
            secondary_y=False,
        )
        fig_chips.add_trace(
            go.Bar(
                x=chips_with_price["date"],
                y=chips_with_price["trust_net"] / 1000,
                name="投信",
                marker_color="#9B59B6",
            ),
            secondary_y=False,
        )
        fig_chips.add_trace(
            go.Bar(
                x=chips_with_price["date"],
                y=chips_with_price["dealer_net"] / 1000,
                name="自營",
                marker_color="#F39C12",
            ),
            secondary_y=False,
        )
        
        # 股價折線(右軸)
        fig_chips.add_trace(
            go.Scatter(
                x=chips_with_price["date"],
                y=chips_with_price["close"],
                name="股價",
                mode="lines",
                line=dict(color="#E74C3C", width=2.5),
            ),
            secondary_y=True,
        )
        
        # 0 線
        fig_chips.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=False)
        
        fig_chips.update_layout(
            height=450,
            barmode="relative",  # 堆疊柱
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=40, r=40, t=40, b=40),
            title="近 60 日 三大法人買賣超 vs 股價"
        )
        fig_chips.update_yaxes(title_text="買賣超(張)", secondary_y=False)
        fig_chips.update_yaxes(title_text="股價", secondary_y=True)
        
        st.plotly_chart(fig_chips, use_container_width=True)
        
        st.caption("💡 **怎麼看**:長條在 0 上方 = 買超、下方 = 賣超。觀察『法人方向 vs 股價方向』是否一致。一致 = 趨勢健康;背離 = 警訊或轉折")
        
        # === 副圖:融資融券走勢 ===
        if not margin.empty:
            st.subheader("💸 融資融券走勢")
            
            fig_margin = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_margin.add_trace(
                go.Scatter(
                    x=margin["date"],
                    y=margin["margin_balance"],
                    name="融資餘額",
                    mode="lines",
                    line=dict(color="#E67E22", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(230, 126, 34, 0.1)",
                ),
                secondary_y=False,
            )
            
            fig_margin.add_trace(
                go.Scatter(
                    x=margin["date"],
                    y=margin["short_balance"],
                    name="融券餘額",
                    mode="lines",
                    line=dict(color="#16A085", width=2),
                ),
                secondary_y=True,
            )
            
            fig_margin.update_layout(
                height=300,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                margin=dict(l=40, r=40, t=20, b=40),
            )
            fig_margin.update_yaxes(title_text="融資餘額(張)", secondary_y=False)
            fig_margin.update_yaxes(title_text="融券餘額(張)", secondary_y=True)
            
            st.plotly_chart(fig_margin, use_container_width=True)
            
            st.caption("💡 **反指標思考**:融資快速攀升 = 散戶熱情高漲(常見短期見頂訊號);融資下降 = 散戶離場(可能築底)")
        
        # === 籌碼解讀文字 ===
        st.markdown("### 📝 今日籌碼速讀")
        
        readings = []
        if foreign_5d > 1000:
            readings.append("✅ 外資 5 日大幅買超(>1000 張),機構看好")
        elif foreign_5d > 0:
            readings.append("🟢 外資 5 日小幅買超")
        elif foreign_5d < -1000:
            readings.append("❌ 外資 5 日大幅賣超(>1000 張),機構撤退")
        elif foreign_5d < 0:
            readings.append("🔴 外資 5 日小幅賣超")
        
        if trust_5d > 0 and foreign_5d > 0:
            readings.append("⭐ 外資+投信同向買超 = 雙法人共識,訊號強")
        elif trust_5d < 0 and foreign_5d < 0:
            readings.append("⚠️ 外資+投信同向賣超 = 雙法人共識撤退")
        
        if margin_latest is not None and margin_30d_ago is not None:
            m_chg = (margin_latest["margin_balance"] - margin_30d_ago["margin_balance"]) / margin_30d_ago["margin_balance"] * 100 if margin_30d_ago["margin_balance"] > 0 else 0
            if m_chg > 20:
                readings.append(f"🚨 融資 30 日大增 {m_chg:.1f}%,散戶過熱訊號(反指標)")
            elif m_chg < -20:
                readings.append(f"💎 融資 30 日大減 {abs(m_chg):.1f}%,散戶離場(常見築底訊號)")
        
        if sh_latest is not None and sh_30d_ago is not None:
            sh_now = sh_latest.get("foreign_holding_ratio")
            sh_old = sh_30d_ago.get("foreign_holding_ratio")
            if pd.notna(sh_now) and pd.notna(sh_old):
                sh_chg = sh_now - sh_old
                if abs(sh_chg) > 1:
                    direction_text = "增持" if sh_chg > 0 else "減持"
                    readings.append(f"🌐 外資 30 日{direction_text} {abs(sh_chg):.2f} 百分點(結構性變化)")
        
        if readings:
            for r in readings:
                st.markdown(f"- {r}")
        else:
            st.caption("籌碼相對清淡,無顯著訊號")
    else:
        st.info("此檔尚無籌碼資料")

        # =====================================================
    # AI 獨立觀察 (Phase 3.3 / 4 整合版)
    # =====================================================
    st.divider()
    st.subheader("🤖 AI 獨立觀察")
    st.caption("AI 會優先參考您輸入的重大事件，並結合市場數據給出觀察。")
    
    # 取使用者論點(只用 time_horizon,不影響 AI 分析)
    thesis_data = supabase.table("theses") \
        .select("*").eq("symbol", selected_symbol).eq("user_id", get_user_id()).execute().data
    current_thesis = thesis_data[0] if thesis_data else {}
    
    primary_horizon = current_thesis.get("time_horizon")
    
    if not primary_horizon:
        st.warning("⚠️ 請先到「📝 投資論點」頁設定『時間框架』,AI 才知道要做哪個時間框架的分析")
    else:
        # ================= 新增的前瞻資訊輸入框 =================
        memory_key = f"persistent_event_{selected_symbol}"
        if memory_key not in st.session_state:
            st.session_state[memory_key] = ""

        def sync_event_memory():
            input_key = f"ai_event_input_{selected_symbol}"
            st.session_state[memory_key] = st.session_state[input_key]

        upcoming_events = st.text_area(
            "📢 補充最新重大事件 (這會強制 AI 優先閱讀)", 
            value=st.session_state[memory_key],
            placeholder="例如: 2603 董事會已決議配息 16 元 / 6862 發行 10 億元 CB 且主力詢圈中...",
            key=f"ai_event_input_{selected_symbol}",
            height=100,
            on_change=sync_event_memory,
            help="輸入的內容會自動儲存，換頁或修改持股資料後回來依然有效。"
        )
        # =======================================================

        # === 載入「最近一次」觀察(從 DB 或 session_state)===
        session_key = f"ai_obs_{selected_symbol}"
        
        if session_key not in st.session_state:
            history_obs = ai_analyzer.load_observations(selected_symbol, limit=1, user_id=get_user_id())
            if history_obs:
                latest = history_obs[0]
                st.session_state[session_key] = {
                    "data": latest["validated_points"],
                    "tokens": {
                        "input": latest.get("input_tokens"),
                        "output": latest.get("output_tokens"),
                        "total": latest.get("total_tokens"),
                    },
                    "model": latest.get("model_used"),
                    "created_at": latest.get("created_at"),
                }
        
        # === 按鈕區 ===
        if session_key in st.session_state:
            button_label = "🔄 重新跑 AI 觀察"
        else:
            button_label = "📊 執行 AI 獨立觀察"
        
        bcol1, bcol2 = st.columns([1, 3])
        with bcol1:
            run_ai = st.button(button_label, type="primary", key=f"run_ai_{selected_symbol}")
        with bcol2:
            st.caption(f"⏳ 預估 30-60 秒 | 💰 約 3-8 美分 (含 thinking tokens)")
        
        if run_ai:
            with st.spinner("🤖 AI 正在結合數據與前瞻事件進行觀察..."):
                # 取資料
                stock_info = next((s for s in stocks if s["symbol"] == selected_symbol), {})
                val_data = load_latest_valuation(get_user_id()).get(selected_symbol, {})
                rev_df_ai = load_monthly_revenue(selected_symbol)
                rev_list = rev_df_ai.to_dict("records") if not rev_df_ai.empty else []
                qfin_df_ai = load_quarterly_financials(selected_symbol)
                qfin_list = qfin_df_ai.to_dict("records") if not qfin_df_ai.empty else []
                
                # 籌碼摘要
                chips_df_ai = load_chips(selected_symbol, days=30)
                margin_df_ai = load_margin(selected_symbol, days=30)
                sh_df_ai = load_shareholding(selected_symbol, days=30)
                
                chips_summary = {}
                if not chips_df_ai.empty:
                    recent_5 = chips_df_ai.tail(5)
                    chips_summary["foreign_5d"] = recent_5["foreign_net"].sum() / 1000
                    chips_summary["trust_5d"] = recent_5["trust_net"].sum() / 1000
                    
                    latest_chip = chips_df_ai.iloc[-1]
                    consec = 0
                    for i in range(len(chips_df_ai) - 1, -1, -1):
                        if (chips_df_ai.iloc[i]["foreign_net"] > 0) == (latest_chip["foreign_net"] > 0):
                            consec += 1
                        else:
                            break
                    chips_summary["consecutive_days"] = consec
                    chips_summary["direction"] = "買超" if latest_chip["foreign_net"] > 0 else "賣超"
                
                if not margin_df_ai.empty and len(margin_df_ai) >= 20:
                    m_now = margin_df_ai.iloc[-1]["margin_balance"]
                    m_old = margin_df_ai.iloc[-20]["margin_balance"]
                    if m_old > 0:
                        chips_summary["margin_change_pct"] = (m_now - m_old) / m_old * 100
                
                if not sh_df_ai.empty:
                    sh_latest = sh_df_ai.iloc[-1].get("foreign_holding_ratio")
                    if pd.notna(sh_latest):
                        chips_summary["foreign_holding"] = round(sh_latest, 2)
                
                # 取資料新鮮度
                freshness = load_data_freshness()
                data_freshness = freshness.get("股價/PE", "未知")
                
                # === 計算 holding_context (持股事實 + 累積股息)===
                all_txns = load_transactions(get_user_id())
                holding_ctx = ai_analyzer.build_holding_context(
                    symbol=selected_symbol,
                    transactions=all_txns,
                    current_price=val_data.get("close", 0),
                )

                # === Phase 4.5: 抓取最近新聞作為市場 context ===
                # 用每小時 bucket 當 cache key (1 小時內不重抓)
                hourly_bucket = int(time.time() // 3600)
                
                @st.cache_data(ttl=3600, show_spinner=False)
                def _fetch_news_for_obs(sym, name, bucket):
                    return news_fetcher.fetch_news(sym, name, days=7, limit=10)
                
                stock_name_for_news = stock_info.get("name", selected_symbol)
                try:
                    news_list = _fetch_news_for_obs(selected_symbol, stock_name_for_news, hourly_bucket)
                except Exception as e:
                    print(f"[app] 新聞抓取失敗: {e}")
                    news_list = []
                
                # 顯示抓到幾則新聞
                if news_list:
                    st.caption(f"📰 自動抓取 {len(news_list)} 則主流媒體新聞作為 AI 分析參考")
                else:
                    st.caption("📰 (本次沒抓到主流媒體新聞,AI 將基於數據與你輸入的事件分析)")

                # === 計算技術指標(只給布林通道,夏普不給 AI)===
                # 設計原因: 夏普值是 252 日歷史指標(體質),
                # 布林通道是 20 日近期指標(動態),
                # 兩者時間尺度不一致,夏普會干擾 AI 的「即時觀察」判斷
                # 夏普值仍在 UI 顯示給「人」做長期體質參考
                metrics_for_ai = {}
                try:
                    price_records_for_ai = load_prices(selected_symbol)
                    if price_records_for_ai and len(price_records_for_ai) >= 20:
                        df_m = pd.DataFrame(price_records_for_ai)
                        df_m["date"] = pd.to_datetime(df_m["date"])
                        df_m = df_m.sort_values("date").reset_index(drop=True)
                        # 只傳布林通道給 AI(時間尺度跟「近期觀察」一致)
                        metrics_for_ai["bollinger"] = metrics.calculate_bollinger_bands(df_m["close"])
                except Exception as e:
                    print(f"[app] 給 AI 的 metrics 計算失敗: {e}")

                # 執行
                result = ai_analyzer.run_observation(
                    symbol=selected_symbol,
                    name=stock_info.get("name", selected_symbol),
                    industry=stock_info.get("industry", "-"),
                    primary_horizon=primary_horizon,
                    valuation={
                        "close": val_data.get("close"),
                        "pe": val_data.get("pe"),
                        "pb": val_data.get("pb"),
                        "dividend_yield": val_data.get("dividend_yield"),
                    },
                    monthly_rev=rev_list,
                    quarterly_fin=qfin_list,
                    chips=chips_summary,
                    data_freshness=data_freshness,
                    holding_context=holding_ctx,
                    upcoming_events=upcoming_events,
                    news_list=news_list,
                    metrics=metrics_for_ai,
                )
                
                if result["success"]:
                    # 存 DB(thesis_snapshot 只是記錄當時的論點,不影響分析)
                    thesis_obj = {
                        "thesis": current_thesis.get("thesis"),
                        "moat": current_thesis.get("moat"),
                        "risks": current_thesis.get("risks"),
                    }
                    review_id = ai_analyzer.save_observation(
                        symbol=selected_symbol,
                        primary_horizon=primary_horizon,
                        result=result,
                        thesis_snapshot=thesis_obj,
                        user_id=get_user_id(),
                    )
                    st.success("✅ AI 觀察完成!")
                    # 加入 created_at 時間戳
                    result["created_at"] = datetime.now().isoformat()
                    st.session_state[session_key] = result
                    st.rerun()
                else:
                    st.error(f"❌ 失敗:{result.get('error')}")
                    if result.get("raw_response"):
                        with st.expander("查看 AI 原始回應"):
                            st.code(result["raw_response"])
        
        # === 顯示結果 ===
        if session_key in st.session_state:
            result = st.session_state[session_key]
            
            # 顯示「上次跑的時間 + 新鮮度提示」
            if result.get("created_at"):
                try:
                    created = pd.to_datetime(result["created_at"])
                    created_naive = created.tz_localize(None) if created.tz else created
                    now = datetime.now()
                    hours_ago = (now - created_naive.to_pydatetime()).total_seconds() / 3600
                    
                    if hours_ago < 1:
                        age_text = f"{int(hours_ago * 60)} 分鐘前"
                        getattr(st, "success")(f"📅 這份觀察是 **{age_text}** 跑的 ({created_naive.strftime('%Y-%m-%d %H:%M')})")
                    elif hours_ago < 24:
                        age_text = f"{int(hours_ago)} 小時前"
                        st.info(f"📅 這份觀察是 **{age_text}** 跑的 ({created_naive.strftime('%Y-%m-%d %H:%M')})")
                    elif hours_ago < 24 * 7:
                        age_text = f"{int(hours_ago / 24)} 天前"
                        st.info(f"📅 這份觀察是 **{age_text}** 跑的 ({created_naive.strftime('%Y-%m-%d %H:%M')})")
                    else:
                        age_text = f"{int(hours_ago / 24)} 天前"
                        st.warning(f"📅 這份觀察是 **{age_text}** 跑的 ({created_naive.strftime('%Y-%m-%d %H:%M')}) ⚠️ 資料可能過期,建議重跑")
                except Exception:
                    pass
            
            render_stress_test_result(
                result["data"],
                tokens=result.get("tokens"),
                model=result.get("model"),
                holding_context=holding_ctx,
            )
        
        # === 歷史 AI 觀察紀錄 ===
        history_list = ai_analyzer.load_observations(selected_symbol, limit=5, user_id=get_user_id())
        if len(history_list) > 1:
            st.divider()
            with st.expander(f"📜 歷史 AI 觀察紀錄 ({len(history_list)} 筆)"):
                for h in history_list:
                    created = pd.to_datetime(h["created_at"]).strftime("%Y-%m-%d %H:%M")
                    tokens_t = h.get("total_tokens", "-")
                    stance_h = h.get("recommendation", "-")
                    stance_label_h = ai_analyzer.STANCE_LABELS.get(stance_h, stance_h) if stance_h else "-"
                    
                    with st.container(border=True):
                        st.caption(f"📅 {created} | 立場: {stance_label_h} | Total tokens: {tokens_t}")
                        if st.button(f"📂 載入此次結果", key=f"load_obs_{h['id']}"):
                            st.session_state[session_key] = {
                                "data": h["validated_points"],
                                "tokens": {
                                    "input": h.get("input_tokens"),
                                    "output": h.get("output_tokens"),
                                    "total": h.get("total_tokens"),
                                },
                                "model": h.get("model_used"),
                                "created_at": h.get("created_at"),
                            }
                            st.rerun()


# ============================================================
# AI 觀察結果顯示 helper (Phase 3.3)
# ============================================================
HORIZON_LABELS = {
    "short": "短期 (1-3 個月)",
    "medium": "中期 (6-12 個月)",
    "long": "長期 (3 年以上)",
}

STANCE_LABELS = {
    "strongly_bullish": ("🟢🟢 強烈看多", "#27AE60"),
    "moderately_bullish": ("🟢 中度看多", "#27AE60"),
    "neutral_lean_bullish": ("🟢 中性偏多", "#52BE80"),
    "neutral": ("⚪ 中性", "#7F8C8D"),
    "neutral_lean_bearish": ("🔴 中性偏空", "#EC7063"),
    "moderately_bearish": ("🔴 中度看空", "#E74C3C"),
    "strongly_bearish": ("🔴🔴 強烈看空", "#C0392B"),
}


def render_stress_test_result(data: dict, tokens: dict = None, model: str = None, holding_context: dict = None):
    """渲染 AI 獨立觀察結果(新結構:無多空辯論)"""
    
    primary_horizon = data.get("primary_horizon", "")
    horizon_label = HORIZON_LABELS.get(primary_horizon, "")
    
    # === Header + Token KPI ===
    st.markdown(f"### 🤖 AI 獨立觀察 ── 主力框架:{horizon_label}")
    
    if tokens:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("輸入 Token", f"{tokens.get('input', 0):,}")
        c2.metric("輸出 Token", f"{tokens.get('output', 0):,}")
        c3.metric("總 Token (含 thinking)", f"{tokens.get('total', 0):,}")
        c4.metric("Model", model or "-")

    # === 持股資訊橫幅(如果使用者持有此檔)===
    if holding_context:
        h = holding_context
        pnl_color = "#E74C3C" if h.get("unrealized_pnl_pct", 0) >= 0 else "#27AE60"
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FFF9E6, #FFFEF7);
            border-left: 6px solid #F39C12;
            padding: 16px 20px;
            border-radius: 8px;
            margin: 16px 0;
        ">
            <div style="font-size:1.1rem; font-weight:700; color:#7D6608; margin-bottom: 8px;">
                💼 此次分析考量了你的持股狀況
            </div>
            <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.95rem;">
                <span><strong>持股</strong>: {h.get('shares', 0):,} 股</span>
                <span><strong>均價</strong>: {h.get('avg_cost', 0)}</span>
                <span><strong>損益</strong>: <span style="color:{pnl_color}; font-weight:600;">{h.get('unrealized_pnl_pct', 0):+.2f}%</span></span>
                <span><strong>累積股息</strong>: {h.get('dividends_received_per_share', 0)} 元/股 (共 {h.get('dividend_events_count', 0)} 次)</span>
                <span><strong>有效成本</strong>: {h.get('effective_cost_per_share', 0)} 元</span>
            </div>
        </div>
        """, unsafe_allow_html=True)    
    
    # ===========================================================
    # 【A】整體判斷
    # ===========================================================
    oj = data.get("overall_judgment", {})
    stance_key = oj.get("stance", "neutral")
    stance_label, stance_color = STANCE_LABELS.get(stance_key, ("⚪ 中性", "#7F8C8D"))
    confidence = oj.get("confidence", "-")
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {stance_color}15, {stance_color}05);
        border-left: 6px solid {stance_color};
        padding: 20px 24px;
        border-radius: 8px;
        margin: 16px 0;
    ">
        <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
            <span style="font-size:1.8rem; font-weight:700; color:{stance_color};">
                {stance_label}
            </span>
            <span style="font-size:1.2rem; color:#555;">
                信心度 <strong style="color:{stance_color}; font-size:1.5rem;">{confidence}/10</strong>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if oj.get("core_reasoning"):
        st.markdown(f"**💡 核心理由**:{oj['core_reasoning']}")
    
    change_signals = oj.get("what_would_change_my_mind", [])
    if change_signals:
        with st.container(border=True):
            st.markdown("**🔄 什麼訊號會讓 AI 改變想法**")
            for s in change_signals:
                st.markdown(f"- {s}")
    
    # ===========================================================
    # 【B】當前處境
    # ===========================================================
    situation = data.get("current_situation", [])
    if situation:
        st.markdown("#### 📖 當前處境")
        with st.container(border=True):
            for s in situation:
                st.markdown(f"- {s}")
    
    # ===========================================================
    # 【C】情境推演
    # ===========================================================
    scenarios = data.get("scenario_analysis", [])
    if scenarios:
        st.markdown("#### 🎭 情境推演")
        st.caption("AI 評估的不同情境發生機率(三個情境機率加總 = 100%)")
        
        # 機率視覺化條
        prob_html = '<div style="display:flex; gap:4px; margin: 12px 0; height: 32px; border-radius: 6px; overflow: hidden;">'
        colors = ["#52BE80", "#F4D03F", "#E74C3C"]
        for i, sc in enumerate(scenarios[:3]):
            prob = sc.get("probability", 0)
            color = colors[i] if i < len(colors) else "#95A5A6"
            prob_html += f'<div style="flex:{prob}; background:{color}; display:flex; align-items:center; justify-content:center; color:white; font-weight:600; font-size:0.9rem;">{prob}%</div>'
        prob_html += "</div>"
        st.markdown(prob_html, unsafe_allow_html=True)
        
        # 三個情境並列
        cols = st.columns(len(scenarios))
        for i, (col, sc) in enumerate(zip(cols, scenarios)):
            color = colors[i] if i < len(colors) else "#95A5A6"
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                        f"<strong style='font-size:1.05rem;'>{sc.get('name', f'情境 {i+1}')}</strong><br>"
                        f"<span style='color:{color}; font-weight:700; font-size:1.4rem;'>{sc.get('probability', 0)}%</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    if sc.get("key_assumptions"):
                        st.markdown("**關鍵假設**")
                        for a in sc["key_assumptions"]:
                            st.markdown(f"- {a}")
                    
                    if sc.get("implications"):
                        st.markdown("**若成真會發生**")
                        for imp in sc["implications"]:
                            st.markdown(f"- {imp}")
    
    # ===========================================================
    # 【D】訊號追蹤
    # ===========================================================
    signals = data.get("signals_to_monitor", [])
    if signals:
        st.markdown("#### 📍 該追蹤的訊號")
        
        critical = [s for s in signals if s.get("importance") == "critical"]
        important = [s for s in signals if s.get("importance") == "important"]
        minor = [s for s in signals if s.get("importance") == "minor"]
        
        if critical:
            with st.container(border=True):
                st.markdown("**🔴 關鍵(必看)**")
                for sig in critical:
                    st.markdown(f"- **{sig.get('signal', '')}**")
                    if sig.get("why_matters"):
                        st.caption(f"   為什麼重要:{sig['why_matters']}")
        
        if important:
            with st.container(border=True):
                st.markdown("**🟡 重要(該看)**")
                for sig in important:
                    st.markdown(f"- {sig.get('signal', '')}")
                    if sig.get("why_matters"):
                        st.caption(f"   為什麼重要:{sig['why_matters']}")
        
        if minor:
            with st.expander("**🟢 次要(有空看)**"):
                for sig in minor:
                    st.markdown(f"- {sig.get('signal', '')}")
                    if sig.get("why_matters"):
                        st.caption(f"   {sig['why_matters']}")
    
    # ===========================================================
    # 質化總結(分析師靈魂)
    # ===========================================================
    summary = data.get("qualitative_summary", "")
    if summary:
        st.markdown("#### 💭 綜合判斷")
        with st.container(border=True):
            st.markdown(summary)
    
    # ===========================================================
    # AI 自身限制
    # ===========================================================
    limits = data.get("ai_self_disclosed_limits", [])
    if limits:
        with st.expander("🔍 AI 自己揭露的分析限制"):
            for l in limits:
                st.markdown(f"- {l}")
    
    # ===========================================================
    # 資料引用
    # ===========================================================
    refs = data.get("data_references", [])
    if refs:
        with st.expander("📊 本次引用的資料(可回去驗證)"):
            for r in refs:
                st.markdown(f"- {r}")
    
    # ===========================================================
    # 新聞引用 (Phase 4.5)
    # ===========================================================
    news_refs = data.get("news_references", [])
    if news_refs:
        with st.expander(f"📰 本次引用的新聞 ({len(news_refs)} 則)"):
            st.caption("AI 在分析中引用了下列新聞,你可以審視 AI 的詮釋是否合理")
            for r in news_refs:
                st.markdown(f"- {r}")


# ============================================================
# 頁面 3: 投資論點
# ============================================================
def page_thesis():
    st.title("📝 投資論點")
    st.caption("記錄你買進的理由,定期 review 論點是否仍然成立")
    
    stocks = load_stocks(get_user_id())
    if not stocks:
        st.warning("尚未有追蹤個股")
        return
    
    stock_options = {f"{s['symbol']} {s['name']}": s['symbol'] for s in stocks}
    
    # 個股選擇放側邊欄(scroll 時也能切換)
    with st.sidebar:
        st.divider()
        st.subheader("📝 選擇個股")
        selected_label = st.selectbox(
            "個股",
            list(stock_options.keys()),
            label_visibility="collapsed",
            key="thesis_stock_selector"
        )
    selected_symbol = stock_options[selected_label]
    
    existing = supabase.table("theses") \
        .select("*").eq("symbol", selected_symbol).eq("user_id", get_user_id()).execute().data
    current = existing[0] if existing else {}
    
    # === Sticky 個股名稱(scroll 時固定在頂部)===
    valuation = load_latest_valuation(get_user_id())
    val = valuation.get(selected_symbol, {})
    current_price = val.get("close")
    
    # 用最新價格 + 漲跌資訊組 sticky header
    prices = load_prices(selected_symbol)
    if prices and len(prices) >= 2:
        df_prices = pd.DataFrame(prices).sort_values("date")
        latest_p = df_prices.iloc[-1]
        prev_p = df_prices.iloc[-2]
        change = latest_p["close"] - prev_p["close"]
        change_pct = (change / prev_p["close"]) * 100
        latest_date = pd.to_datetime(latest_p["date"])
        # 用既有的 helper 函式
        render_sticky_stock_header(
            selected_label,
            {"close": latest_p["close"], "date": latest_date},
            change,
            change_pct
        )
    
    # ===== 就緒度檢查(在表單上方顯示)=====
    has_thesis = bool((current.get("thesis") or "").strip())
    has_horizon = bool(current.get("time_horizon"))
    has_moat = bool((current.get("moat") or "").strip())
    has_risks = bool((current.get("risks") or "").strip())
    
    ready_for_stress_test = has_thesis and has_horizon
    
    cols = st.columns(2)
    with cols[0]:
        if current.get("updated_at"):
            last_update = pd.to_datetime(current["updated_at"]).strftime("%Y-%m-%d")
            st.info(f"📅 論點最後更新:{last_update}")
        else:
            st.warning("⚠️ 這檔股票尚未填寫投資論點")
    with cols[1]:
        if current.get("last_reviewed_at"):
            last_review = pd.to_datetime(current["last_reviewed_at"]).strftime("%Y-%m-%d")
            days_ago = (datetime.now() - pd.to_datetime(current["last_reviewed_at"]).tz_localize(None)).days
            st.info(f"👀 最後檢視:{last_review} ({days_ago} 天前)")
    
    # ===== 就緒度面板 =====
    with st.container(border=True):
        st.markdown("##### 📋 Stress Test 就緒度")
        rd1, rd2, rd3, rd4, rd5 = st.columns(5)
        rd1.markdown(f"{'✅' if has_thesis else '❌'} 核心論點\n\n*必要*")
        rd2.markdown(f"{'✅' if has_horizon else '❌'} 時間框架\n\n*必要*")
        rd3.markdown(f"{'✅' if has_moat else '⚪'} 護城河\n\n*選填*")
        rd4.markdown(f"{'✅' if has_risks else '⚪'} 主要風險\n\n*選填*")
        rd5.markdown(f"{'✅' if (current.get('strategy_note') or '').strip() else '⚪'} 策略補充\n\n*選填*")
        if ready_for_stress_test:
            st.success("✅ 已達 Stress Test 執行條件")
        else:
            missing = []
            if not has_thesis:
                missing.append("核心論點")
            if not has_horizon:
                missing.append("時間框架")
            st.error(f"❌ 還缺必要欄位:{', '.join(missing)}")
    
    st.divider()
    
    with st.form(f"thesis_form_{selected_symbol}", clear_on_submit=False):
        # ===== 必填區 =====
        st.subheader("🎯 核心論點 :red[*必填]")
        thesis = st.text_area(
            "為什麼買這檔?看到什麼機會?",
            value=current.get("thesis", "") or "",
            height=120,
            placeholder="例如:看好 AI 浪潮帶動先進製程需求,公司 2nm 領先一個世代...",
            label_visibility="visible"
        )
        
        st.subheader("⏳ 時間框架 :red[*必填]")
        horizon_options = {
            "": "(請選擇)",
            "short": "🚀 短期 (1-3 個月) - 看技術面/籌碼動能",
            "medium": "📊 中期 (6-12 個月) - 看基本面趨勢",
            "long": "🏛️ 長期 (3 年以上) - 看護城河/股息",
        }
        current_horizon = current.get("time_horizon", "") or ""
        time_horizon = st.selectbox(
            "你打算抱多久?這會決定 AI 分析的主力框架",
            options=list(horizon_options.keys()),
            format_func=lambda k: horizon_options[k],
            index=list(horizon_options.keys()).index(current_horizon) if current_horizon in horizon_options else 0,
        )
        
        st.divider()
        
        # ===== 選填區 =====
        st.markdown("##### 以下欄位皆為選填,但填寫越完整,Stress Test 結果越有對照價值")
        
        st.subheader("🏰 護城河 :gray[(選填)]")
        moat = st.text_area(
            "這家公司有什麼別人取代不了的優勢?",
            value=current.get("moat", "") or "",
            height=100,
            placeholder="例如:技術領先、客戶轉換成本高、規模經濟、專利..."
        )
        
        st.subheader("⚠️ 主要風險 :gray[(選填)]")
        risks = st.text_area(
            "什麼情況會打破論點?",
            value=current.get("risks", "") or "",
            height=100,
            placeholder="例如:1.中國競爭追上 2.AI 需求趨緩 3.地緣政治..."
        )
        
        st.subheader("📝 策略補充 :gray[(選填)]")
        strategy_note = st.text_area(
            "你對這檔的特殊操作策略(會餵給 AI 當『約束條件』,不是立場)",
            value=current.get("strategy_note", "") or "",
            height=80,
            placeholder="例如:只進不出、股息複利再投入、目標 0 成本 / 等月營收 YoY 轉正才加碼 / 跌破 60MA 出場一半...",
            help="這是『策略邊界』(行為約束),不要寫『我覺得會漲』這種立場性陳述"
        )
        
        st.divider()
        
        # ===== 行動 / 風控 =====
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🎯 目前判斷")
            view_options = ["", "加碼", "持有", "觀察", "減碼", "出場"]
            current_view_value = current.get("current_view", "") or ""
            view_index = view_options.index(current_view_value) if current_view_value in view_options else 0
            current_view = st.selectbox(
                "你現在打算怎麼做?",
                options=view_options,
                index=view_index,
                help="加碼=想買更多、持有=維持現狀、觀察=還在想、減碼=想賣一些、出場=想全賣"
            )
        with col2:
            st.subheader("🛡️ 個人停損價")
            stop_loss = st.number_input(
                "跌破多少你會出場?(0 表示不設停損)",
                value=float(current.get("stop_loss") or 0),
                min_value=0.0, step=0.5, format="%.2f",
            )
        
        st.divider()
        col_a, col_b, _ = st.columns([1, 1, 3])
        with col_a:
            submitted = st.form_submit_button("💾 儲存", type="primary", use_container_width=True)
        with col_b:
            mark_reviewed = st.form_submit_button("👀 標記已檢視", use_container_width=True)
        
        if submitted:
            data = {
                "symbol": selected_symbol,
                "user_id": get_user_id(),
                "thesis": thesis or None,
                "moat": moat or None,
                "risks": risks or None,
                "strategy_note": strategy_note or None,
                "time_horizon": time_horizon if time_horizon else None,
                "current_view": current_view if current_view else None,
                "stop_loss": stop_loss if stop_loss > 0 else None,
                "updated_at": datetime.now().isoformat(),
            }
            if not existing:
                data["created_at"] = datetime.now().isoformat()
            supabase.table("theses").upsert(data).execute()
            st.cache_data.clear()
            st.success("✅ 論點已儲存!")
            st.rerun()
        
        if mark_reviewed:
            supabase.table("theses").upsert({
                "symbol": selected_symbol,
                "user_id": get_user_id(),
                "last_reviewed_at": datetime.now().isoformat()
            }).execute()
            st.cache_data.clear()
            st.success("✅ 已標記為已檢視!")
            st.rerun()
    
    # ===== 停損監控 =====
    valuation = load_latest_valuation(get_user_id())
    current_price = valuation.get(selected_symbol, {}).get("close")
    
    if current_price and current.get("stop_loss"):
        st.divider()
        st.subheader("🛡️ 停損監控")
        sl = current["stop_loss"]
        diff_pct = (current_price - sl) / sl * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("現價", f"{current_price:,.2f}")
        col2.metric("停損價", f"{sl:.2f}")
        col3.metric("距停損", f"{diff_pct:+.2f}%")
        if current_price <= sl:
            st.error(f"🚨 已跌破停損價!現價 {current_price} ≤ 停損 {sl}")
        elif diff_pct < 5:
            st.warning(f"⚠️ 接近停損價,緩衝僅剩 {diff_pct:.2f}%")
    
    # ===== 對照 AI 觀察(進階)=====
    st.divider()
    st.subheader("🔍 對照 AI 觀察")
    st.caption("把你寫的論點 vs 最近一次 AI 獨立觀察,列出『角度差異』(只列差異,不評對錯)")
    
    if not has_thesis:
        st.warning("⚠️ 請先填寫核心論點,才能進行對照")
    else:
        # 撈這檔最近一次 AI 觀察
        last_obs_list = ai_analyzer.load_observations(selected_symbol, limit=1, user_id=get_user_id())
        
        if not last_obs_list:
            st.info(
                "👉 這檔尚未跑過 AI 獨立觀察。\n\n"
                "請先到「📈 個股技術分析」頁底部執行 AI 觀察,跑完後再回到這頁進行論點對照。"
            )
        else:
            last_obs = last_obs_list[0]
            obs_created = pd.to_datetime(last_obs["created_at"]).strftime("%Y-%m-%d %H:%M")
            obs_stance = last_obs.get("recommendation", "")
            obs_stance_label = ai_analyzer.STANCE_LABELS.get(obs_stance, obs_stance) if obs_stance else "-"
            
            st.info(
                f"📊 將對照最近一次 AI 觀察:\n\n"
                f"- 跑於 {obs_created}\n"
                f"- AI 立場:{obs_stance_label}"
            )
            
            comp_button = st.button(
                "🔍 跑論點對照",
                key=f"compare_thesis_{selected_symbol}",
                type="primary",
            )
            
            if comp_button:
                with st.spinner("🤖 AI 正在比對角度差異..."):
                    user_thesis_for_comp = {
                        "thesis": current.get("thesis", ""),
                        "moat": current.get("moat", ""),
                        "risks": current.get("risks", ""),
                        "strategy_note": current.get("strategy_note", ""),
                    }
                    comp_result = ai_analyzer.run_thesis_comparison(
                        ai_observation=last_obs["validated_points"],
                        user_thesis=user_thesis_for_comp,
                    )
                    if comp_result["success"]:
                        st.session_state[f"comp_result_{selected_symbol}"] = comp_result
                    else:
                        st.error(f"❌ 對照失敗: {comp_result.get('error')}")
            
            # 顯示對照結果
            if f"comp_result_{selected_symbol}" in st.session_state:
                comp = st.session_state[f"comp_result_{selected_symbol}"]
                cdata = comp["data"]
                
                with st.container(border=True):
                    if cdata.get("summary"):
                        st.info(f"**📌 對照摘要**:{cdata['summary']}")
                    
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        ai_only = cdata.get("ai_mentioned_user_didnt", [])
                        if ai_only:
                            st.markdown("**🤖 AI 提到、你沒寫到的角度**")
                            for item in ai_only:
                                st.markdown(f"- {item}")
                    with cc2:
                        user_only = cdata.get("user_mentioned_ai_didnt", [])
                        if user_only:
                            st.markdown("**📝 你寫到、AI 沒提到的角度**")
                            for item in user_only:
                                st.markdown(f"- {item}")
                    
                    aligned = cdata.get("both_aligned_on", [])
                    if aligned:
                        st.markdown("**🤝 雙方都提到的角度**")
                        for item in aligned:
                            st.markdown(f"- {item}")
                    
                    strategy_rel = cdata.get("user_strategy_relevance", [])
                    if strategy_rel:
                        st.markdown("**🎯 你的策略 vs AI 觀點**")
                        for item in strategy_rel:
                            st.markdown(f"- {item}")
                    
                    st.caption(f"📊 對照 Token: {comp.get('tokens', {}).get('total', '-')}")

def page_transactions():
    """交易與追蹤管理"""
    st.title("⚙️ 交易管理")
    st.caption("管理你的交易紀錄跟追蹤清單")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 新增交易", "📋 交易紀錄", "💰 股利管理", "📈 追蹤清單管理"])
    
    # ============================================
    # Tab 1: 新增交易
    # ============================================
    with tab1:
        st.subheader("新增一筆交易")
        
        stocks = load_stocks(get_user_id())
        has_tracked = bool(stocks)
        
        # === 模式選擇:從現有股票選 vs 新增股票 ===
        if has_tracked:
            mode = st.radio(
                "選擇模式",
                ["📌 從追蹤清單選擇", "➕ 新增還沒追蹤的股票"],
                horizontal=True,
                key="add_txn_mode",
            )
        else:
            # 完全沒追蹤股票 → 直接走「新增」模式
            mode = "➕ 新增還沒追蹤的股票"
            st.info("👋 第一次新增交易?直接輸入持股資訊,系統會自動加入追蹤清單並抓取資料")
        
        is_new_stock = (mode == "➕ 新增還沒追蹤的股票")
        
        # === Form 外:股票識別欄位(這樣選擇變化能即時反應)===
        if is_new_stock:
            st.markdown("**📈 新股票資訊**")
            col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
            with col_s1:
                new_symbol = st.text_input("代號 *", placeholder="例如 2317", max_chars=6, key="new_sym_in_txn")
            with col_s2:
                new_name = st.text_input("名稱 *", placeholder="例如 鴻海", key="new_name_in_txn")
            with col_s3:
                new_industry = st.text_input("產業(選填)", placeholder="例如 電子代工", key="new_ind_in_txn")
            st.caption("⏳ 儲存後會自動跑 FinMind 同步近 3 年資料(約 30 秒)")
        else:
            stock_options = {f"{s['symbol']} {s['name']}": s['symbol'] for s in stocks}
            selected_label = st.selectbox("個股 *", list(stock_options.keys()), key="sel_existing_stock")
            symbol_from_select = stock_options[selected_label]
        
        # === Form 內:交易欄位 ===
        with st.form("new_txn_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                action = st.selectbox("動作 *", ["buy", "sell"], format_func=lambda x: "🟢 買進" if x == "buy" else "🔴 賣出")
                txn_date = st.date_input("交易日期 *", value=datetime.now().date())
                shares = st.number_input("股數 *", min_value=1, value=1000, step=1)
            with col2:
                price = st.number_input("價格 *", min_value=0.01, value=100.0, step=0.5, format="%.2f")
                amount = shares * price
                default_fee = max(20, round(amount * 0.001425))
                fee = st.number_input("手續費", min_value=0, value=int(default_fee), step=1, help="預設 0.1425% 手續費,最低 20 元")
                
                # 賣出才有交易稅
                tax = 0
                if action == "sell":
                    tax = round(amount * 0.003)
                    st.caption(f"💡 賣出證交稅 (0.3%): {tax:,} 元 (自動)")
            
            note = st.text_input("備註(選填)", placeholder="例如:第一次建倉、跌破均線出場...")
            
            # 預覽
            net_amount = amount + (fee if action == "buy" else -fee - tax)
            st.markdown(f"""
            **交易預覽**
            - 股數 × 價格 = `{shares:,} × {price:.2f}` = **{amount:,.0f}**
            - 手續費: `{fee:,}` {'(買進加上)' if action == 'buy' else '(賣出扣除)'}
            {f"- 交易稅: `{tax:,}` (賣出扣除)" if action == "sell" else ""}
            - **{'買進總成本' if action == 'buy' else '賣出實得'}**: `{net_amount:,.0f}` 元
            """)
            
            submitted = st.form_submit_button(
                "💾 儲存交易" + (" + 同步資料" if is_new_stock else ""),
                type="primary",
                use_container_width=True,
            )
            
            if submitted:
                # === 決定 symbol ===
                if is_new_stock:
                    sym_input = (new_symbol or "").strip()
                    name_input = (new_name or "").strip()
                    ind_input = (new_industry or "").strip() or "未分類"
                    
                    if not sym_input or not name_input:
                        st.error("❌ 代號跟名稱必填")
                        st.stop()
                    
                    # 檢查當前 user 是否已有此股
                    existing = supabase.table("stocks").select("symbol").eq("symbol", sym_input).eq("user_id", get_user_id()).execute().data
                    if existing:
                        st.error(f"⚠️ {sym_input} 已在你的追蹤清單,請改選「📌 從追蹤清單選擇」模式")
                        st.stop()
                    
                    final_symbol = sym_input
                    
                    # 1. 加入 stocks
                    try:
                        supabase.table("stocks").insert({
                            "symbol": final_symbol,
                            "user_id": get_user_id(),
                            "name": name_input,
                            "industry": ind_input,
                        }).execute()
                    except Exception as e:
                        st.error(f"❌ 加入追蹤清單失敗: {e}")
                        st.stop()
                    
                    # 2. 跑 FinMind 同步(等完才繼續)
                    with st.spinner(f"🔄 正在抓取 {final_symbol} {name_input} 近 3 年資料(約 30 秒)..."):
                        try:
                            result = subprocess.run(
                                [sys.executable, "-c", 
                                 f"from data_pipeline import sync_symbol; sync_symbol('{final_symbol}')"],
                                capture_output=True, text=True, timeout=180
                            )
                            if result.returncode != 0:
                                st.warning(f"⚠️ FinMind 同步未完全成功,但股票已加入。可稍後到側邊欄「同步最新資料」重試")
                                with st.expander("查看同步 log"):
                                    st.code(result.stderr[-1000:] if result.stderr else result.stdout[-1000:])
                        except subprocess.TimeoutExpired:
                            st.warning("⏰ 同步超過 3 分鐘,請稍後手動同步")
                        except Exception as e:
                            st.warning(f"⚠️ 同步異常: {e}")
                else:
                    final_symbol = symbol_from_select
                
                # === 寫入 transactions ===
                try:
                    supabase.table("transactions").insert({
                        "symbol": final_symbol,
                        "user_id": get_user_id(),
                        "date": str(txn_date),
                        "action": action,
                        "shares": int(shares),
                        "price": float(price),
                        "fee": int(fee),
                        "tax": int(tax),
                        "note": note or None,
                    }).execute()
                    st.cache_data.clear()
                    if is_new_stock:
                        st.success(f"✅ 完成!已新增追蹤 {final_symbol} 並儲存交易")
                    else:
                        st.success(f"✅ 已儲存:{txn_date} {final_symbol} {action.upper()} {shares:,} 股 @ {price}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 儲存交易失敗: {e}")
    
    # ============================================
    # Tab 2: 交易紀錄(列表 + 編輯 + 刪除)
    # ============================================
    with tab2:
        st.subheader("所有交易紀錄")
        
        txns = supabase.table("transactions").select("*").eq("user_id", get_user_id()).order("date", desc=True).execute().data
        
        if not txns:
            st.info("還沒有交易紀錄,到「新增交易」分頁建立第一筆")
        else:
            df_txns = pd.DataFrame(txns)
            stocks_map = {s["symbol"]: s["name"] for s in load_stocks(get_user_id())}
            df_txns["name"] = df_txns["symbol"].map(stocks_map)
            df_txns["amount"] = df_txns["shares"] * df_txns["price"]
            df_txns["action_label"] = df_txns["action"].map(lambda a: "🟢 買" if a == "buy" else "🔴 賣")
            
            # 篩選
            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                filter_symbols = ["全部"] + sorted(df_txns["symbol"].unique().tolist())
                filter_symbol = st.selectbox("篩選個股", filter_symbols, key="filter_symbol")
            with col_f2:
                filter_action = st.selectbox("篩選動作", ["全部", "buy", "sell"], 
                                              format_func=lambda x: "全部" if x == "全部" else ("🟢 買進" if x == "buy" else "🔴 賣出"),
                                              key="filter_action")
            
            df_view = df_txns.copy()
            if filter_symbol != "全部":
                df_view = df_view[df_view["symbol"] == filter_symbol]
            if filter_action != "全部":
                df_view = df_view[df_view["action"] == filter_action]
            
            st.caption(f"顯示 {len(df_view)} / {len(df_txns)} 筆")
            
            # 顯示表格
            display = df_view[["date", "symbol", "name", "action_label", "shares", "price", "amount", "fee", "tax", "note", "id"]].copy()
            display = display.rename(columns={
                "date": "日期", "symbol": "代號", "name": "名稱",
                "action_label": "動作", "shares": "股數", "price": "價格",
                "amount": "金額", "fee": "手續費", "tax": "交易稅", "note": "備註",
                "id": "ID"
            })
            
            st.dataframe(
                display.drop(columns=["ID"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "股數": st.column_config.NumberColumn(format="localized"),
                    "價格": st.column_config.NumberColumn(format="%.2f"),
                    "金額": st.column_config.NumberColumn(format="localized"),
                    "手續費": st.column_config.NumberColumn(format="localized"),
                    "交易稅": st.column_config.NumberColumn(format="localized"),
                }
            )
            
            st.divider()
            
            # 刪除交易
            st.subheader("🗑️ 刪除交易")
            st.caption("選擇要刪除的交易(刪除後無法復原)")
            
            txn_options = {
                f"{r['date']} | {r['symbol']} {stocks_map.get(r['symbol'], '')} | {r['action_label']} {r['shares']:,} @ {r['price']:.2f}": r["id"]
                for _, r in df_view.iterrows()
            }
            
            if txn_options:
                col_d1, col_d2 = st.columns([3, 1])
                with col_d1:
                    selected_for_delete = st.selectbox("選擇交易", list(txn_options.keys()), key="del_txn_select")
                with col_d2:
                    st.markdown("&nbsp;")  # 空行對齊
                    if st.button("🗑️ 刪除", type="secondary", use_container_width=True):
                        try:
                            txn_id = txn_options[selected_for_delete]
                            supabase.table("transactions").delete().eq("id", txn_id).eq("user_id", get_user_id()).execute()
                            st.cache_data.clear()
                            st.success("✅ 已刪除")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 刪除失敗: {e}")
            
            # 編輯模式 (整合股息手動覆寫)
            st.divider()
            with st.expander("✏️ 編輯現有交易與股利覆寫"):
             st.caption("選擇要編輯的交易，修改內容或更新今年預估股息")
    
    # 保護:當 user 沒有交易紀錄時, txn_options 不會被定義,這裡補一個空 dict 避免 UnboundLocalError
    if 'txn_options' not in dir():
        txn_options = {}
    
    if txn_options:
        edit_label = st.selectbox("選擇要編輯的交易", list(txn_options.keys()), key="edit_select")
        edit_id = txn_options[edit_label]
        edit_row = df_view[df_view["id"] == edit_id].iloc[0]
        edit_sym = edit_row["symbol"] # 取得該筆交易的股票代號
        
        with st.form("edit_txn_form"):
            st.write(f"### 📦 編輯 {edit_sym} 交易內容")
            st.caption("💡 想設定該股的『今年預估股利』或補登除息?到「💰 股利管理」分頁")
            ec1, ec2 = st.columns(2)
            with ec1:
                e_date = st.date_input("日期", value=pd.to_datetime(edit_row["date"]).date())
                e_action = st.selectbox("動作", ["buy", "sell"], 
                                        index=0 if edit_row["action"] == "buy" else 1,
                                        format_func=lambda x: "🟢 買進" if x == "buy" else "🔴 賣出")
            with ec2:
                e_shares = st.number_input("股數", min_value=1, value=int(edit_row["shares"]), step=1)
                e_price = st.number_input("價格", min_value=0.01, value=float(edit_row["price"]), step=0.5, format="%.2f")
            
            ec3, ec4 = st.columns(2)
            with ec3:
                e_fee = st.number_input("手續費", min_value=0, value=int(edit_row["fee"]), step=1)
            with ec4:
                e_tax = st.number_input("交易稅", min_value=0, value=int(edit_row.get("tax", 0)), step=1)
            
            e_note = st.text_input("備註", value=edit_row.get("note") or "")
            
            if st.form_submit_button("💾 儲存所有變更", type="primary"):
                try:
                    # 更新交易紀錄 (transactions 表)
                    supabase.table("transactions").update({
                        "date": str(e_date),
                        "action": e_action,
                        "shares": int(e_shares),
                        "price": float(e_price),
                        "fee": int(e_fee),
                        "tax": int(e_tax),
                        "note": e_note or None,
                    }).eq("id", edit_id).eq("user_id", get_user_id()).execute()

                    # 清除快取並刷新
                    st.cache_data.clear()
                    st.success(f"✅ {edit_sym} 交易紀錄已更新")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 更新失敗: {e}")
    
    # ============================================
    # Tab 3: 💰 股利管理 (Phase 4.7)
    # ============================================
    with tab3:
        st.subheader("💰 股利管理")
        st.caption("管理你追蹤股票的股利資訊:預估年股利(用於 KPI) + 補登除息事件(用於含息成本計算)")
        
        stocks_list = load_stocks(get_user_id())
        if not stocks_list:
            st.info("尚未有追蹤股票,請先到「📈 追蹤清單管理」加入個股")
        else:
            # 選股票
            stock_option_map = {f"{s['symbol']} {s['name']}": s['symbol'] for s in stocks_list}
            div_label = st.selectbox(
                "選擇要管理的股票",
                list(stock_option_map.keys()),
                key="div_mgmt_stock"
            )
            div_symbol = stock_option_map[div_label]
            
            st.divider()
            
            # === 區塊 1: 今年預估年股利 (manual_dividend) ===
            st.markdown("#### 📊 今年預估年股利")
            st.caption("覆寫系統自動抓取的數字 (例如:FinMind 抓到去年的,但你知道今年宣告 X 元)。填 0 = 改回自動抓取。")
            
            # 撈現有 manual_dividend
            stock_res = supabase.table("stocks") \
                .select("manual_dividend") \
                .eq("symbol", div_symbol) \
                .eq("user_id", get_user_id()) \
                .execute()
            current_manual = float(stock_res.data[0].get("manual_dividend") or 0) if stock_res.data else 0.0
            
            # 顯示自動抓取的值供對照
            auto_div = get_latest_dividend(div_symbol)
            st.caption(f"🤖 系統自動抓取最近一次除息: **{auto_div} 元/股** (作為參考)")
            
            col_md1, col_md2 = st.columns([1, 2])
            with col_md1:
                new_manual_div = st.number_input(
                    "今年預估發放現金股利 (元/股)",
                    min_value=0.0,
                    value=current_manual,
                    step=0.1,
                    key=f"manual_div_{div_symbol}"
                )
            with col_md2:
                st.markdown("&nbsp;")  # 空行對齊
                if st.button(f"💾 儲存「今年預估股利」", key=f"save_manual_{div_symbol}"):
                    try:
                        supabase.table("stocks").update({
                            "manual_dividend": new_manual_div
                        }).eq("symbol", div_symbol).eq("user_id", get_user_id()).execute()
                        st.cache_data.clear()
                        if new_manual_div == 0:
                            st.success(f"✅ 已改回自動抓取 ({auto_div} 元/股)")
                        else:
                            st.success(f"✅ {div_symbol} 預估股利更新為 {new_manual_div} 元/股")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 儲存失敗: {e}")
            
            st.divider()
            
            # === 區塊 2: 補登除息 (insert dividends) ===
            st.markdown("#### 💵 補登除息(歷史事件)")
            st.caption(
                "用於 FinMind 還沒抓到的除息資料(通常公司除息後 1-2 週才會被收錄)。"
                "**補登後立刻反映到「累積已領股息」與「含息成本」計算**。"
            )
            
            with st.form(f"add_dividend_{div_symbol}", clear_on_submit=True):
                col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
                with col_d1:
                    new_ex_date = st.date_input(
                        "除息日",
                        value=datetime.now().date(),
                        help="股價開始反映除息的日期(不是錢入帳的日期)"
                    )
                with col_d2:
                    new_cash_div = st.number_input(
                        "現金股利 (元/股)",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        help="每股配發的現金股利金額"
                    )
                with col_d3:
                    new_stock_div = st.number_input(
                        "股票股利 (元/股)",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        help="多數股票為 0,有配股才填"
                    )
                
                submitted = st.form_submit_button("➕ 新增除息事件", type="primary")
                
                if submitted:
                    if new_cash_div == 0 and new_stock_div == 0:
                        st.error("❌ 現金股利跟股票股利至少要填一個")
                    else:
                        try:
                            div_record = {
                                "symbol": div_symbol,
                                "ex_date": str(new_ex_date),
                                "cash_dividend": float(new_cash_div),
                                "stock_dividend": float(new_stock_div),
                                "total_dividend": float(new_cash_div) + float(new_stock_div),
                            }
                            supabase.table("dividends").upsert(div_record).execute()
                            st.cache_data.clear()
                            st.success(
                                f"✅ {div_symbol} 補登成功 "
                                f"(除息日 {new_ex_date} / 現金 {new_cash_div} / 股票 {new_stock_div})"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 新增失敗: {e}")
            
            st.divider()
            
            # === 區塊 3: 該股的除息歷史 ===
            st.markdown(f"#### 📜 {div_symbol} 除息歷史")
            div_history = get_all_dividends(div_symbol)
            
            if not div_history:
                st.info("尚無除息紀錄")
            else:
                # 顯示最近 10 筆,反向(最新在上)
                recent_divs = sorted(div_history, key=lambda d: d.get("ex_date", ""), reverse=True)[:10]
                df_div = pd.DataFrame(recent_divs)
                
                if "ex_date" in df_div.columns:
                    df_div = df_div.rename(columns={
                        "ex_date": "除息日",
                        "cash_dividend": "現金股利(元/股)",
                        "stock_dividend": "股票股利(元/股)",
                        "total_dividend": "總股利(元/股)",
                    })
                    # 篩需要的欄位
                    display_cols = [c for c in ["除息日", "現金股利(元/股)", "股票股利(元/股)", "總股利(元/股)"] if c in df_div.columns]
                    st.dataframe(df_div[display_cols], use_container_width=True, hide_index=True)
                    st.caption(f"顯示最近 {len(recent_divs)} 筆 (總共 {len(div_history)} 筆)")
                
                # 提供刪除功能(用 SQL,因為刪除少用,放在 expander)
                with st.expander("⚠️ 刪除某筆除息紀錄(誤輸入時用)"):
                    st.caption("輸入要刪除的除息日(YYYY-MM-DD),按下按鈕後不可恢復")
                    del_col1, del_col2 = st.columns([2, 1])
                    with del_col1:
                        del_date = st.text_input(
                            "除息日",
                            placeholder="2026-06-17",
                            key=f"del_date_{div_symbol}"
                        )
                    with del_col2:
                        st.markdown("&nbsp;")
                        if st.button("🗑️ 刪除", key=f"del_btn_{div_symbol}"):
                            if not del_date:
                                st.error("請輸入除息日")
                            else:
                                try:
                                    supabase.table("dividends") \
                                        .delete() \
                                        .eq("symbol", div_symbol) \
                                        .eq("ex_date", del_date) \
                                        .execute()
                                    st.cache_data.clear()
                                    st.success(f"✅ 已刪除 {div_symbol} {del_date} 的除息紀錄")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 刪除失敗: {e}")
    
    # ============================================
    # Tab 4: 追蹤清單管理
    # ============================================
    with tab4:
        st.subheader("📈 追蹤清單管理")
        
        stocks = load_stocks(get_user_id())
        st.markdown(f"**目前追蹤 {len(stocks)} 檔個股**")
        
        # 顯示現有清單
        if stocks:
            stocks_df = pd.DataFrame(stocks)[["symbol", "name", "industry"]]
            stocks_df.columns = ["代號", "名稱", "產業"]
            st.dataframe(stocks_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 新增追蹤
        st.markdown("#### ➕ 新增追蹤股票")
        st.caption("輸入台股代號,系統會自動跑 FinMind 抓取近 3 年股價/PE/月營收/季報/籌碼")
        
        with st.form("add_stock_form", clear_on_submit=True):
            col_n1, col_n2, col_n3 = st.columns([1, 1, 2])
            with col_n1:
                new_symbol = st.text_input("代號 *", placeholder="例如 2330", max_chars=6)
            with col_n2:
                new_name = st.text_input("名稱 *", placeholder="例如 台積電")
            with col_n3:
                new_industry = st.text_input("產業", placeholder="例如 半導體(選填)")
            
            add_submitted = st.form_submit_button("➕ 新增並同步", type="primary", use_container_width=True)
            
            if add_submitted:
                if not new_symbol or not new_name:
                    st.error("❌ 代號跟名稱必填")
                else:
                    new_symbol = new_symbol.strip()
                    # 檢查當前 user 是否已有此股
                    existing = supabase.table("stocks").select("symbol").eq("symbol", new_symbol).eq("user_id", get_user_id()).execute().data
                    if existing:
                        st.warning(f"⚠️ {new_symbol} 已在追蹤清單中")
                    else:
                        try:
                            # 1. 新增到 stocks 表 (含 user_id)
                            supabase.table("stocks").insert({
                                "symbol": new_symbol,
                                "user_id": get_user_id(),
                                "name": new_name.strip(),
                                "industry": new_industry.strip() or "未分類",
                            }).execute()
                            st.success(f"✅ 已加入 {new_symbol} {new_name}")
                            
                            # 2. 跑 FinMind 同步
                            st.info(f"🔄 正在抓取 {new_symbol} 近 3 年資料...這需要約 30 秒")
                            with st.spinner("同步中..."):
                                try:
                                    result = subprocess.run(
                                        [sys.executable, "-c", 
                                         f"from data_pipeline import sync_symbol; sync_symbol('{new_symbol}')"],
                                        capture_output=True, text=True, timeout=180
                                    )
                                    if result.returncode == 0:
                                        st.success("✅ 資料同步完成!")
                                        with st.expander("查看 log"):
                                            st.code(result.stdout[-2000:])
                                    else:
                                        st.error(f"❌ 同步失敗,但股票已加入。可手動重跑同步")
                                        st.code(result.stderr[-1000:])
                                except subprocess.TimeoutExpired:
                                    st.error("⏰ 超過 3 分鐘,請稍後到側邊欄按「同步最新資料」")
                                except Exception as e:
                                    st.error(f"❌ 同步異常: {e}")
                            
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 新增失敗: {e}")
        
        st.divider()
        
        # 移除追蹤
        st.markdown("#### 🗑️ 移除追蹤股票")
        st.caption("⚠️ 警告:移除股票會同時刪除該檔的所有交易、論點、AI 觀察紀錄")
        
        if stocks:
            stock_to_remove_options = {f"{s['symbol']} {s['name']}": s['symbol'] for s in stocks}
            
            col_r1, col_r2 = st.columns([3, 1])
            with col_r1:
                to_remove_label = st.selectbox(
                    "選擇要移除的股票", 
                    list(stock_to_remove_options.keys()),
                    key="remove_stock"
                )
            with col_r2:
                st.markdown("&nbsp;")
                
                # 用 confirmation 機制
                confirm_key = f"confirm_remove_{stock_to_remove_options[to_remove_label]}"
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False
                
                if not st.session_state[confirm_key]:
                    if st.button("🗑️ 移除", type="secondary", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    if st.button("⚠️ 確認移除", type="primary", use_container_width=True):
                        try:
                            sym_to_remove = stock_to_remove_options[to_remove_label]
                            uid = get_user_id()
                            # 只刪當前 user 的個人資料,市場資料(daily_prices 等)是共用,不刪
                            supabase.table("transactions").delete().eq("symbol", sym_to_remove).eq("user_id", uid).execute()
                            supabase.table("theses").delete().eq("symbol", sym_to_remove).eq("user_id", uid).execute()
                            supabase.table("thesis_reviews").delete().eq("symbol", sym_to_remove).eq("user_id", uid).execute()
                            supabase.table("stocks").delete().eq("symbol", sym_to_remove).eq("user_id", uid).execute()
                            
                            st.session_state[confirm_key] = False
                            st.cache_data.clear()
                            st.success(f"✅ 已從你的追蹤清單移除 {to_remove_label} (市場資料保留供共用)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 移除失敗: {e}")
                            st.session_state[confirm_key] = False


# ============================================================
# 主程式
# ============================================================
PAGES = {
    "📊 投資組合總覽": page_portfolio_overview,
    "📈 個股技術分析": page_stock_detail,
    "📝 投資論點": page_thesis,
    "⚙️ 交易管理": page_transactions,
}

with st.sidebar:
    st.title("📈 持股分析")
    page = st.radio("頁面", list(PAGES.keys()))
    st.divider()
    
    # === 資料新鮮度 ===
    freshness = load_data_freshness()
    with st.expander("📅 資料最新日期", expanded=False):
        for k, v in freshness.items():
            st.caption(f"**{k}**:{v if v else '無資料'}")
    
    st.divider()
    
    # === 同步資料按鈕 ===
    if st.button("📥 同步最新資料", use_container_width=True, help="跑 data_pipeline.py 抓 FinMind 最新資料(約 30 秒)"):
        with st.spinner("正在同步...請稍候 30-60 秒"):
            try:
                result = subprocess.run(
                    [sys.executable, "data_pipeline.py"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    st.success("✅ 同步完成!")
                    st.cache_data.clear()
                    with st.expander("查看 log"):
                        st.code(result.stdout[-2000:])
                    st.rerun()
                else:
                    st.error(f"❌ 失敗")
                    st.code(result.stderr[-1000:])
            except subprocess.TimeoutExpired:
                st.error("⏰ 超過 5 分鐘,可能是網路問題")
            except Exception as e:
                st.error(f"❌ 例外:{e}")
    
    if st.button("🔄 清除快取", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

PAGES[page]()