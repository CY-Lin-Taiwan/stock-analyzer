"""
AI 分析模組 (Phase 4)

新增:
- holding_context 參數(持股事實 + 累積股息)
- AI 對「持股者」給不同視角的觀點
- 不影響 AI 獨立判斷(只是聚焦)
"""
import os
import json
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from supabase import create_client

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# API 逾時(秒)。pro 級模型 + 長 prompt 生成完整 JSON,正常約 20~60 秒;
# 超過就多半是連線或服務端問題。不設 timeout 會無限等待 ——
# 畫面只會一直轉,沒有錯誤也不會中止,無法判斷是慢還是掛了。
API_TIMEOUT = 180

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

_supabase = None
def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase


HORIZON_LABELS = {
    "short": "短期 (1-3 個月)",
    "medium": "中期 (6-12 個月)",
    "long": "長期 (3 年以上)",
}

STANCE_LABELS = {
    "strongly_bullish": "強烈看多",
    "moderately_bullish": "中度看多",
    "neutral_lean_bullish": "中性偏多",
    "neutral": "中性",
    "neutral_lean_bearish": "中性偏空",
    "moderately_bearish": "中度看空",
    "strongly_bearish": "強烈看空",
}


# ============================================================
# 工具函式:計算累積已領股息
# ============================================================
def _db_retry(fn, tries=3, delay=0.4, label=""):
    """
    Supabase 查詢的重試包裝。

    連線偶爾會出現 httpx.ReadError: Resource temporarily unavailable ——
    那是 TCP 瞬斷而非程式錯誤,重試一次通常就過。但若沒有保護,
    單一次瞬斷會讓整個頁面掛掉。

    頁面載入時的 DB 查詢已增加不少(還原股價、公司行為、股息品質、
    持續性、大盤基準…),瞬斷機率相應變高,故統一加重試。
    """
    import time as _t
    last = None
    for n in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if n < tries - 1:
                _t.sleep(delay * (n + 1))
    print(f"[ai_analyzer] DB 查詢失敗{(' ' + label) if label else ''}: {last}")
    return None


def calculate_accumulated_dividends(
    symbol: str,
    initial_position_date: str,
    current_shares: int,
    transactions: Optional[list] = None,
) -> dict:
    """計算「初始建倉日之後」實際領到的股息
    
    如果有傳 transactions,會用精準版算法(對齊每次除息日的當下持股)
    若沒傳,fallback 用簡化版(目前持股 × 累積股息元/股)
    """
    sb = _get_supabase()
    
    res = _db_retry(
        lambda: sb.table("dividends")
        .select("ex_date,cash_dividend,total_dividend")
        .eq("symbol", symbol)
        .gt("ex_date", initial_position_date)
        .order("ex_date")
        .execute(),
        label=f"dividends {symbol}",
    )

    divs = (res.data or []) if res is not None else []
    
    if not divs:
        return {
            "events_count": 0,
            "total_per_share": 0,
            "total_received": 0,
            "events": [],
        }
    
    # === 精準版:對齊每次除息日的當下持股 ===
    if transactions:
        # 整理該檔交易為時間序列
        txns_sorted = []
        for t in transactions:
            if t.get("symbol") != symbol:
                continue
            date_str = t.get("date")
            action = t.get("action", "buy")
            shares = float(t.get("shares", 0))
            if not date_str or shares <= 0:
                continue
            txns_sorted.append({
                "date": date_str,
                "action": action.lower(),
                "shares": shares,
            })
        txns_sorted.sort(key=lambda x: x["date"])
        
        total_received = 0.0
        events = []
        
        for div in divs:
            ex_date = div.get("ex_date")
            cash_div = float(div.get("cash_dividend", 0) or 0)
            if not ex_date or cash_div <= 0:
                continue
            
            # 算除息日當下持股
            shares_at_ex = 0.0
            for txn in txns_sorted:
                if txn["date"] < ex_date:
                    if txn["action"] == "buy":
                        shares_at_ex += txn["shares"]
                    elif txn["action"] == "sell":
                        shares_at_ex -= txn["shares"]
            
            if shares_at_ex <= 0:
                continue
            
            received = shares_at_ex * cash_div
            total_received += received
            events.append({
                "date": ex_date,
                "cash_dividend": cash_div,
                "shares_at_that_time": shares_at_ex,
                "received": received,
            })
        
        total_per_share = (total_received / current_shares) if current_shares > 0 else 0
        
        return {
            "events_count": len(events),
            "total_per_share": round(total_per_share, 2),
            "total_received": round(total_received, 0),
            "events": events,
        }
    
    # === Fallback:簡化版(沒傳 transactions 時用) ===
    total_per_share = sum(d.get("cash_dividend", 0) or 0 for d in divs)
    total_received = total_per_share * current_shares
    
    return {
        "events_count": len(divs),
        "total_per_share": round(total_per_share, 2),
        "total_received": round(total_received, 0),
        "events": [
            {
                "date": d["ex_date"],
                "cash_dividend": d.get("cash_dividend", 0),
            }
            for d in divs
        ],
    }


def build_holding_context(
    symbol: str,
    transactions: list,
    current_price: float,
) -> Optional[dict]:
    """從 transactions 算出該檔的持股 context"""
    if not transactions:
        return None
    
    # 篩這檔的交易
    txns = [t for t in transactions if t["symbol"] == symbol]
    if not txns:
        return None
    
    # 算淨持股
    buy = [t for t in txns if t["action"] == "buy"]
    sell = [t for t in txns if t["action"] == "sell"]
    
    total_buy_shares = sum(t["shares"] for t in buy)
    total_buy_cost = sum(t["shares"] * t["price"] + (t.get("fee") or 0) for t in buy)
    total_sell_shares = sum(t["shares"] for t in sell)
    
    current_shares = total_buy_shares - total_sell_shares
    
    if current_shares <= 0:
        return None  # 沒持股(已全賣)
    
    avg_cost = total_buy_cost / total_buy_shares if total_buy_shares > 0 else 0
    total_cost = current_shares * avg_cost
    current_value = current_shares * current_price
    unrealized_pnl = current_value - total_cost
    unrealized_pnl_pct = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    initial_txns = [t for t in buy if t.get("is_initial_position")]
    if initial_txns:
        initial_date = min(t["date"] for t in initial_txns)
    else:
        initial_date = min(t["date"] for t in buy)
    
    # 計算累積已領股息(精準版:對齊每次除息日的當下持股)
    div_info = calculate_accumulated_dividends(symbol, initial_date, current_shares, transactions=txns)
    
    # 算「有效成本」(扣除已領股息)
    effective_cost_per_share = avg_cost - div_info["total_per_share"]
    effective_cost_total = effective_cost_per_share * current_shares
    effective_cost_pnl_pct = ((current_price - effective_cost_per_share) / effective_cost_per_share) * 100 if effective_cost_per_share > 0 else 0
    
    initial_date_obj = datetime.strptime(initial_date, "%Y-%m-%d")
    holding_days = (datetime.now() - initial_date_obj).days
    
    return {
        "shares": current_shares,
        "avg_cost": round(avg_cost, 2),
        "total_cost": round(total_cost, 0),
        "current_price": current_price,
        "current_value": round(current_value, 0),
        "unrealized_pnl": round(unrealized_pnl, 0),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "initial_position_date": initial_date,
        "holding_days": holding_days,
        "dividends_received_per_share": div_info["total_per_share"],
        "dividends_received_total": div_info["total_received"],
        "dividend_events_count": div_info["events_count"],
        "effective_cost_per_share": round(effective_cost_per_share, 2),
        "effective_cost_total": round(effective_cost_total, 0),
        "effective_cost_pnl_pct": round(effective_cost_pnl_pct, 2),
    }


# ============================================================
# 主功能:AI 獨立觀察
# ============================================================
def build_observation_prompt(
    symbol: str,
    name: str,
    industry: str,
    primary_horizon: str,
    valuation: dict,
    monthly_rev: list,
    quarterly_fin: list,
    chips: dict,
    data_freshness: str = None,
    holding_context: Optional[dict] = None,
    upcoming_events: str = None,
    news_list: Optional[list] = None,
    metrics: Optional[dict] = None,
) -> str:
    """組裝 AI 獨立觀察 prompt"""
    # 靜默 fallback 很危險:若 time_horizon 為空或 key 改名,
    # AI 會被當成「中期」分析,而使用者設定的可能是長期 —— 不會有任何警告。
    if primary_horizon in HORIZON_LABELS:
        primary_label = HORIZON_LABELS[primary_horizon]
    else:
        primary_label = ("⚠️ 未設定(以下分析預設為中期 6-12 個月;"
                         "請在 `ai_self_disclosed_limits` 明確指出"
                         "「使用者未設定時間框架」)")
        print(f"[ai_analyzer] ⚠️ 未知的 time_horizon: {primary_horizon!r},"
              f"已退回中期預設", flush=True)
    
    # 整理月營收
    rev_text = "(無資料)"
    if monthly_rev:
        rev_lines = []
        for r in monthly_rev[-6:]:
            yoy = r.get("revenue_yoy")
            yoy_str = f"YoY {yoy:+.1f}%" if yoy is not None else "YoY N/A"
            rev_lines.append(f"  - {r['year_month']}: {r['revenue']/1e8:,.1f} 億 ({yoy_str})")
        all_revs = [r["revenue"] for r in monthly_rev if r.get("revenue")]
        if all_revs:
            rev_max = max(all_revs) / 1e8
            rev_min = min(all_revs) / 1e8
            rev_lines.append(f"  - [歷史對照] 最高 {rev_max:,.1f} 億 / 最低 {rev_min:,.1f} 億 (近 {len(all_revs)} 月)")
        rev_text = "\n".join(rev_lines)
    
    # 整理季報
    fin_text = "(無資料)"
    if quarterly_fin:
        fin_lines = []
        for q in quarterly_fin[-4:]:
            eps = q.get("eps", "N/A")
            gm = q.get("gross_margin")
            roe = q.get("roe")
            gm_str = f"{gm:.2f}%" if gm is not None else "N/A"
            roe_str = f"{roe:.2f}%" if roe is not None else "N/A"
            fin_lines.append(
                f"  - {q['year_quarter']}: EPS {eps} | 毛利率 {gm_str} | ROE {roe_str}"
            )
        all_eps = [q["eps"] for q in quarterly_fin if q.get("eps") is not None]
        if all_eps:
            eps_max = max(all_eps)
            eps_min = min(all_eps)
            fin_lines.append(f"  - [歷史對照] EPS 最高 {eps_max:.2f} / 最低 {eps_min:.2f} (近 {len(all_eps)} 季)")
        fin_text = "\n".join(fin_lines)
    
    # 籌碼
    chips_text = "(無資料)"
    if chips:
        chips_text = (
            f"  - 外資 5 日累計: {chips.get('foreign_5d', 0):+,.0f} 張\n"
            f"  - 投信 5 日累計: {chips.get('trust_5d', 0):+,.0f} 張\n"
            f"  - 連續 {chips.get('consecutive_days', 0)} 天{chips.get('direction', '無')}\n"
            f"  - 融資 30 日變化: {chips.get('margin_change_pct', 0):+.1f}%\n"
            f"  - 外資持股比: {chips.get('foreign_holding', 'N/A')}%"
        )
    
    # === 持股 context 區塊 ===
    holding_block = ""
    if holding_context:
        h = holding_context
        annual_div_est = valuation.get("dividend_yield", 0) * valuation.get("close", 0) / 100 if valuation.get("dividend_yield") else 0
        yoc = (annual_div_est / h["avg_cost"]) * 100 if h["avg_cost"] > 0 else 0
        
        holding_block = f"""

# 使用者持股狀況 (這是「客觀事實」,不是使用者的立場)

⚠️ 重要:這個資訊只是要你「**從持股者視角**」分析,不是要你迎合使用者立場。
   使用者持有 ≠ 看多。可能被套牢、可能想加碼、可能想減碼。
   你的角色是「**對持股者有用的獨立觀點**」,不是「**為持股者背書**」。

## 持股事實
- 持股: {h['shares']:,} 股 (約 {h['shares']/1000:.0f} 張)
- 平均成本: {h['avg_cost']} 元/股
- 總成本: {h['total_cost']:,.0f} 元
- 現價: {h['current_price']}
- 目前市值: {h['current_value']:,.0f} 元
- 未實現損益: {h['unrealized_pnl']:+,.0f} 元 ({h['unrealized_pnl_pct']:+.2f}%)
- 系統初始建倉日: {h['initial_position_date']} (持有 {h['holding_days']} 天)

## 累積已領股息 (從初始建倉日之後)
- 已除息次數: {h['dividend_events_count']} 次
- 累積每股股息: {h['dividends_received_per_share']} 元
- 累積總股息: {h['dividends_received_total']:,.0f} 元

## 真實報酬(扣除已領股息)
- 有效成本: {h['effective_cost_per_share']} 元/股 (= 均價 {h['avg_cost']} - 累積股息 {h['dividends_received_per_share']})
- 扣除股息後的損益: {h['effective_cost_pnl_pct']:+.2f}%
- 成本殖利率 (Yield on Cost, 預估): {yoc:.2f}%

## 分析視角要求

請從「**持股者的決策框架**」角度分析:
1. 對「持有 vs 加碼 vs 減碼」三種行動的判讀(只給分析,不下決策)
2. 當前位階對持股者的意義(獲利保護點? 加碼進場點? 出場觀察點?)
3. 「成本殖利率 vs 現價殖利率」對長期策略的意義
4. 持股時間框架下要追蹤的訊號
"""
    
    # === 處理前瞻事件 ===
    event_text = ""
    if upcoming_events and upcoming_events.strip():
        event_text = f"\n## 近期重大事件 / 前瞻資訊 (使用者主動輸入,極重要)\n- {upcoming_events}\n"
    
    # === 處理自動抓取的新聞 ===
    news_text = ""
    if news_list:
        news_lines = ["", "## 近期主流媒體報導 (系統自動抓取,僅供參考)", ""]
        for i, n in enumerate(news_list, 1):
            source = n.get('source', '未知')
            age = n.get('age', '')
            title = n.get('title', '')
            news_lines.append(f"{i}. [{source} · {age}] {title}")
        news_text = "\n".join(news_lines) + "\n"
    
    # === 技術狀態 (合成後的一句話 + 可信度) ===
    # 不傳 %B / K / D / 斜率等原始數字,只傳 Python 端合成的結論:
    #   1. %B 和 KD 都在回答「價格在近期區間的哪個位置」,高度重疊;
    #      分開餵會讓 AI 把同一件事當成兩個獨立證據,假性提高信心度
    #   2. 三個指標各自強制引用,AI 會變成清單朗讀,稀釋質化分析深度
    #   3. 判斷邏輯留在可被檢查的 Python 裡,不是丟給 AI 自由心證
    metrics_text = "(無資料)"
    tech_confidence = None
    if metrics:
        ts = metrics.get("tech_state")
        if ts and ts.get("has_data"):
            tech_confidence = ts.get("confidence")
            lines = [f"- {ts['text']}"]
            lines.append(f"- 訊號可信度: {tech_confidence} ({ts.get('cross_reason', '')})")
            if ts.get("market_sync"):
                lines.append(
                    "- ⚠️ 此訊號與大盤同步,屬全市場走勢。**不得**把它描述成"
                    "該個股專屬的轉強/轉弱訊號,也不得據此調整對個股的判斷"
                )
            if metrics.get("price_adjust_reliable") is False:
                lines.append("- ⚠️ 警告: 還原股價驗證未通過,以上技術指標可能失真,請降低權重")
            metrics_text = "\n".join(lines)

    # === 股息品質 / 持續性 ===
    # 刻意不傳填息率百分比:樣本小、窗口人為設定,包裝成乾淨的百分比
    # 會被 AI 當成硬證據。只傳結構性結論與該響的警報。
    dividend_text = ""
    dividend_must_mention = False
    if metrics:
        dsig = metrics.get("dividend_signal")
        if dsig and dsig.get("lines"):
            dividend_must_mention = dsig.get("must_mention", False)
            dividend_text = ("\n## 股息品質\n" + "\n".join(dsig["lines"]) + "\n")
        ssig = metrics.get("sustainability_signal")
        if ssig and ssig.get("lines"):
            if ssig.get("must_mention"):
                dividend_must_mention = True
            dividend_text += ("\n## 股息持續性(配息未來發得出來嗎)\n"
                              + "\n".join(ssig["lines"]) + "\n")
    
    freshness_note = f"(資料最新更新至 {data_freshness})" if data_freshness else ""
    today_str = datetime.now().strftime("%Y-%m-%d")

    # === 處理 f-string 限制 (將帶有引號與大括號的 JSON 結構拉到字串外) ===
    holder_json = ""
    qualitative_desc = ""
    holder_rule = ""
    if holding_context:
        holder_json = """  "holder_perspective": {
    "action_framework": "對『持有 vs 加碼 vs 減碼』三種行動的分析(不下決策,只給框架)",
    "current_position_meaning": "當前位階對持股者的意義",
    "yoc_strategy_note": "YoC vs 現價殖利率 對策略的意義",
    "holder_signals": ["持股者特別該追蹤的訊號 1", "訊號 2"]
  },"""
        qualitative_desc = "。若使用者持股,從持股者視角給靈魂分析"
        holder_rule = "10. 持股者視角下,要在 holder_perspective 區塊給具體框架"
    
    prompt = f"""你是一位有膽識的台股獨立分析師。今天是 {today_str}。

# 任務

對 **{symbol} {name}** 給出你眼中的觀點。

# 核心原則(極為重要,違反 = 失敗)

1. **必須明確表態**:明確的整體立場(看多/看空/中性 + 信心度)。「兩面討好」是廢話。
2. **不是要你下決策**:給「判斷傾向」,不是叫使用者買賣
3. **保持開放**:明確列出「什麼訊號會讓你改變想法」
4. **前三區用列點**(處境/情境/訊號),不要堆砌敘述。質化分析放最終「綜合判斷」
5. **情境必須給機率**:三個情境機率加總 = 100%
6. **訊號用重要性標籤**:🔴 關鍵 / 🟡 重要 / 🟢 次要
7. **使用最新資料** {freshness_note}
8. **景氣循環股特別警覺**:基期效應、PE 反指標、循環位置
9. **科技成長股特別警覺**:估值已透支多少未來、競爭格局
{"10. **持股者視角**:見下方持股區塊指引" if holding_context else ""}
11. **新聞使用守則**:若下方有「近期主流媒體報導」區塊,你必須:
    a. 區分「實質事件」vs「股價評論/猜測」── 只把實質事件納入分析(法說會、新訂單、政策、營收公告等)
    b. 不被單一新聞的情緒/立場主導(媒體有立場,你要客觀)
    c. 對撞:若新聞跟客觀數據矛盾,以數據為準,但要說明「為何媒體這樣解讀」
12. **引用透明**:若你的分析有用到新聞資訊,必須在 `news_references` 欄位明確列出「引用第 N 則新聞 + 怎麼用它」
12.5. **持股損益不得影響分析結論**:
    a. 虧損**不是**「更該加碼」或「更該續抱」的理由;獲利也**不是**「更該續抱」
       或「該獲利了結」的理由。持股損益是 context,不是論據
    b. 檢驗方式:如果把同一份數據給一個「完全沒持股」的人看,
       你的結論會不會不同?會的話,你就是在遷就部位而非分析數據
    c. **禁止對套牢部位使用缺乏依據的安慰性語言** ——
       「黎明前」「終將回歸」「不該放棄」「地心引力會把股價拉回」
       這類話若沒有具體、可驗證的依據支撐,一律不得使用
    d. 若數據顯示結構性問題(例如純價格報酬長期為負、配息侵蝕本金),
       **不得用「投資看的是未來」這類說法帶過** —— 必須正面說明
       那個問題是否已經改善,以及用什麼指標判斷
    e. 對套牢部位,`what_would_change_my_mind` 必須包含
       「什麼情況下應該認賠出場」的具體條件,不可只寫看多的翻轉條件

12.8. **綜合判斷必須是全面的**:`qualitative_summary` 是整份分析的核心,
    必須把所有有資料的面向都納入,而不是只挑好講故事的講。
    a. 常見的偷懶模式:只寫基本面 + 籌碼(因為有新聞可以呼應),
       完全不提技術狀態與股息品質 —— 這不算完成分析
    b. `technical_read` 是技術面的**明細**,不是它的替代品。
       綜合判斷仍必須把技術狀態納入敘事,說明它與基本面是**互相佐證**
       還是**互相矛盾**
    c. 面向之間的**矛盾**比一致更值得寫。例如「營收與毛利率創高,
       但股價與技術面弱勢」本身就是最重要的發現,要正面處理而不是各說各話

13. **技術狀態**:若下方有「近期技術狀態」區塊,`technical_read` 四個欄位全部必填。
    另外三條原則:
    a. 若含「鈍化」,必須說明方向意義 —— 高檔鈍化 = 趨勢延續,
       此期間的死亡交叉多為假訊號,不是賣出訊號;低檔鈍化 = 跌勢未止,
       不是「超賣就會反彈」
    b. **不得單獨改變 `stance`**。技術面是短線位階,立場要由基本面
       (營收、財務、籌碼、產業循環位置)決定;技術面只能影響
       `holder_perspective` 的加減碼框架與 `signals_to_monitor`
    c. 若標示「與大盤同步」,那是 beta 不是 alpha ——
       寫成「該股出現轉弱訊號」是錯的,應寫「大盤回檔帶動,
       該股尚無個股層級的技術訊號」

{"14. **股息(強制)**:上方股息區塊出現 ⚠️ 警報。提到殖利率、配息或存股相關判斷時,**必須**一併說明該警報的內容與影響,不可只報好消息。注意:配息金額穩定不等於配息可持續 —— 若警報指出配息靠保留盈餘硬撐,必須明講。" if dividend_must_mention else "14. **股息(條件式)**:若上方股息區塊出現 ⚠️ 警報,才需在分析中提及並說明影響;沒有警報就不必特別著墨,不要為了交差而硬寫。"}

# 個股
- {symbol} {name} ({industry})

# 客觀資料

## 估值
- 現價: {valuation.get('close', 'N/A')}
- PE: {valuation.get('pe', 'N/A')}
- PB: {valuation.get('pb', 'N/A')}
- 殖利率: {valuation.get('dividend_yield', 'N/A')}%

## 近期技術狀態 (布林位階 + 市場狀態 + KD,均使用還原股價)
{metrics_text}
{dividend_text}

## 近 6 個月營收
{rev_text}

## 近 4 季財務
{fin_text}

## 近期籌碼面
{chips_text}
{holding_block}
{event_text}
{news_text}

# 主力時間框架

{primary_label}

# 輸出格式

**只輸出純 JSON,不要 markdown**:

{{
  "primary_horizon": "{primary_horizon}",
  
  "overall_judgment": {{
    "stance": "從七選一: strongly_bullish / moderately_bullish / neutral_lean_bullish / neutral / neutral_lean_bearish / moderately_bearish / strongly_bearish",
    "confidence": "1-10 整數",
    "core_reasoning": "為什麼這個立場?2-3 句質化但簡潔",
    "what_would_change_my_mind": ["訊號 1", "訊號 2", "訊號 3"]
  }},
  
  "current_situation": [
    "處境列點 1", "處境列點 2", "處境列點 3", "處境列點 4"
  ],
  
  "scenario_analysis": [
    {{
      "name": "情境 A 名稱",
      "probability": 50,
      "key_assumptions": ["假設 1", "假設 2"],
      "implications": ["若成真會發生 1", "若成真會發生 2"]
    }},
    {{
      "name": "情境 B 名稱",
      "probability": 30,
      "key_assumptions": ["..."],
      "implications": ["..."]
    }},
    {{
      "name": "情境 C 名稱",
      "probability": 20,
      "key_assumptions": ["..."],
      "implications": ["..."]
    }}
  ],
  
  "signals_to_monitor": [
    {{"importance": "critical", "signal": "🔴 訊號 1", "why_matters": "..."}},
    {{"importance": "critical", "signal": "🔴 訊號 2", "why_matters": "..."}},
    {{"importance": "important", "signal": "🟡 訊號 1", "why_matters": "..."}},
    {{"importance": "important", "signal": "🟡 訊號 2", "why_matters": "..."}},
    {{"importance": "minor", "signal": "🟢 訊號 1", "why_matters": "..."}}
  ],
  
{holder_json}
  "technical_read": {{
    "market_state": "引用市場狀態(上升趨勢 / 下降趨勢 / 區間震盪 / 帶寬收縮),並說明它讓位階的意義變成什麼",
    "position": "引用布林位階與 KD 位階的數字與描述",
    "signal_confidence": "可信度標籤(高 / 中高 / 低 / 不適用)+ 它代表什麼。『不適用』= 沒有交叉可評,不是訊號很弱",
    "impact": "以上對你的判斷有什麼影響。若認為影響不大,要說明為什麼(例如長線存股),不可略過"
  }},
  "qualitative_summary": "綜合判斷(2-3 段,真正的質化分析。**必須整合以下所有面向**:基本面(營收/財務)、籌碼面、技術狀態、股息品質、產業循環位置、持股 context。有訊號的面向不可略過,沒有資料的面向要明說。不是逐項條列,而是把它們織成一個連貫的判斷 —— 說明哪些面向互相佐證、哪些互相矛盾{qualitative_desc})",
  
  "ai_self_disclosed_limits": [
    "我這次分析的限制 1", "限制 2"
  ],
  
  "data_references": [
    "本次引用的關鍵數據點 1", "數據點 2",
    "(技術狀態已在 technical_read 欄位,此處不必重複)"
  ],
  
  "news_references": [
    "(若有引用新聞) 引用第 N 則 [來源] 標題: 怎麼用它,例如:佐證/補充/反證了 X"
  ]
}}

# 嚴格規則

1. 整體立場必填(不能空白逃避)
2. 三情境機率加總 = 100%
3. what_would_change_my_mind 至少 3 條
4. current_situation 4-6 個列點
5. signals_to_monitor 至少 5 個,含 2 個 🔴
6. qualitative_summary 是質化分析容器
7. 不寫「建議買進/賣出」
8. 不要編造數字
9. 不寫多空辯論結構
10. news_references 如果這次分析沒用到新聞,給空陣列 []。若用了,必須明確說「怎麼用」
11. technical_read 四個欄位必填(若有提供技術狀態),不可留空或寫「無」
12. 技術狀態不得單獨改變 stance;可信度「低」的訊號不可作為有力證據
13. 引用配息區間時,不得以單一極端年度(最低或最高那年)代表未來配息水準;
    應以中位數為錨點。早期年度可能因減資、產業循環位置而不可比
{holder_rule}
"""
    return prompt


def audit_observation(data: dict, metrics: Optional[dict] = None,
                      holding_context: Optional[dict] = None) -> List[str]:
    """
    生成後檢查:規則有沒有真的被遵守。

    為什麼需要:
      prompt 規則是「軟要求」—— 模型漏掉不會報錯、不會有任何反饋。
      實測就發生過:股息訊號送了「賺股息、賠價差」,綜合判斷卻只用
      一句「投資看的是未來」帶過;技術狀態送了布林與 KD,質化分析
      完全沒提。加更多規則只會稀釋既有規則,不會提高遵守率。

      這支不強迫模型照做,而是讓「沒照做」變得看得見 ——
      失效可見,比多寫三條規則有用。

    Returns: 未通過的檢查項目說明(空 list = 全部通過)
    """
    issues = []
    summary = data.get("qualitative_summary", "") or ""
    m = metrics or {}

    # 1) 技術狀態有送 → 質化分析應該要提到
    if m.get("tech_state", {}).get("has_data"):
        kws = ("布林", "%B", "位階", "KD", "鈍化", "交叉", "趨勢", "帶寬", "技術")
        if not any(k in summary for k in kws):
            issues.append("綜合判斷未提及技術狀態(布林/KD/市場狀態)")
        tr = data.get("technical_read") or {}
        empty = [k for k in ("market_state", "position",
                             "signal_confidence", "impact") if not tr.get(k)]
        if empty:
            issues.append(f"technical_read 欄位未填:{'、'.join(empty)}")

    # 2) 股息有送 → 質化分析應該要提到
    if (m.get("dividend_signal", {}).get("lines")
            or m.get("sustainability_signal", {}).get("lines")):
        kws = ("配息", "股息", "殖利率", "填息", "配發", "除息")
        if not any(k in summary for k in kws):
            issues.append("綜合判斷未提及股息品質或配息持續性")

    # 3) 有 ⚠️ 警報 → 一定要處理,不可略過
    warned = (m.get("dividend_signal", {}).get("must_mention")
              or m.get("sustainability_signal", {}).get("must_mention"))
    if warned:
        kws = ("賠價差", "侵蝕", "本金", "老本", "未填", "衰退", "警", "風險")
        if not any(k in summary for k in kws):
            issues.append("有股息 ⚠️ 警報,但綜合判斷未正面處理")

    # 4) 套牢部位 → 必須有認賠條件
    pnl = (holding_context or {}).get("unrealized_pnl_pct")
    if isinstance(pnl, (int, float)) and pnl < -10:
        changes = " ".join(
            data.get("overall_judgment", {}).get("what_would_change_my_mind", []))
        if not any(k in changes for k in ("認賠", "停損", "出場", "賣出", "退出")):
            issues.append("持股虧損逾 10%,但未給出任何認賠/出場條件")

    # 5) 立場必填
    if not data.get("overall_judgment", {}).get("stance"):
        issues.append("overall_judgment.stance 未填")

    return issues


def run_observation(
    symbol: str,
    name: str,
    industry: str,
    primary_horizon: str,
    valuation: dict,
    monthly_rev: list,
    quarterly_fin: list,
    chips: dict,
    data_freshness: str = None,
    holding_context: Optional[dict] = None,
    upcoming_events: str = None,
    news_list: Optional[list] = None,
    metrics: Optional[dict] = None,
    model_name: str = None,
) -> dict:
    """執行 AI 獨立觀察"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    prompt = build_observation_prompt(
        symbol, name, industry, primary_horizon,
        valuation, monthly_rev, quarterly_fin, chips, data_freshness,
        holding_context, upcoming_events, news_list, metrics
    )
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": API_TIMEOUT},
        )
        
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "data": None,
                "error": f"AI 回傳非 JSON 格式: {e}",
                "tokens": {},
                "model": model_name,
                "raw_response": response.text[:1000],
            }
        
        tokens = {}
        if hasattr(response, "usage_metadata"):
            _i = response.usage_metadata.prompt_token_count or 0
            _o = response.usage_metadata.candidates_token_count or 0
            _t = response.usage_metadata.total_token_count or 0
            tokens = {
                "input": _i,
                "output": _o,
                "total": _t,
                # gemini-3.x pro 是 thinking 模型:回答前會先推理,
                # 而推理量隨 prompt 複雜度暴增,是生成變慢的主因。
                # total 減去 input/output 就是思考 token。
                "thinking": max(0, _t - _i - _o),
            }
        
        return {
            "success": True,
            "data": data,
            "error": None,
            "tokens": tokens,
            "model": model_name,
            "raw_response": None,
        }
    
    except Exception as e:
        if any(k in str(e).lower() for k in ("timeout", "deadline", "504")):
            return {
                "success": False, "data": None,
                "error": (f"AI 逾時({API_TIMEOUT} 秒未回應)——"
                          f"可能是服務端壅塞或該模型暫時不可用。"
                          f"請稍後再試,或改用其他模型。"),
                "tokens": {}, "model": model_name, "raw_response": None,
            }
        return {
            "success": False,
            "data": None,
            "error": str(e),
            "tokens": {},
            "model": model_name,
            "raw_response": None,
        }


# 別名(向後相容)
run_independent_observation = run_observation


# ============================================================
# 進階:論點對照
# ============================================================
def build_thesis_comparison_prompt(ai_observation: dict, user_thesis: dict) -> str:
    """組裝論點對照 prompt"""
    qualitative = ai_observation.get("qualitative_summary", "")
    overall = ai_observation.get("overall_judgment", {})
    stance = overall.get("stance", "")
    stance_label = STANCE_LABELS.get(stance, stance)
    
    ai_views_text = f"## AI 整體立場\n{stance_label} (信心度 {overall.get('confidence', '-')}/10)\n\n"
    ai_views_text += f"核心理由: {overall.get('core_reasoning', '')}\n\n"
    ai_views_text += "## AI 對當前處境的觀察\n"
    for sit in ai_observation.get("current_situation", []):
        ai_views_text += f"- {sit}\n"
    ai_views_text += "\n## AI 看到的情境\n"
    for sc in ai_observation.get("scenario_analysis", []):
        ai_views_text += f"- {sc.get('name', '')} ({sc.get('probability', 0)}%)\n"
    
    prompt = f"""# 任務:角度差異對照(不評對錯)

## AI 獨立觀察的核心

### 質化總結
{qualitative}

{ai_views_text}

## 使用者的論點

### 核心論點
{user_thesis.get('thesis', '(未填寫)')}

### 護城河
{user_thesis.get('moat', '(未填寫)')}

### 主要風險
{user_thesis.get('risks', '(未填寫)')}

### 個人策略
{user_thesis.get('strategy_note', '(未填寫)')}

# 輸出格式 (純 JSON)

{{
  "ai_mentioned_user_didnt": ["..."],
  "user_mentioned_ai_didnt": ["..."],
  "both_aligned_on": ["..."],
  "user_strategy_relevance": ["..."],
  "summary": "1-2 句總結"
}}

# 規則

1. 不評對錯
2. 不做新分析
3. 角度差異要描述性
4. 每個列表 3-5 個就夠
"""
    return prompt


def run_thesis_comparison(ai_observation: dict, user_thesis: dict, model_name: str = None) -> dict:
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    prompt = build_thesis_comparison_prompt(ai_observation, user_thesis)
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": API_TIMEOUT},
        )
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            return {"success": False, "data": None, "error": f"非 JSON: {e}", "tokens": {}, "model": model_name, "raw_response": response.text[:1000]}
        
        tokens = {}
        if hasattr(response, "usage_metadata"):
            _i = response.usage_metadata.prompt_token_count or 0
            _o = response.usage_metadata.candidates_token_count or 0
            _t = response.usage_metadata.total_token_count or 0
            tokens = {
                "input": _i,
                "output": _o,
                "total": _t,
                # gemini-3.x pro 是 thinking 模型:回答前會先推理,
                # 而推理量隨 prompt 複雜度暴增,是生成變慢的主因。
                # total 減去 input/output 就是思考 token。
                "thinking": max(0, _t - _i - _o),
            }
        return {"success": True, "data": data, "error": None, "tokens": tokens, "model": model_name, "raw_response": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e), "tokens": {}, "model": model_name, "raw_response": None}


# ============================================================
# DB 存取
# ============================================================
def save_observation(symbol: str, primary_horizon: str, result: dict, thesis_snapshot: Optional[dict] = None, user_id: str = None) -> Optional[str]:
    if not result.get("success"):
        return None
    sb = _get_supabase()
    data = result["data"]
    record = {
        "symbol": symbol,
        "user_id": user_id,
        "thesis_snapshot": (thesis_snapshot or {}).get("thesis"),
        "moat_snapshot": (thesis_snapshot or {}).get("moat"),
        "risks_snapshot": (thesis_snapshot or {}).get("risks"),
        "validated_points": data,
        "challenged_points": [],
        "refuted_points": [],
        "overall_assessment": data.get("qualitative_summary"),
        "recommendation": data.get("overall_judgment", {}).get("stance"),
        "model_used": result.get("model"),
        "input_tokens": result.get("tokens", {}).get("input"),
        "output_tokens": result.get("tokens", {}).get("output"),
        "total_tokens": result.get("tokens", {}).get("total"),
    }
    res = sb.table("thesis_reviews").insert(record).execute()
    if res.data:
        return res.data[0]["id"]
    return None


def load_observations(symbol: str, limit: int = 5, user_id: str = None):
    sb = _get_supabase()
    query = sb.table("thesis_reviews").select("*").eq("symbol", symbol)
    if user_id:
        query = query.eq("user_id", user_id)
    res = query.order("created_at", desc=True).limit(limit).execute()
    return res.data or []