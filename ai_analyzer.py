"""
AI 分析模組 (Phase 4)

新增:
- holding_context 參數(持股事實 + 累積股息)
- AI 對「持股者」給不同視角的觀點
- 不影響 AI 獨立判斷(只是聚焦)
"""
import os
import json
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from supabase import create_client

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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
def calculate_accumulated_dividends(
    symbol: str,
    initial_position_date: str,
    current_shares: int,
) -> dict:
    """計算「初始建倉日之後」實際領到的股息"""
    sb = _get_supabase()
    
    res = sb.table("dividends") \
        .select("ex_date,cash_dividend,total_dividend") \
        .eq("symbol", symbol) \
        .gt("ex_date", initial_position_date) \
        .order("ex_date") \
        .execute()
    
    divs = res.data or []
    
    if not divs:
        return {
            "events_count": 0,
            "total_per_share": 0,
            "total_received": 0,
            "events": [],
        }
    
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
    
    # 計算累積已領股息
    div_info = calculate_accumulated_dividends(symbol, initial_date, current_shares)
    
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
    primary_label = HORIZON_LABELS.get(primary_horizon, "中期 (6-12 個月)")
    
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
    
    # === 處理技術指標 (夏普值 + 布林通道) ===
    metrics_text = "(無資料)"
    if metrics:
        sharpe = metrics.get("sharpe")
        bb = metrics.get("bollinger")
        
        lines = []
        if sharpe is not None:
            lines.append(f"- 個股夏普值 (1 年, RF=1.5%): {sharpe}")
        else:
            lines.append("- 個股夏普值: 資料不足")
        
        if bb:
            lines.append(
                f"- 布林通道: 現價 {bb.get('current')} / "
                f"上軌 {bb.get('upper')} / 中軌 {bb.get('middle')} / 下軌 {bb.get('lower')}"
            )
            pb = bb.get('percent_b')
            if pb is not None:
                lines.append(f"- 布林位階 %B = {pb}% (0%=下軌, 50%=中軌, 100%=上軌)")
            lines.append(f"- 位階描述: {bb.get('position')}")
        else:
            lines.append("- 布林通道: 資料不足")
        
        metrics_text = "\n".join(lines)
    
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

# 個股
- {symbol} {name} ({industry})

# 客觀資料

## 估值
- 現價: {valuation.get('close', 'N/A')}
- PE: {valuation.get('pe', 'N/A')}
- PB: {valuation.get('pb', 'N/A')}
- 殖利率: {valuation.get('dividend_yield', 'N/A')}%

## 技術指標 (近期)
{metrics_text}

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
  "qualitative_summary": "綜合判斷(2-3 段,真正的質化分析。展現分析師思考深度{qualitative_desc})",
  
  "ai_self_disclosed_limits": [
    "我這次分析的限制 1", "限制 2"
  ],
  
  "data_references": [
    "本次引用的關鍵數據點 1", "數據點 2"
  ],
  
  "news_references": [
    "(若有引用新聞) 引用第 N 則 [來源] 標題: 怎麼用它,例如：佐證/補充/反證了 X"
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
{holder_rule}
"""
    return prompt


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
            }
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
            tokens = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count,
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
            }
        )
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            return {"success": False, "data": None, "error": f"非 JSON: {e}", "tokens": {}, "model": model_name, "raw_response": response.text[:1000]}
        
        tokens = {}
        if hasattr(response, "usage_metadata"):
            tokens = {
                "input": response.usage_metadata.prompt_token_count,
                "output": response.usage_metadata.candidates_token_count,
                "total": response.usage_metadata.total_token_count,
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