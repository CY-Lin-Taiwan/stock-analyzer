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
        event_text = f"\n## 近期重大事件 / 前瞻資訊 (極重要：這是尚未反映在上方客觀數據的最新事實)\n- {upcoming_events}\n"
    
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
   - 但「明確表態」≠「刻意找碴」
   - 你是「中立審計員」,不是「強制反對者」
   - 看到證據強就支持,看到風險大就警示,看到模糊就明說「資料不足」
   - 若財務數據與籌碼面同時極佳,該說 validated 就說 validated,然後轉向「未來什麼條件下這個榮景會結束?」(預演破局),而不是強行找碴
2. **不是要你下決策**:給「判斷傾向」,不是叫使用者買賣
3. **保持開放**:明確列出「什麼訊號會讓你改變想法」
4. **前三區用列點**(處境/情境/訊號),不要堆砌敘述。質化分析放最終「綜合判斷」
5. **情境必須給機率**:三個情境機率加總 = 100%
6. **訊號用重要性標籤**:🔴 關鍵 / 🟡 重要 / 🟢 次要
7. **使用最新資料** {freshness_note}
8. **動態異常偵測(取代靜態模板)**:
   - 看到極端數據組合(PE 極低 + 殖利率極高、PE 極高 + 毛利突破歷史、ROE 異常高、營收 YoY 異常等)時,**禁止直接套用「景氣循環頂部」「估值泡沫」「成長股泡沫」這種模板下結論**
   - 改問:「當前定價反映了市場什麼預期? 這個預期跟基本面數據是否相符?」
   - 並列討論兩種競爭性解釋:
     A. 市場錯殺/錯漲(預期過度悲觀或樂觀)
     B. 市場正在定價一個你還沒看到的長期變數(隱藏利空/利多)
   - 你的任務是評估「哪個解釋更能解釋當前數據異常」,而非直接套模板
   - 例子:若 PE 6 + 殖利率 7% + 毛利率近 4 季穩定 + 現金部位充足 → 不要說「景氣循環末端」,要問「市場為什麼給這個價? 是錯殺還是看到結構性風險?」

9. **歷史軌跡 = 舉證責任,不是預測公式**:
   - 歷史軌跡有警示價值,但不是「必然重演」的預測
   - 當數據出現歷史罕見組合時,先描述「歷史上類似情境通常如何發展」(承認歷史重力)
   - 然後問:「現在跟歷史最大的差別是什麼? 是否有足夠新變數能克服歷史均值回歸引力?」
   - 對「典範轉移」(AI 浪潮、地緣政治重構、商業模式變革等)保持開放
   - 但要求嚴格舉證:新變數要具體可驗證,不能只是「這次不一樣」的感覺
{"10. **持股者視角**:見下方持股區塊指引" if holding_context else ""}

# 個股
- {symbol} {name} ({industry})

# 客觀資料

## 估值
- 現價: {valuation.get('close', 'N/A')}
- PE: {valuation.get('pe', 'N/A')}
- PB: {valuation.get('pb', 'N/A')}
- 殖利率: {valuation.get('dividend_yield', 'N/A')}%

## 近 6 個月營收
{rev_text}

## 近 4 季財務
{fin_text}

## 近期籌碼面
{chips_text}
{holding_block}
{event_text}

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
    "what_would_change_my_mind": ["訊號 1", "訊號 2", "訊號 3"],
    "breaking_conditions": [
      {{
        "condition": "在什麼總經/產業/公司層級的變數下,目前的判斷會失效(具體可驗證,不要寫『若情況惡化』這種廢話)",
        "monitor_signal": "對應該追蹤的具體訊號(例如:單季毛利率跌破 X%、CSP 法說 capex 指引下修等)",
        "severity": "若條件成真的影響強度: 致命 / 重大 / 中度"
      }},
      {{
        "condition": "另一個破局條件",
        "monitor_signal": "...",
        "severity": "..."
      }}
    ]
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
  "qualitative_summary": "綜合判斷(2-3 段,真正的質化分析,必須展現以下對撞思考:(1)目前市場定價反映什麼共識? (2)你的觀察跟市場共識有何不同? (3)為什麼你被某邊說服或仍保留懷疑? 不要堆砌數據,要展現分析師的判斷力{qualitative_desc})",
  
  "ai_self_disclosed_limits": [
    "我這次分析的限制 1", "限制 2"
  ],
  
  "data_references": [
    "本次引用的關鍵數據點 1", "數據點 2"
  ]
}}

# 嚴格規則

1. 整體立場必填(不能空白逃避)
2. 三情境機率加總 = 100%
3. what_would_change_my_mind 至少 3 條
4. **breaking_conditions 至少 2 條**:具體可驗證的破局條件 + 對應追蹤訊號
5. current_situation 4-6 個列點
6. signals_to_monitor 至少 5 個,含 2 個 🔴
7. qualitative_summary 是質化分析容器,需展現「市場共識 vs 你的觀察」的對撞思考
8. 不寫「建議買進/賣出」
9. 不要編造數字
10. 不寫多空辯論結構
11. **不要套用通案模板**:看到極端值時並列討論「錯殺 vs 隱藏利空」兩種可能
12. **歷史軌跡用於舉證,不用於下結論**:有歷史相似情境時要說明,但允許「新變數可克服歷史引力」的可能
13. **事件與數據的整合詮釋(通用,所有個股適用)**:
    當 upcoming_events 包含結構性事件(擴廠期/CB 發行/閉鎖期/業務轉型/購併/法說 guidance 重大調整/重大訴訟/監管變化等),量化數據(營收/EPS/毛利)的詮釋必須結合事件 context,禁止直接套衰退或泡沫模板。
    
    處理框架(三步):
    
    Step 1 - 識別事件類型,推論「典型財務 pattern」:
      - 「擴廠期/Capex 高峰期」 → 折舊增加、營收暫時持平或下滑、毛利可能下降
      - 「CB 發行/閉鎖期」 → 籌碼異動、稀釋預期、特定人鎖碼可能控盤
      - 「業務轉型期」 → 舊業務萎縮、新業務未放量、營收 YoY 暫時難看
      - 「景氣循環下行段」 → 營收 YoY 衰退、毛利收縮、配息可能下修
      - 「景氣循環上行段」 → 營收 YoY 暴增但基期低、需警覺循環頂點
      - 「法說 guidance 重大調整」 → 預期校準,需重評
      - 「重大訴訟/監管」 → 區分「一次性影響」vs「結構性影響」
    
    Step 2 - 對照當前量化數據是「符合 pattern」還是「異常」:
      - 符合 pattern(例:擴廠期月營收下降是合理):
        → 不能套衰退模板
        → 該追蹤的是「pattern 演進進度」(投產時間? 新業務 ramp up 速度? 同業比較?)
      - 偏離 pattern(例:擴廠期但同業沒下滑、或下滑幅度遠超同業):
        → 才是真警訊,須警告使用者
    
    Step 3 - 給出「pattern-aware」的判讀:
      - 明確說明「我認知這是 X 期,典型 pattern 是 Y」
      - 分析「當前是否符合 Y 或偏離 Y」
      - 偏離才是異常,符合就是「進度中」
    
    禁止行為:
    - 看到月營收 -50% 直接套衰退模板,忽略 upcoming_events 提到的擴廠/轉型 context
    - 看到 CB 發行直接套「公司缺錢」的負面詮釋,忽略「籌資擴張」可能
    - 看到擴廠直接套「產能過剩風險」,忽略「配合需求成長」可能
    - 看到 PE 異常低就套「景氣循環頂部」,忽略可能的結構性轉變
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
    model_name: str = None,
) -> dict:
    """執行 AI 獨立觀察"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    prompt = build_observation_prompt(
        symbol, name, industry, primary_horizon,
        valuation, monthly_rev, quarterly_fin, chips, data_freshness,
        holding_context, upcoming_events
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
def save_observation(symbol: str, primary_horizon: str, result: dict, thesis_snapshot: Optional[dict] = None) -> Optional[str]:
    if not result.get("success"):
        return None
    sb = _get_supabase()
    data = result["data"]
    record = {
        "symbol": symbol,
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


def load_observations(symbol: str, limit: int = 5):
    sb = _get_supabase()
    res = sb.table("thesis_reviews").select("*").eq("symbol", symbol).order("created_at", desc=True).limit(limit).execute()
    return res.data or []