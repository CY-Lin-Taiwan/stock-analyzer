"""
持股島 — 把「組合現況」畫成一座等角島嶼
==========================================
畫面語言:
  天氣   = 大盤 0050 的中期狀態(get_regime)
  大樓   = 個股;樓高與佔地 = 市值權重,顏色 = combine_state() 的合成狀態
  特效   = 只看短期(幾天到幾週),且跟大盤對照:
           冒煙 = 跌破三線且均線下彎 1~4 天,或大盤也一起走弱
           起火 = 跌破三線且均線下彎持續 5 天以上,而大盤沒有一起走弱
           金光 = 個股上升趨勢,而大盤不是
           配息等長期體質問題不上島,放在個股分頁的提醒
  街區   = 報酬相關性超過門檻的持股歸同一區(與產業無關)
  協作線 = 相關性 ≥ 0.3 的配對
  洋流   = 外資 + 投信近 5 日買賣超

設計原則:
  - 不看使用者損益決定任何視覺(套牢不會讓樓起火、也不會讓樓變矮)
  - 只用 metrics.py 已經算好的結果,判斷邏輯不搬進 JS
  - 畫面本體是 island_template.html(純前端,可單獨開啟看範例資料),
    這支只負責把真實資料組成 JSON 注入
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit.components.v1 as components

TEMPLATE = Path(__file__).with_name("island_template.html")
PLACEHOLDER = "/*__ISLAND_DATA__*/null"


def _num(v, nd=1):
    """NaN / None → None,其餘四捨五入"""
    try:
        f = float(v)
        return None if f != f else round(f, nd)
    except (TypeError, ValueError):
        return None


def build_payload(
    rows_state: List[Dict],
    extras: Dict[str, Dict],
    correlation: Optional[Dict],
    market_ctx: Optional[Dict],
    log: List[Tuple[str, str]],
    theme: Optional[str] = None,
    asof: Optional[str] = None,
    market_breakdown: bool = False,
) -> Dict:
    """
    Args:
        rows_state:  _portfolio_states() 的 rows,且已併入 meta(name/weight/industry/pnl_pct)
        extras:      _island_extras() 的逐檔資料(f5/t5/div_warn)
        correlation: metrics.analyze_correlation() 的輸出
        market_ctx:  load_market_context() 的輸出
        log:         [(YYYY-MM-DD, 文字), ...]
    """
    holdings = []
    for r in rows_state:
        ex = extras.get(r["symbol"], {})
        holdings.append({
            "sym": r["symbol"],
            "name": r.get("name") or r["symbol"],
            "ind": r.get("industry") or "-",
            "w": _num(r.get("weight"), 1) or 0,
            "regime": r.get("regime", "—"),
            "zone": r.get("zone", "—"),
            "state": r.get("state", "—"),
            "meaning": r.get("state_meaning", ""),
            "pnl": _num(r.get("pnl_pct"), 1) or 0,
            "div": _num(r.get("div_share"), 0),
            "f5": int(ex.get("f5") or 0),
            "t5": int(ex.get("t5") or 0),
            "kind": etf_kind(r["symbol"], r.get("name") or ""),
            # 長期配息狀況:只在個股分頁顯示,不影響特效
            "divNote": "；".join(x for x in (ex.get("div_warn"), ex.get("erosion")) if x) or None,
            # 破線狀態:{"days": 連續天數, "text": 說明} 或 None
            "brk": ex.get("breakdown"),
        })
    holdings.sort(key=lambda h: -h["w"])

    corr = {}
    for p in (correlation or {}).get("pairs", []):
        corr[f"{p['a']}|{p['b']}"] = p["corr"]

    return {
        "live": True,
        "asof": str(asof)[:10] if asof else None,
        "theme": theme if theme in ("light", "dark") else None,
        "market": {"regime": (market_ctx or {}).get("regime_label") or "資料不足",
                   "breakdown": bool(market_breakdown)},
        "fireDays": BREAK_FIRE_DAYS,
        "holdings": holdings,
        "corr": corr,
        "log": [[d[5:].replace("-", "/"), t] for d, t in log],
    }


# ===== 破線 =====
BREAK_FIRE_DAYS = 5         # 破線持續幾個交易日才從冒煙升級成起火
MA20_DROP_PCT = 2.0         # 20 日線 10 天內至少下跌這個 %,才算「明顯下彎」
MA60_DROP_PCT = 0.5         # 60 日線 10 天內至少下跌這個 %
SLOPE_DAYS = 10


def breakdown_status(close) -> Optional[Dict]:
    """
    跌破三線且均線明顯下彎,並算出已經持續幾個交易日。

    條件(每一天都要成立):
      - 收盤低於 5 / 20 / 60 日均線
      - 20 日線比 10 天前低 2% 以上、60 日線比 10 天前低 0.5% 以上
    只要有一天不成立(例如站回任一條均線),天數就歸零。

    為什麼「下彎」要有幅度:只寫「比幾天前低」的話,低 0.01% 也算,
    震盪盤裡隨便一次回檔就成立 —— 模擬顯示沒有任何趨勢的股價
    也有約兩成的日子會符合,火就失去意義。

    Args:
        close: 還原收盤價序列(pandas Series,舊到新)
    Returns: {"days": int, "text": str},目前未破線回 None
    """
    try:
        if close is None or len(close) < 60 + SLOPE_DAYS + 1:
            return None
        c = close.astype(float).reset_index(drop=True)
        ma5, ma20, ma60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
        cond = ((c < ma5) & (c < ma20) & (c < ma60)
                & ((ma20 / ma20.shift(SLOPE_DAYS) - 1) * 100 <= -MA20_DROP_PCT)
                & ((ma60 / ma60.shift(SLOPE_DAYS) - 1) * 100 <= -MA60_DROP_PCT))
        days = 0
        for v in reversed(cond.tolist()):
            if v:
                days += 1
            else:
                break
        if not days:
            return None
        return {"days": days,
                "text": (f"收盤 {c.iloc[-1]:,.2f} 跌破 5 日線 {ma5.iloc[-1]:,.2f}、"
                         f"20 日線 {ma20.iloc[-1]:,.2f}、60 日線 {ma60.iloc[-1]:,.2f},"
                         f"均線明顯下彎")}
    except Exception:
        return None


# ===== ETF 類型 =====
# 不同類型「出問題」的意思不同,不能全部拿 0050 來比:
#   market   市值型(0050 等):它本身就是大盤,不會「比大盤差」→ 最多冒煙
#   bond     債券(代號 B 結尾):跟股市無關 → 不跟大盤對照,只看自己破線
#   leverage 槓桿 / 反向(L / R 結尾):本來就放大或反著走 → 不套用特效
#   active   主動式(A 結尾)、high_div 高股息、theme 主題 → 照一般規則
#   stock    個股
# 名稱判斷難免漏網,可以在這裡手動指定:{"代號": "類型"}
ETF_KIND_OVERRIDE: Dict[str, str] = {}
MARKET_ETFS = {"0050", "006208", "006203", "006204", "00692", "00850"}


def etf_kind(symbol: str, name: str = "") -> str:
    sym = (symbol or "").upper()
    if sym in ETF_KIND_OVERRIDE:
        return ETF_KIND_OVERRIDE[sym]
    if not sym.startswith("00"):
        return "stock"
    if sym.endswith("B"):
        return "bond"
    if sym.endswith("L") or sym.endswith("R"):
        return "leverage"
    if sym.endswith("A"):
        return "active"
    if sym in MARKET_ETFS or any(k in name for k in ("台灣50", "臺灣50")):
        return "market"
    if any(k in name for k in ("高股息", "高息", "股息")):
        return "high_div"
    return "theme"


def _effect(h: Dict, mkt: str, mkt_breakdown: bool = False) -> Optional[str]:
    """與模板裡 effectOf() 相同的規則"""
    kind = h.get("kind", "stock")
    if kind == "leverage":
        return None
    brk = h.get("brk")
    mkt_weak = mkt_breakdown or mkt == "下降趨勢"
    if brk:
        if kind == "market":
            return "smoke"
        if kind != "bond" and mkt_weak:
            return "smoke"
        return "fire" if brk["days"] >= BREAK_FIRE_DAYS else "smoke"
    if kind in ("stock", "high_div", "theme", "active") \
            and h["regime"] == "上升趨勢" and mkt != "上升趨勢":
        return "gold"
    return None


def _districts(payload: Dict, thr: float) -> List[List[Dict]]:
    """依相關性門檻分群(union-find),與模板裡 clusters() 相同"""
    parent = {h["sym"]: h["sym"] for h in payload["holdings"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in payload["corr"].items():
        a, b = k.split("|")
        if r >= thr and a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: Dict[str, List[Dict]] = {}
    for h in payload["holdings"]:
        groups.setdefault(find(h["sym"]), []).append(h)
    return sorted(groups.values(), key=lambda g: -sum(h["w"] for h in g))


def summarize(payload: Dict, thr: float = 0.40) -> Tuple[str, str]:
    """
    標題列的兩句話:街區結論 + 特效清單。
    放在 iframe 外面當區塊說明,所以在 Python 端算(以預設門檻 0.40 為準);
    使用者在島上拉門檻時,滑桿旁會即時顯示新的街區數。
    """
    hs = payload["holdings"]
    if not hs:
        return "", ""
    n, n_ind = len(hs), len({h["ind"] for h in hs})
    cl = _districts(payload, thr)
    big = round(sum(h["w"] for h in cl[0]), 1)
    t = f"{thr:.2f}"
    if len(cl) == 1:
        head = f"{n} 檔、{n_ind} 個產業，在相關性 {t} 的門檻下**全部是同一個街區**，報酬幾乎一起動。"
    elif len(cl) == n:
        head = f"{n} 檔各自成一區，在相關性 {t} 的門檻下**沒有明顯一起動的持股**。最大一棟佔 {big}%。"
    elif len(cl) < n_ind:
        head = f"{n} 檔、{n_ind} 個產業，在相關性 {t} 的門檻下其實只有 **{len(cl)} 個街區**。最大的街區佔 {big}%。"
    else:
        head = (f"{n} 檔分成 **{len(cl)} 個街區**（相關性門檻 {t}），街區數和產業數相當，"
                f"產業分散大致反映在報酬上。最大的街區佔 {big}%。")

    mkt = payload["market"]["regime"]
    names = {"fire": "🔥 起火", "smoke": "💨 冒煙", "gold": "✨ 金光"}
    fx = {k: [] for k in names}
    for h in hs:
        k = _effect(h, mkt, payload["market"].get("breakdown", False))
        if k:
            fx[k].append(h["name"])
    fx_line = "　".join(f"{names[k]} {'、'.join(v)}" for k, v in fx.items() if v)
    return head, fx_line or "島上目前沒有特效，持股都跟著大盤走"


def render(payload: Dict, height: int = 720) -> None:
    """注入資料並嵌入頁面。iframe 高度固定,手機上會在框內捲動。"""
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, data, 1)
    components.html(html, height=height, scrolling=True)
