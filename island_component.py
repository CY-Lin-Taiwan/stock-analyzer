"""
持股島 — 把「組合現況」畫成一座等角島嶼
==========================================
畫面語言:
  天氣   = 大盤 0050 的中期狀態(get_regime)
  大樓   = 個股;樓高與佔地 = 市值權重,顏色 = combine_state() 的合成狀態
  特效   = 個股相對大盤:起火(自己的問題)/ 冒煙(大環境拖累)/ 金光(自己在漲)
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
            "divWarn": ex.get("div_warn"),
            "erosion": ex.get("erosion"),
            "breakdown": ex.get("breakdown"),
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
        "holdings": holdings,
        "corr": corr,
        "log": [[d[5:].replace("-", "/"), t] for d, t in log],
    }


DIV_TYPE_PCT = 40   # 配息佔近 3 年報酬 ≥ 40% 視為配息型(與個股頁分級一致)
MA_SLOPE_DAYS = 5   # 均線「下彎」= 比 N 個交易日前低


def ma_breakdown(close) -> Optional[str]:
    """
    跌破三線且均線下彎:收盤低於 5/20/60 日均線,且 20 日、60 日均線都比 5 天前低。

    為什麼另外加這條:原本的「下降趨勢」看 20 日均線 10 天斜率 ≥ 2%,
    反應很慢 —— 股價急跌兩成,20 日線可能才剛開始彎,還判不出下降趨勢。
    這條是看盤的人最直覺的「破線」判斷,反應快得多。

    Args:
        close: 還原收盤價序列(pandas Series,舊到新)
    Returns: 說明文字,未觸發回 None
    """
    try:
        if close is None or len(close) < 60 + MA_SLOPE_DAYS:
            return None
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        c = float(close.iloc[-1])
        m5, m20, m60 = float(ma5.iloc[-1]), float(ma20.iloc[-1]), float(ma60.iloc[-1])
        below = c < m5 and c < m20 and c < m60
        down = (m20 < float(ma20.iloc[-1 - MA_SLOPE_DAYS])
                and m60 < float(ma60.iloc[-1 - MA_SLOPE_DAYS]))
        if below and down:
            return (f"收盤 {c:,.2f} 跌破 5 日線 {m5:,.2f}、20 日線 {m20:,.2f}、"
                    f"60 日線 {m60:,.2f},且均線下彎")
    except Exception:
        return None
    return None


def _effect(h: Dict, mkt: str, mkt_breakdown: bool = False) -> Optional[str]:
    """與模板裡 effectOf() 相同的規則"""
    div_type = h.get("div") is not None and h["div"] >= DIV_TYPE_PCT
    if div_type and (h.get("divWarn") or h.get("erosion")):
        return "fire"
    if h.get("breakdown"):
        return "smoke" if mkt_breakdown else "fire"
    if h["regime"] == "下降趨勢":
        return "smoke" if mkt == "下降趨勢" else "fire"
    if h["regime"] == "上升趨勢" and mkt != "上升趨勢":
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
