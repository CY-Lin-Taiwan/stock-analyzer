"""
Price Adjust - 還原股價(除權息 / 減資)
==========================================
用途: 把 FinMind 的「原始收盤價」還原成連續序列,供技術指標使用。

為什麼需要:
  資料庫存的是原始收盤價,除權息當天會跳空。以長榮 2603 為例,
  2023-06-30 配息 70 元(參考價 155 → 85),當天實際漲停填息,
  但 KD 的 9 日高點仍記得除息前的 155,導致 K 值被壓在 23.6,
  真實值應為 80.2 —— 訊號完全反向。布林帶寬同時假性擴張 20 天。

設計原則:
  - 向後還原 (back-adjust):事件日「之前」的價格乘上調整因子,
    **最新一天永遠維持真實價格不變**。
  - 調整因子取自交易所公告參考價,不自己算股利公式:
        r = reference_price / before_price
    (不可用 after_price —— 那是除息當天實際收盤,含當天市場波動)
  - 自我檢查用交易所規則:還原後除息日報酬必須落在 ±10% 漲跌幅內。
  - 失敗 graceful:對不上就不調整,並回報 reliable=False,
    讓下游把指標標成不可信,而不是給一個錯的數字。

不做的事:
  - 不寫 DB(還原價不落地,由 app 端 cache 即時算)
  - 不處理股票分割 / 現金增資(台股罕見,由覆蓋率檢查抓出來人工處理)
"""
from typing import List, Dict, Optional, Tuple

import pandas as pd


# === 參數 ===
PRICE_LIMIT = 0.10          # 台股單日漲跌幅限制 10%
LIMIT_MARGIN = 0.005        # 驗證留 0.5% 餘裕(參考價四捨五入)
ALIGN_TOLERANCE = 0.01      # before_price 與 DB 前收的容許誤差 1%
OHLC_COLS = ("open", "high", "low", "close")


def _to_date_str(v) -> str:
    """把 date / datetime / str 統一成 'YYYY-MM-DD'"""
    return str(v)[:10]


def build_adjusted_ohlc(
    df: pd.DataFrame,
    actions: List[Dict],
) -> Tuple[pd.DataFrame, Dict]:
    """
    建立還原後的 OHLC 序列。

    Args:
        df: 原始價格,需含 date(datetime64) + open/high/low/close,
            已按 date 升冪排序。
        actions: 公司行為列表,每筆需含:
            - action_date  : 'YYYY-MM-DD' (或 date/datetime)
            - kind         : '除權息' / '減資'
            - before_price : 事件前收盤價
            - reference_price : 交易所公告參考價

    Returns:
        (adjusted_df, report)

        adjusted_df: df 的副本,OHLC 已還原,額外多一欄 adj_factor。
                     失敗或無事件 → 回原始 df 副本(adj_factor 全為 1.0)。

        report: {
          "reliable": bool,          # False = 下游應把指標標成不可信
          "applied":  [...],         # 成功套用的事件(含 r、驗證後報酬)
          "skipped":  [...],         # 被跳過的事件 + 原因
          "warnings": [str, ...],
          "total_factor": float,     # 最舊一筆的累積因子
        }
    """
    report = {"reliable": True, "applied": [], "skipped": [],
              "warnings": [], "total_factor": 1.0}

    if df is None or df.empty:
        report["warnings"].append("價格序列為空")
        return (df.copy() if df is not None else pd.DataFrame()), report

    out = df.copy().reset_index(drop=True)
    missing = [c for c in OHLC_COLS if c not in out.columns]
    if missing:
        report["reliable"] = False
        report["warnings"].append(f"價格序列缺少欄位 {missing},無法還原")
        out["adj_factor"] = 1.0
        return out, report

    if not actions:
        out["adj_factor"] = 1.0
        return out, report

    # 日期 → 列號
    idx = {_to_date_str(d): i for i, d in enumerate(out["date"])}

    # -------- 1) 逐筆驗證,算出可用的調整因子 --------
    usable = []
    for a in actions:
        ds = _to_date_str(a.get("action_date"))
        kind = a.get("kind", "?")
        tag = f"{ds} {kind}"

        try:
            before = float(a["before_price"])
            ref = float(a["reference_price"])
        except (TypeError, ValueError, KeyError):
            report["skipped"].append({"event": tag, "reason": "價格欄位缺失或非數值"})
            continue

        if before <= 0 or ref <= 0:
            report["skipped"].append({"event": tag, "reason": "價格為 0 或負值"})
            continue

        i = idx.get(ds)
        if i is None:
            # 事件早於價格序列起點 → 序列裡本來就沒有這個缺口,無須調整
            report["skipped"].append({"event": tag, "reason": "不在價格序列範圍內(無缺口需修補)"})
            continue
        if i == 0:
            report["skipped"].append({"event": tag, "reason": "位於序列第一天,無前收可驗證"})
            continue

        # 一致性:FinMind 的事件前收盤價要對得上我們 DB 的前一交易日
        prev_close = float(out.iloc[i - 1]["close"])
        drift = abs(before - prev_close) / prev_close if prev_close else 1.0
        if drift > ALIGN_TOLERANCE:
            report["reliable"] = False
            report["skipped"].append({
                "event": tag,
                "reason": f"前收不符:公告 {before} vs DB {prev_close}(差 {drift*100:.2f}%)"
            })
            continue

        r = ref / before

        # 交易所規則驗證:還原後,事件日報酬必須在漲跌幅限制內。
        # 原始報酬 = close/prev_close - 1(含股利缺口)
        # 還原報酬 = close/ref - 1(缺口已剔除)
        adj_ret = float(out.iloc[i]["close"]) / ref - 1.0
        if abs(adj_ret) > PRICE_LIMIT + LIMIT_MARGIN:
            report["reliable"] = False
            report["skipped"].append({
                "event": tag,
                "reason": f"還原後報酬 {adj_ret*100:+.2f}% 超出 ±10% 漲跌幅,"
                          f"參考價可能有誤"
            })
            continue

        usable.append({"row": i, "r": r})
        report["applied"].append({
            "event": tag, "r": round(r, 6),
            "before": before, "reference": ref,
            "raw_return_pct": round((float(out.iloc[i]["close"]) / prev_close - 1) * 100, 2),
            "adj_return_pct": round(adj_ret * 100, 2),
        })

    if not usable:
        out["adj_factor"] = 1.0
        return out, report

    # -------- 2) 累積因子:第 i 筆 = 所有「事件日 > i」的 r 相乘 --------
    factor = pd.Series(1.0, index=out.index)
    for u in sorted(usable, key=lambda x: x["row"], reverse=True):
        factor.iloc[:u["row"]] *= u["r"]

    for c in OHLC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce") * factor
    out["adj_factor"] = factor
    report["total_factor"] = round(float(factor.iloc[0]), 6)

    # -------- 3) 收尾保證:最新一天不可被改動 --------
    if abs(float(out.iloc[-1]["close"]) - float(df.iloc[-1]["close"])) > 1e-6:
        report["reliable"] = False
        report["warnings"].append("最新收盤價被改動 —— 還原方向有誤,請勿使用")

    return out, report


def check_dividend_coverage(
    dividends: List[Dict],
    actions: List[Dict],
    df: pd.DataFrame,
) -> List[str]:
    """
    覆蓋率檢查:確認 dividends 表裡每筆股利都有對應的公司行為紀錄。

    為什麼用這個方向,而不是掃價格找大跌日:
      長榮 2603 三年內有 16 個單日跌幅 > 5% 的日子,實測全部是真實波動
      (關稅衝擊、日圓套利平倉等),沒有一筆是資料缺口。用價格門檻掃描
      會每年噴十幾個假警告,警告就失去意義。反過來從股利紀錄查,
      false positive 幾乎為零,而真正該抓的「有股利卻沒還原」一定抓到。

    Args:
        dividends: DB dividends 表,需含 ex_date / cash_dividend / stock_dividend
        actions:   corporate_actions 表
        df:        價格序列(用來界定檢查範圍)

    Returns:
        警告訊息列表,無問題回 []
    """
    if not dividends or df is None or df.empty:
        return []

    lo = _to_date_str(df.iloc[0]["date"])
    hi = _to_date_str(df.iloc[-1]["date"])
    action_years = {_to_date_str(a.get("action_date"))[:4] for a in (actions or [])}

    warnings = []
    for d in dividends:
        ex = _to_date_str(d.get("ex_date"))
        if not ex or ex < lo or ex > hi:
            continue  # 超出價格序列範圍,不影響還原

        cash = float(d.get("cash_dividend") or 0)
        stock = float(d.get("stock_dividend") or 0)
        if cash <= 0 and stock <= 0:
            continue  # 沒配息,不會有缺口

        # 用「年度」比對而非精確日期 —— dividends.ex_date 可能是公告日暫代,
        # 不可信;但同一年只會有一次除權息,年度比對足夠且不會誤判。
        if ex[:4] not in action_years:
            warnings.append(
                f"{ex[:4]} 年有股利紀錄(現金 {cash} / 股票 {stock})"
                f"但找不到對應的除權息參考價,該年度前後的技術指標可能失真"
            )

    return warnings


def summarize(report: Dict) -> str:
    """把 report 轉成一行人看的摘要(給 UI caption 用)"""
    n = len(report.get("applied", []))
    if not report.get("reliable", True):
        return "⚠️ 還原失敗,技術指標不可信"
    if n == 0:
        return "無除權息事件,使用原始價格"
    return f"已還原 {n} 次除權息(權息還原)"
