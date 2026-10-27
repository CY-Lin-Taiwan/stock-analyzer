"""
起火條件診斷 —— 為什麼這檔有 / 沒有起火?
=============================================
把持股島「破線」的每個條件,逐日攤開最近 15 個交易日:

  ① 收盤 < 5 日線
  ② 收盤 < 20 日線
  ③ 收盤 < 60 日線
  ④ 20 日線比 5 天前低
  ⑤ 60 日線比 5 天前低

五個都成立的那天才算「破線」,連續 5 天破線才起火。
快速通道:①②③ 成立,而且 20 天內從高點跌 20% 以上 → 直接起火。
0050 也在走弱時,以上都只算冒煙。

最後也會檢查金光:20 日線往上、收盤站在 20 日線上,而大盤不是上升趨勢。

⚠️ 只讀不寫,零 API 成本。

執行:
    python3 check_fire.py 2308
"""
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

import price_adjust
import island_component as ic

load_dotenv()
SEP = "=" * 92


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "2308"
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    rows = (sb.table("daily_prices").select("date,open,high,low,close")
            .eq("symbol", symbol).order("date").execute().data)
    if not rows:
        print(f"❌ 找不到 {symbol} 的股價")
        return
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    acts = (sb.table("corporate_actions").select("*")
            .eq("symbol", symbol).order("action_date").execute().data or [])
    adj, _ = price_adjust.build_adjusted_ohlc(df, acts)

    c = adj["close"].astype(float).reset_index(drop=True)
    ma5, ma20, ma60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    n = ic.SLOPE_DAYS
    high20 = c.rolling(ic.CRASH_DAYS, min_periods=1).max()
    drop = (c / high20 - 1) * 100
    s20 = (ma20 / ma20.shift(n) - 1) * 100
    s60 = (ma60 / ma60.shift(n) - 1) * 100

    ok = lambda b: "✅" if b else "❌"
    print(f"\n{SEP}\n起火條件診斷 — {symbol}(還原股價)\n{SEP}")
    print(f"{'日期':<11}{'收盤':>9}{'5MA':>9}{'20MA':>9}{'60MA':>9}"
          f"{f'20MA{n}日':>10}{f'60MA{n}日':>10}{'距20日高':>10}   ①②③④⑤  判斷")
    # 天數要從整段資料累計,不能只從表上第一天開始算
    streaks, streak = [], 0
    for i in range(len(c)):
        allok = (c[i] < ma5[i]) and (c[i] < ma20[i]) and (c[i] < ma60[i]) and s20[i] < 0 and s60[i] < 0
        streak = streak + 1 if allok else 0
        streaks.append(streak)
    for i in range(max(0, len(c) - 15), len(c)):
        cond = [c[i] < ma5[i], c[i] < ma20[i], c[i] < ma60[i],
                s20[i] < 0, s60[i] < 0]
        streak = streaks[i]
        crash = all(cond[:3]) and drop[i] <= -ic.CRASH_PCT
        if crash:
            verdict = f"急跌 {abs(drop[i]):.0f}% → 起火"
        elif streak:
            verdict = f"破線第 {streak} 天 → " + ("起火" if streak >= ic.BREAK_FIRE_DAYS else "冒煙")
        else:
            verdict = "—"
        print(f"{adj['date'].iloc[i]:%Y-%m-%d} {c[i]:>9.2f}{ma5[i]:>9.2f}{ma20[i]:>9.2f}"
              f"{ma60[i]:>9.2f}{s20[i]:>+9.2f}%{s60[i]:>+9.2f}%{drop[i]:>+9.1f}%   "
              + "".join(ok(x) for x in cond)
              + f"  {verdict}")

    i = len(c) - 1
    slope10 = (ma20[i] / ma20[i - 10] - 1) * 100 if i >= 10 else float("nan")
    print(f"\n【金光檢查】20 日線 10 天 {slope10:+.2f}%(≥ +2% 才算上升趨勢)、"
          f"收盤 {c[i]:,.2f} {'≥' if c[i] >= ma20[i] else '<'} 20 日線 {ma20[i]:,.2f}")
    if slope10 >= 2 and c[i] < ma20[i]:
        print("  → 均線還在往上,但股價已跌破 20 日線:舊規則會亮金光,新規則不會")
    print(f"\n{SEP}")
    print(f"連續 {ic.BREAK_FIRE_DAYS} 天破線,或跌破三線且 {ic.CRASH_DAYS} 天內跌 {ic.CRASH_PCT:.0f}% 以上 → 起火;"
          f"0050 也在走弱時只會冒煙")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
