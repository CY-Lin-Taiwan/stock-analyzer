"""
News Fetcher - 抓取個股相關新聞(白名單版)
===========================================
用途: 為 AI 觀察自動補充「市場前瞻性事件」context
      AI 會主動納入新聞做綜合判斷

設計原則:
  - 用 Google News RSS 的 site: 指令直接鎖定白名單
    (避免被 CMoney / 同學會 / 業配文 洗版)
  - 只抓「標題 + 來源 + 時間」,不抓內文
    (節省 token、避免幻覺)
  - 失敗 graceful: 抓不到 → 回空 list,不影響 AI 觀察
"""
import socket

import feedparser
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from typing import List, Dict, Optional


# 抓新聞的逾時(秒)。feedparser 底層用 urllib,預設 socket timeout 是 None,
# 也就是「無限等待」—— Google News 連不上時會永遠卡住,而呼叫端
# (AI 觀察)只會顯示一直在跑,查不出原因。
# 抓不到就回空 list,不影響分析 —— 新聞是加分項,不該卡死整個流程。
NEWS_TIMEOUT = 10


# === 白名單財經媒體網域 ===
TRUSTED_DOMAINS = [
    "cnyes.com",              # 鉅亨網
    "money.udn.com",          # 經濟日報
    "udn.com",                # 聯合新聞網
    "ctee.com.tw",            # 工商時報
    "cna.com.tw",             # 中央社
    "technews.tw",            # 科技新報
    "businessweekly.com.tw",  # 商業周刊
    "moneydj.com",            # MoneyDJ
    "wealth.com.tw",          # 財訊
    "bnext.com.tw",           # 數位時代
    "businesstoday.com.tw",   # 今周刊
    "mirrormedia.mg",         # 鏡週刊
]


def _format_age(published_dt: Optional[datetime]) -> str:
    """把發布時間轉成『N 天前 / N 小時前』方便閱讀"""
    if not published_dt:
        return "時間未知"
    
    now = datetime.now(timezone.utc)
    if published_dt.tzinfo is None:
        published_dt = published_dt.replace(tzinfo=timezone.utc)
    
    delta = now - published_dt
    days = delta.days
    hours = delta.seconds // 3600
    
    if days >= 1:
        return f"{days} 天前"
    elif hours >= 1:
        return f"{hours} 小時前"
    else:
        return "剛剛"


def _parse_published(entry) -> Optional[datetime]:
    """從 RSS entry 解析發布時間,失敗回 None"""
    try:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def _build_site_query(name: str, symbol: str = "") -> str:
    """組 Google News query 用 site: 鎖定白名單"""
    site_parts = [f"site:{domain}" for domain in TRUSTED_DOMAINS]
    site_filter = " OR ".join(site_parts)
    keyword = f"{name} {symbol}".strip()
    return f"{keyword} ({site_filter})"


def _extract_domain_from_link(link: str) -> str:
    """從連結解析來源網域(備用)"""
    try:
        parsed = urlparse(link)
        return parsed.netloc.replace('www.', '')
    except Exception:
        return ''


def fetch_news(symbol: str, name: str, days: int = 7, limit: int = 10) -> List[Dict]:
    """
    抓取個股最近新聞(白名單版,自動過濾雜訊)
    
    Args:
        symbol: 股票代號 (例如 "2603")
        name: 股票名稱 (例如 "長榮")
        days: 抓最近幾天 (預設 7 天)
        limit: 最多保留幾則 (預設 10 則)
    
    Returns:
        List[Dict]: 新聞列表,每筆包含:
          - title: 標題
          - link: 原文連結
          - source: 顯示用的來源名稱
          - published_dt: datetime 物件
          - age: 「N 小時前」這種文字
        
        失敗 / 抓不到 → 回空 list []
    """
    if not symbol or not name:
        return []
    
    query = _build_site_query(name, symbol)
    encoded_query = quote(query)
    
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={encoded_query}"
        f"&hl=zh-TW"
        f"&gl=TW"
        f"&ceid=TW:zh-Hant"
    )
    
    try:
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            return []
        
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        news_list = []
        
        for entry in feed.entries[:limit * 2]:
            published_dt = _parse_published(entry)
            
            if published_dt and published_dt.timestamp() < cutoff:
                continue
            
            title = entry.get('title', '').strip()
            source_display = ''
            if ' - ' in title:
                parts = title.rsplit(' - ', 1)
                title = parts[0].strip()
                source_display = parts[1].strip() if len(parts) > 1 else ''
            
            link = entry.get('link', '')
            
            news_list.append({
                'title': title,
                'link': link,
                'source': source_display or _extract_domain_from_link(link),
                'published_dt': published_dt,
                'age': _format_age(published_dt),
            })
            
            if len(news_list) >= limit:
                break
        
        return news_list
        
    except Exception as e:
        print(f"[news_fetcher] 抓取失敗 {symbol} {name}: {e}")
        return []


# 產業關鍵字正規化。
#
# ⚠️ 產業欄位是使用者手動填的自由文字,無法預先窮舉。
# 所以策略是:**優先直接用使用者填的字**,只有在明顯查不到的情況
# (過長的複合名稱,例如「電子IC導線架」「被動元件MLCC」)才做映射。
# 下表只是常見情況的補救,不是完整清單 —— 真正的解法是把「實際用了
# 哪些關鍵字」顯示給使用者看,讓他自己調整產業欄位的寫法。
INDUSTRY_KEYWORDS = {
    "航運": "航運", "航運業": "航運", "貨櫃": "航運",
    "半導體": "半導體", "半導體業": "半導體",
    "電子零組件": "電子零組件", "電子零組件業": "電子零組件",
    "被動元件MLCC": "被動元件", "被動元件": "被動元件",
    "功率元件": "功率半導體", "電子IC導線架": "半導體封測",
    "PCB": "PCB", "印刷電路板": "PCB", "載板": "ABF載板",
    "金融": "金融股", "金融保險": "金融股", "銀行": "金融股",
    "光電": "光電", "電腦及週邊設備": "電腦周邊",
    "通信網路": "網通", "其他電子": "電子股",
    "鋼鐵": "鋼鐵", "塑膠": "塑化", "紡織": "紡織",
    "生技醫療": "生技", "汽車": "汽車零組件",
}


MAX_SEARCHABLE_LEN = 5     # 超過這個長度的產業名,新聞標題裡幾乎不會出現


def normalize_industries(industries: List[str], limit: int = 3) -> List[str]:
    """
    把產業欄位轉成新聞查得到的關鍵字。

    順序:
      1. 名稱夠短(≤5 字)→ 直接用,不動使用者填的字
      2. 太長 → 先查對照表,再試子字串比對
      3. 都不行 → 仍然用原字(至少誠實反映使用者的輸入)
    """
    out = []
    for ind in (industries or []):
        ind = (ind or "").strip()
        if not ind or ind == "-":
            continue

        if len(ind) <= MAX_SEARCHABLE_LEN:
            kw = ind                       # 使用者填得夠精簡就直接用
        else:
            kw = INDUSTRY_KEYWORDS.get(ind)
            if not kw:
                for k, v in INDUSTRY_KEYWORDS.items():
                    if k in ind:
                        kw = v
                        break
            kw = kw or ind                 # 仍找不到就用原字

        if kw not in out:
            out.append(kw)
        if len(out) >= limit:
            break
    return out


def fetch_industry_news(industries: List[str], days: int = 3,
                        limit: int = 10) -> List[Dict]:
    """
    抓「產業層級」的新聞 —— 給組合分析用。

    組合分析要看的是環境,不是個股(個股新聞在個股頁已經有了)。
    產業新聞才回答「為什麼這幾檔一起動」,那是組合層級該問的問題。

    Args:
        industries: 產業名稱(建議傳權重最大的幾個)
    """
    kws = normalize_industries(industries, limit=3)
    if not kws:
        print("[news_fetcher] 產業新聞:沒有可用的產業關鍵字")
        return []
    site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS)
    query = f"({' OR '.join(kws)}) ({site_filter})"
    out = _search_news(query, days=days, limit=limit,
                       label=f"產業新聞({'、'.join(kws)})")
    # 把實際用的關鍵字掛在結果上,讓 UI 顯示給使用者 ——
    # 產業欄位是他填的,他才知道該怎麼改才查得到
    for item in out:
        item["_keywords"] = kws
    if not out:
        return [{"_keywords": kws, "_empty": True}]
    return out


def _search_news(query: str, days: int, limit: int, label: str) -> List[Dict]:
    """共用的 RSS 查詢與解析"""
    rss_url = (f"https://news.google.com/rss/search?q={quote(query)}"
               f"&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    try:
        _old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(NEWS_TIMEOUT)
        try:
            feed = feedparser.parse(rss_url)
        finally:
            socket.setdefaulttimeout(_old)
        if not feed.entries:
            print(f"[news_fetcher] {label}:0 則")
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        out = []
        for e in feed.entries[:limit * 3]:
            dt = _parse_published(e)
            if dt and dt.timestamp() < cutoff:
                continue
            title = e.get("title", "").strip()
            src = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title, src = parts[0].strip(), parts[1].strip()
            out.append({"title": title, "source": src or "",
                        "age": _format_age(dt), "link": e.get("link", "")})
            if len(out) >= limit:
                break
        print(f"[news_fetcher] {label}:{len(out)} 則")
        return out
    except Exception as e:
        print(f"[news_fetcher] {label} 失敗: {e}")
        return []


def fetch_portfolio_news(names: List[str], days: int = 3,
                         limit: int = 10) -> List[Dict]:
    """
    抓「跟持股相關」的新聞 —— 給組合分析用。

    先前用「台股 加權指數 大盤」這種泛用關鍵字查,結果常常抓不到 ——
    白名單媒體的標題不見得含那些字。改用實際持股名稱(取權重最大的幾檔)
    以 OR 串成單一查詢,一次呼叫抓完,而且結果直接跟組合相關。

    Args:
        names: 持股名稱(建議傳權重最大的 3~5 檔)
    """
    names = [n for n in (names or []) if n][:5]
    if not names:
        return []
    # 連字號在 Google 搜尋語法裡是「排除」運算子 ——
    # 「三集瑞-KY」可能被解析成「三集瑞 且 不含 KY」,整個查詢就歪了。
    # 統一去掉,用主名稱查即可。
    names = [n.split("-")[0].strip() for n in names]
    site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS)
    keyword = " OR ".join(names)          # 不加引號,降低查詢複雜度
    query = f"({keyword}) ({site_filter})"
    rss_url = (f"https://news.google.com/rss/search?q={quote(query)}"
               f"&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    try:
        _old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(NEWS_TIMEOUT)
        try:
            feed = feedparser.parse(rss_url)
        finally:
            socket.setdefaulttimeout(_old)
        if not feed.entries:
            # 退路:合併查詢失敗時,改用「個股查詢」逐檔抓再合併。
            # 個股查詢的字串簡單得多,實測穩定;成本是多幾次呼叫,
            # 所以只對權重最大的兩檔做。
            print(f"[news_fetcher] 組合新聞:合併查詢無結果,改逐檔抓")
            merged, seen = [], set()
            for nm in names[:2]:
                for item in fetch_news(nm, nm, days=days, limit=5):
                    if item["title"] in seen:
                        continue
                    seen.add(item["title"])
                    merged.append(item)
            print(f"[news_fetcher] 組合新聞(逐檔):{len(merged)} 則")
            return merged[:limit]
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        out = []
        for e in feed.entries[:limit * 3]:
            dt = _parse_published(e)
            if dt and dt.timestamp() < cutoff:
                continue
            title = e.get("title", "").strip()
            src = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title, src = parts[0].strip(), parts[1].strip()
            out.append({"title": title, "source": src or "",
                        "age": _format_age(dt), "link": e.get("link", "")})
            if len(out) >= limit:
                break
        print(f"[news_fetcher] 組合新聞:{len(out)} 則")
        return out
    except Exception as e:
        print(f"[news_fetcher] 組合新聞抓取失敗: {e}")
        return []


def fetch_market_news(days: int = 3, limit: int = 8) -> List[Dict]:
    """
    抓「大盤層級」的新聞 —— 給組合分析用。

    為什麼需要:工具裡所有數值資料都是已發生的(價格、籌碼、財報)。
    新聞是唯一帶有「市場預期」與「敘事」的來源,而先前只用在個股觀察,
    組合層級完全沒有 —— 於是 AI 只能就已發生的數字重組,
    產不出任何前瞻性的內容。

    刻意抓比較短的天數(3 天):組合分析要的是「現在的氛圍」,
    不是一週前的新聞。
    """
    site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DOMAINS)
    query = f"台股 加權指數 大盤 ({site_filter})"
    rss_url = (f"https://news.google.com/rss/search?q={quote(query)}"
               f"&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    try:
        _old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(NEWS_TIMEOUT)
        try:
            feed = feedparser.parse(rss_url)
        finally:
            socket.setdefaulttimeout(_old)
        if not feed.entries:
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        out = []
        for e in feed.entries[:limit * 2]:
            dt = _parse_published(e)
            if dt and dt.timestamp() < cutoff:
                continue
            title = e.get("title", "").strip()
            src = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title, src = parts[0].strip(), parts[1].strip()
            out.append({"title": title, "source": src or "",
                        "age": _format_age(dt), "link": e.get("link", "")})
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        print(f"[news_fetcher] 大盤新聞抓取失敗: {e}")
        return []


def format_news_for_ai(news_list: List[Dict]) -> str:
    """
    把新聞列表格式化成 AI prompt 用的文字
    """
    if not news_list:
        return ""
    
    lines = [f"過去 7 天主流財經媒體報導 (共 {len(news_list)} 則):", ""]
    for i, n in enumerate(news_list, 1):
        source = n.get('source', '未知來源')
        age = n.get('age', '時間未知')
        title = n.get('title', '')
        lines.append(f"{i}. [{source} · {age}] {title}")
    
    return "\n".join(lines)


# 測試用
if __name__ == "__main__":
    print("=" * 70)
    print("測試: 抓長榮 2603 最近 7 天新聞 (site: 白名單版)")
    print("=" * 70)
    
    results = fetch_news("2603", "長榮", days=7, limit=10)
    
    if not results:
        print("\n❌ 沒抓到白名單新聞")
        print("可能原因:")
        print("  1. 該股最近主流財經媒體沒報導")
        print("  2. Google News 暫時連不上")
        print("  3. 白名單太窄(可考慮加更多網域)")
    else:
        for i, n in enumerate(results, 1):
            print(f"\n{i}. [{n['source']}] {n['title']}")
            print(f"   {n['age']}")
        
        print("\n" + "=" * 70)
        print(f"✅ 共抓到 {len(results)} 則新聞")
        
        print("\n" + "=" * 70)
        print("AI 用格式預覽:")
        print("=" * 70)
        print(format_news_for_ai(results))