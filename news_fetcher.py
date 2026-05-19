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
import feedparser
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from typing import List, Dict, Optional


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