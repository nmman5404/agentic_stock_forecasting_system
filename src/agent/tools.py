import feedparser
import urllib.parse
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from utils.logger import get_logger

logger = get_logger("AgentTools")

def _get_queries_for_ticker(ticker: str) -> list[str]:
    """Map mã cổ phiếu với các truy vấn liên quan để tối đa hóa độ phủ tin tức."""
    query_map = {
        "VIC": ['"VIC" "Vingroup"', '"VinFast" "Vingroup"', '"cổ phiếu VIC"'],
        "VHM": ['"VHM" "Vinhomes"', '"Vinhomes" "cổ phiếu"'],
        "VRE": ['"VRE" "Vincom Retail"', '"Vincom Retail" "cổ phiếu"'],
        "VPL": ['"VPL" "Vinpearl"', '"Vinpearl" "Vingroup"'],
    }
    return query_map.get(ticker, [f'"{ticker}" "cổ phiếu"', f'"{ticker}" "chứng khoán"'])

def tool_search_google_news(ticker: str, max_items: int = 5, lookback_days: int = 7) -> str:
    """Tìm kiếm tin tức trên Google News RSS, có lọc trùng lặp và giới hạn thời gian."""
    logger.info("Calling Google News RSS Tool | ticker=%s", ticker)
    queries = _get_queries_for_ticker(ticker)
    
    all_items = []
    seen_titles = set()
    
    # Tính thời điểm cắt đứt (cutoff) để không lấy tin quá cũ
    cutoff_date = datetime.now().astimezone() - timedelta(days=lookback_days)

    for q in queries:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(q)}&hl=vi&gl=VN&ceid=VN:vi"
        try:
            feed = feedparser.parse(url)
            for entry in getattr(feed, "entries", []):
                title = entry.get('title', '').strip()
                if not title or title in seen_titles:
                    continue
                
                # Xử lý thời gian và lọc tin cũ
                pub_str = entry.get('published', entry.get('updated', ''))
                try:
                    pub_dt = parsedate_to_datetime(pub_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.astimezone()
                    if pub_dt < cutoff_date:
                        continue  # Bỏ qua tin cũ hơn 7 ngày
                except Exception:
                    pass # Nếu không parse được ngày thì vẫn giữ lại cho an toàn
                
                seen_titles.add(title)
                all_items.append(f"- [{pub_str}] {title}")
                
        except Exception as exc:
            logger.warning("Google News Query failed | query=%s | error=%s", q, exc)

    if not all_items:
        logger.info("Google News RSS Tool | No recent news found for %s", ticker)
        return "Không có tin tức nào mới trên thị trường trong 7 ngày qua."
    
    logger.info("Google News RSS Tool | Found %d unique recent items for %s", len(all_items), ticker)
    
    # Trả về tối đa max_items bài báo mới nhất
    return "\n".join(all_items[:max_items])