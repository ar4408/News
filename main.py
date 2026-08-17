import os
import sys
import logging
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# RSS feeds
RSS_FEEDS = {
    "技術": "https://news.google.com/rss/search?q=AI+%E8%87%AA%E5%8B%95%E5%8C%96+%E6%8A%80%E8%A1%93&hl=ja&gl=JP&ceid=JP:ja",
    "経済": "https://news.google.com/rss/search?q=IT+%E7%B5%8C%E6%B8%88%E5%8B%95%E5%90%91&hl=ja&gl=JP&ceid=JP:ja",
}

# Notion API settings
NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_VERSION = "2022-06-28"

# GenAI model
GENAI_MODEL = "gemini-3.6-flash"


def get_env_vars():
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not NOTION_API_KEY:
        missing.append("NOTION_API_KEY")
    if not NOTION_DATABASE_ID:
        missing.append("NOTION_DATABASE_ID")

    if missing:
        logger.error("Missing environment variables: %s", ", ".join(missing))
        sys.exit(1)

    return GEMINI_API_KEY, NOTION_API_KEY, NOTION_DATABASE_ID


def fetch_existing_urls(notion_api_key, database_id):
    """Fetch all existing URLs from Notion database to prevent duplicates."""
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    
    existing_urls = set()
    has_more = True
    start_cursor = None
    
    try:
        while has_more:
            url = f"https://api.notion.com/v1/databases/{database_id}/query"
            data = {"page_size": 100}
            if start_cursor:
                data["start_cursor"] = start_cursor
            
            r = requests.post(url, headers=headers, json=data, timeout=30)
            if r.status_code != 200:
                logger.error("Failed to fetch existing URLs: %s %s", r.status_code, r.text)
                break
            
            response = r.json()
            for result in response.get("results", []):
                url_prop = result.get("properties", {}).get("URL", {})
                if url_prop.get("type") == "url":
                    page_url = url_prop.get("url")
                    if page_url:
                        existing_urls.add(page_url)
            
            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")
        
        logger.info("Found %d existing URLs in Notion", len(existing_urls))
    except Exception as e:
        logger.error("Error fetching existing URLs: %s", e)
    
    return existing_urls


def deduplicate_entries(entries):
    """Deduplicate entries by URL."""
    seen_urls = set()
    deduped = []
    
    for entry in entries:
        link = entry.get("link", "")
        if link and link not in seen_urls:
            seen_urls.add(link)
            deduped.append(entry)
        elif not link:
            # Entries without URL are kept as is
            deduped.append(entry)
    
    if len(entries) > len(deduped):
        logger.info("Deduplicated %d entries -> %d entries", len(entries), len(deduped))
    
    return deduped


def fetch_latest_entries(feed_url, limit=2):
    logger.info("Fetching RSS feed: %s", feed_url)
    d = feedparser.parse(feed_url)
    entries = []
    for e in d.entries[:limit]:
        title = e.get("title")
        link = e.get("link")
        published = e.get("published")
        entries.append({"title": title, "link": link, "published": published})
    logger.info("Found %d entries", len(entries))
    return entries


def init_genai_client(api_key):
    """Initialize GenAI client using the new `google-genai` package.

    Returns a `generate(prompt)` function or `None` on failure.
    """
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error("Failed to initialize genai client: %s", e)
        return None

    def generate(prompt):
        try:
            response = client.models.generate_content(model=GENAI_MODEL, contents=prompt)
            # Prefer `.text` if available; fall back to string form
            return getattr(response, "text", None) or str(response)
        except Exception:
            logger.exception("GenAI generation failed")
            return ""

    return generate


def generate_detailed_content(generate_func, title, url):
    """Generate detailed summary and background knowledge for the article."""
    prompt = (
        f"以下のニュース記事について、詳細な解説を日本語で作成してください。\n"
        f"記事タイトル: {title}\n"
        f"URL: {url}\n\n"
        "以下の4つの要素を分析して抽出してください。JSON形式で返してください。\n\n"
        "1. 【詳細要約】\n"
        "   - ニュースの背景や文脈\n"
        "   - 具体的な数値や事実\n"
        "   - 今後の影響や課題\n"
        "   - 200～400文字程度\n\n"
        "2. 【補足・背景知識・用語解説】\n"
        "   - 記事に出てくる専門用語の説明\n"
        "   - 業界の背景知識\n"
        "   - 初心者向けの補足説明\n"
        "   - 箇条書き形式（3～5項目）\n\n"
        "3. 【重要度】\n"
        "   - 記事の重要度を「★1」「★2」「★3」のいずれかで判定\n"
        "   - ★3: 重要性が極めて高い（業界・キャリア・経済全体への大きな影響）\n"
        "   - ★2: 中程度の重要性（注目すべき動き、トレンド、施策）\n"
        "   - ★1: 参考程度（軽微な更新、ニッチなトピック）\n\n"
        "4. 【関連テーマ】\n"
        "   - 最大3つまでのタグを選択\n"
        "   - 候補: 「生成AI」「基盤モデル」「マクロ経済」「IT業界」「キャリア」「スタートアップ」「ガバナンス」「セキュリティ」\n"
        "   - タグは正確にカテゴリ名を返してください\n\n"
        "必ずJSONのみを返してください（マークダウン記号なし）。以下の構造で返してください:\n"
        '{\n'
        '  "detailed_summary": "詳細要約テキスト",\n'
        '  "background_knowledge": ["項目1", "項目2", "項目3"],\n'
        '  "importance": "★2",\n'
        '  "related_themes": ["テーマ1", "テーマ2"]\n'
        '}\n'
    )
    logger.info("Generating detailed content for: %s", title)
    resp = generate_func(prompt)
    if not resp:
        logger.error("Empty response from GenAI for title: %s", title)
        return {
            "detailed_summary": "",
            "background_knowledge": [],
            "importance": "★2",
            "related_themes": ["IT・テクノロジー"]
        }
    
    # Try to parse JSON response
    try:
        import json
        import re
        
        # Remove markdown code blocks (```json ... ```)
        cleaned_resp = re.sub(r'```\s*json\s*', '', resp, flags=re.IGNORECASE)
        cleaned_resp = re.sub(r'```\s*', '', cleaned_resp)
        cleaned_resp = cleaned_resp.strip()
        
        # Extract JSON from response (in case there's extra text)
        json_start = cleaned_resp.find('{')
        json_end = cleaned_resp.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned_resp[json_start:json_end]
            content = json.loads(json_str)
            
            # Ensure default values exist
            if "importance" not in content or not content.get("importance"):
                content["importance"] = "★2"
            if "related_themes" not in content or not content.get("related_themes"):
                content["related_themes"] = ["IT・テクノロジー"]
            if "background_knowledge" not in content:
                content["background_knowledge"] = []
            if "detailed_summary" not in content:
                content["detailed_summary"] = ""
            
            logger.info("Generated detailed content for: %s (importance: %s, themes: %s)", 
                       title, content.get("importance"), content.get("related_themes"))
            return content
    except Exception as e:
        logger.warning("Failed to parse JSON response: %s. Raw response: %s", e, resp[:200])
    
    # Fallback: return structure with default values
    return {
        "detailed_summary": resp,
        "background_knowledge": [],
        "importance": "★2",
        "related_themes": ["IT・テクノロジー"]
    }


def build_notion_blocks(detailed_content):
    """Build Notion blocks from detailed content with toggle structure."""
    blocks = []
    
    # Introductory text
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": "🧠 読む前の思考"}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": "（タイトルから背景や影響を1秒だけ推測してみよう）"}}
            ]
        }
    })
    
    # Toggle: 詳細要約
    summary_text = detailed_content.get("detailed_summary", "")
    toggle_children_summary = []
    if summary_text:
        toggle_children_summary.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": summary_text}}]
            }
        })
    
    blocks.append({
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": "📄 詳細要約"}}],
            "children": toggle_children_summary
        }
    })
    
    # Toggle: 補足・背景知識・用語解説
    background_items = detailed_content.get("background_knowledge", [])
    toggle_children_background = []
    
    if background_items:
        for item in background_items:
            toggle_children_background.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": item}}]
                }
            })
    else:
        toggle_children_background.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": "（背景知識なし）"}}]
            }
        })
    
    blocks.append({
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": "💡 補足・背景知識・用語解説"}}],
            "children": toggle_children_background
        }
    })
    
    # Callout: 思考アウトプット
    blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [
                {"type": "text", "text": {"content": "✍️ 思考アウトプット"}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": "（My Take & So What?）"}}
            ],
            "icon": {"emoji": "✍️"},
            "color": "purple_background"
        }
    })
    
    # Callout content: 想定とのギャップ・発見
    blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [
                {"type": "text", "text": {"content": "💡 "}, "annotations": {}},
                {"type": "text", "text": {"content": "想定とのギャップ・発見"}, "annotations": {"bold": True}}
            ],
            "icon": {"emoji": "💡"},
            "color": "blue_background"
        }
    })
    
    # Callout content: So What?（自分・業務への影響/活かせること）
    blocks.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [
                {"type": "text", "text": {"content": "🚀 "}, "annotations": {}},
                {"type": "text", "text": {"content": "So What?（自分・業務への影響/活かせること）"}, "annotations": {"bold": True}}
            ],
            "icon": {"emoji": "🚀"},
            "color": "yellow_background"
        }
    })
    
    return blocks


def create_notion_page(notion_api_key, database_id, title, url, category, detailed_content):
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    jst = timezone(timedelta(hours=9))
    today = datetime.now(jst).date().isoformat()

    # Build content blocks
    children = build_notion_blocks(detailed_content)

    # Extract properties from detailed_content with validation
    importance = detailed_content.get("importance", "★2") or "★2"
    related_themes = detailed_content.get("related_themes", []) or ["IT・テクノロジー"]
    
    # Ensure related_themes is a list and contains valid strings
    if not isinstance(related_themes, list):
        related_themes = [related_themes]
    related_themes = [str(t).strip() for t in related_themes if t]
    if not related_themes:
        related_themes = ["IT・テクノロジー"]

    # Build properties
    properties = {
        "名前": {"title": [{"text": {"content": title}}]},
        "URL": {"url": url},
        "カテゴリ": {"select": {"name": category}},
        "日付": {"date": {"start": today}},
        "ステータス": {"status": {"name": "未読"}},
        "重要度": {"select": {"name": importance}},
        "関連テーマ": {
            "multi_select": [{"name": theme} for theme in related_themes]
        }
    }

    data = {
        "parent": {"database_id": database_id},
        "properties": properties,
        "children": children,
    }

    logger.info("Creating Notion page for: %s (importance: %s, themes: %s)", 
               title, importance, related_themes)
    r = requests.post(NOTION_API_URL, headers=headers, json=data, timeout=30)
    if r.status_code not in (200, 201):
        logger.error("Failed to create Notion page: %s %s", r.status_code, r.text)
        return False
    logger.info("Notion page created: %s", r.json().get("id"))
    return True


def main():
    GEMINI_API_KEY, NOTION_API_KEY, NOTION_DATABASE_ID = get_env_vars()

    generate_func = init_genai_client(GEMINI_API_KEY)
    if generate_func is None:
        logger.error("GenAI client could not be initialized. Exiting.")
        sys.exit(1)

    # Fetch existing URLs from Notion to prevent duplicates
    existing_urls = fetch_existing_urls(NOTION_API_KEY, NOTION_DATABASE_ID)

    for category, feed_url in RSS_FEEDS.items():
        entries = fetch_latest_entries(feed_url, limit=2)
        
        # Deduplicate entries from RSS feed itself
        entries = deduplicate_entries(entries)
        
        for e in entries:
            title = e["title"] or "(無題)"
            link = e["link"] or ""
            
            # Skip if URL already exists in Notion
            if link and link in existing_urls:
                logger.info("Skipping duplicate URL: %s", link)
                continue
            
            try:
                detailed_content = generate_detailed_content(generate_func, title, link)
            except Exception as ex:
                logger.exception("Error generating content for %s: %s", title, ex)
                detailed_content = {"detailed_summary": "", "background_knowledge": []}

            try:
                ok = create_notion_page(NOTION_API_KEY, NOTION_DATABASE_ID, title, link, category, detailed_content)
                if not ok:
                    logger.error("Failed to push to Notion for: %s", title)
            except Exception as ex:
                logger.exception("Error creating Notion page for %s: %s", title, ex)

            time.sleep(1)  # gentle pacing between items

    logger.info("All done.")


if __name__ == "__main__":
    main()
