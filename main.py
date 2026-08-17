import os
import sys
import logging
import time
import json
import re
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
            deduped.append(entry)
    
    if len(entries) > len(deduped):
        logger.info("Deduplicated %d entries -> %d entries", len(entries), len(deduped))
    
    return deduped


def fetch_latest_entries(feed_url, limit=5):
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
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error("Failed to initialize genai client: %s", e)
        return None

    def generate(prompt):
        try:
            response = client.models.generate_content(model=GENAI_MODEL, contents=prompt)
            return getattr(response, "text", None) or str(response)
        except Exception:
            logger.exception("GenAI generation failed")
            return ""

    return generate


def generate_detailed_content(generate_func, title, url):
    """Generate structured analysis based on efficiency elements."""
    prompt = (
        f"以下のニュース記事について、短時間で深く理解するための構造化分析を行ってください。\n"
        f"記事タイトル: {title}\n"
        f"URL: {url}\n\n"
        "以下の項目を解析し、必ずJSON形式のみで出力してください（マークダウン記号は含めないでください）。\n\n"
        "1. 【結論】: 1〜2文で最重要ファクトを記述（\"conclusion\"）\n"
        "2. 【背景】: 事象の経緯や背景を箇条書き2〜3項目で記述（\"background\"）\n"
        "3. 【影響】: 業界・技術・経済への波及効果を箇条書き2〜3項目で記述（\"impact\"）\n"
        "4. 【キー数値・ファクト】: 金額、割合、件数などの重要な数値を箇条書きで抽出（\"key_numbers\"）\n"
        "5. 【事実と意見の整理】: 「事実（確定事項）」と「意見・予測」を区別して記述（\"fact_and_opinion\"）\n"
        "6. 【専門用語・背景解説】: 重要な用語や解説（箇条書き2〜3項目）（\"glossary\"）\n"
        "7. 【重要度】: 「★1」「★2」「★3」のいずれかで判定（\"importance\"）\n"
        "8. 【関連テーマ】: 最大3つまでのタグ（\"related_themes\"）\n"
        "   候補: 「生成AI」「基盤モデル」「マクロ経済」「IT業界」「キャリア」「スタートアップ」「ガバナンス」「セキュリティ」\n\n"
        "出力フォーマット（JSONのみ）:\n"
        '{\n'
        '  "conclusion": "結論テキスト",\n'
        '  "background": ["背景1", "背景2"],\n'
        '  "impact": ["影響1", "影響2"],\n'
        '  "key_numbers": ["数値データ1", "数値データ2"],\n'
        '  "fact_and_opinion": {"fact": ["事実1"], "opinion": ["意見・予測1"]},\n'
        '  "glossary": ["用語解説1", "用語解説2"],\n'
        '  "importance": "★2",\n'
        '  "related_themes": ["生成AI", "IT業界"]\n'
        '}\n'
    )
    logger.info("Generating detailed content for: %s", title)
    resp = generate_func(prompt)
    if not resp:
        logger.error("Empty response from GenAI for title: %s", title)
        return {
            "conclusion": title,
            "background": [],
            "impact": [],
            "key_numbers": [],
            "fact_and_opinion": {"fact": [], "opinion": []},
            "glossary": [],
            "importance": "★2",
            "related_themes": ["IT業界"]
        }
    
    try:
        cleaned_resp = re.sub(r'```\s*json\s*', '', resp, flags=re.IGNORECASE)
        cleaned_resp = re.sub(r'```\s*', '', cleaned_resp).strip()
        
        json_start = cleaned_resp.find('{')
        json_end = cleaned_resp.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned_resp[json_start:json_end]
            content = json.loads(json_str)
            
            content.setdefault("conclusion", title)
            content.setdefault("background", [])
            content.setdefault("impact", [])
            content.setdefault("key_numbers", [])
            content.setdefault("fact_and_opinion", {"fact": [], "opinion": []})
            content.setdefault("glossary", [])
            content.setdefault("importance", "★2")
            content.setdefault("related_themes", ["IT業界"])
            
            logger.info("Generated detailed content for: %s (importance: %s)", title, content.get("importance"))
            return content
    except Exception as e:
        logger.warning("Failed to parse JSON response: %s", e)
    
    return {
        "conclusion": title,
        "background": [resp],
        "impact": [],
        "key_numbers": [],
        "fact_and_opinion": {"fact": [], "opinion": []},
        "glossary": [],
        "importance": "★2",
        "related_themes": ["IT業界"]
    }


def build_notion_blocks(content):
    """Build Notion blocks with structured analysis, strictly excluding manual entry spaces."""
    blocks = []
    
    # 1. 結論 Callout
    conclusion = content.get("conclusion", "")
    if conclusion:
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [
                    {"type": "text", "text": {"content": "📌 結論（要点）: "}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": conclusion}}
                ],
                "icon": {"emoji": "⚡"},
                "color": "blue_background"
            }
        })
    
    # 2. 構造分析 Toggle
    toggle_children = []
    
    # 背景
    background = content.get("background", [])
    if background:
        toggle_children.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": "🔍 背景・経緯"}}]}
        })
        for item in background:
            toggle_children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]}
            })

    # 影響
    impact = content.get("impact", [])
    if impact:
        toggle_children.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": "🚀 今後の影響・波及効果"}}]}
        })
        for item in impact:
            toggle_children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]}
            })

    # キー数値
    key_numbers = content.get("key_numbers", [])
    if key_numbers:
        toggle_children.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": "📊 主要な数値・ファクト"}}]}
        })
        for item in key_numbers:
            toggle_children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]}
            })

    # 事実 vs 意見
    fo = content.get("fact_and_opinion", {})
    facts = fo.get("fact", []) if isinstance(fo, dict) else []
    opinions = fo.get("opinion", []) if isinstance(fo, dict) else []
    if facts or opinions:
        toggle_children.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": "⚖️ 事実（Fact）と意見（Opinion）の整理"}}]}
        })
        for f in facts:
            toggle_children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "[事実] "}, "annotations": {"bold": True, "color": "green"}},
                        {"type": "text", "text": {"content": f}}
                    ]
                }
            })
        for o in opinions:
            toggle_children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "[意見/予測] "}, "annotations": {"bold": True, "color": "orange"}},
                        {"type": "text", "text": {"content": o}}
                    ]
                }
            })

    blocks.append({
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": "📄 構造化要約・背景分析"}}],
            "children": toggle_children
        }
    })

    # 3. 用語解説 Toggle
    glossary = content.get("glossary", [])
    if glossary:
        glossary_children = []
        for item in glossary:
            glossary_children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": item}}]}
            })
        blocks.append({
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": [{"type": "text", "text": {"content": "💡 専門用語・補足解説"}}],
                "children": glossary_children
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

    children = build_notion_blocks(detailed_content)

    importance = detailed_content.get("importance", "★2") or "★2"
    related_themes = detailed_content.get("related_themes", []) or ["IT業界"]
    
    if not isinstance(related_themes, list):
        related_themes = [related_themes]
    related_themes = [str(t).strip() for t in related_themes if t]
    if not related_themes:
        related_themes = ["IT業界"]

    properties = {
        "名前": {"title": [{"text": {"content": title}}]},
        "URL": {"url": url},
        "カテゴリ": {"select": {"name": category}},
        "日付": {"date": {"start": today}},
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

    existing_urls = fetch_existing_urls(NOTION_API_KEY, NOTION_DATABASE_ID)

    for category, feed_url in RSS_FEEDS.items():
        entries = fetch_latest_entries(feed_url, limit=5)
        entries = deduplicate_entries(entries)
        
        for e in entries:
            title = e["title"] or "(無題)"
            link = e["link"] or ""
            
            if link and link in existing_urls:
                logger.info("Skipping duplicate URL: %s", link)
                continue
            
            try:
                detailed_content = generate_detailed_content(generate_func, title, link)
            except Exception as ex:
                logger.exception("Error generating content for %s: %s", title, ex)
                detailed_content = {"conclusion": title}

            try:
                ok = create_notion_page(NOTION_API_KEY, NOTION_DATABASE_ID, title, link, category, detailed_content)
                if not ok:
                    logger.error("Failed to push to Notion for: %s", title)
            except Exception as ex:
                logger.exception("Error creating Notion page for %s: %s", title, ex)

            time.sleep(1)

    logger.info("All done.")


if __name__ == "__main__":
    main()