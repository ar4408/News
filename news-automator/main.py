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


def generate_summary(generate_func, title, url):
    prompt = (
        f"以下の制約で日本語の要約を作成してください。\n"
        f"記事タイトル: {title}\n"
        f"URL: {url}\n\n"
        "制約:\n"
        "- 箇条書きを3つ出力すること。\n"
        "- 各箇条書きは40文字以内に収めること。\n"
        "- 箇条書きは1行に1つ、先頭を" + '・' + "で始めること。\n"
        "- 余計な説明は付けず、箇条書きのみを出力すること。\n"
    )
    logger.info("Generating summary for: %s", title)
    resp = generate_func(prompt)
    if not resp:
        logger.error("Empty response from GenAI for title: %s", title)
        return ""
    # Post-process: keep only three lines and ensure they are short
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    # Take first 3 non-empty lines
    bullets = lines[:3]
    # If they don't start with ・, normalize
    normalized = []
    for b in bullets:
        if not b.startswith("・"):
            b = "・" + b.lstrip("- ")
        # Truncate to 40 chars
        if len(b) > 40:
            b = b[:37] + "..."
        normalized.append(b)
    summary_text = "\n".join(normalized)
    logger.info("Generated summary:\n%s", summary_text)
    return summary_text


def create_notion_page(notion_api_key, database_id, title, url, category, summary_text):
    headers = {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    jst = timezone(timedelta(hours=9))
    today = datetime.now(jst).date().isoformat()

    data = {
        "parent": {"database_id": database_id},
        "properties": {
            "名前": {"title": [{"text": {"content": title}}]},
            "URL": {"url": url},
            "カテゴリ": {"select": {"name": category}},
            "日付": {"date": {"start": today}},
        },
        "children": [
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": "AI要約"}}]},
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": summary_text}}]},
            },
        ],
    }

    logger.info("Creating Notion page for: %s", title)
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

    for category, feed_url in RSS_FEEDS.items():
        entries = fetch_latest_entries(feed_url, limit=2)
        for e in entries:
            title = e["title"] or "(無題)"
            link = e["link"] or ""
            try:
                summary = generate_summary(generate_func, title, link)
            except Exception as ex:
                logger.exception("Error generating summary for %s: %s", title, ex)
                summary = ""

            try:
                ok = create_notion_page(NOTION_API_KEY, NOTION_DATABASE_ID, title, link, category, summary)
                if not ok:
                    logger.error("Failed to push to Notion for: %s", title)
            except Exception as ex:
                logger.exception("Error creating Notion page for %s: %s", title, ex)

            time.sleep(1)  # gentle pacing between items

    logger.info("All done.")


if __name__ == "__main__":
    main()
