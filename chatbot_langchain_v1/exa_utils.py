# exa_utils.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

EXA_API_KEY = os.getenv("EXA_API_KEY")

SEARCH_URL = "https://api.exa.ai/search"
CONTENTS_URL = "https://api.exa.ai/contents"

HEADERS = {
    "x-api-key": EXA_API_KEY,
    "Content-Type": "application/json",
}


def run_exa_search_and_fetch(query: str):
    print("🔥 Using Exa Search + Contents...")

    # -------------------------------
    # 1) SEARCH
    # -------------------------------
    resp = requests.post(
        SEARCH_URL,
        json={
            "query": query,
            "useAutoprompt": True,
            "numResults": 5
        },
        headers=HEADERS,
    )

    data = resp.json()

    urls = [r.get("url") for r in data.get("results", []) if r.get("url")]
    print("🔗 Exa URLs found:", urls)

    if not urls:
        return "", []

    # -------------------------------
    # 2) CONTENTS (FETCH)
    # -------------------------------
    fetch_resp = requests.post(
        CONTENTS_URL,
        json={
            "urls": urls,
            "text": True,              # NEW — returns extracted text
            "summary": True,           # NEW — returns AI-generated summary
            "summaryMaxLength": 4000,  # NEW — summary length
        },
        headers=HEADERS,
    )

    fetch_data = fetch_resp.json()
    # print("📥 Contents Response:", fetch_data)

    blocks = []
    sources = []
    idx = 1

    for item in fetch_data.get("results", []):
        title = item.get("title", "Unknown")
        url = item.get("url")
        text = item.get("text")
        summary = item.get("summary")

        # Prefer extracted text, fallback to summary
        content = text or summary

        if not content:
            continue

        blocks.append(f"[{idx}] {content}")
        sources.append({"index": idx, "title": title, "url": url})
        idx += 1

    if not blocks:
        print("⚠️ No content or summary returned.")
        return "", []

    return "\n\n".join(blocks), sources
