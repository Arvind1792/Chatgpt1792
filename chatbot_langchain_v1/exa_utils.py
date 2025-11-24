# exa_utils.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

EXA_API_KEY = os.getenv("EXA_API_KEY")

SEARCH_URL = "https://api.exa.ai/search"
FETCH_URL = "https://api.exa.ai/contents"

HEADERS = {
    "x-api-key": EXA_API_KEY,
    "Content-Type": "application/json",
}


def run_exa_search_and_fetch(query: str):
    """
    Exa Search -> Fetch full text.
    Returns:
        context_with_numbers (str): text chunks numbered [1], [2], ...
        sources (list): [{index, title, url}]
    """

    print("🔥 Using Exa Search + Fetch...")

    # STEP 1 — SEARCH
    search_body = {
        "query": query,
        "useAutoprompt": True,
        "numResults": 5,
    }

    resp = requests.post(SEARCH_URL, json=search_body, headers=HEADERS)
    try:
        search_data = resp.json()
    except Exception:
        print("❌ EXA search: non-JSON response")
        return "", []

    if "error" in search_data:
        print("❌ Exa Search Error:", search_data["error"])
        return "", []

    ids = [r.get("id") for r in search_data.get("results", []) if r.get("id")]
    if not ids:
        print("❌ Exa returned no results")
        return "", []

    # STEP 2 — FETCH CONTENT
    fetch_body = {
        "ids": ids,
        "includeText": True,
        "numCharacters": 4000,
    }

    fetch_resp = requests.post(FETCH_URL, json=fetch_body, headers=HEADERS)
    try:
        fetch_data = fetch_resp.json()
    except Exception:
        print("❌ EXA fetch: non-JSON response")
        return "", []

    if "error" in fetch_data:
        print("❌ Exa Fetch Error:", fetch_data["error"])
        return "", []

    numbered_blocks = []
    sources = []
    idx = 1

    for item in fetch_data.get("results", []):
        title = item.get("title", "Unknown")
        url = item.get("url", "")
        text = item.get("text", "")

        if not text or not text.strip():
            continue

        # for the LLM context
        numbered_blocks.append(f"[{idx}] {text}")

        # for displaying sources
        sources.append(
            {
                "index": idx,
                "title": title,
                "url": url,
            }
        )

        idx += 1

    if not numbered_blocks:
        return "", []

    full_context = "\n\n".join(numbered_blocks)
    return full_context, sources
