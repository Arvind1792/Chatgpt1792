# exa_utils.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

EXA_API_KEY = os.getenv("EXA_API_KEY")

SEARCH_URL = "https://api.exa.ai/search"
FETCH_URL = "https://api.exa.ai/contents"
# ANSWER_URL = "https://api.exa.ai/answer"  # Not used in this example

HEADERS = {
    "x-api-key": EXA_API_KEY,
    "Content-Type": "application/json"
}


def run_exa_search(query: str) -> str:
    """
    Uses Exa Search + Fetch to retrieve full webpage text.
    """

    print("🔥 Using EXA AI fallback...")

    # STEP 1 — SEARCH
    search_body = {
        "query": query,
        "useAutoprompt": True,
        "numResults": 5,
    }

    search_resp = requests.post(SEARCH_URL, json=search_body, headers=HEADERS)
    search_data = search_resp.json()
    # print(search_data)
    if "error" in search_data:
        print("❌ Exa search error:", search_data["error"])
        return ""

    result_ids = [r["id"] for r in search_data.get("results", [])]

    if not result_ids:
        print("❌ Exa returned no results")
        return ""

    # STEP 2 — FETCH CONTENT
    fetch_body = {
        "ids": result_ids,
        "includeText": True,
        "numCharacters": 3000,   # GET FULL TEXT
    }

    fetch_resp = requests.post(FETCH_URL, json=fetch_body, headers=HEADERS)
    fetch_data = fetch_resp.json()

    if "error" in fetch_data:
        print("❌ Exa fetch error:", fetch_data["error"])
        return ""

    context_parts = []

    for item in fetch_data.get("results", []):
        title = item.get("title", "")
        text = item.get("text", "")

        if text.strip():
            context_parts.append(f"[Web: {title}]\n{text}")
    # print(context_parts)
    return "\n\n".join(context_parts)
