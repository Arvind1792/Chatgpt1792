# cqc.py
import re

def clean_text(t: str) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


def should_use_exa(query: str, docs: list) -> bool:
    """
    Decide if Pinecone retrieval is weak and we should fallback to Exa AI.
    This is the heart of the Context Quality Check (CQC).
    """

    # 1) No documents returned
    if len(docs) == 0:
        print("CQC: No documents retrieved → Exa needed")
        return True

    # Extract text
    chunk_texts = [
        clean_text(
            d.metadata.get("text")
            or d.page_content
            or ""
        ) for d in docs
    ]

    # 2) All chunks empty
    if all(t == "" for t in chunk_texts):
        print("CQC: All retrieved chunks empty → Exa needed")
        return True

    # 3) Total context too small (< 150 chars)
    combined = " ".join(chunk_texts).strip()
    if len(combined) < 150:
        print("CQC: Context too small → Exa needed")
        return True

    # 4) Keyword overlap check (semantic relevance)
    q_words = [w for w in query.lower().split() if len(w) > 3]
    hits = sum(1 for w in q_words if w in combined.lower())

    if hits == 0:
        print("CQC: Query keywords not found in context → Exa needed")
        return True

    # 5) PASS — Pinecone context is good
    print("CQC: Pinecone context OK")
    return False
