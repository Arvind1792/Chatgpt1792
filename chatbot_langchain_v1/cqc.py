# cqc.py  -- LLM-based Context Quality Check (CQC)

from typing import List
from langchain_core.documents import Document


def _format_docs_for_cqc(
    docs: List[Document],
    max_docs: int = 4,
    max_chars_per_doc: int = 700,
) -> str:
    """
    Build a compact context string for the LLM judge.
    We don't need full chunks, just enough to decide if it's relevant.
    """
    parts = []
    for i, d in enumerate(docs[:max_docs], start=1):
        src = d.metadata.get("source", "Unknown")
        text = d.metadata.get("text") or d.page_content or ""
        text = text.strip().replace("\n", " ")

        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."

        parts.append(f"[{i} | Source: {src}]\n{text}")

    return "\n\n".join(parts)


def should_use_exa(
    query: str,
    docs: List[Document],
    llm,
    debug: bool = False,
) -> bool:
    """
    LLM-based Context Quality Check.

    Returns:
        True  -> context is NOT sufficient, use Exa
        False -> context is sufficient, use PDFs (Pinecone RAG)

    Logic:
    - If no docs -> immediately use Exa
    - Otherwise ask the LLM:
        "Is this context enough to answer fully and accurately?"
    """

    # 1) No docs -> must fall back
    if not docs:
        if debug:
            print("CQC-LLM: No docs retrieved -> USE_EXA")
        return True

    # 2) Build compact context view
    context_view = _format_docs_for_cqc(docs)
    print("context_view :- ",context_view)
    judge_prompt = f"""
You are a context quality judge in a RAG system.

You are given:
- A user question
- Several retrieved context chunks from a vector database (PDFs)

Your ONLY job is to decide whether these chunks contain enough information
to answer the question fully and accurately.

Rules:
1. If the context clearly contains enough info to answer the question,
   respond with exactly: USE_PDF
2. If the context is missing key information, is too vague, unrelated,
   or only partially helpful, respond with exactly: USE_EXA
3. Do NOT provide an explanation. ONLY respond with one token:
   USE_PDF or USE_EXA

--------------------
Context Chunks:
{context_view}
--------------------

Question:
{query}

Your answer (USE_PDF or USE_EXA):
"""

    resp = llm.invoke(judge_prompt)
    decision = resp.content.strip().upper()
    
    if debug:
        print(f"CQC-LLM decision raw: {decision}")

    if decision.startswith("USE_PDF"):
        return False  # i.e. don't use Exa
    else:
        # Default to Exa if anything weird
        return True
