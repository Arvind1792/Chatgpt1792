# chat_langchain_v1.py
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore

from exa_utils import run_exa_search_and_fetch
from cqc import should_use_exa

load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


def build_retriever_and_llm():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )

    return retriever, llm


def build_pdf_context_with_numbers(docs):
    """
    Build context like:
    [1] (Source: file1.pdf) chunk text
    [2] (Source: file2.pdf) chunk text
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "Unknown")
        txt = d.metadata.get("text") or d.page_content or ""
        blocks.append(f"[{i}] (Source: {src}) {txt}")
    return "\n\n".join(blocks)


def ask_llm_pdf_with_citations(llm, context: str, query: str) -> str:
    """
    LLM answers from PDF context using inline numeric citations [1], [2], ...
    """

    prompt = f"""
You are a RAG assistant over a set of PDF documents.

You MUST obey these rules:

1. Use ONLY the information in the numbered PDF context chunks below.
2. Use inline numeric citations like [1], [2], [3] in your answer, where the
   number refers to the context chunk number.
3. If the answer is not clearly present in the context, say:
   "I don't know. The answer is not clearly available in the provided documents. Please check the PDFs."
4. Do NOT use outside knowledge. Do NOT hallucinate.

------------------------
NUMBERED PDF CONTEXT:
{context}
------------------------

Question:
{query}

Answer (with inline citations):
"""

    resp = llm.invoke(prompt)
    return resp.content


def ask_llm_exa_with_citations(llm, exa_context: str, query: str, exa_sources: list) -> str:
    """
    LLM answers from Exa web context using inline numeric citations [1], [2], ...
    """

    sources_block = "\n".join(
        f"[{s['index']}] {s['title']} - {s['url']}" for s in exa_sources
    )

    prompt = f"""
You are a factual assistant over WEB sources provided by Exa.

Rules:
1. Use ONLY the numbered web context chunks below.
2. Use inline numeric citations like [1], [2], [3] in your answer.
3. The numbers refer to the SOURCES list.
4. If the answer is not clearly present in the context, say:
   "I don't know. The answer is not clearly available in the provided sources."
5. Do NOT use outside knowledge.

------------------------
NUMBERED WEB CONTEXT:
{exa_context}
------------------------

SOURCES:
{sources_block}

Question:
{query}

Answer (with inline citations):
"""

    resp = llm.invoke(prompt)
    return resp.content


def main():
    print("🚀 Loading retriever + LLM (Azure OpenAI + Pinecone)...")
    retriever, llm = build_retriever_and_llm()
    print("✨ Ready. Type your question (or 'exit').\n")

    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break

        # 1) Retrieve from Pinecone
        docs = retriever.invoke(q)

        # 2) Decide PDF vs Exa using LLM-based CQC
        use_exa = should_use_exa(q, docs, llm, debug=True)  

        if use_exa:
            print("📡 Using Exa web fallback...")
            exa_context, exa_sources = run_exa_search_and_fetch(q)

            if not exa_context:
                print("❌ Exa could not retrieve useful context.")
                print(
                    "I don't know. The answer is not clearly available in the provided documents or web results."
                )
                continue

            answer = ask_llm_exa_with_citations(llm, exa_context, q, exa_sources)

            print("\n🔵 ANSWER (from Exa + LLM):")
            print(answer)

            print("\n🌐 SOURCES:")
            for s in exa_sources:
                print(f"  [{s['index']}] {s['title']} - {s['url']}")
        else:
            print("📘 Using PDF (Pinecone RAG)...")
            pdf_context = build_pdf_context_with_numbers(docs)
            answer = ask_llm_pdf_with_citations(llm, pdf_context, q)

            print("\n🔵 ANSWER (from PDFs):")
            print(answer)

            print("\n📁 PDF CHUNK SOURCES:")
            for i, d in enumerate(docs, start=1):
                print(f"  [{i}] {d.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()
