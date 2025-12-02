
import os
# import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from exa_utils import run_exa_search_and_fetch
from cqc import should_use_exa
load_dotenv()

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

from langchain_core.messages import BaseMessage

from datetime import datetime, timedelta

def generate_sas_url(container_name: str, blob_name: str, hours_valid: int = 6):
    """
    Generates a read-only SAS URL for a private Azure Blob.
    """
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    account_key = os.getenv("AZURE_STORAGE_KEY")

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=hours_valid)
    )

    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"

# ---------------------------
# RAG building
# ---------------------------
# @st.cache_resource(show_spinner=True)
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
    print("llm is working...")
    return retriever, llm


# ---------------------------
# PDF Context Builder
# ---------------------------
def build_pdf_context_with_numbers(docs):
    """
    Produces:
    [1] (Source: URL) text
    [2] (Source: URL) text
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "Unknown URL")
        txt = d.metadata.get("text") or d.page_content or ""
        blocks.append(f"[{i}] (Source: {src}) {txt}")
    return "\n\n".join(blocks)


# ---------------------------
# PDF Answer with Inline Citations
# ---------------------------
def ask_llm_pdf_with_citations(llm, context: str, query: str,pdf_sources: list) -> str:
    """
    Returns answer with inline numbered citations [1], [2], etc.
    And clickable links in the Sources list.
    """

    numbered_sources_text = "\n".join(
        f"[{s['index']}] <{s['source']}>"
        for s in pdf_sources
    )

    prompt = f"""
You are a strict RAG assistant. Answer ONLY from the PDF context provided.

Follow these RULES:

1. When using information from the text, add an inline citation in NUMBERED format:
   Like this → [1], [2], [3].

2. Do NOT use URLs inline.
   ONLY use numbered citations.

3. You MUST use the numbering exactly as given below:

   {numbered_sources_text}

4. Each numbered citation corresponds to a source:
   - [1] = {pdf_sources[0]['source']}   (automatically linked in Streamlit)
   - [2] = {pdf_sources[1]['source']}   (if exists)
   - etc.

5. DO NOT make up numbers. Use only the numbers from the list.
6. At the end of your answer include:

   **Sources Used:**
   [1] <URL>
   [2] <URL>

7. If answer cannot be derived from context, say:
   "I don't know. The answer is not available in the provided PDFs."

------------------------
------------------------
PDF CONTEXT:
{context}
------------------------

Question:
{query}

Answer (with inline citations):
"""
    resp = llm.invoke(prompt)
    return resp.content


# ---------------------------
# Exa Answer with Citations
# ---------------------------
def ask_llm_exa_with_citations(llm, exa_context: str, query: str, exa_sources: list):
    sources_block = "\n".join(
        f"[{s['index']}] {s['title']} - {s['url']}"
        for s in exa_sources
    )

    prompt = f"""
You are a factual assistant using ONLY the Exa web context.

Rules:
1. Use ONLY the provided context.
2. When referencing a sentence, include:
   **[Source: <title>]**
3. Multiple citations are allowed.
4. At the end produce a **Sources Used** section with titles+urls.
5. If not available in context, say:
   "I don't know. The answer is not in the provided documents."

------------------------
WEB CONTEXT:
{exa_context}
------------------------

SOURCES:
{sources_block}

Question:
{query}

Answer:
"""
    resp = llm.invoke(prompt)
    return resp.content


# ---------------------------
# Decision + Answering Logic
# ---------------------------
def answer_query(query: str):
    retriever, llm = build_retriever_and_llm()
    docs = retriever.invoke(query)
    print(f"[DEBUG] Retrieved {len(docs)} docs from Pinecone.")
    # print(docs)
    # Should fallback to Exa?
    use_exa = should_use_exa(query, docs, llm, debug=False)

    if use_exa:
        exa_context, exa_sources = run_exa_search_and_fetch(query)
        if not exa_context:
            return (
                "I don't know. The answer is not clearly available in the provided PDFs or web.",
                {"type": "none"},
            )

        answer = ask_llm_exa_with_citations(llm, exa_context, query, exa_sources)
        return answer, {"type": "exa", "exa_sources": exa_sources}

    # PDF Answer Mode
    pdf_context = build_pdf_context_with_numbers(docs)
    pdf_sources = []
    for i, d in enumerate(docs):
        raw_url = d.metadata.get("source", "")
        # Extract blob name from URL:
        # URL: https://account.blob.core.windows.net/container/my.pdf
        blob_name = raw_url.split("/")[-1]
        print("blob_name :- ",blob_name)
        sas_url = generate_sas_url(AZURE_CONTAINER, blob_name)

        pdf_sources.append({
            "index": i + 1,
            "source": sas_url
        })
    answer = ask_llm_pdf_with_citations(llm, pdf_context, query,pdf_sources)

    

    return answer, {"type": "pdf", "pdf_sources": pdf_sources}


@tool
def ask_pdf_question(query: str) -> str:
    """
    Answers a question using the PDF RAG system + Exa fallback.
    """
    answer,sources = answer_query(query)
    return answer
