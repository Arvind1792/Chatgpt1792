# summary_agent.py  (Final LC v1 Compatible Version)

import os
import tempfile
from typing import List

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.agents import create_agent


# ======================================================
# 1. LOAD ENVIRONMENT
# ======================================================

load_dotenv()

AZURE_STORAGE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")


# ======================================================
# 2. LLM PROVIDER
# ======================================================

def get_llm() -> AzureChatOpenAI:
    """
    Azure ChatOpenAI model (LangChain v1).
    """
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.2,
    )


# ======================================================
# 3. DOWNLOAD PDF FROM AZURE
# ======================================================

def download_pdf(pdf_name: str) -> str:
    """
    Download PDF from Azure Blob → returns local path.
    """
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)
    container = service.get_container_client(AZURE_CONTAINER)
    blob = container.get_blob_client(pdf_name)

    if not blob.exists():
        raise FileNotFoundError(f"PDF '{pdf_name}' not found in Azure container.")

    stream = blob.download_blob()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(stream.readall())
        return tmp.name


# ======================================================
# 4. LOAD AND SPLIT PDF
# ======================================================

def load_and_split(local_pdf_path: str) -> List[Document]:
    """
    Load PDF → split into chunks for summarization.
    """
    loader = PyPDFLoader(local_pdf_path)
    pages = loader.load()  # One Document per page

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_documents(pages)


# ======================================================
# 5. LCEL MAP → REDUCE SUMMARIZATION
# ======================================================

def build_medium_summary_chain(llm):
    """
    LCEL-only summarizer:
    - Map step: small summary for each chunk
    - Reduce step: merge into medium summary (1–3 paragraphs)
    """

    # MAP (summarize each chunk)
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this part of a PDF clearly:"),
        ("human", "{context}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    # REDUCE (combine chunk summaries)
    reduce_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Combine the partial summaries into a summary "
            "Write natural flowing text, no bullet points."
        ),
        ("human", "{context}")
    ])
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # Final map → reduce pipeline
    def map_reduce(documents: List[Document]) -> str:
        # 1. Map step over chunks
        mapped_summaries = [
            map_chain.invoke({"context": d.page_content})
            for d in documents
        ]

        # 2. Combine mapped summaries
        combined_text = "\n\n".join(mapped_summaries)

        # 3. Reduce to final summary
        final_summary = reduce_chain.invoke({"context": combined_text})
        return final_summary.strip()

    return map_reduce

def build_hierarchical_summary_chain(llm):
    """
    Multi-level hierarchical summarizer.
    Works even for 200+ page PDFs.
    """

    # ----- MAP STEP -----
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this part of a PDF clearly."),
        ("human", "{context}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    # ----- REDUCE STEP -----
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize these partial summaries into a coherent summary."),
        ("human", "{context}")
    ])
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # Helper: Group lists
    def group(items, n=10):
        for i in range(0, len(items), n):
            yield items[i:i+n]

    def map_reduce(documents: List[Document]) -> str:
        # 1) MAP over chunks
        mapped = [map_chain.invoke({"context": d.page_content}) for d in documents]

        # 2) HIERARCHICAL REDUCE
        # Level 1: Batch-wise summaries
        batch_summaries = []
        for batch in group(mapped, 10):  # Groups of 10 chunks
            combined = "\n\n".join(batch)
            batch_summary = reduce_chain.invoke({"context": combined})
            batch_summaries.append(batch_summary)
            print("batch 1 completed")

        # 3) Level 2: Final summary
        final_combined = "\n\n".join(batch_summaries)
        print("Prepairing final_summary")
        final_summary = reduce_chain.invoke({"context": final_combined})
        
        return final_summary.strip()

    return map_reduce


# ======================================================
# 6. TOOL: summarize_pdf
# ======================================================

@tool
def summarize_pdf(pdf_name: str) -> str:
    """
    PDF summary tool .
    """

    llm = get_llm()

    # STEP 1 — Download from Azure
    local_pdf = download_pdf(pdf_name)

    # STEP 2 — Load + split into chunks
    docs = load_and_split(local_pdf)
    if not docs:
        return f"No content extracted from PDF '{pdf_name}'."

    # STEP 3 — LCEL summarization pipeline
    summary_chain = build_hierarchical_summary_chain(llm)
    # summary_chain = build_medium_summary_chain(llm)

    # STEP 4 — Run summarization
    summary = summary_chain(docs)
    return summary



if __name__ == "__main__":
    agent = get_summary_agent()

    print("🚀 PDF Summarizer Agent Ready")
    print("Try: summarize hackaidea.pdf\n")

    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break

        result = agent.invoke({
            "messages": [{"role": "user", "content": q}]
        })

        msg = result["messages"][-1]["content"]
        print("\nAgent:", msg, "\n")
