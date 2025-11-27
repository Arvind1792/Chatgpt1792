# app_v1.py - Streamlit UI for RAG + Exa fallback with inline numeric citations

import os
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore

from exa_utils import run_exa_search_and_fetch
from cqc import should_use_exa
from summary_agent import get_medium_summary_agent
from azure.storage.blob import BlobServiceClient
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

from langchain_core.messages import BaseMessage

def extract_agent_output(response):
    """
    Handles all possible output types from create_agent.invoke().
    Supports: AIMessage, BaseMessage, dict with 'messages', string.
    """
    # Case 1: AIMessage / BaseMessage
    if isinstance(response, BaseMessage):
        return response.content

    # Case 2: dict with messages[] (graph-style)
    if isinstance(response, dict):
        # If result has messages list
        if "messages" in response and isinstance(response["messages"], list):
            msgs = response["messages"]
            if msgs:
                last = msgs[-1]
                # last might be BaseMessage or dict
                return getattr(last, "content", str(last))

        # Generic fallback
        return str(response)

    # Case 3: raw string
    if isinstance(response, str):
        return response

    # Final fallback
    return str(response)


# ---------------------------
# RAG building
# ---------------------------
@st.cache_resource(show_spinner=True)
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
def ask_llm_pdf_with_citations(llm, context: str, query: str) -> str:
    prompt = f"""
You are a RAG assistant answering ONLY from the PDF context provided.

Rules:
1. Use ONLY the content inside the given context.
2. When you use any sentence from the context, add an inline citation:
   **[Source: <URL>]**
3. If multiple URLs support the same sentence, include multiple citations.
4. At the end of your answer produce:

   **Sources Used**
   - <URL1>
   - <URL2>
   - ...

5. If the answer cannot be derived from the context, say:
   "I don't know. The answer is not available in the provided PDFs."

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
def answer_query(query: str, retriever, llm):
    docs = retriever.invoke(query)
    print(f"[DEBUG] Retrieved {len(docs)} docs from Pinecone.")
    print(docs)
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
    answer = ask_llm_pdf_with_citations(llm, pdf_context, query)

    pdf_sources = [
        {
            "index": i + 1,
            "source": d.metadata.get("source", "Unknown URL")
        }
        for i, d in enumerate(docs)
    ]

    return answer, {"type": "pdf", "pdf_sources": pdf_sources}


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="GenAI RAG Bot (PDF + Exa)",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 GenAI RAG Chatbot (PDF + Exa with Inline Citations)")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown(f"**Pinecone Index:** `{PINECONE_INDEX}`")
        if st.button("🧹 Clear Chat History"):
            st.session_state.history = []
            st.rerun()

    retriever, llm = build_retriever_and_llm()

    # Replay chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # Show citations
            src = msg.get("sources")
            if not src:
                continue

            if src["type"] == "pdf":
                with st.expander("📁 PDF Sources"):
                    for s in src.get("pdf_sources", []):
                        url = s["source"]
                        st.markdown(f"[{s['index']}] 📄 [Open PDF]({url})")

            elif src["type"] == "exa":
                with st.expander("🌐 Web Sources (Exa)"):
                    for s in src.get("exa_sources", []):
                        st.markdown(f"[{s['index']}] **{s['title']}**")
                        st.markdown(f"[Open]({s['url']})")

    # # ======================================================
    # # PDF Summarizer Section (New)
    # # ======================================================
    # st.subheader("📄 PDF Summarizer (Medium Summary)")



    # def list_pdfs_in_azure():
    #     conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    #     container = os.getenv("AZURE_CONTAINER")
    #     service = BlobServiceClient.from_connection_string(conn)
    #     container_client = service.get_container_client(container)
    #     return [b.name for b in container_client.list_blobs() if b.name.lower().endswith(".pdf")]

    # pdf_files = list_pdfs_in_azure()

    # if pdf_files:
    #     selected_pdf = st.selectbox("Choose PDF to summarize:", pdf_files)

    #     if st.button("Summarize Selected PDF"):
    #         st.info(f"Summarizing: {selected_pdf}")

    #         agent = get_medium_summary_agent()
    #         response = agent.invoke({
    #             "messages": [
    #                 {"role": "user", "content": f"Summarize {selected_pdf}"}
    #             ]
    #         })

    #         st.success("Summary:")
    #         summary_text = extract_agent_output(response)
    #         st.write(summary_text)
    # else:
    #     st.warning("No PDFs found in Azure Blob Storage")

    # User query
    user_query = st.chat_input("Ask something...")
   
    if user_query:
        st.session_state.history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_text, source_info = answer_query(user_query, retriever, llm)
                st.write(answer_text)
                # st.write(response["messages"][-1]["content"])
                # Show sources immediately
                if source_info["type"] == "pdf":
                    with st.expander("📁 PDF Sources"):
                        for s in source_info.get("pdf_sources", []):
                            st.markdown(f"[{s['index']}] 📄 [Open PDF]({s['source']})")

                elif source_info["type"] == "exa":
                    with st.expander("🌐 Web Sources (Exa)"):
                        for s in source_info.get("exa_sources", []):
                            st.markdown(f"[{s['index']}] **{s['title']}**")
                            st.markdown(f"[Open]({s['url']})")

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": source_info,
                    }
                )


if __name__ == "__main__":
    main()
