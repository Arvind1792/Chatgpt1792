# app_v1.py - Streamlit UI for RAG + Exa fallback with inline numeric citations

import os
import streamlit as st
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


def build_pdf_context_with_numbers(docs):
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "Unknown")
        txt = d.metadata.get("text") or d.page_content or ""
        # The model only needs the content; we include (Source: ...) for safety
        blocks.append(f"[{i}] (Source: {src}) {txt}")
    return "\n\n".join(blocks)


def ask_llm_pdf_with_citations(llm, context: str, query: str) -> str:
    prompt = f"""
You are a RAG assistant over a set of PDF documents.

Rules:
1. Answer ONLY using the provided context.
2. When referencing any information from context, insert an inline citation in this format:
   **[Source: filename.pdf]**
3. Multiple citations may appear in one sentence if needed.
4. At the end add a **Sources Used** section listing each unique citation used.
5. If the answer is not fully supported by context, reply:
   "I don't know. The answer is not clearly available in the provided documents. Please check the PDFs."

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
    sources_block = "\n".join(
        f"[{s['index']}] {s['title']} - {s['url']}" for s in exa_sources
    )

    prompt = f"""
You are a factual assistant over WEB sources provided by Exa.

Rules:
1. Answer ONLY using the provided context.
2. When referencing any information from context, insert an inline citation in this format:
   **[ Source: title]**
3. Multiple citations may appear in one sentence if needed.
4. At the end add a **Sources Used** section listing each unique citation used.
5. If the answer is not fully supported by context, reply:
   "I don't know. The answer is not clearly available in the provided documents. Please check the PDFs."


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


def answer_query(query: str, retriever, llm):
    """
    Returns:
      answer_text: str
      source_info: dict with keys:
           - "type": "pdf" or "exa"
           - "pdf_sources": list of {"index", "source"}   (for pdf)
           - "exa_sources": list of {"index", "title", "url"} (for exa)
    """

    # 1) retrieve from Pinecone
    docs = retriever.invoke(query)
    print(docs)
    # 2) Check context quality using LLM-based judge
    use_exa = should_use_exa(query, docs, llm, debug=False)
    
    if use_exa:
        exa_context, exa_sources = run_exa_search_and_fetch(query)
        print(exa_context,exa_sources)
        if not exa_context:
            return (
                "I don't know. The answer is not clearly available in the provided documents or web results.",
                {"type": "none"},
            )

        answer = ask_llm_exa_with_citations(llm, exa_context, query, exa_sources)
        return answer, {"type": "exa", "exa_sources": exa_sources}

    # 3) PDF mode
    pdf_context = build_pdf_context_with_numbers(docs)
    answer = ask_llm_pdf_with_citations(llm, pdf_context, query)

    pdf_sources = [
        {"index": i + 1, "source": d.metadata.get("source", "Unknown")}
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
    # st.markdown(
    #     "• First tries to answer from your **PDF knowledge base** (Pinecone).\n"
    #     "• If context is weak, it falls back to **Exa web search+fetch**.\n"
    #     "• All answers use **inline numeric citations** like `[1]`, `[2]`."
    # )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown(f"**Pinecone Index:** `{PINECONE_INDEX}`")
        if st.button("🧹 Clear Chat History"):
            st.session_state.history = []
            st.rerun()

    with st.spinner("Loading retriever + LLM..."):
        retriever, llm = build_retriever_and_llm()

    # Show chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                src_info = msg.get("sources")
                if not src_info:
                    continue
                if src_info["type"] == "pdf":
                    pdf_sources = src_info.get("pdf_sources", [])
                    if pdf_sources:
                        with st.expander("📁 PDF Sources"):
                            for s in pdf_sources:
                                st.markdown(f"[{s['index']}] `{s['source']}`")
                elif src_info["type"] == "exa":
                    exa_sources = src_info.get("exa_sources", [])
                    if exa_sources:
                        with st.expander("🌐 Web Sources (Exa)"):
                            for s in exa_sources:
                                title = s.get("title", "Unknown")
                                url = s.get("url", "")
                                st.markdown(f"[{s['index']}] **{title}**")
                                if url:
                                    st.markdown(f"[Open]({url})")

    # New query
    user_query = st.chat_input("Ask a question...")

    if user_query:
        # display user msg
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.history.append(
            {"role": "user", "content": user_query}
        )

        # assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_text, source_info = answer_query(user_query, retriever, llm)
                st.write(answer_text)

                # Also display sources for this turn
                if source_info["type"] == "pdf":
                    pdf_sources = source_info.get("pdf_sources", [])
                    if pdf_sources:
                        with st.expander("📁 PDF Sources"):
                            for s in pdf_sources:
                                st.markdown(f"[{s['index']}] `{s['source']}`")
                elif source_info["type"] == "exa":
                    exa_sources = source_info.get("exa_sources", [])
                    if exa_sources:
                        with st.expander("🌐 Web Sources (Exa)"):
                            for s in exa_sources:
                                title = s.get("title", "Unknown")
                                url = s.get("url", "")
                                st.markdown(f"[{s['index']}] **{title}**")
                                if url:
                                    st.markdown(f"[Open]({url})")

                # push assistant msg + sources to history
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": source_info,
                    }
                )


if __name__ == "__main__":
    main()
