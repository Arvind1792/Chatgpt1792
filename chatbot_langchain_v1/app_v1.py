# app.py - Streamlit UI for RAG Chatbot (LangChain v1 LCEL)
import os
from dotenv import load_dotenv

import streamlit as st

from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from cqc import should_use_exa
from exa_utils import run_exa_search

load_dotenv()

# Azure OpenAI keys
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

# Pinecone keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


def format_docs(docs):
    parts = []
    print(docs)
    for d in docs:
        src = d.metadata.get("source")
        txt = d.page_content or ""
        parts.append(f"[Source: {src}]\n{txt}")
    return "\n\n".join(parts)


@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def build_rag_chain():
    # 1) Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 2) Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # 3) Vector store + retriever
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4) Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a RAG assistant over a set of PDF documents.

Rules:
1. ONLY use the information in the provided context.
2. If you cannot answer fully from the context, say:
   "I don't know. The answer is not clearly available in the provided documents. Please check the PDFs."
3. Do not use external knowledge.
4. Do not hallucinate.

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer:
"""
    )

    # 5) LLM  ✅ we will return this
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )

    # 6) Context chain: retriever → format_docs
    context_chain = retriever | RunnableLambda(format_docs)

    # 7) LCEL RAG pipeline
    rag_chain = (
        RunnableMap(
            {
                "question": RunnablePassthrough(),
                "context": context_chain,
            }
        )
        | prompt
        | llm
    )

    # IMPORTANT: return llm as well
    return rag_chain, retriever, llm

def answer_query(query, rag_chain, retriever, llm):
    """
    Unified RAG + Exa fallback answer logic.
    Returns: (exa_context_or_None, answer_text, docs)
    """

    # Step 1: Retrieve from Pinecone
    docs = retriever.get_relevant_documents(query)

    # Step 2: Check context quality (CQC)
    use_exa = should_use_exa(query, docs)

    # Step 3: If Pinecone weak → Exa fallback
    if use_exa:
        exa_context = run_exa_search(query)

        final_prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer.

Context:
{exa_context}

Question:
{query}

If the answer is not clearly in the context, say:
"I don't know. The answer is not clearly available in the provided documents or web results."
"""

        resp = llm.invoke(final_prompt)
        answer_text = getattr(resp, "content", str(resp))

        return exa_context, answer_text, docs

    # Step 4: Pinecone context is OK → use normal RAG chain
    resp = rag_chain.invoke(query)
    answer_text = getattr(resp, "content", str(resp))

    return None, answer_text, docs
def main():
    st.set_page_config(page_title="GenAI RAG Bot", page_icon="🤖", layout="wide")

    st.title("🤖 GenAI RAG Chatbot (LangChain v1)")
    st.markdown(
        "Ask questions based on your **PDF knowledge base** "
        "(Azure OpenAI + Pinecone)."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown(f"**Pinecone Index:** `{PINECONE_INDEX}`")
        st.markdown("**Top K Chunks:** 5")
        if st.button("🧹 Clear Chat History"):
            st.session_state.history = []
            st.success("History cleared")

    with st.spinner("Loading RAG pipeline..."):
        rag_chain, retriever, llm = build_rag_chain()


    # Show history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("context"):
                with st.expander("📚 Retrieved Context"):
                    for i, ctx in enumerate(msg["context"], start=1):
                        st.markdown(f"**Chunk {i}** — *{ctx['source']}*")
                        st.write(ctx["text"])
                        st.markdown("---")

    user_query = st.chat_input("Type your question here...")

    if user_query:
        # Show user question
        with st.chat_message("user"):
            st.write(user_query)

        # Run RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                exa_used_context, answer, docs = answer_query(
                    user_query, rag_chain, retriever, llm
                )


                if exa_used_context:
                    st.warning("⚠️ Pinecone context weak → Used EXA AI fallback")
                answer = getattr(answer, "content", str(answer))

                # Get context docs
                docs = retriever.get_relevant_documents(user_query)
                # st.write(docs)
                ctx_list = [
                    {
                        "source": d.metadata.get("source"),
                        "text": d.page_content,
                    }
                    for d in docs
                ]

                st.write(answer)
                with st.expander("📚 Retrieved Context Chunks"):
                    for i, ctx in enumerate(ctx_list, start=1):
                        st.markdown(f"**Chunk {i}** — *{ctx['source']}*")
                        st.write(ctx["text"])
                        st.markdown("---")

                # Save to history
                st.session_state.history.append(
                    {"role": "user", "content": user_query}
                )
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "context": ctx_list,
                    }
                )


if __name__ == "__main__":
    main()
