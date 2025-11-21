# app.py - Streamlit UI for RAG Chatbot (Azure OpenAI + Pinecone)
import os
from dotenv import load_dotenv

import streamlit as st

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env
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


@st.cache_resource(show_spinner=True)
def load_rag_pipeline():
    """Build and cache the RAG pipeline: embeddings + vector store + QA chain."""
    # 1. Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 2. Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    # 3. Vector store
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",  # must match metadata["text"] from ingest
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. LLM
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )

    # 5. RAG Chain


    prompt_template = """
    You are a question-answering assistant.

    You MUST follow these rules:

    1. Only answer using the information explicitly present in the retrieved context.
    2. If the answer is not fully supported by the context, you MUST say:
    "I don't know. The answer is not clearly available in the provided documents. Please check the PDFs."
    3. Do NOT use outside knowledge.
    4. Do NOT guess. Do NOT hallucinate.
    5. Be concise.

    ---------------------
    Context from PDFs:
    {context}
    ---------------------

    Question: {question}

    Answer:
    """

    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )


    return qa_chain


def main():
    st.set_page_config(
        page_title="GenAI RAG Bot",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 GenAI RAG Chatbot")
    st.markdown(
        "Ask questions based on your **PDF knowledge base** "
        "(ingested into Pinecone using Azure OpenAI embeddings)."
    )

    # Initialize session state for chat history
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {"role": "user"/"assistant", "content": "...", "context": [...]}

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("**Backend:** Azure OpenAI + Pinecone + LangChain")
        st.markdown(f"**Index:** `{PINECONE_INDEX}`")
        st.markdown("**Top K Chunks:** 5")

        if st.button("🧹 Clear Chat History"):
            st.session_state.history = []
            st.success("Chat history cleared.")

    # Load RAG pipeline (cached)
    with st.spinner("Loading RAG pipeline..."):
        qa_chain = load_rag_pipeline()

    # Chat input
    user_query = st.chat_input("Type your question here...")

    # Display previous conversation
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # If assistant and has context, show expandable context
            if msg["role"] == "assistant" and msg.get("context"):
                with st.expander("📚 Show retrieved context"):
                    for i, ctx in enumerate(msg["context"], start=1):
                        st.markdown(f"**Chunk {i}** — *{ctx['source']}*")
                        st.write(ctx["text"])
                        st.markdown("---")

    if user_query:
        # Show user message
        with st.chat_message("user"):
            st.write(user_query)

        # Run RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain({"query": user_query})

                answer = result["result"]
                sources = result["source_documents"]

                # Prepare context info
                ctx_list = []
                for doc in sources:
                    ctx_list.append(
                        {
                            "source": doc.metadata.get("source"),
                            "text": doc.page_content,
                        }
                    )

                # Display answer
                st.write(answer)

                # Display context nicely
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
