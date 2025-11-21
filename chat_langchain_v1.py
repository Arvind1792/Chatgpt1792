# chat_langchain_v1.py
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from langchain_pinecone import PineconeVectorStore

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda

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
    """Convert retrieved docs into a single context string."""
    parts = []
    for d in docs:
        src = d.metadata.get("source")
        txt = d.metadata.get("text") or ""
        parts.append(f"[Source: {src}]\n{txt}")
    return "\n\n".join(parts)


def build_rag_chain():
    """Build LCEL-based RAG pipeline (retriever + prompt + LLM)."""

    # 1) Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 2) Pinecone + VectorStore + Retriever
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",  # must match metadata["text"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print(retriever)
    # 3) Strict RAG Prompt – NO hallucinations
    rag_prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant using Retrieval-Augmented Generation (RAG).

You MUST follow these rules:

1. ONLY use the information explicitly present in the context below.
2. If the answer is not clearly and fully supported by the context,
   you MUST reply exactly:
   "I don't know. The answer is not clearly available in the provided documents. Please check the PDFs."
3. Do NOT use outside knowledge.
4. Do NOT guess or hallucinate.
5. Answer concisely.

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer:
"""
    )

    # 4) Azure GPT-4o LLM
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.5,
    )

    # 5) Context chain: retriever → format_docs
    context_chain = retriever | RunnableLambda(format_docs)

    # 6) LCEL RAG pipeline
    rag_chain = (
        RunnableMap(
            {
                "question": RunnablePassthrough(),
                "context": context_chain,
            }
        )
        | rag_prompt
        | llm
    )

    return rag_chain, retriever


def main():
    print("🚀 Loading RAG pipeline (Azure OpenAI + Pinecone + LangChain v1)...")
    rag_chain, retriever = build_rag_chain()
    print("✨ Chatbot ready! Type your question, or 'exit' to quit.\n")

    while True:
        query = input("\n🔶 Your Question: ")
        if query.strip().lower() in ("exit", "quit"):
            print("👋 Bye!")
            break

        # Invoke chain → returns a ChatMessage
        answer_message = rag_chain.invoke(query)
        answer_text = getattr(answer_message, "content", str(answer_message))

        # Also fetch retrieved documents to show context
        docs = retriever.get_relevant_documents(query)

        print("\n🔵 ANSWER:")
        print(answer_text)

        print("\n📚 RETRIEVED CONTEXT CHUNKS:")
        if not docs:
            print("No documents retrieved from Pinecone.")
        else:
            for i, doc in enumerate(docs, start=1):
                print(f"\n----- Chunk {i} -----")
                print(f"📄 Source PDF: {doc.metadata.get('source')}")
                print("🧩 Content:")
                print(doc.metadata.get("text"))
                print("------------------------")


if __name__ == "__main__":
    main()
