# chat_langchain.py
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA

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


def load_rag_pipeline():

    print("[1] Initializing Azure embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

    print("[2] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    print("[3] Creating vector store...")
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("[4] Loading Azure GPT-4o model...")
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    print("\n✨ Chatbot ready! Ask me anything about your PDFs.\n")
    return qa


if __name__ == "__main__":

    qa = load_rag_pipeline()

    while True:
        query = input("\n🔶 Your Question (type 'exit' to quit): ")

        if query.lower() in ["exit", "quit"]:
            break

        result = qa({"query": query})

        print("\n🔵 Answer:")
        print(result["result"])

        print("\n📚 Retrieved Context Chunks:")
        for i, doc in enumerate(result["source_documents"], start=1):
            print(f"\n----- Chunk {i} -----")
            print(f"Source PDF: {doc.metadata.get('source')}")
            print("Content:")
            print(doc.page_content)
            print("------------------------")
