# ingest_langchain.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Azure OpenAI keys
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# Pinecone keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


def load_pdfs(folder_path):
    """Load all PDFs and convert them to LangChain Document objects."""
    docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = file
            docs.extend(pdf_docs)
    return docs


def ingest_data():
    print("\n🔹 Step 1 - Loading PDFs...")
    documents = load_pdfs("data")
    print(f"[INFO] Loaded {len(documents)} pages from PDF folder.")

    print("\n🔹 Step 2 - Splitting into text chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Total chunks created: {len(chunks)}")

    print("\n🔹 Step 3 - Initializing Azure Embedding model...")
    embed_model = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    print("\n🔹 Step 4 - Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Auto-create Pinecone index if it doesn't exist
    if PINECONE_INDEX not in pc.list_indexes().names():
        print(f"[INFO] Creating Pinecone index: {PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=3072,  # required for text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX)

    print("\n🔹 Step 5 - Uploading embeddings to Pinecone using LangChain...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embed_model,
        index_name=PINECONE_INDEX
    )

    print("\n✅ DONE — All chunks uploaded to Pinecone successfully!")
    print(f"   → Index name: {PINECONE_INDEX}")


if __name__ == "__main__":
    ingest_data()
