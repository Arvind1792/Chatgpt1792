# ingest_langchain.py (Optimized for Azure OpenAI + Pinecone)
import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --------------------------
# Azure OpenAI Configuration
# --------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# --------------------------
# Pinecone Configuration
# --------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
DIMENSION = 3072  # Use 1536 if using "text-embedding-3-small"

# --------------------------
# Load PDFs
# --------------------------
def load_pdfs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            full = os.path.join(folder_path, file)
            print(f"[PDF] Loading {file}")
            loader = PyPDFLoader(full)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = file
            docs.extend(pdf_docs)
    return docs


# --------------------------
# Retry Handler
# --------------------------
def embed_with_retry(embedding_model, texts, retries=5, delay=30):
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(texts)
        except Exception as e:
            if "RateLimit" in str(e) or "429" in str(e):
                wait = delay * (attempt + 1)
                print(f"⚠️ Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("❌ Could not embed after several retries")


# --------------------------
# Ingestion Pipeline
# --------------------------
def ingest_data():
    print("\n============================")
    print("📥 PDF → Pinecone Ingestion")
    print("============================\n")

    # 1. Load PDFs
    documents = load_pdfs("data")
    print(f"[INFO] Loaded {len(documents)} pages.")

    # 2. Split into text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"[INFO] Final chunk count: {len(chunks)}")

    # 3. Initialize Azure Embeddings
    embed_model = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 4. Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Auto-create index
    if PINECONE_INDEX not in pc.list_indexes().names():
        print(f"[PINECONE] Creating index {PINECONE_INDEX}...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX)

    # 5. Upload vectors in batches
    print("\n[UPLOAD] Uploading vectors...")
    batch_size = 50

    all_texts = [c.page_content for c in chunks]
    
    # all_metadata = [
    #     {"text": c.page_content, "source": c.metadata["source"]} for c in chunks
    # ]
    all_metadata = [
    {
        "text": c.page_content,      # the chunk text
        "source": c.metadata["source"]
    }
    for c in chunks
]


    for start in range(0, len(all_texts), batch_size):
        end = start + batch_size
        batch_texts = all_texts[start:end]
        batch_meta = all_metadata[start:end]

        print(f"➡ Embedding batch {start} → {end}")

        vectors = embed_with_retry(embed_model, batch_texts)
        ids = [f"vec-{i}" for i in range(start, end)]

        index.upsert(vectors=zip(ids, vectors, batch_meta))
        print(f"   ✔ Uploaded {len(ids)} vectors")

    print("\n✅ Ingestion Completed Successfully!\n")


if __name__ == "__main__":
    ingest_data()
