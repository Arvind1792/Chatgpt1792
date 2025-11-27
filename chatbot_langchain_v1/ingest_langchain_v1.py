import os
import time
import tempfile
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ------------------------------
# Azure Storage Configuration
# ------------------------------
AZURE_STORAGE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")
AZURE_PDF_BASE_URL = os.getenv("AZURE_PDF_BASE_URL")   # https://<acct>.blob.core.windows.net/<container>

# ------------------------------
# Pinecone Configuration
# ------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
DIMENSION = 3072  # text-embedding-3-large


# ------------------------------
# Load PDFs from Azure Blob
# ------------------------------
def load_pdfs_from_azure():
    print("\n🔗 Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)
    container = blob_service_client.get_container_client(AZURE_CONTAINER)

    pdf_files = [b.name for b in container.list_blobs() if b.name.lower().endswith(".pdf")]
    print(f"📄 Found PDFs in Azure: {pdf_files}")

    docs = []

    for filename in pdf_files:
        print(f"⬇ Downloading: {filename}")
        blob_client = container.get_blob_client(filename)

        # Download bytes
        data = blob_client.download_blob().readall()

        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Load using PyPDFLoader (requires file path)
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        for p in pages:
            p.metadata["source"] = f"{AZURE_PDF_BASE_URL}/{filename}"

        docs.extend(pages)

    return docs


# ------------------------------
# Retry Wrapper for Embeddings
# ------------------------------
def embed_with_retry(model, texts, retries=5, delay=15):
    for attempt in range(retries):
        try:
            return model.embed_documents(texts)
        except Exception as e:
            if "429" in str(e) or "RateLimit" in str(e):
                wait = delay * (attempt + 1)
                print(f"⚠️ Rate limit hit. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("❌ Embedding failed after retries")


# ------------------------------
# Ingest Pipeline
# ------------------------------
def ingest_data():
    print("\n============================")
    print("📥 AZURE PDF → PINECONE")
    print("============================\n")

    # Step 1: Load PDFs
    documents = load_pdfs_from_azure()
    print(f"[INFO] Loaded {len(documents)} pages.")

    # Step 2: Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"[INFO] Produced {len(chunks)} chunks.")

    # Add chunk_id for inline citation formatting
    for idx, c in enumerate(chunks, start=1):
        c.metadata["chunk_id"] = idx

    # Step 3: Azure embeddings
    print("🧠 Initializing Azure embeddings...")
    embed_model = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    # Step 4: Pinecone
    print("🔗 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Auto-create index
    if PINECONE_INDEX not in pc.list_indexes().names():
        print(f"🟢 Creating index {PINECONE_INDEX}...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX)
    print(f"🟦 Pinecone index ready: {PINECONE_INDEX}")

    # Step 5: Upload chunks in batches
    batch_size = 40
    print("\n⬆ Uploading vectors to Pinecone...\n")

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch = chunks[start:end]

        # Required fields for retriever
        texts = [c.page_content for c in batch]
        sources = [c.metadata.get("source") for c in batch]
        chunk_ids = [c.metadata.get("chunk_id") for c in batch]

        # Embed batch
        print(f"➡ Embedding batch {start}–{end}")
        vectors = embed_with_retry(embed_model, texts)

        # Vector IDs
        ids = [f"vec-{i}" for i in range(start, end)]

        # FIXED: Pinecone metadata structure MUST contain "text"
        metadata = [
            {
                "text": txt,              # required
                "source": src,            # azure PDF URL
                "chunk_id": cid,
            }
            for txt, src, cid in zip(texts, sources, chunk_ids)
        ]

        # Upload to Pinecone
        index.upsert(
            vectors=[
                {"id": _id, "values": vec, "metadata": meta}
                for _id, vec, meta in zip(ids, vectors, metadata)
            ]
        )

        print(f"   ✔ Uploaded {len(ids)} vectors")

    print("\n🎉 DONE! All Azure PDF chunks embedded & stored in Pinecone.\n")

if __name__ == "__main__":
    ingest_data()
