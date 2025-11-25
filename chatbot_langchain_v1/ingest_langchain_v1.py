# ingest_langchain_v1.py
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
DIMENSION = 3072  # For text-embedding-3-large


# --------------------------
# Load PDFs
# --------------------------
def load_pdfs(folder_path: str):
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

# AZURE_PDF_BASE_URL = os.getenv("AZURE_PDF_BASE_URL")
# def load_pdfs_from_azure(folder_path):
#     docs = []
#     for file in os.listdir(folder_path):
#         if file.lower().endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(folder_path, file))
#             pdf_pages = loader.load()

#             for p in pdf_pages:
#                 p.metadata["source"] = f"{AZURE_PDF_BASE_URL}/{file}"
            
#             docs.extend(pdf_pages)
    
#     return docs


# --------------------------
# Retry Handler for Embeddings
# --------------------------
def embed_with_retry(embedding_model, texts, retries: int = 5, delay: int = 15):
    """
    Embeds text safely with automatic retry on 429 rate limit errors.
    """
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(texts)

        except Exception as e:
            msg = str(e)
            if "RateLimit" in msg or "429" in msg:
                wait = delay * (attempt + 1)
                print(f"⚠️ Rate limit hit. Retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("❌ Could not embed after several retries.")


# --------------------------
# Ingestion Pipeline
# --------------------------
def ingest_data():
    print("\n============================")
    print("📥 PDF → Pinecone Ingestion")
    print("============================\n")

    # 1. Load PDFs
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_DIR = os.path.join(BASE_DIR, "..", "data")

    documents = load_pdfs(PDF_DIR)

    # documents = load_pdfs("data")
    print(f"[INFO] Loaded {len(documents)} pages.")

    # 2. Split into text chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"[INFO] Final chunk count: {len(chunks)}")

    # Assign chunk_id for inline citations
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = idx

    # 3. Initialize Azure Embeddings
    print("\n[EMBED] Initializing Azure Embeddings...")
    embed_model = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 4. Connect to Pinecone
    print("\n[PINECONE] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Auto-create index if needed
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX not in existing_indexes:
        print(f"[PINECONE] Creating index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX)
    print(f"[PINECONE] Using index: {PINECONE_INDEX}")

    # 5. Upload vectors in batches
    print("\n[UPLOAD] Uploading vectors in batches...")
    batch_size = 50

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch = chunks[start:end]

        batch_texts = [c.page_content for c in batch]
        batch_sources = [c.metadata.get("source") for c in batch]
        batch_chunk_ids = [c.metadata.get("chunk_id") for c in batch]

        print(f"   → Embedding batch {start} to {end}")

        vectors = embed_with_retry(embed_model, batch_texts)

        ids = [f"vec-{i}" for i in range(start, end)]

        metadata = [
            {
                "source": src,
                "text": txt,
                "chunk_id": cid,
            }
            for src, txt, cid in zip(batch_sources, batch_texts, batch_chunk_ids)
        ]

        index.upsert(
            vectors=[
                {"id": _id, "values": vec, "metadata": meta}
                for _id, vec, meta in zip(ids, vectors, metadata)
            ]
        )

        print(f"      ✔ Uploaded {len(ids)} vectors")

    print("\n🎉 DONE! All chunks uploaded to Pinecone.\n")


if __name__ == "__main__":
    ingest_data()
