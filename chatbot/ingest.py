import os
import pickle
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader

# 1) Load environment variables from .env
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and EMBEDDING_DEPLOYMENT):
    raise RuntimeError("Please set Azure OpenAI values in .env")


client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def extract_text_from_pdf(path: str) -> str:
    """Read a PDF and return all text."""
    reader = PdfReader(path)
    pages_text: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split long text into overlapping chunks.

    - max_chars: maximum characters per chunk
    - overlap: characters of overlap between consecutive chunks
    """
    cleaned = " ".join(text.split())  # collapse whitespace
    chunks: List[str] = []
    start = 0
    n = len(cleaned)

    if n == 0:
        return chunks

    while start < n:
        end = min(start + max_chars, n)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap
        if start >= n:
            break
    return chunks

def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Call Azure OpenAI embeddings API for a batch of texts.
    Returns a numpy array of shape (batch_size, embedding_dim).
    """
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_DEPLOYMENT,  # deployment name
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")



def build_index(
    pdf_folder: str = "data",
    embeddings_path: str = "embeddings.npy",
    meta_path: str = "chunks.pkl",
    batch_size: int = 16,
) -> None:
    """
    Walk through pdf_folder, extract & chunk text,
    compute embeddings, and save embeddings + metadata.
    """
    all_chunks: List[Dict] = []
    all_texts: List[str] = []

    # 3) Collect chunks from all PDFs
    
    for root, _, files in os.walk(pdf_folder):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(root, fname)
            print(f"[INFO] Reading PDF: {full_path}")

            raw_text = extract_text_from_pdf(full_path)
            chunks = chunk_text(raw_text)

            for chunk in chunks:
                if not chunk.strip():
                    continue
                all_chunks.append(
                    {
                        "text": chunk,
                        "source": fname,  # you can also store full_path
                    }
                )
                all_texts.append(chunk)

    print(f"[INFO] Total text chunks: {len(all_texts)}")

    if not all_texts:
        print("[WARN] No text found in PDFs. Nothing to index.")
        return

    # 4) Embed in batches
    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        print(f"[INFO] Embedding batch {i} – {i + len(batch) - 1}")
        emb = embed_batch(batch)
        all_embeddings.append(emb)

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"[INFO] Embeddings shape: {embeddings_matrix.shape}")

    # 5) Save embeddings + metadata to disk
    np.save(embeddings_path, embeddings_matrix)
    with open(meta_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[DONE] Saved {embeddings_matrix.shape[0]} embeddings to {embeddings_path}")
    print(f"[DONE] Saved metadata for {len(all_chunks)} chunks to {meta_path}")

if __name__ == "__main__":
    build_index()

