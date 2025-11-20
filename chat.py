# chat.py
import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

# 1) Load env vars
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and EMBEDDING_DEPLOYMENT and CHAT_DEPLOYMENT):
    raise RuntimeError("Please set all Azure OpenAI values in .env")

# 2) Create Azure OpenAI client (same client used for embeddings + chat)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# 3) Load embeddings + metadata from disk
EMBEDDINGS_PATH = "embeddings.npy"
META_PATH = "chunks.pkl"

if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(META_PATH)):
    raise RuntimeError(
        "Embeddings not found. Run ingest.py first to generate embeddings and metadata."
    )

embeddings_matrix: np.ndarray = np.load(EMBEDDINGS_PATH).astype("float32")
with open(META_PATH, "rb") as f:
    chunks_meta: List[Dict] = pickle.load(f)

print(f"[INFO] Loaded embeddings: {embeddings_matrix.shape}")
print(f"[INFO] Loaded chunks metadata: {len(chunks_meta)} entries")

def embed_query(query: str) -> np.ndarray:
    """Get embedding vector for the user query."""
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_DEPLOYMENT,
    )
    vec = np.array(response.data[0].embedding, dtype="float32")
    return vec  # shape (dim,)

def cosine_similarities(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query_vec and each row in doc_matrix.
    Returns 1D array of scores.
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-10

    doc_norms = np.linalg.norm(doc_matrix, axis=1) + eps
    q_norm = np.linalg.norm(query_vec) + eps

    scores = (doc_matrix @ query_vec) / (doc_norms * q_norm)
    return scores  # higher = more similar

def search_similar_chunks(question: str, top_k: int = 5) -> List[Dict]:
    """Return top_k most similar chunks for the question."""
    q_vec = embed_query(question)
    scores = cosine_similarities(q_vec, embeddings_matrix)

    # Get indices of top_k scores
    top_idx = np.argsort(scores)[::-1][:top_k]

    results: List[Dict] = []
    for rank, idx in enumerate(top_idx, start=1):
        meta = chunks_meta[idx]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "text": meta["text"],
                "source": meta["source"],
            }
        )
    return results

def build_rag_prompt(question: str, contexts: List[Dict]) -> Tuple[str, str]:
    """
    Build system + user messages for GPT-4o using retrieved chunks.
    """
    # Concatenate context
    context_blocks = []
    for c in contexts:
        block = f"[Source {c['rank']} – {c['source']}]\n{c['text']}"
        context_blocks.append(block)
    context_text = "\n\n".join(context_blocks)

    system_message = (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n"
        "If the answer is not clearly in the context, say you don't know and suggest checking the PDFs.\n"
        "Always mention the source numbers (like Source 1, Source 2) that you used."
    )
    # print(context_text)
    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer strictly based on the context above."
    )

    return system_message, user_message

def answer_question(question: str) -> Tuple[str, List[Dict]]:
    """
    Full RAG pipeline for a single question:
    - retrieve similar chunks
    - call GPT-4o with context
    - return answer and used chunks
    """
    retrieved = search_similar_chunks(question, top_k=5)
    # print(retrieved)
    sys_msg, usr_msg = build_rag_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,  # deployment name for GPT-4o
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ],
        temperature=0.5,
    )

    answer = response.choices[0].message.content
    return answer, retrieved

if __name__ == "__main__":
    print("=== Local PDF RAG Bot (Azure OpenAI GPT-4o) ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input("Your question: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break

        answer, sources = answer_question(user_q)

        print("\n--- Answer ---")
        print(answer)

        print("\n--- Sources used ---")
        for s in sources:
            print(f"[Source {s['rank']}] {s['source']} (similarity = {s['score']:.4f}) {s['text']}")
            print("\n")
        print("\n")
