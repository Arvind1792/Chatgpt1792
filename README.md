# GenAI RAG Project with Azure OpenAI + Pinecone + LangChain

This project demonstrates a **production-grade Retrieval-Augmented Generation (RAG) system** using:

* **Azure OpenAI GPT-4o** for response generation
* **Azure OpenAI Embeddings (text-embedding-3-large)** for vector embeddings
* **Pinecone Cloud (v3)** as the vector database
* **LangChain** for document loading, chunking, embeddings, vector search, and orchestration

It loads PDFs from a local folder, ingests them into Pinecone, and allows interactive query answering using GPT-4o—grounded entirely in your documents.

---

## 🚀 Features

✔️ Fully cloud-based vector search using **Pinecone**
✔️ High-quality embeddings (3072-dim) using **Azure OpenAI**
✔️ Clean RAG architecture with **separate ingest & chat pipelines**
✔️ Context-aware answers with **source PDF references**
✔️ Automatic chunking of large PDF documents
✔️ Modern, non-deprecated LangChain integrations
✔️ Easy to extend (add UI, FastAPI, Streamlit, etc.)

---

## 📁 Project Structure

```
GenAiTask/
│
├── data/                   # Place your PDFs here
│     file1.pdf
│     file2.pdf
│
├── ingest_langchain.py     # Run once: preprocess + embed + upload to Pinecone
├── chat_langchain.py       # Run anytime: query the RAG chatbot
├── .env                    # Environment variables
├── README.md               # Project documentation
└── .venv/                  # Virtual environment (ignored by git)
```

---

## 🛠️ Installation

### 1. Clone the repository

```
 git clone <your_repo_url>
 cd GenAiTask
```

### 2. Create a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

If you do not have a requirements file yet, you can export:

```
pip freeze > requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root with the following:

```
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your key>
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_GPT4O_DEPLOYMENT=gpt-4o

# Pinecone
PINECONE_API_KEY=<your pinecone key>
PINECONE_INDEX=genai-rag-index
```

Make sure your Pinecone index dimension matches your embedding model:

* `text-embedding-3-large` → **3072 dimensions**
* `text-embedding-3-small` → **1536 dimensions**

---

## 📥 Step 1 — Ingest PDFs into Pinecone

Place all PDFs inside the `data/` folder.

Then run:

```
python ingest_langchain.py
```

This will:

* Load PDFs
* Convert to pages
* Chunk text
* Generate embeddings
* Upload vectors + metadata into Pinecone

You only run ingestion **once**, unless you add new PDFs.

---

## 💬 Step 2 — Chat with Your Knowledge Base

Run the chatbot:

```
python chat_langchain.py
```

Now you can ask:

```
"What does the report say about Indian stock market regulation?"
"Summarize the section about the film industry's economic impact."
```

The model will:

* Retrieve relevant chunks from Pinecone
* Generate an answer using GPT-4o
* Show the **exact text chunks** retrieved
* Show **source PDF names**

---

## 🧠 How It Works (Architecture)

```
    Local PDFs
        ↓
  PyPDF Loader
        ↓
  Text Splitter (1000 chars, 200 overlap)
        ↓
  Azure Embeddings (3072-dim)
        ↓
   Pinecone Index (cloud)
        ↓
   Retriever (k=5)
        ↓
   GPT-4o (Azure)
        ↓
    Final Answer + Sources
```

This ensures answers always come from your documents.

---

## 📌 Requirements

* Python 3.9+ recommended
* Azure OpenAI resource with GPT-4o + Embeddings deployed
* Pinecone Cloud account (Starter free tier is fine)
* 5–10 PDFs for testing

---

## 📦 Dependencies

going into your requirements.txt:

```
langchain
langchain-openai
langchain-community
langchain-pinecone
pinecone
pypdf
python-dotenv
```

---

