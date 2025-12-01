# app_v1.py - Streamlit UI for RAG + Exa fallback with inline numeric citations

import os
import streamlit as st
from dotenv import load_dotenv

# from summary_agent import get_medium_summary_agent
# from azure.storage.blob import BlobServiceClient
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

from langchain_core.messages import BaseMessage

from summary_agent import summarize_pdf_medium
from qna import ask_pdf_question

import re

def extract_numbered_sources_from_answer(answer: str):
    """
    Extracts:
    [1] <https://url.com>
    [2] <https://url.com>
    from the 'Sources Used:' block in the answer.
    Returns a list: [{"index": 1, "source": "https://url.com"}, ...]
    """
    pattern = r'\[(\d+)\]\s*<([^>]+)>'
    matches = re.findall(pattern, answer)

    sources = []
    for idx, url in matches:
        sources.append({"index": int(idx), "source": url})
    return sources


def make_citations_clickable(answer: str) -> str:
    pdf_sources = extract_numbered_sources_from_answer(answer)

    for src in pdf_sources:
        idx = str(src["index"])
        url = src["source"]

        # Match [1], [ 1 ], [1 ]
        pattern = r'\[\s*' + re.escape(idx) + r'\s*\]'
        replacement = f'<a href="{url}" target="_blank">[{idx}]</a>'

        answer = re.sub(pattern, replacement, answer)

    return answer

def extract_agent_output(response):
    """
    Handles all possible output types from create_agent.invoke().
    Supports: AIMessage, BaseMessage, dict with 'messages', string.
    """
    # Case 1: AIMessage / BaseMessage
    if isinstance(response, BaseMessage):
        return response.content

    # Case 2: dict with messages[] (graph-style)
    if isinstance(response, dict):
        # If result has messages list
        if "messages" in response and isinstance(response["messages"], list):
            msgs = response["messages"]
            if msgs:
                last = msgs[-1]
                # last might be BaseMessage or dict
                return getattr(last, "content", str(last))

        # Generic fallback
        return str(response)

    # Case 3: raw string
    if isinstance(response, str):
        return response

    # Final fallback
    return str(response)



def get_smart_pdf_agent():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.2,
    )

    system_prompt = """
You are a smart PDF assistant.

Your capabilities:
1. If the user asks to SUMMARIZE a PDF:
    - Identify the relevant PDF name.
    - Call the summarize_pdf_medium(pdf_name) tool.

2. If the user asks ANY QUESTION about the content of the PDFs:
    - Call the ask_pdf_question(query) tool.

Rules:
- NEVER hallucinate.
- NEVER rewrite or paraphrase tool output.
- NEVER summarize tool output.
- NEVER remove citations like [1], [2], [Source: URL], etc.
- NEVER generate your own explanation instead of the tool output.
- When a tool returns a result, your final answer MUST be identical to the tool output.
- If you are unsure whether user wants summary or QnA, ASK the user.
- ALWAYS use the correct tool instead of answering directly.
"""

    agent = create_agent(
        model=llm,
        tools=[summarize_pdf_medium, ask_pdf_question],
        system_prompt=system_prompt,
    )

    return agent

# ---------------------------
# RAG building
# ---------------------------
# @st.cache_resource(show_spinner=True)

# ---------------------------
# Streamlit UI
# ---------------------------
# def main():
#     st.set_page_config(
#         page_title="GenAI RAG Bot (PDF + Exa)",
#         page_icon="🤖",
#         layout="wide",
#     )

#     st.title("🤖 GenAI RAG Chatbot (PDF + Exa with Inline Citations)")

#     if "history" not in st.session_state:
#         st.session_state.history = []

#     with st.sidebar:
#         st.header("⚙️ Settings")
#         st.markdown(f"**Pinecone Index:** `{PINECONE_INDEX}`")
#         if st.button("🧹 Clear Chat History"):
#             st.session_state.history = []
#             st.rerun()



#     # Replay chat history
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])
#     # User query
#     user_query = st.chat_input("Ask something...")
   
#     if user_query:
#         st.session_state.history.append({"role": "user", "content": user_query})
#         with st.chat_message("user"):
#             st.write(user_query)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 agent = get_smart_pdf_agent()

#                 response = agent.invoke({
#                     "messages": [
#                         {"role": "user", "content": user_query}
#                     ]
#                 })
#                 print("Agent response:", response)
#                 answer = extract_agent_output(response)

#                 st.write(answer)
#                 with st.expander("📁 PDF Sources"):
#                     for s in pdf_sources:
#                          st.markdown(f"[{s['index']}] ➜ [{s['source']}]({s['source']})")

#                 st.session_state.history.append(
#                     {
#                         "role": "assistant",
#                         "content": answer,
#                         "sources": answer,
#                     }
#                 )

def main():
    st.set_page_config(
        page_title="GenAI RAG Bot (PDF + Exa)",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Smart Agent ")

    if "history" not in st.session_state:
        st.session_state.history = []

    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown(f"**Pinecone Index:** `{PINECONE_INDEX}`")
        if st.button("🧹 Clear Chat History"):
            st.session_state.history = []
            st.rerun()

    # REPLAY HISTORY
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

            # show sources if present
            sources = msg.get("sources", [])
            if sources:
                with st.expander("📁 PDF Sources"):
                    for s in sources:
                        st.markdown(
                            f"[{s['index']}] ➜ [{s['source']}]({s['source']})",
                            unsafe_allow_html=True
                        )

    # USER INPUT
    user_query = st.chat_input("Ask something...")

    if user_query:
        # Store user message
        st.session_state.history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.write(user_query)

        # ASSISTANT
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                agent = get_smart_pdf_agent()

                response = agent.invoke({
                    "messages": [{"role": "user", "content": user_query}]
                })


                # Extract PDF sources if tool returned them
                tool_messages = response.get("messages", [])
                pdf_sources = []

                for m in tool_messages:
                    if hasattr(m, "tool_calls"):
                        continue
                    if m.__class__.__name__ == "ToolMessage":
                        # Parse sources from tool message metadata or content
                        # Modify this if your tool returns structured sources
                        pass

                # # DISPLAY ANSWER (markdown so [1] becomes clickable)
                # st.markdown(final_answer, unsafe_allow_html=True)
                final_answer = extract_agent_output(response)
                final_answer_clickable = make_citations_clickable(final_answer)

                st.markdown(final_answer_clickable, unsafe_allow_html=True)

                # DISPLAY SOURCE LINKS
                if pdf_sources:
                    with st.expander("📁 PDF Sources"):
                        for s in pdf_sources:
                            st.markdown(
                                f"[{s['index']}] ➜ [{s['source']}]({s['source']})",
                                unsafe_allow_html=True
                            )

                # Save in chat history
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": final_answer,
                        "sources": pdf_sources
                    }
                )


if __name__ == "__main__":
    main()



    # # ======================================================
    # # PDF Summarizer Section (New)
    # # ======================================================
    # st.subheader("📄 PDF Summarizer (Medium Summary)")



    # def list_pdfs_in_azure():
    #     conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    #     container = os.getenv("AZURE_CONTAINER")
    #     service = BlobServiceClient.from_connection_string(conn)
    #     container_client = service.get_container_client(container)
    #     return [b.name for b in container_client.list_blobs() if b.name.lower().endswith(".pdf")]

    # pdf_files = list_pdfs_in_azure()

    # if pdf_files:
    #     selected_pdf = st.selectbox("Choose PDF to summarize:", pdf_files)

    #     if st.button("Summarize Selected PDF"):
    #         st.info(f"Summarizing: {selected_pdf}")

    #         agent = get_medium_summary_agent()
    #         response = agent.invoke({
    #             "messages": [
    #                 {"role": "user", "content": f"Summarize {selected_pdf}"}
    #             ]
    #         })

    #         st.success("Summary:")
    #         summary_text = extract_agent_output(response)
    #         st.write(summary_text)
    # else:
    #     st.warning("No PDFs found in Azure Blob Storage")
