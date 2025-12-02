
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


from summary_agent import summarize_pdf
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

def build_debug_trace(response) -> str:
    """
    Build a step-by-step debug trace from the agent response returned by create_agent.
    It shows:
    - user / assistant messages
    - tool calls (which tool + arguments)
    - tool results
    """
    lines = []

    if isinstance(response, dict) and "messages" in response:
        msgs = response["messages"]
    else:
        # If it's not the graph-style dict, just dump the raw object
        return f"Raw response (no messages list):\n{repr(response)}"

    for i, m in enumerate(msgs, start=1):
        # Try to get a role/type label
        role = getattr(m, "type", None) or getattr(m, "role", m.__class__.__name__)
        lines.append(f"Step {i}: [{role}]")

        # AIMessage with tool_calls → model decided to call a tool
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            lines.append("  → Tool call(s):")
            for tc in m.tool_calls:
                name = tc.get("name", "unknown_tool")
                args = tc.get("args", {})
                lines.append(f"    - Tool: {name}")
                lines.append(f"      Args: {args}")

        # ToolMessage → tool's result
        elif isinstance(m, ToolMessage):
            tool_name = getattr(m, "name", "unknown_tool")
            content = m.content
            # Shorten long content for the trace
            if isinstance(content, str) and len(content) > 400:
                content_preview = content[:400] + "... [truncated]"
            else:
                content_preview = content
            lines.append(f"  → Tool result from '{tool_name}':")
            lines.append(f"    {content_preview}")

        else:
            # Normal system/user/assistant messages
            content = getattr(m, "content", "")
            if isinstance(content, list):
                # Sometimes content can be a list of parts; join them
                content = " ".join(str(p) for p in content)
            if isinstance(content, str) and len(content) > 400:
                content_preview = content[:400] + "... [truncated]"
            else:
                content_preview = content
            lines.append(f"  Content: {content_preview}")

        lines.append("")  # blank line between steps

    return "\n".join(lines)



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
        azure_endpoint=os.getenv("AZURE_OPENAI_AGENT_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_AGENT_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_GPT4_1_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_AGENT_API_VERSION")
    )

    system_prompt = """
You are a smart PDF assistant. You MUST obey the tool-usage rules exactly as written below.

Your capabilities:
1. If the user asks to SUMMARIZE a PDF:
    - Identify the relevant PDF name.
    - Call the summarize_pdf(pdf_name) tool.
   

2. If the user asks ANY QUESTION about the content of the PDFs:
    - Call the ask_pdf_question(query) tool.
    
Rules:
1. You are allowed to use **ONLY ONE TOOL CALL** per user request.
   - Never call two tools.
   - Never combine tasks.
   - Never answer multi-intent questions.

2. You MUST decide which ONE tool to call:
   - If the user asks to **summarize a PDF**, call: summarize_pdf(pdf_name)
   - If the user asks a **question about PDF content**, call: ask_pdf_question(query)

3. If the user mixes multiple tasks in one message (example:
   "Summarize ext.pdf AND what is India’s GDP?")
   → You MUST NOT choose a tool.
   → You MUST reply asking the user to choose **ONE task only**.
   Example:
   "You asked for two different tasks. I can only perform one tool action at a time. Do you want a PDF summary or a PDF question?"

4. You MUST NOT answer ANYTHING using your own knowledge.
   - NO reasoning.
   - NO explanations.
   - NO extra text.
   - NO combining tool output with your own text.

5. When a tool returns output:
   - Your final answer MUST be **exactly and only** the tool output.
   - Do NOT paraphrase.
   - Do NOT summarize.
   - Do NOT add extra words.

6. NEVER hallucinate.
7. NEVER generate citations on your own. Only pass through citations returned by tools.
8. NEVER remove citations from tool output.
9. NEVER rewrite, clean, or correct tool output.
10. If the user does not specify which PDF for a summary, ask:
    "Please specify the PDF name you want summarized."

-----------------------------------------
🎯 BEHAVIOR SUMMARY
-----------------------------------------
- ALWAYS choose exactly ONE tool.
- If multiple tasks appear → ask user to choose ONE.
- Final answer MUST equal tool output exactly.
- No extra words, no creativity, no combination.
-----------------------------------------

"""

    agent = create_agent(
        model=llm,
        tools=[summarize_pdf, ask_pdf_question],
        system_prompt=system_prompt,
    )

    return agent

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
    print(user_query)
    if user_query:
        # Store user message
        st.session_state.history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.write(user_query)

        # ASSISTANT
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                agent = get_smart_pdf_agent()

                # You can also pass a config if you want later
                response = agent.invoke({
                    "messages": [{"role": "user", "content": user_query}]
                })

                # 🔹 Build debug trace before extracting final answer
                debug_trace = build_debug_trace(response)

                final_answer = extract_agent_output(response)
                final_answer_clickable = make_citations_clickable(final_answer)

                st.markdown(final_answer_clickable, unsafe_allow_html=True)

                # 🔹 Show agent internal steps
                with st.expander("🛠 Agent debug trace"):
                    st.text(debug_trace)

                # You can still wire real pdf_sources later if you return them from tools
                pdf_sources = []

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
                        "content": final_answer_clickable,
                        "sources": pdf_sources,
                        "debug_trace": debug_trace,
                    }
                )


if __name__ == "__main__":
    main()
