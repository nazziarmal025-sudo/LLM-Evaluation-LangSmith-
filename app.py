from __future__ import annotations

from typing import TypedDict, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch  # ✅ NEW
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


# -------------------------
# 1) Define the graph state
# -------------------------
class AppState(TypedDict, total=False):
    question: str
    search_results: List[Dict[str, Any]]
    answer: str


# -------------------------
# 2) Tools + HF model
# -------------------------
search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

HF_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

hf_llm = HuggingFaceEndpoint(
    repo_id=HF_REPO_ID,
    task="text-generation",
    max_new_tokens=350,
    do_sample=False,
    provider="auto",
)
llm = ChatHuggingFace(llm=hf_llm)


# -------------------------
# 3) Graph nodes
# -------------------------
def web_search_node(state: AppState) -> AppState:
    q = state["question"]

    # ✅ New TavilySearch expects {"query": "..."}
    resp = search_tool.invoke({"query": q})

    # resp usually looks like {"query":..., "results":[...], ...}
    results = resp.get("results", [])
    return {"search_results": results}


def summarize_node(state: AppState) -> AppState:
    q = state["question"]
    results = state.get("search_results", [])

    compact = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content"),
        }
        for r in results
    ]

    prompt = f"""
You are a helpful assistant.

User question:
{q}

Web search snippets:
{compact}

Write the final answer in this exact format:

- Bullet 1 (very simple)
- Bullet 2 (very simple)
- Bullet 3 (very simple)
- Bullet 4 (optional)

Explanation (1-2 short lines, very simple):
<your explanation>

Sources:
- <url>
- <url>
"""

    msg = llm.invoke(prompt)
    return {"answer": msg.content}


# -------------------------
# 4) Build the LangGraph
# -------------------------
builder = StateGraph(AppState)
builder.add_node("web_search", web_search_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "web_search")
builder.add_edge("web_search", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# -------------------------
# 5) Streamlit UI
# -------------------------
st.set_page_config(page_title="LangGraph + Tavily + HF", layout="wide")
st.title("🔎 Web Search Summarizer (LangGraph + LangSmith)")

question = st.text_input("Ask a question", placeholder="e.g. What is the EU AI Act?")

if st.button("Search & Summarize"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the web and summarizing..."):
            result = graph.invoke({"question": question.strip()})

        st.subheader("Answer")
        st.markdown(result["answer"])

        with st.expander("🔍 Raw Tavily results"):
            st.json(result.get("search_results", []))
