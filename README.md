# LangGraph + Tavily Web Search Summarizer (Streamlit + Hugging Face + LangSmith)

A simple Streamlit app that:
- Takes a user question
- Uses Tavily to search the web
- Summarizes results into very simple bullet points using a Hugging Face LLM
- Traces and monitors runs in LangSmith

## Tech Stack
- Streamlit (UI)
- LangGraph (workflow graph)
- LangChain (LLM + tool wiring)
- Tavily (web search)
- Hugging Face Inference (LLM)
- LangSmith (tracing/monitoring)

## Project Structure
├── app.py
├── requirements.txt
├── .env # NOT committed (contains keys)
└── .gitignor

## 1) Prerequisites
- Python 3.10+ recommended
- Tavily API key
- Hugging Face token
- LangSmith API key (optional but recommended for tracing)

## 2) Install (VS Code / Terminal)

### Create and activate venv

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

## 3) Install (VS Code / Terminal)
pip install -r requirements.txt

3) Environment Variables

Create a .env file in the project root:

# Tavily
TAVILY_API_KEY=YOUR_TAVILY_KEY

# Hugging Face
HUGGINGFACEHUB_API_TOKEN=YOUR_HF_TOKEN

# LangSmith (optional tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGSMITH_KEY
LANGCHAIN_PROJECT=langgraph-streamlit

4) Run the App
streamlit run app.py


Open the URL printed in your terminal (usually):

http://localhost:8501

5) LangSmith Tracing

If you set:

LANGCHAIN_TRACING_V2=true

LANGCHAIN_API_KEY=...

then your graph + tool + LLM calls will appear in LangSmith under:

LANGCHAIN_PROJECT