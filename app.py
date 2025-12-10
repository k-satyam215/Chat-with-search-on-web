import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Chat With Search"

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

# ---------- Premium CSS ----------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #020617 40%, #000000 100%);
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Main container width */
    .main > div {
        padding-top: 1.5rem;
    }

    /* Title spacing */
    h1 {
        font-weight: 700 !important;
        letter-spacing: 0.03em;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        padding: 0.25rem 0;
    }
    [data-testid="stChatMessage"] > div {
        border-radius: 18px;
        padding: 0.8rem 1rem;
        backdrop-filter: blur(16px);
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,64,175,0.7));
        border: 1px solid rgba(148,163,184,0.3);
        box-shadow: 0 18px 45px rgba(15,23,42,0.7);
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage-User"] > div {
        background: linear-gradient(135deg, rgba(59,130,246,0.95), rgba(37,99,235,0.9));
        border: 1px solid rgba(191,219,254,0.8);
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 999px !important;
        padding: 0.9rem 1.2rem !important;
        background: rgba(15,23,42,0.7) !important;
        border: 1px solid rgba(148,163,184,0.6) !important;
        color: #e5e7eb !important;
    }
    [data-testid="stChatInput"] button {
        border-radius: 999px !important;
        padding: 0.5rem 1.2rem !important;
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 30px rgba(22,163,74,0.5);
    }

    /* Sidebar glass card */
    section[data-testid="stSidebar"] > div {
        background: rgba(15,23,42,0.85);
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(51,65,85,0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Tools ----------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        placeholder="Enter your Groq API key",
        type="password",
        help="Your key is used only for this session and is not stored.",
    )
    st.markdown("---")
    st.caption("🔎 LangChain + Groq powered AI research agent.")

# ---------- Header ----------
st.markdown(
    """
    <div style="padding: 0.4rem 0 1.2rem 0;">
        <h1>🔎 AI Research Agent with Web Search</h1>
        <p style="color:#9ca3af; max-width:780px; font-size:0.95rem;">
            Ask any programming, AI, or research question and the agent will combine web search, Wikipedia, 
            and Arxiv papers to generate concise, well‑reasoned answers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Chat History ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I’m an AI research assistant. What would you like to explore today?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------- Chat Input ----------
prompt = st.chat_input("Type your question here...")

if prompt:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to start chatting.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
    )

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False,
        )
        response = search_agent.run(
            st.session_state.messages,
            callbacks=[st_cb],
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
