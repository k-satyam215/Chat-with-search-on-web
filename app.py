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

st.markdown(
    """
    <style>
    /* Warm gradient background */
    .stApp {
        background: radial-gradient(circle at top left, #3b1f2b 0, #120b18 40%, #050308 100%);
        color: #f5f5f4;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Main container padding */
    .main > div {
        padding-top: 1.5rem;
    }

    h1 {
        font-weight: 700 !important;
        letter-spacing: 0.04em;
    }

    /* Chat bubbles: glass + warm accent */
    [data-testid="stChatMessage"] {
        padding: 0.25rem 0;
    }
    [data-testid="stChatMessage"] > div {
        border-radius: 18px;
        padding: 0.85rem 1rem;
        backdrop-filter: blur(18px);
        background: linear-gradient(135deg, rgba(24,16,32,0.92), rgba(58,34,55,0.85));
        border: 1px solid rgba(248, 250, 252, 0.06);
        box-shadow: 0 18px 45px rgba(15,15,23,0.85);
    }
    /* User bubble: soft amber + rose */
    [data-testid="stChatMessage"][data-testid="stChatMessage-User"] > div {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.96), rgba(244, 114, 182, 0.95));
        border: 1px solid rgba(254, 243, 199, 0.9);
        color: #111827;
        font-weight: 500;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 999px !important;
        padding: 0.95rem 1.3rem !important;
        background: rgba(15,10,20,0.92) !important;
        border: 1px solid rgba(148, 132, 178, 0.7) !important;
        color: #f5f5f4 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #a1a1aa !important;
    }
    [data-testid="stChatInput"] button {
        border-radius: 999px !important;
        padding: 0.55rem 1.4rem !important;
        background: linear-gradient(135deg, #f97316, #ec4899) !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 30px rgba(236, 72, 153, 0.55);
    }

    /* Sidebar glass card */
    section[data-testid="stSidebar"] > div {
        background: rgba(13, 10, 18, 0.96);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(55, 48, 71, 0.9);
    }

    /* Sidebar text tweaks */
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
    }

    /* Subheader / description text */
    .description-text {
        color: #d4d4d8;
        font-size: 0.95rem;
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
        <p class="description-text" style="max-width:780px;">
            Ask any programming, AI, or research question and the agent will combine web search, Wikipedia,
            and Arxiv papers to give focused, research‑grade answers.
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
