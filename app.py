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


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Groq API Key",
    placeholder="Enter your Groq API key",
    type="password",
    help="Your key is used only for this session and is not stored."
)

st.sidebar.markdown("---")
st.sidebar.caption("LangChain + Groq powered AI research agent.")

st.title("🔎 AI Research Agent with Web Search")
st.write(
    "Ask any programming, AI, or research question and the agent will search the web, "
    "Wikipedia, and Arxiv to generate a concise answer."
)

# --------- Chat History ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I’m an AI research assistant. How can I help you today?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

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
            expand_new_thoughts=False
        )
        response = search_agent.run(
            st.session_state.messages,
            callbacks=[st_cb]
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
