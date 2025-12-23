import streamlit as st
import os
import time
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="The AI Newsroom", page_icon="📰")
st.title("🤖 The AI Newsroom Agent")
st.markdown("Enter a topic, and I will **Research**, **Write**, and **Edit** a report for you.")

# --- 2. SETUP SECRETS ---
# Try to load from .env, but allow user to input key if missing
load_dotenv()
default_key = os.getenv("OPENROUTER_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
    model_id = st.selectbox("Select Model", [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3-8b-instruct:free",
        "microsoft/phi-3-medium-128k-instruct:free"
    ])
    
    if not api_key:
        st.warning("Please enter your API Key to proceed.")
        st.stop()

# --- 3. DEFINE THE AGENT (Cached) ---
# We use @st.cache_resource so we don't rebuild the graph on every button click
@st.cache_resource
def build_graph(api_key, model):
    
    # Setup LLM
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    search = DuckDuckGoSearchRun()

    # Define State
    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    # Retry Helper
    def run_with_retry(func, *args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                time.sleep(2)
        return "Error: API Busy"

    # Nodes
    def researcher_node(state: AgentState):
        query = state["messages"][-1].content
        try:
            res = search.invoke(query)
        except:
            res = "No results found."
        return {"messages": [HumanMessage(content=f"FACTS:\n{res}")]}

    def writer_node(state: AgentState):
        prompt = [
            SystemMessage(content="You are a Journalist. Write a 200-word article based on these facts."),
            state["messages"][-1]
        ]
        response = run_with_retry(llm.invoke, prompt)
        # Handle case where retry failed
        if isinstance(response, str): 
            return {"messages": [AIMessage(content="I apologize, the AI service is currently unavailable.")]}
        return {"messages": [response]}

    # Build Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()

# Build the app
app = build_graph(api_key, model_id)

# --- 4. CHAT INTERFACE ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("What should I research today?"):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent
    with st.chat_message("assistant"):
        status_container = st.status("🧠 Agent is thinking...", expanded=True)
        
        try:
            inputs = {"messages": [HumanMessage(content=prompt)]}
            final_response = ""
            
            # Stream the events
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Update status box
                    status_container.write(f"✅ Finished step: **{key}**")
                    
                    if key == "researcher":
                        status_container.markdown(f"Found search data...")
                    
                    if key == "writer":
                        final_response = value["messages"][-1].content

            status_container.update(label="Done!", state="complete", expanded=False)
            
            # Show Final Answer
            st.markdown("### 📰 Your Report")
            st.markdown(final_response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")