import os
import time
from dotenv import load_dotenv 
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. SETUP
load_dotenv() 
api_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "google/gemini-2.0-flash-exp:free"

llm = ChatOpenAI(
    model=MODEL_ID,
    openai_api_key=api_key,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0
)

search = DuckDuckGoSearchRun()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- RETRY LOGIC ---
def run_with_retry(func, *args, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 5
                print(f"   [Traffic] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries reached.")

# --- NODES ---

def researcher_node(state: AgentState):
    print("\n--- [1] RESEARCHER ---")
    query = state["messages"][-1].content
    print(f" > Searching: {query}")
    try:
        res = search.invoke(query)
    except:
        res = "No results."
    return {"messages": [HumanMessage(content=f"FACTS:\n{res}")]}

def writer_node(state: AgentState):
    print("\n--- [2] WRITER ---")
    # We deliberately ask for a LONG article to trigger the Editor later
    prompt = [
        SystemMessage(content="Write a very detailed, comprehensive blog post (at least 300 words) about these facts."),
        state["messages"][-1] 
    ]
    response = run_with_retry(llm.invoke, prompt)
    return {"messages": [response]}

def editor_node(state: AgentState):
    print("\n--- [3] EDITOR (Quality Control) ---")
    print(" > The article was too long. Condensing it now...")
    last_message = state["messages"][-1]
    
    prompt = [
        SystemMessage(content="You are an Editor. The following article is too long. Summarize it into a punchy 100-word version."),
        last_message
    ]
    response = run_with_retry(llm.invoke, prompt)
    return {"messages": [response]}

# --- THE ROUTER (The Logic) ---
def quality_control(state: AgentState):
    """Checks the word count of the writer's draft."""
    last_message = state["messages"][-1]
    content = last_message.content
    word_count = len(content.split())
    
    print(f"   [Check] Word count: {word_count}")
    
    if word_count > 200:
        return "editor" # Too long? Go to Editor
    else:
        return END      # Good? Finish

# --- GRAPH BUILD ---
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")

# CONDITIONAL EDGE: From Writer, we don't go to END. We go to the Router.
workflow.add_conditional_edges(
    "writer",
    quality_control, # The function to decide
    {
        "editor": "editor", # Map return value 'editor' -> editor node
        END: END            # Map return value END -> End
    }
)

workflow.add_edge("editor", END)

app = workflow.compile()

# --- RUN ---
topic = "The history of the NVIDIA GPU architecture"
print(f"Request: {topic}")
inputs = {"messages": [HumanMessage(content=topic)]}

final_output = ""
try:
    for output in app.stream(inputs):
        for key, value in output.items():
            # We want the LAST message from the LAST node that ran
            if value.get("messages"):
                final_output = value["messages"][-1].content
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*30)
print("FINAL PUBLICATION")
print("="*30)
print(final_output)