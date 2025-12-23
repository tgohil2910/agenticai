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

# We go back to Gemini because we know the ID is correct, just busy.
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

# 2. HELPER FUNCTION: THE "RETRY" WRAPPER
# This attempts the API call up to 5 times if it hits a traffic jam (429)
def run_with_retry(func, *args, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg: # "Traffic Jam" error
                wait_time = (attempt + 1) * 5 # Wait 5s, then 10s, then 15s...
                print(f"   [Traffic Jam] Model busy. Waiting {wait_time}s to retry...")
                time.sleep(wait_time)
            else:
                raise e # If it's a real crash (not traffic), raise it.
    raise Exception("Max retries reached. The free model is too busy right now.")

# 3. RESEARCHER NODE
def researcher_node(state: AgentState):
    print("--- [1] RESEARCHER is working ---")
    last_message = state["messages"][-1]
    query = last_message.content
    print(f" > Searching for: {query}")
    
    # Search usually doesn't fail, but good to be safe
    try:
        search_result = search.invoke(query)
    except:
        search_result = "Search tool failed. No results found."
        
    return {"messages": [HumanMessage(content=f"HERE ARE THE FACTS I FOUND:\n{search_result}")]}

# 4. WRITER NODE
def writer_node(state: AgentState):
    print("--- [2] WRITER is working ---")
    messages = state["messages"]
    
    prompt = [
        SystemMessage(content="You are a senior tech journalist. Write a short, engaging blog post based ONLY on the facts provided below."),
        messages[-1] 
    ]
    
    print(" > Generating article (this might take a moment)...")
    
    # WE USE THE RETRY WRAPPER HERE
    response = run_with_retry(llm.invoke, prompt)
    
    return {"messages": [response]}

# 5. BUILD GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# 6. RUN
topic = "The release of GPT-5 rumors and news"
print(f"Topic: {topic}\n")

inputs = {"messages": [HumanMessage(content=topic)]}

final_article = ""
# We wrap the main loop in a try/except just in case
try:
    for output in app.stream(inputs):
        for key, value in output.items():
            if key == "writer":
                final_article = value["messages"][-1].content
except Exception as e:
    print(f"\nCRITICAL FAILURE: {e}")

print("\n" + "="*30)
print("FINAL ARTICLE")
print("="*30)
print(final_article)