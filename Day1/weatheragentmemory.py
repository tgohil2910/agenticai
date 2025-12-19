import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver # <--- NEW IMPORT

# --- CONFIGURATION ---
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "google/gemini-2.0-flash-exp:free"

# 1. SETUP TOOLS & BRAIN
search = DuckDuckGoSearchRun()
tools = [search]

llm = ChatOpenAI(
    model=MODEL_ID,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# 2. SETUP GRAPH
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# --- MEMORY UPGRADE ---
memory = MemorySaver() # <--- Initialize Memory
app = workflow.compile(checkpointer=memory) # <--- Attach it here!

# 3. RUN IT (With a "Thread ID")
# The 'thread_id' is like a User Session ID. 
# As long as you use the same ID, the agent remembers you.
config = {"configurable": {"thread_id": "user_123"}}

print("--- Turn 1: Teaching the Agent ---")
input1 = {"messages": [HumanMessage(content="Hi, my name is Neo.")]}
for update in app.stream(input1, config=config):
    for key, value in update.items():
        if key == "agent":
             print(f"Agent: {value['messages'][-1].content}")

print("\n--- Turn 2: Testing Memory ---")
# Notice we don't repeat the name here. We just ask "What is my name?"
input2 = {"messages": [HumanMessage(content="What is my name?")]}
for update in app.stream(input2, config=config):
    for key, value in update.items():
        if key == "agent":
             print(f"Agent: {value['messages'][-1].content}")