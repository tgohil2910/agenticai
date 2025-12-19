import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode # We use the pre-built node now!

# --- CONFIGURATION ---
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "google/gemini-2.0-flash-exp:free"

# 1. SETUP THE REAL TOOL
# DuckDuckGo allows the agent to search the real web
search = DuckDuckGoSearchRun()

# We wrap it as a "Tool" for LangChain
@tool
def web_search(query: str):
    """Search the internet for real-time information."""
    return search.invoke(query)

tools = [web_search]

# 2. SETUP THE BRAIN
llm = ChatOpenAI(
    model=MODEL_ID,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0
)
# Bind the tool so the LLM knows it has internet access
llm_with_tools = llm.bind_tools(tools)

# 3. SETUP STATE
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. SETUP NODES
def chatbot(state: AgentState):
    """The Brain Node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Note: We don't need to write 'tool_executor' manually anymore. 
# LangGraph has a pre-built 'ToolNode' that handles the execution for us.
tool_node = ToolNode(tools)

# 5. SETUP LOGIC
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 6. BUILD GRAPH
workflow = StateGraph(AgentState)

workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node) # Using the pre-built node

workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # Loop back to brain after searching

app = workflow.compile()

# 7. RUN IT (Ask a REAL question)
# Try asking something that happened recently so you know it's not training data
question = "What was the score again?"
print(f"USER: {question}")
print("--- Agent Thinking ---")

inputs = {"messages": [HumanMessage(content=question)]}

final_answer = ""
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished Node: {key}")
        if key == "agent":
            # Keep track of the latest text response
            final_answer = value["messages"][-1].content

print("\n--- Final Answer ---")
print(final_answer)