import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- CONFIGURATION ---
# We use the same OpenRouter credentials
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "google/gemini-2.0-flash-exp:free"

# 1. SETUP THE BRAIN
# We use LangChain's ChatOpenAI but point it to OpenRouter
llm = ChatOpenAI(
    model=MODEL_ID,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0
)

# 2. DEFINE THE TOOL
# The @tool decorator handles all that JSON schema stuff for you automatically!
@tool
def get_weather(location: str):
    """Call to get the current weather."""
    # Mock data
    if "tokyo" in location.lower():
        return "10 degrees Celsius"
    elif "ny" in location.lower():
        return "22 degrees Celsius"
    return "Unknown weather location"

# Bind the tool to the model so the LLM knows it exists
llm_with_tools = llm.bind_tools([get_weather])

# 3. DEFINE THE STATE (The "Memory")
# This tracks the conversation history automatically
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. DEFINE THE NODES (The "Workers")

def chatbot(state: AgentState):
    """The Brain Node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_executor(state: AgentState):
    """The Tool Node (Runs the tool)"""
    # LangGraph usually has a pre-built 'ToolNode', but let's build it manually 
    # to understand what's happening.
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    results = []
    for t in tool_calls:
        print(f" > [GRAPH ACTIVITY] Calling Tool: {t['name']}")
        # This finds the function '@tool' we defined earlier and runs it
        if t['name'] == "get_weather":
            output = get_weather.invoke(t['args'])
            
            # We must return a ToolMessage back to the graph
            from langchain_core.messages import ToolMessage
            results.append(ToolMessage(tool_call_id=t['id'], content=output))
            
    return {"messages": results}

# 5. DEFINE THE LOGIC (The "Manager")
def should_continue(state: AgentState):
    """Decides: Go to tools? Or end?"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 6. BUILD THE GRAPH
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_executor)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent") # LOOP: After tools, go back to agent!

# Compile
app = workflow.compile()

# 7. RUN IT
print("--- Starting LangGraph Agent ---")
inputs = {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}

final_response = None

# Stream the updates
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished Node: {key}")
        # Capture the final response from the agent node
        if key == "agent":
            final_response = value["messages"][-1].content

print("\n--- Final Result ---")
print(final_response)