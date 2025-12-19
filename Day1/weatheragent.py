import json
from openai import OpenAI

# 1. SETUP: Initialize the client pointing to OpenRouter
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="sk-or-v1-67e1514abf83a4095dc9f961a0483b44d0b5662614c5b26b0fef11b5e20a94356", 
)

# --- CONFIGURATION ---
# If this model ID fails, try one of these backups:
# 1. "meta-llama/llama-3.3-70b-instruct:free"
# 2. "mistralai/mistral-7b-instruct:free"
MODEL_ID = "google/gemini-2.0-flash-exp:free" 
# ---------------------

# 2. THE TOOL: Define a hard-coded Python function
def get_current_weather(location, unit="celsius"):
    """Get the current weather in a given location"""
    print(f" > [TOOL ACTIVITY] Checking weather for {location} in {unit}...")
    
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

# 3. THE SCHEMA
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# 4. THE EXECUTION
def run_conversation(user_query):
    print(f"USER: {user_query}")
    messages = [{"role": "user", "content": user_query}]

    # Step A: Send query + tool definitions
    print(f"... Thinking (Step A) using {MODEL_ID} ...")
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
        max_tokens=1000  # Safety limit
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step B: Check if the LLM decided to use the tool
    if tool_calls:
        print("AI: I need to use a tool to answer this.")
        
        # Step C: Execute the tool
        available_functions = {"get_current_weather": get_current_weather}
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            
            # Step D: Give the tool output BACK to the LLM
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
            
        # Step E: Get the final natural language answer
        print("... Synthesizing Answer (Step E) ...")
        final_response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=1000 # Safety limit
        )
        return final_response.choices[0].message.content
    else:
        return response_message.content

# Run it
print("-" * 50)
final_answer = run_conversation("What's the weather like in Tokyo right now?")
print(f"AI: {final_answer}")