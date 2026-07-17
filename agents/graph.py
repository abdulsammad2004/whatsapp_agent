import os
from dotenv import load_dotenv
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from agents.state import CinemaAgentState
from agents.router import conditional_router
from agents.tools import fetch_movies, fetch_showtimes

load_dotenv()

# =====================================================================
# 1. INITIALIZE THE DUAL-BRAIN ENGINES
# =====================================================================

# Brain A: Strict & Deterministic (Used ONLY for tool selection logic)
llm_strict = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm_strict.bind_tools([fetch_movies, fetch_showtimes])

# Brain B: Charming & Creative (Used for writing ALL conversational responses)
llm_charming = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.65, # Raised to bring back excellent conversational flavor
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================================
# 2. DEFINE DUAL-POWERED WORKFLOW NODES
# =====================================================================

def chat_node(state: CinemaAgentState):
    """Handles general greetings and casual talk with maximum personality."""
    system_instruction = (
        "You are a friendly, conversational WhatsApp support agent for Cue Cinema in Lahore. "
        "Keep your responses short, cheerful, and highly scannable for mobile screens. "
        "Match the user's conversation style and language naturally (English, Roman Urdu, Urdu, etc.). "
        "Use local friendly slang (like 'boss g', 'yaar', 'jaan g') when appropriate. "
        "Format using simple double asterisks (**text**) for bolding key words. Never use triple asterisks."
    )
    
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = llm_charming.invoke(messages) # Uses the charming engine
    return {"messages": [response], "booking_status": "browsing"}

def showtime_node(state: CinemaAgentState):
    """Uses the strict brain to pull database info, then uses the charming brain to talk."""
    strict_instruction = (
        "Analyze the user request. If they want general movie listings, you must call 'fetch_movies'. "
        "If they ask about timings for a specific film, you must call 'fetch_showtimes'."
    )
    
    charming_instruction = (
        "You are the Showtimes Assistant for Cue Cinema.\n"
        "Take the raw database tool output provided in the history and summarize it beautifully for the user.\n"
        "Rules:\n"
        "1. Write in a lively, friendly tone matching the user's language (English or Roman Urdu).\n"
        "2. Format with clean bullet points (•) and emojis.\n"
        "3. Rely strictly on the data. If a requested time or slot isn't in the tool output, politely say it's not available.\n"
        "4. Use simple double asterisks (**text**) for emphasis. Never use triple asterisks."
    )
    
    # Step 1: Use the strict model to evaluate tools
    ai_message = llm_with_tools.invoke([SystemMessage(content=strict_instruction)] + state["messages"])
    
    if ai_message.tool_calls:
        updated_messages = list(state["messages"])
        updated_messages.append(ai_message)
        
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "fetch_movies":
                tool_output = fetch_movies.invoke(tool_args)
            elif tool_name == "fetch_showtimes":
                tool_output = fetch_showtimes.invoke(tool_args)
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."
                
            tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"], name=tool_name)
            updated_messages.append(tool_msg)
            
        # Step 2: Pass database content to the charming engine for localized summary phrasing
        final_response = llm_charming.invoke([SystemMessage(content=charming_instruction)] + updated_messages)
        return {"messages": [ai_message, final_response], "booking_status": "selecting_showtime"}
        
    # Local Fallback Safeguard: If tool_calls was skipped by mistake, force execution
    print("\n🛠️  [Dual-Brain Routing Fallback]: Forcing manual tool query...")
    last_user_msg = state["messages"][-1].content.lower()
    if "spider" in last_user_msg or "makri" in last_user_msg:
        tool_output = fetch_showtimes.invoke({"movie_title_query": "Spider-Man"})
    elif "odyssey" in last_user_msg:
        tool_output = fetch_showtimes.invoke({"movie_title_query": "The Odyssey"})
    else:
        tool_output = fetch_movies.invoke({})
        
    final_response = llm_charming.invoke([
        SystemMessage(content=charming_instruction),
        AIMessage(content=f"Raw Data Context: {tool_output}")
    ])
    return {"messages": [final_response], "booking_status": "selecting_showtime"}

def booking_node(state: CinemaAgentState):
    """Coordinates seat reservation collections cleanly using the charming voice engine."""
    system_instruction = (
        "You are the Booking Assistant for Cue Cinema. "
        "Help the user provide booking confirmation info. Keep it short, conversational, and use simple double asterisks."
    )
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = llm_charming.invoke(messages) # Uses the charming engine
    return {"messages": [response], "booking_status": "holding_seats"}

# =====================================================================
# 3. GRAPH CONSTITUTION
# =====================================================================
builder = StateGraph(CinemaAgentState)

builder.add_node("chat_node", chat_node)
builder.add_node("showtime_node", showtime_node)
builder.add_node("booking_node", booking_node)

builder.add_conditional_edges(START, conditional_router, {
    "chat_node": "chat_node",
    "showtime_node": "showtime_node",
    "booking_node": "booking_node"
})

builder.add_edge("chat_node", END)
builder.add_edge("showtime_node", END)
builder.add_edge("booking_node", END)

cinema_app = builder.compile()