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
# 1. INITIALIZE ULTRA-INTELLIGENT SINGLE ENGINE
# =====================================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3, # Dropped slightly for tighter instruction adherence
    groq_api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools([fetch_movies, fetch_showtimes])

# =====================================================================
# 2. DEFINE SMART WORKFLOW NODES WITH STRICT LANGUAGE GATING
# =====================================================================

def chat_node(state: CinemaAgentState):
    """Handles friendly greetings and general assistance with zero language blending."""
    system_instruction = (
        "You are a welcoming WhatsApp customer support agent for Cue Cinema in Lahore.\n\n"
        "STRICT LANGUAGE SEPARATION RULES:\n"
        "1. ENGLISH MODE: If the user greets or speaks to you in English (e.g., 'hey', 'hello', 'how are you'), you must respond entirely in standard, natural English. Do NOT use Pakistani slang words like 'boss g', 'yaar', or 'jaan g' when speaking English.\n"
        "2. ROMAN URDU MODE: Only if the user addresses you in Roman Urdu (e.g., 'kesa hai', 'baat sun'), switch to natural Roman Urdu. In this mode, you are encouraged to use local friendly terms like 'boss g' or 'yaar'.\n"
        "3. Keep responses short, clean, and perfectly scannable for mobile. Never use triple asterisks (***)."
    )
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response], "booking_status": "browsing"}

def showtime_node(state: CinemaAgentState):
    """Queries live schedules, maintaining clean language separation."""
    system_instruction = (
        "You are the Showtimes Assistant for Cue Cinema.\n\n"
        "STRICT LANGUAGE SEPARATION RULES:\n"
        "1. ENGLISH MODE: If the chat history shows the user is asking questions in English, write the movie listings and responses entirely in clean English. No 'boss g' or 'yaar'.\n"
        "2. ROMAN URDU MODE: If the user is communicating in Roman Urdu, format the entire schedule summary in natural Roman Urdu using local slang contextually.\n\n"
        "CORE TASK INSTRUCTIONS:\n"
        "- If they want to know what's playing, call 'fetch_movies'. If they ask about a specific movie, call 'fetch_showtimes'.\n"
        "- Ground your response strictly in the tool data. If a specific time, movie, or slot is not explicitly returned by the tool, declare it unavailable. Never invent data.\n"
        "- Format using clean bullet points (•) and double asterisks (**text**) for bold words. No triple asterisks."
    )
    
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    ai_message = llm_with_tools.invoke(messages)
    
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
            
        final_response = llm.invoke([SystemMessage(content=system_instruction)] + updated_messages)
        return {"messages": [ai_message, final_response], "booking_status": "selecting_showtime"}
        
    return {"messages": [ai_message], "booking_status": "selecting_showtime"}

def booking_node(state: CinemaAgentState):
    """Coordinates ticket reservation flows while remaining strictly grounded and language-accurate."""
    system_instruction = (
        "You are the Booking Assistant for Cue Cinema.\n\n"
        "STRICT LANGUAGE SEPARATION RULES:\n"
        "1. If the user is speaking English, keep the booking collection process entirely in standard English. Do NOT drop Urdu slang terms like 'boss g' or 'yaar' into English dialogue.\n"
        "2. If the user is speaking Roman Urdu, coordinate the details in smooth Roman Urdu.\n\n"
        "CORE TASK INSTRUCTIONS:\n"
        "- Assist the user in completing ticket booking parameters.\n"
        "- Never invent seat structures, pricing information, or movie names out of thin air. Keep it short and clean."
    )
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    ai_message = llm_with_tools.invoke(messages)
    
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
            
        final_response = llm.invoke([SystemMessage(content=system_instruction)] + updated_messages)
        return {"messages": [ai_message, final_response], "booking_status": "selecting_showtime"}
        
    return {"messages": [ai_message], "booking_status": "holding_seats"}

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