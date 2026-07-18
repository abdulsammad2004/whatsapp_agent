import os
from dotenv import load_dotenv
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from agents.state import CinemaAgentState
from agents.router import conditional_router
from agents.tools import fetch_movies, fetch_showtimes, reserve_seats

load_dotenv()

# =====================================================================
# 1. INITIALIZE ENGINE & GLOBAL TOOL REGISTRY
# =====================================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2, 
    groq_api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools([fetch_movies, fetch_showtimes, reserve_seats])

TOOL_MAP = {
    "fetch_movies": fetch_movies,
    "fetch_showtimes": fetch_showtimes,
    "reserve_seats": reserve_seats
}

# =====================================================================
# 2. DEFINE SYSTEM WORKFLOW NODES
# =====================================================================

def chat_node(state: CinemaAgentState):
    """Handles friendly greetings and basic pleasantries while strictly avoiding hallucinations."""
    system_instruction = (
        "You are a welcoming WhatsApp customer support agent for Cue Cinema in Lahore.\n\n"
        "STRICT LANGUAGE SEPARATION RULES:\n"
        "1. ENGLISH MODE: If the user speaks English, respond entirely in standard English. Do NOT use Pakistani slang words like 'boss g', 'yaar', or 'jaan g'.\n"
        "2. ROMAN URDU MODE: If the user addresses you in Roman Urdu, switch entirely to Roman Urdu and feel free to use local slang like 'boss g' or 'yaar'.\n\n"
        "ANTI-HALLUCINATION RULE:\n"
        "- Never guess, name, or invent any movie titles or genres. If the user asks about movies or showtimes, politely guide them to check the live schedule listings."
    )
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response], "booking_status": "browsing"}

def showtime_node(state: CinemaAgentState):
    """Data Retrieval Node: Forces tool invocation first, falling back to a strictly grounded conversational guide."""
    data_extraction_instruction = (
        "You are the database access layer for Cue Cinema. Your sole job is to execute the correct tool call.\n"
        "1. If the user wants general listings, you MUST call 'fetch_movies'.\n"
        "2. If the user mentions a specific movie name, you MUST call 'fetch_showtimes' with that title.\n"
        "Do not answer with conversational text. You must trigger a tool call JSON object."
    )
    
    formatting_instruction = (
        "You are the Showtimes Assistant for Cue Cinema.\n"
        "Review the chat history. If a tool was executed, summarize its raw data beautifully using bullet points and emojis.\n"
        "CRITICAL SAFETY RULE: If NO tool output is present in the history, it means the movie name was unclear or a code is missing. "
        "In this case, do NOT confirm bookings or invent showtimes/prices. Instead, politely ask the user to clarify the movie name or provide a valid showtime code.\n\n"
        "STRICT FORMATTING:\n"
        "- Match the user's input language (English or Roman Urdu) completely.\n"
        "- Use simple double asterisks (**text**) for bold words. Never use triple asterisks (***)."
    )
    
    messages = [SystemMessage(content=data_extraction_instruction)] + state["messages"]
    ai_message = llm_with_tools.invoke(messages)
    
    if ai_message.tool_calls:
        updated_messages = list(state["messages"])
        updated_messages.append(ai_message)
        next_status = "selecting_showtime"
        
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name in TOOL_MAP:
                tool_output = TOOL_MAP[tool_name].invoke(tool_args)
                if tool_name == "reserve_seats":
                    next_status = "awaiting_payment"
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."
                
            tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"], name=tool_name)
            updated_messages.append(tool_msg)
            
        final_response = llm.invoke([SystemMessage(content=formatting_instruction)] + updated_messages)
        return {"messages": [ai_message, final_response], "booking_status": next_status}
        
    conversational_response = llm.invoke([SystemMessage(content=formatting_instruction)] + state["messages"])
    return {"messages": [conversational_response], "booking_status": "selecting_showtime"}

def booking_node(state: CinemaAgentState):
    """Transactional Node: Protects checkout flows and blocks duplicate reservations post-checkout."""
    status = state.get("booking_status", "browsing")
    
    # 🛑 STATE PROTECTION GATE: If the ticket is already locked down, ban tool usage entirely!
    if status == "awaiting_payment":
        payment_reminder_instruction = (
            "You are the Booking Assistant for Cue Cinema.\n"
            "The user's tickets are already successfully held in the database and they have received their secure checkout link.\n"
            "If the user says 'ok' or confirms, simply remind them politely to click the secure payment link provided above to finalize their purchase and receive their e-tickets.\n"
            "CRITICAL: Do NOT execute any more tools, do NOT alter the showtime details, and do NOT create new links. Keep it brief and mirror their language perfectly."
        )
        messages = [SystemMessage(content=payment_reminder_instruction)] + state["messages"]
        response = llm.invoke(messages) # Calls the raw LLM with zero tool access
        return {"messages": [response], "booking_status": "awaiting_payment"}

    # Standard Transactional Mode (Before reservation is locked)
    tool_instruction = (
        "You are the transaction processing layer for Cue Cinema. Your sole objective is to trigger the reservation backend.\n"
        "If a user provides a showtime code or confirms their booking parameters, execute the 'reserve_seats' tool immediately.\n"
        "Do not write conversational sentences until the tool has executed."
    )
    
    formatting_instruction = (
        "You are the Booking Assistant for Cue Cinema.\n"
        "Review the chat history. If 'reserve_seats' was successfully executed, display the billing breakdown and payment link cleanly.\n"
        "CRITICAL SAFETY RULE: If the 'reserve_seats' tool was NOT executed, ask them clearly to provide their specific showtime code.\n\n"
        "LIMITATIONS:\n"
        "1. NEVER ask for, accept, or let the customer type credit card numbers or CVVs in this chat.\n"
        "2. Mirror the user's input language perfectly. Keep it short and punchy."
    )
    
    messages = [SystemMessage(content=tool_instruction)] + state["messages"]
    ai_message = llm_with_tools.invoke(messages)
    
    if ai_message.tool_calls:
        updated_messages = list(state["messages"])
        updated_messages.append(ai_message)
        next_status = "holding_seats"
        
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name in TOOL_MAP:
                tool_output = TOOL_MAP[tool_name].invoke(tool_args)
                if tool_name == "reserve_seats":
                    next_status = "awaiting_payment" # Switch status on success!
                elif tool_name == "fetch_movies" or tool_name == "fetch_showtimes":
                    next_status = "selecting_showtime"
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."
                
            tool_msg = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"], name=tool_name)
            updated_messages.append(tool_msg)
            
        final_response = llm.invoke([SystemMessage(content=formatting_instruction)] + updated_messages)
        return {"messages": [ai_message, final_response], "booking_status": next_status}
        
    conversational_response = llm.invoke([SystemMessage(content=formatting_instruction)] + state["messages"])
    return {"messages": [conversational_response], "booking_status": "holding_seats"}

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