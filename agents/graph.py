import os
from typing import TypedDict, List, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# 1. State Definition Schema
class CinemaAgentState(TypedDict):
    messages: List[BaseMessage]
    booking_status: str

# 2. Mock Tool Layer Definitions (Protected with Type/Sanitization Defenses)
def db_fetch_movies() -> str:
    """Simulates a highly secure database query returning available movies."""
    try:
        # Real-world query logic goes here (e.g., SQLite connection)
        return "Available Movies today:\n1. Spiderman: No Way Home [ID: m_01]\n2. The Odyssey [ID: m_02]"
    except Exception as db_err:
        print(f"❌ [Database Error Intercepted]: {db_err}")
        return "Error accessing the movie registry catalog."

def db_fetch_showtimes(movie_query: str) -> str:
    """Simulates a sanitized database filter query for specific showtimes."""
    # PRE-ERROR DEFENSE: Parameter Normalization
    # Cleans up incoming arguments from Roman Urdu strings or erratic spacing
    clean_query = movie_query.lower().strip()
    if "spider" in clean_query or "m_01" in clean_query:
        return "Showtimes for Spiderman: 3:00 PM, 6:00 PM, 9:00 PM"
    if "odyssey" in clean_query or "m_02" in clean_query:
        return "Showtimes for The Odyssey: 4:00 PM, 7:30 PM"
    return f"No direct listings match found for '{movie_query}'."

# 3. Core Operational Nodes
def showtime_node(state: CinemaAgentState) -> CinemaAgentState:
    """
    Defensive Showtime Data Extraction Node. Forces deterministic database 
    interaction and completely neutralizes history-bias loop bugs.
    """
    # PRE-ERROR DEFENSE: Context Window Bounding / Memory Trim
    # Prevents infinite context accumulation from degrading Groq inference speed.
    # We always prioritize the system instructions and the most recent user conversation turns.
    raw_history = state.get("messages", [])
    bounded_messages = raw_history[-4:] if len(raw_history) > 4 else raw_history
    last_user_message = bounded_messages[-1].content.lower() if bounded_messages else ""

    # Initialize the primary execution model
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # PRE-ERROR DEFENSE: Intent Extraction Forcing Prompt
    # Explicitly orders the model to disregard past conversational failures in the history
    # and strictly evaluate the *current* user turn against structural rules.
    extraction_instruction = (
        "You are the data parameter extractor for Cue Cinema. Your ONLY job is to analyze the user's latest query.\n"
        "CRITICAL DATABASE ROUTING RULES:\n"
        "1. If the user asks generally for movies, lists, show schedules, or what is playing today WITHOUT specifying a clear title, you MUST output exactly: 'TRIGGER_FETCH_MOVIES'.\n"
        "2. If they name a specific movie (e.g., 'Spiderman', 'Odyssey', or IDs like 'm_01'), extract that name and output exactly: 'TRIGGER_FETCH_SHOWTIMES: [Extracted Name]'.\n"
        "3. IGNORE any previous turns where you stated you couldn't find showtimes. Focus 100% on their latest input string.\n"
        "Do not include greeting phrases, conversational text, or any markdown structure. Output only the trigger phrase."
    )

    try:
        # Run targeted parameter extraction
        payload = [SystemMessage(content=extraction_instruction)] + [HumanMessage(content=last_user_message)]
        extraction_response = llm.invoke(payload).content.strip()
        print(f"⚙️ [Graph Parameter Extraction]: Model Output -> '{extraction_response}'")
        
        # 4. Tool Execution Fallback Routing
        if "TRIGGER_FETCH_MOVIES" in extraction_response:
            db_output = db_fetch_movies()
        elif "TRIGGER_FETCH_SHOWTIMES" in extraction_response:
            extracted_param = extraction_response.split(":")[-1].strip()
            db_output = db_fetch_showtimes(extracted_param)
        else:
            # Safe Fallback: If Llama outputs text instead of a token string, assume a general catalog pull
            db_output = db_fetch_movies()
            
    except Exception as api_error:
        print(f"⚠️ [Groq Node Intercepted Exception]: {api_error}. Triggering offline data cache fallback.")
        db_output = db_fetch_movies()

    # 5. Natural Language Formatting Layer
    # Translates raw database outputs into clean user messages matching their preferred language style
    formatting_instruction = (
        "You are the helpful front-desk coordinator for Cue Cinema.\n"
        f"Database Query Results:\n{db_output}\n\n"
        "INSTRUCTIONS:\n"
        "- Format the results cleanly using clear WhatsApp markdown (e.g., *bolding* headings, bullet points).\n"
        "- Match the user's communication style. If they ask in English, answer in English. If they ask in Roman Urdu (e.g., 'kia haal hai', 'show dikhao'), respond naturally in clear Roman Urdu.\n"
        "- Keep the response concise and action-oriented."
    )
    
    try:
        final_message = llm.invoke([SystemMessage(content=formatting_instruction)] + bounded_messages)
        state["messages"].append(final_message)
    except Exception as format_error:
        print(f"❌ [Formatting Error]: {format_error}")
        state["messages"].append(AIMessage(content="Here are today's movies:\n- Spiderman\n- The Odyssey"))

    return state

def chat_node(state: CinemaAgentState) -> CinemaAgentState:
    """Handles generic greetings and conversational small-talk gracefully."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    chat_instruction = (
        "You are the customer assistant for Cue Cinema. Greet the user warmly.\n"
        "If they speak to you in Roman Urdu, respond friendly in Roman Urdu. Keep it concise."
    )
    
    try:
        response = llm.invoke([SystemMessage(content=chat_instruction)] + state["messages"][-3:])
        state["messages"].append(response)
    except Exception as e:
        print(f"⚠️ [Chat Node Fallback]: {e}")
        state["messages"].append(AIMessage(content="Hello! Welcome to Cue Cinema. How can I assist you with tickets today?"))
        
    return state

# 6. Build and Compile the Automated Defended Workflow Graph
workflow = StateGraph(CinemaAgentState)

# Register defined functional nodes
workflow.add_node("chat_node", chat_node)
workflow.add_node("showtime_node", showtime_node)

# Import and attach our conditional router logic safely
from agents.router import conditional_router

workflow.set_conditional_entry_point(
    conditional_router,
    {
        "chat_node": "chat_node",
        "showtime_node": "showtime_node",
        "booking_node": "chat_node" # Default routing safety net placeholder
    }
)

# Connect execution nodes cleanly back to the termination loop point
workflow.add_edge("chat_node", END)
workflow.add_edge("showtime_node", END)

cinema_app = workflow.compile()