import os
from typing import Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# 1. Import the real shared state schema (instead of a local inline one)
from agents.state import CinemaAgentState

# 2. Import the real database-backed tools (instead of mock functions)
from agents.tools import fetch_movies, fetch_showtimes

# 3. Core Operational Nodes
def showtime_node(state: CinemaAgentState) -> CinemaAgentState:
    """
    Defensive Showtime Data Extraction Node. Forces deterministic database 
    interaction and completely neutralizes history-bias loop bugs.
    """
    # PRE-ERROR DEFENSE: Context Window Bounding / Memory Trim
    raw_history = state.get("messages", [])
    bounded_messages = raw_history[-4:] if len(raw_history) > 4 else raw_history
    last_user_message = bounded_messages[-1].content.lower() if bounded_messages else ""

    # Initialize the primary execution model
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # PRE-ERROR DEFENSE: Intent Extraction Forcing Prompt
    extraction_instruction = (
        "You are the data parameter extractor for Cue Cinema. Your ONLY job is to analyze the user's latest query.\n"
        "CRITICAL DATABASE ROUTING RULES:\n"
        "1. If the user asks generally for movies, lists, show schedules, or what is playing today WITHOUT specifying a clear title, you MUST output exactly: 'TRIGGER_FETCH_MOVIES'.\n"
        "2. If they name a specific movie (e.g., 'Spiderman', 'Odyssey', or IDs like 'm_01'), extract that name and output exactly: 'TRIGGER_FETCH_SHOWTIMES: [Extracted Name]'.\n"
        "3. IGNORE any previous turns where you stated you couldn't find showtimes. Focus 100% on their latest input string.\n"
        "Do not include greeting phrases, conversational text, or any markdown structure. Output only the trigger phrase."
    )

    try:
        payload = [SystemMessage(content=extraction_instruction)] + [HumanMessage(content=last_user_message)]
        extraction_response = llm.invoke(payload).content.strip()
        print(f"⚙️ [Graph Parameter Extraction]: Model Output -> '{extraction_response}'")

        # 4. Tool Execution Routing — now hitting the REAL SQLite-backed tools
        if "TRIGGER_FETCH_MOVIES" in extraction_response:
            db_output = fetch_movies.invoke({})
        elif "TRIGGER_FETCH_SHOWTIMES" in extraction_response:
            extracted_param = extraction_response.split(":")[-1].strip()
            db_output = fetch_showtimes.invoke({"movie_title_query": extracted_param})
        else:
            # Safe Fallback: If Llama outputs text instead of a token string, assume a general catalog pull
            db_output = fetch_movies.invoke({})

    except Exception as api_error:
        print(f"⚠️ [Groq Node Intercepted Exception]: {api_error}. Triggering offline data cache fallback.")
        try:
            db_output = fetch_movies.invoke({})
        except Exception as db_fallback_error:
            print(f"❌ [Database Fallback Also Failed]: {db_fallback_error}")
            db_output = "Movie listings are temporarily unavailable."

    # 5. Natural Language Formatting Layer — friendly Roman Urdu/English mix
    formatting_instruction = (
        "You are 'Cue', the friendly WhatsApp buddy for Cue Cinema — texting like a helpful friend, not a formal front desk.\n"
        f"Database Query Results:\n{db_output}\n\n"
        "LANGUAGE RULES:\n"
        "- Default to natural Roman Urdu + English mix as spoken in Lahore (e.g., 'Yeh dekh bhai, aaj ke shows ready hain!').\n"
        "- If the user's message was mostly English, keep your reply mostly English but casual — not stiff.\n"
        "- If the user wrote in Roman Urdu, lean more Urdu in your reply.\n"
        "FORMATTING:\n"
        "- Use WhatsApp markdown: *bold* for movie names/headings, bullet points (-) for showtime lists.\n"
        "- Keep it snappy and casual — no long paragraphs, no corporate tone.\n"
        "- End with a light, friendly nudge, like asking which show they want or if they need anything else."
    )

    try:
        final_message = llm.invoke([SystemMessage(content=formatting_instruction)] + bounded_messages)
        state["messages"].append(final_message)
    except Exception as format_error:
        print(f"❌ [Formatting Error]: {format_error}")
        state["messages"].append(AIMessage(content=f"Yahan hain latest listings:\n{db_output}"))

    return state

def chat_node(state: CinemaAgentState) -> CinemaAgentState:
    """Handles generic greetings and conversational small-talk gracefully."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    chat_instruction = (
        "You are 'Cue', the friendly WhatsApp buddy for Cue Cinema — not a formal assistant, more like a helpful friend texting back.\n"
        "LANGUAGE RULES:\n"
        "- Default to a natural Roman Urdu + English mix, the way young people in Lahore actually text (e.g., 'Kya haal hai! Movies check karne hain? Bata dena bhai.').\n"
        "- If the user writes in pure English, reply mostly in English but keep it casual, with the occasional light Urdu word if it fits naturally (e.g., 'yaar', 'bhai', 'theek hai').\n"
        "- If the user writes in Roman Urdu, mirror that closely — lean more Urdu than English.\n"
        "- Never sound like a corporate script. No 'Dear valued customer' energy — think casual friend, not call center.\n"
        "TONE:\n"
        "- Short, warm, a little playful. Use casual punctuation and the occasional emoji (🎬🍿) where it fits.\n"
        "- Keep it to 1-2 short sentences max for greetings/small talk."
    )

    try:
        response = llm.invoke([SystemMessage(content=chat_instruction)] + state["messages"][-3:])
        state["messages"].append(response)
    except Exception as e:
        print(f"⚠️ [Chat Node Fallback]: {e}")
        state["messages"].append(AIMessage(content="Heyy! Welcome to Cue Cinema 🎬 Kya dekhna hai aaj?"))

    return state

# 6. Build and Compile the Automated Defended Workflow Graph
workflow = StateGraph(CinemaAgentState)

workflow.add_node("chat_node", chat_node)
workflow.add_node("showtime_node", showtime_node)

from agents.router import conditional_router

workflow.set_conditional_entry_point(
    conditional_router,
    {
        "chat_node": "chat_node",
        "showtime_node": "showtime_node",
        "booking_node": "chat_node"  # Default routing safety net placeholder — real booking_node still TODO
    }
)

workflow.add_edge("chat_node", END)
workflow.add_edge("showtime_node", END)

cinema_app = workflow.compile()