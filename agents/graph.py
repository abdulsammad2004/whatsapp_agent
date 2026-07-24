import os
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from agents.state import CinemaAgentState
from agents.tools import fetch_movies, fetch_showtimes, reserve_seats


class BookingExtraction(BaseModel):
    """Extracts the showtime code and ticket count the user wants to book."""
    showtime_id: str = Field(
        default="UNKNOWN",
        description="The internal showtime code (e.g. 'st_201') that matches what the user described. Output exactly 'UNKNOWN' if you cannot confidently match one."
    )
    num_tickets: int = Field(
        default=0,
        description="Number of tickets the user wants. Output 0 if not mentioned anywhere."
    )


def showtime_node(state: CinemaAgentState) -> CinemaAgentState:
    raw_history = state.get("messages", [])
    bounded_messages = raw_history[-4:] if len(raw_history) > 4 else raw_history
    last_user_message = bounded_messages[-1].content.lower() if bounded_messages else ""

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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

        if "TRIGGER_FETCH_MOVIES" in extraction_response:
            db_output = fetch_movies.invoke({})
        elif "TRIGGER_FETCH_SHOWTIMES" in extraction_response:
            extracted_param = extraction_response.split(":")[-1].strip()
            db_output = fetch_showtimes.invoke({"movie_title_query": extracted_param})
            state["movie_title"] = extracted_param
            state["available_showtimes_context"] = db_output
    # Clear any previously selected showtime — a new movie query means the old
    # selection is stale and must NOT be silently reused by booking_node.
            state["showtime_id"] = ""
            state["num_tickets"] = 0
        else:
            db_output = fetch_movies.invoke({})

    except Exception as api_error:
        print(f"⚠️ [Groq Node Intercepted Exception]: {api_error}. Triggering offline data cache fallback.")
        try:
            db_output = fetch_movies.invoke({})
        except Exception as db_fallback_error:
            print(f"❌ [Database Fallback Also Failed]: {db_fallback_error}")
            db_output = "Movie listings are temporarily unavailable."

    formatting_instruction = (
        "You are 'Cue', the friendly WhatsApp buddy for Cue Cinema — texting like a helpful friend, not a formal front desk.\n"
        f"Database Query Results:\n{db_output}\n\n"
        "LANGUAGE RULES (STRICT MIRRORING):\n"
        "- Reply in the SAME language the user just used. Do not switch languages on your own.\n"
        "- If the user wrote in English, reply in English only.\n"
        "- If the user wrote in Roman Urdu (or mixed), reply in natural Roman Urdu + English mix as spoken in Lahore.\n"
        "GROUNDING RULES (CRITICAL — DO NOT BREAK):\n"
        "- ONLY state facts that literally appear in the Database Query Results above. Never invent showtimes, screens, prices, or seat numbers that aren't in that data.\n"
        "- This cinema does NOT have individual seat selection — only a TOTAL SEAT COUNT per showtime. NEVER mention specific seat numbers or rows. If asked about seats, say only how many are available in total.\n"
        "- NEVER say a booking or reservation is confirmed. You are only showing information.\n"
        "- CRITICAL: NEVER show internal codes like 'st_201' to the user. Describe each showtime naturally by TIME and SCREEN/LOCATION only (e.g. 'the 4:00 PM show at Gold Screen 1'). The codes are for internal use only.\n"
        "FORMATTING:\n"
        "- Use WhatsApp markdown: *bold* for movie names/headings, bullet points (-) for showtime lists.\n"
        "- Keep it snappy and casual — no long paragraphs, no corporate tone.\n"
        "- End by naturally asking which showTIME they'd like (describe it, e.g. 'the 4pm one or the 8:30pm one?') and how many tickets — never mention a code."
    )

    try:
        final_message = llm.invoke([SystemMessage(content=formatting_instruction)] + bounded_messages)
        state["messages"].append(final_message)
    except Exception as format_error:
        print(f"❌ [Formatting Error]: {format_error}")
        state["messages"].append(AIMessage(content=f"Here's what's showing:\n{db_output}"))

    return state


def booking_node(state: CinemaAgentState) -> CinemaAgentState:
    """
    Real booking node. Matches the user's NATURAL-LANGUAGE description (e.g.
    "the 4pm one", "gold screen", "the earlier show") against the hidden
    showtime data captured earlier in this session, resolving it to a real
    showtime_id internally. The user never needs to know or say any code.
    """
    raw_history = state.get("messages", [])
    bounded_messages = raw_history[-8:] if len(raw_history) > 8 else raw_history
    user_phone = state.get("user_phone", "unknown")

    known_showtime_id = state.get("showtime_id", "")
    known_num_tickets = state.get("num_tickets", 0)
    available_context = state.get("available_showtimes_context", "")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    try:
        structured_llm = llm.with_structured_output(BookingExtraction)
        extraction_prompt = (
            "You are a booking parameter extractor for Cue Cinema.\n"
            f"Here is the showtime data that was shown to the user earlier (codes are internal, the user never saw them):\n{available_context}\n\n"
            f"Already confirmed earlier in this session (may be empty): showtime_id='{known_showtime_id}', num_tickets={known_num_tickets}\n"
            "The user will describe their choice NATURALLY (e.g. 'the 4pm one', 'gold screen', 'the earlier show', 'that one you said') "
            "— match their description against the showtime data above and output the matching internal showtime_id.\n"
            "If the current messages don't restate a choice, use the already-known value shown above instead of guessing.\n"
            "If you truly cannot match anything confidently, output 'UNKNOWN'."
        )
        payload = [SystemMessage(content=extraction_prompt)] + bounded_messages
        extraction = structured_llm.invoke(payload)
        print(f"⚙️ [Booking Extraction]: showtime_id='{extraction.showtime_id}' | num_tickets={extraction.num_tickets}")

        showtime_id = extraction.showtime_id if extraction.showtime_id != "UNKNOWN" else known_showtime_id
        num_tickets = extraction.num_tickets if extraction.num_tickets > 0 else (known_num_tickets or 1)

        if not showtime_id:
            state["messages"].append(AIMessage(
                content="Which showtime would you like — and for which movie? Just tell me the time that works for you 🎬"
            ))
            return state

    except Exception as e:
        print(f"⚠️ [Booking Extraction Error]: {e}")
        state["messages"].append(AIMessage(content="Sorry, something went wrong on my end — could you try that again? 🙏"))
        return state

    state["showtime_id"] = showtime_id
    state["num_tickets"] = num_tickets

    try:
        result = reserve_seats.invoke({
            "showtime_id": showtime_id,
            "num_tickets": num_tickets,
            "user_phone": user_phone
        })
    except Exception as db_error:
        print(f"❌ [Booking DB Error]: {db_error}")
        state["messages"].append(AIMessage(content="There's an issue with the booking system right now — please try again shortly 🙏"))
        return state

    if not result.get("success"):
        state["messages"].append(AIMessage(content=f"❌ {result.get('message')}"))
        return state

    state["booking_status"] = "holding_seats"
    reply = (
        f"🎬 *{result['movie_title']}* — {result['show_time']} ({result['screen']})\n"
        f"Tickets: {result['num_tickets']} | Total: {result['total_amount']} PKR\n\n"
        f"Click below to confirm your booking 👇\n{result['payment_link']}"
    )
    state["messages"].append(AIMessage(content=reply))
    return state


def chat_node(state: CinemaAgentState) -> CinemaAgentState:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    chat_instruction = (
        "You are 'Cue', the friendly WhatsApp buddy for Cue Cinema — not a formal assistant, more like a helpful friend texting back.\n"
        "LANGUAGE RULES (STRICT MIRRORING):\n"
        "- Default to ENGLISH. Only switch to a Roman Urdu + English mix if the USER writes in Roman Urdu first. Never initiate in Roman Urdu.\n"
        "- Once the user has written in Roman Urdu, you can mirror that naturally for the rest of the conversation.\n"
        "- Never sound like a corporate script. Think casual friend, not call center.\n"
        "GROUNDING RULES (CRITICAL — DO NOT BREAK):\n"
        "- You have NO access to real movie/showtime/booking data in this node. NEVER state specific showtimes, prices, seat numbers, or confirm any booking here.\n"
        "- Never mention internal codes like 'st_201' — the user should never see these.\n"
        "- If the user asks about seats, a specific booking status, or anything requiring real data, gently redirect them to ask about movies/showtimes.\n"
        "- This cinema does NOT support individual seat selection at all — never mention seat letters/numbers/rows.\n"
        "TONE:\n"
        "- Short, warm, a little playful. Use casual punctuation and the occasional emoji (🎬🍿) where it fits.\n"
        "- Keep it to 1-2 short sentences max for greetings/small talk."
    )

    try:
        response = llm.invoke([SystemMessage(content=chat_instruction)] + state["messages"][-3:])
        state["messages"].append(response)
    except Exception as e:
        print(f"⚠️ [Chat Node Fallback]: {e}")
        state["messages"].append(AIMessage(content="Hey! How can I help you today? 🎬"))

    return state


workflow = StateGraph(CinemaAgentState)

workflow.add_node("chat_node", chat_node)
workflow.add_node("showtime_node", showtime_node)
workflow.add_node("booking_node", booking_node)

from agents.router import conditional_router

workflow.set_conditional_entry_point(
    conditional_router,
    {
        "chat_node": "chat_node",
        "showtime_node": "showtime_node",
        "booking_node": "booking_node"
    }
)

workflow.add_edge("chat_node", END)
workflow.add_edge("showtime_node", END)
workflow.add_edge("booking_node", END)

cinema_app = workflow.compile()