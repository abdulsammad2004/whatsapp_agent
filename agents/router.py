import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from agents.state import CinemaAgentState

load_dotenv()

class IntentSchema(BaseModel):
    """Structured analysis of the user's latest message to determine workflow routing."""
    intent: Literal["check_showtimes", "book_ticket", "general_chat"] = Field(
        description=(
            "Choose 'check_showtimes' if the customer is asking about what movies are playing, show schedules, or timings. "
            "Choose 'book_ticket' if they want to select seats, provide booking info, or hold tickets. "
            "Choose 'general_chat' for greetings, casual chit-chat, or generic cinema info."
        )
    )
    extracted_movie: str = Field(
        default="",
        description="The clean name of the movie the user mentioned (e.g. 'Spider-Man', 'The Odyssey'). If none, leave empty."
    )

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

structured_llm = llm.with_structured_output(IntentSchema)

def conditional_router(state: CinemaAgentState) -> Literal["showtime_node", "booking_node", "chat_node"]:
    """Dynamically routes traffic, ensuring operational states are preserved during confirmations."""
    status = state.get("booking_status", "browsing")
    last_message = state["messages"][-1].content if state["messages"] else ""
    clean_msg = last_message.lower().strip()
    
    # 1. Handle affirmative shortcuts mid-funnel to prevent state leaking
    if status in ["selecting_showtime", "holding_seats", "awaiting_payment"]:
        if clean_msg in ["ok", "yes", "yup", "haan", "theek hai", "sure", "confirm"]:
            return "booking_node" if status in ["holding_seats", "awaiting_payment"] else "showtime_node"

    # 2. Run live AI intent mapping
    try:
        analysis = structured_llm.invoke(last_message)
        
        if analysis.intent == "check_showtimes":
            return "showtime_node"
            
        if analysis.intent == "book_ticket":
            return "booking_node"
            
        if analysis.intent == "general_chat" and status not in ["selecting_showtime", "holding_seats", "awaiting_payment"]:
            return "chat_node"
            
    except Exception as e:
        print(f"\n⚠️ [Router Intent Analysis Issue]: {e}")
        
    # 3. Fallback to active state machine tracking
    if status == "selecting_showtime":
        return "showtime_node"
    elif status in ["holding_seats", "awaiting_payment"]:
        return "booking_node"
        
    return "chat_node"