import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from agents.state import CinemaAgentState

# Load .env configurations
load_dotenv()

# =====================================================================
# 1. DEFINE STRUCTURED OUTPUT SCHEMA
# =====================================================================
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

# =====================================================================
# 2. INITIALIZE GROQ LLM BRAIN
# =====================================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Bind our strict structured output schema directly to the model
structured_llm = llm.with_structured_output(IntentSchema)

# =====================================================================
# 3. CONSTITUTE THE COGNITIVE ROUTER
# =====================================================================
def conditional_router(state: CinemaAgentState) -> Literal["showtime_node", "booking_node", "chat_node"]:
    """Dynamically routes traffic by evaluating incoming user intent via Groq."""
    status = state.get("booking_status", "browsing")
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # 1. ALWAYS run live Groq intent mapping first
    try:
        analysis = structured_llm.invoke(last_message)
        
        if analysis.intent == "check_showtimes":
            return "showtime_node"
            
        if analysis.intent == "book_ticket":
            return "booking_node"
            
        if analysis.intent == "general_chat" and status != "holding_seats":
            return "chat_node"
            
    except Exception as e:
        print(f"\n⚠️ [Router Intent Analysis Issue]: {e}")
        
    # 2. Fallback to active state machine tracking if intent is ambiguous
    if status == "selecting_showtime":
        return "showtime_node"
    elif status == "holding_seats":
        return "booking_node"
        
    return "chat_node"