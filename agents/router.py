import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# 1. Structured Output Schema with Robust Fallback Documentation
class IntentAnalyzer(BaseModel):
    """Analyzes the user's incoming message to categorize their immediate action query."""
    intent: Literal["general_chat", "check_showtimes", "book_ticket"] = Field(
        default="general_chat",
        description="Categorize the user text. 'check_showtimes' for inquiries about movies, lists, timings, or playing schedules. 'book_ticket' for payment or reservation requests. Use 'general_chat' for greetings, casual remarks, typos, or unclear single words."
    )

def conditional_router(state: dict) -> str:
    """
    Defensive Traffic Router optimized for Groq. Maps out execution flow based on intent 
    extraction and handles conversational trap escapes cleanly.
    """
    messages = state.get("messages", [])
    booking_status = state.get("booking_status", "browsing")
    
    if not messages:
        return "chat_node"
        
    # Isolate the latest incoming user payload turn
    last_message = messages[-1].content.strip()
    
    # PRE-ERROR DEFENSE: Instant Escape for Punctuation or Empty Inputs
    # Bypasses Groq API completely for raw characters or short greetings to save rate limits
    if last_message in ["?", "??", "hello", "hey", "hi", "assalam alaikum", "aoa"] or len(last_message) <= 2:
        print("🛡️ [Pre-emptive Router Defense]: Short query or punctuation detected. Forcing 'chat_node'.")
        return "chat_node"

    # PRE-ERROR DEFENSE: Missing API Key Guard
    if not os.getenv("GROQ_API_KEY"):
        print("❌ [Critical Environment Error]: GROQ_API_KEY is completely missing from your .env file!")
        return "chat_node"

    # Initialize Groq LLM with deterministic temperature
    # Using the high-powered Llama 3.3 70b model for accurate structured extraction
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        max_retries=2 # Pre-emptively retry if Groq hits a brief connection blip
    ) 
    
    # Bind the structured Pydantic parser to Llama
    structured_llm = llm.with_structured_output(IntentAnalyzer)
    
    # 2. Try-Except Gatekeeping Core for Groq API
    try:
        # Request structured categorization schema from Llama
        analysis = structured_llm.invoke(last_message)
        extracted_intent = analysis.intent
        print(f"🔮 [Groq Router Analysis] Extracted Intent: '{extracted_intent}' | Session Status: '{booking_status}'")
        
    except Exception as groq_exception:
        # PRE-ERROR DEFENSE: Groq Rate Limit (429) or JSON Parsing Failure Recovery
        # If Groq throttles your requests or fails to format the JSON string properly,
        # catch the crash gracefully and fallback to normal conversation instead of throwing a 500 error.
        print(f"⚠️ [Groq Router Exception Intercepted]: {groq_exception}. Defaulting safely to chat node.")
        return "chat_node"

    # 3. Defensive Decoupling Resolution Mapping
    # PRE-ERROR DEFENSE: Sticky Trap Override
    # Break out of any persistent node state loops immediately if the intent points to regular conversation.
    if extracted_intent == "general_chat":
        return "chat_node"
        
    # Route to transactional nodes based on intent analysis
    if extracted_intent == "check_showtimes":
        return "showtime_node"
        
    if extracted_intent == "book_ticket":
        return "booking_node"

    # Ultimate safety net fallback
    return "chat_node"