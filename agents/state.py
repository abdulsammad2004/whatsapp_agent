from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class CinemaAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_phone: str
    movie_title: str
    showtime_id: str
    num_tickets: int
    available_showtimes_context: str   # NEW — internal-only, holds codes for matching, never shown to user
    selected_seats: list[str]
    booking_status: str