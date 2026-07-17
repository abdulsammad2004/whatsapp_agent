from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class CinemaAgentState(TypedDict):
    # 'add_messages' ensures new texts are appended to history instead of overwriting it
    messages: Annotated[list[BaseMessage], add_messages]
    
    # These tracking keys will use default overwrite behavior
    user_phone: str
    movie_title: str
    showtime_id: str
    selected_seats: list[str]
    booking_status: str  # e.g., 'browsing', 'selecting_showtime', 'holding_seats', 'completed'