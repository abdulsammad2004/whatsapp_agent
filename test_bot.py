from langchain_core.messages import HumanMessage
from agents.graph import cinema_app
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agents.graph import cinema_app

def run_test_chat():
    print("🚀 Cue Cinemas at your Service!")
    print("Type 'exit' or 'quit' to stop the test.\n")
    
    current_state = {
        "messages": [],
        "user_phone": "03160460983",
        "movie_title": "",
        "showtime_id": "",
        "selected_seats": [],
        "booking_status": "browsing"
    }

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Shutting down test simulator. Catch ya later!")
            break
            
        if not user_input.strip():
            continue

        current_state["messages"].append(HumanMessage(content=user_input))
        output_state = cinema_app.invoke(current_state)

        # Parse output cleanly in case Gemini returns block elements
        latest_reply = output_state["messages"][-1].content
        if isinstance(latest_reply, list):
            text_parts = []
            for part in latest_reply:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif hasattr(part, "text"):
                    text_parts.append(part.text)
                elif isinstance(part, str):
                    text_parts.append(part)
            latest_reply = "\n".join(text_parts)

        print(f"\n🤖 Bot:\n{latest_reply}\n")
        print(f"⚙️  [Current System State] status: '{output_state['booking_status']}'")
        print("-" * 50)

        current_state = output_state

if __name__ == "__main__":
    run_test_chat()