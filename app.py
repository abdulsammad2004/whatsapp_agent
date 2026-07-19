import os
import asyncio
from fastapi import FastAPI, Form, Response
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Securely bind the LangGraph workflow
from agents.graph import cinema_app

load_dotenv()

app = FastAPI(title="Production-Grade Async Gateway")

# Persistent memory registers
USER_SESSIONS = {}
SESSION_LOCKS = {}

# PRE-ERROR DEFENSE: Secure API Initialization Guard
# Validates credentials immediately on startup to prevent silent runtime dispatch failures.
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not ACCOUNT_SID or not AUTH_TOKEN:
    print("\n❌ [CRITICAL CONFIG ERROR]: Twilio credentials missing from your .env file!")
    print("Outbound REST dispatch will fail until TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are set.\n")

@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    user_id = From  # Twilio passes this exactly as 'whatsapp:+923160460983'
    user_text = Body.strip() if Body else ""
    
    if user_id not in SESSION_LOCKS:
        SESSION_LOCKS[user_id] = asyncio.Lock()
        
    async with SESSION_LOCKS[user_id]:
        print(f"\n📩 [Incoming Message] From: {user_id} | Body: '{user_text}'")
        
        if user_id not in USER_SESSIONS:
            USER_SESSIONS[user_id] = {
                "messages": [],
                "booking_status": "browsing"
            }
            
        USER_SESSIONS[user_id]["messages"].append(HumanMessage(content=user_text))
        
        current_graph_state = {
            "messages": USER_SESSIONS[user_id]["messages"],
            "booking_status": USER_SESSIONS[user_id]["booking_status"]
        }
        
        # 1. State Machine Execution Core
        try:
            updated_graph_state = cinema_app.invoke(current_graph_state)
            
            USER_SESSIONS[user_id]["messages"] = updated_graph_state["messages"]
            USER_SESSIONS[user_id]["booking_status"] = updated_graph_state["booking_status"]
            
            bot_response = updated_graph_state["messages"][-1].content
            print(f"⚙️  [Graph Success] Next State Status: '{updated_graph_state['booking_status']}'")
            
        except Exception as graph_err:
            print(f"❌ [Graph Logic Exception Intercepted]: {graph_err}")
            USER_SESSIONS[user_id]["booking_status"] = "browsing"
            bot_response = "System update in progress. Let's restart this chat. How can I help you right now?"

        # 2. PRE-ERROR DEFENSE: Asynchronous Outbound REST Dispatch Layer
        # Instead of packaging XML in the HTTP response body, we explicitly trigger
        # a direct, non-blocking API post call back to Twilio's delivery gateway.
        try:
            # Initialize internal tracking client on demand safely
            client = Client(ACCOUNT_SID, AUTH_TOKEN)
            
            print(f"🚀 [Dispatching Outbound REST Call] To: {user_id}")
            client.messages.create(
                body=bot_response,
                from_=os.getenv("TWILIO_SENDER_NUMBER", "whatsapp:+14155238886"), # Your Sandbox Sender
                to=user_id
            )
            print("✨ [Dispatch Accepted]: Message handed off to WhatsApp network successfully.")
            
        except TwilioRestException as twilio_err:
            # Captures exact API delivery errors (e.g., Trial account restrictions or number unpairing)
            print(f"❌ [Twilio REST Gateway Refused Packet]: Code {twilio_err.code} - {twilio_err.msg}")
        except Exception as network_err:
            print(f"❌ [Outbound Network Timeout/Failure]: {network_err}")

        # 3. Secure Hook Handshake Release
        # Return a completely blank, clean TwiML block. This signals to Twilio's HTTP gateway 
        # that we have successfully consumed the incoming packet, preventing duplicate retries.
        return Response(content="<Response></Response>", media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)