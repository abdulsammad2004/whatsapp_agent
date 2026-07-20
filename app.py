import os
import asyncio
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Re-link your functional LangGraph system
from agents.graph import cinema_app

load_dotenv()

app = FastAPI(title="Restored Cinema Gateway")

USER_SESSIONS = {}
SESSION_LOCKS = {}

@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    user_id = From
    user_text = Body.strip() if Body else ""
    
    if user_id not in SESSION_LOCKS:
        SESSION_LOCKS[user_id] = asyncio.Lock()
        
    async with SESSION_LOCKS[user_id]:
        print(f"\n📩 [Incoming Payload] From: {user_id} | Text: '{user_text}'")
        
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
        
        # Run execution loop through Groq & LangGraph safely
        try:
            updated_graph_state = cinema_app.invoke(current_graph_state)
            
            USER_SESSIONS[user_id]["messages"] = updated_graph_state["messages"]
            USER_SESSIONS[user_id]["booking_status"] = updated_graph_state["booking_status"]
            
            bot_response = updated_graph_state["messages"][-1].content
            
        except Exception as graph_err:
            print(f"❌ [Logic Error]: {graph_err}")
            bot_response = "System update in progress. Let's restart. How can I help you?"

        # Generate standard TwiML XML
        twilio_response = MessagingResponse()
        twilio_response.message(bot_response)
        
        raw_xml = str(twilio_response)
        print(f"📤 [Generated XML Response]: {raw_xml}")
        
        # PRE-ERROR DEFENSE: Strict Type Header Enforcement
        # Forcing application/xml encoding explicitly ensures the proxy server 
        # doesn't corrupt the string format down the line.
        return Response(content=raw_xml, media_type="text/xml")

# 🚀 THE CRITICAL INFRASTRUCTURE PIECE
# This tells Python to kick off the Uvicorn engine when running via 'python app.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)