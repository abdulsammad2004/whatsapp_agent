import os
import asyncio
import sqlite3
import httpx
from fastapi import FastAPI, Form, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from agents.graph import cinema_app
from jinja2 import Environment, FileSystemLoader

load_dotenv()

app = FastAPI(title="Restored Cinema Gateway")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env = Environment(loader=FileSystemLoader("templates"), cache_size=0)

DB_PATH = "database/cinema_ops.db"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

USER_SESSIONS = {}
SESSION_LOCKS = {}

WELCOME_MESSAGE = (
    "Hello! Welcome to Cue Cinema. 🎬\n"
    "I can help you check movies, showtimes, or book tickets — just tell me what you're looking for!"
)

@app.post("/webhook")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    user_id = From
    user_text = Body.strip() if Body else ""

    if user_id not in SESSION_LOCKS:
        SESSION_LOCKS[user_id] = asyncio.Lock()

    async with SESSION_LOCKS[user_id]:
        print(f"\n📩 [Incoming Payload] From: {user_id} | Text: '{user_text}'")

        # First-ever message in this session → send fixed welcome, skip the graph this turn
        if user_id not in USER_SESSIONS:
            USER_SESSIONS[user_id] = {
                "messages": [],
                "booking_status": "browsing",
                "user_phone": user_id
            }
            twilio_response = MessagingResponse()
            twilio_response.message(WELCOME_MESSAGE)
            print(f"👋 [First Contact]: Sending welcome message to {user_id}")
            return Response(content=str(twilio_response), media_type="text/xml")

        USER_SESSIONS[user_id]["messages"].append(HumanMessage(content=user_text))

        current_graph_state = {
            "messages": USER_SESSIONS[user_id]["messages"],
            "booking_status": USER_SESSIONS[user_id]["booking_status"],
            "user_phone": USER_SESSIONS[user_id]["user_phone"]
        }

        try:
            updated_graph_state = cinema_app.invoke(current_graph_state)
            USER_SESSIONS[user_id]["messages"] = updated_graph_state["messages"]
            USER_SESSIONS[user_id]["booking_status"] = updated_graph_state["booking_status"]
            bot_response = updated_graph_state["messages"][-1].content
        except Exception as graph_err:
            print(f"❌ [Logic Error]: {graph_err}")
            bot_response = "System update in progress. Let's restart. How can I help you?"

        twilio_response = MessagingResponse()
        twilio_response.message(bot_response)
        raw_xml = str(twilio_response)
        print(f"📤 [Generated XML Response]: {raw_xml}")

        return Response(content=raw_xml, media_type="text/xml")


@app.get("/pay/{booking_id}", response_class=HTMLResponse)
async def payment_page(request: Request, booking_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT b.booking_id, b.selected_seats, b.total_amount, b.payment_status,
               m.title, s.show_time, s.screen
        FROM bookings b
        JOIN showtimes s ON b.showtime_id = s.showtime_id
        JOIN movies m ON s.movie_id = m.movie_id
        WHERE b.booking_id = ?
    ''', (booking_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return HTMLResponse("<h1>Booking not found</h1>", status_code=404)

    booking_id, seats, total, status, title, show_time, screen = row

    return templates.TemplateResponse(
    request, "pay.html", {
        "booking_id": booking_id,
        "movie_title": title,
        "show_time": show_time,
        "screen": screen,
        "num_tickets": seats,
        "total_amount": total,
        "status": status
    }
)

BRIDGE_URL = "http://127.0.0.1:3000"  # your Node bridge's local server
@app.post("/pay/{booking_id}/confirm", response_class=HTMLResponse)
async def confirm_payment(request: Request, booking_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_phone, payment_status FROM bookings WHERE booking_id = ?", (booking_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return HTMLResponse("<h1>Booking not found</h1>", status_code=404)

    user_phone, current_status = row

    if current_status != "confirmed":
        cursor.execute("UPDATE bookings SET payment_status = 'confirmed' WHERE booking_id = ?", (booking_id,))
        conn.commit()
    conn.close()

    # Use localhost, not the public ngrok URL — bridge and FastAPI are on the same machine,
    # so this avoids an unnecessary round-trip through the public internet.
    ticket_image_url = "http://127.0.0.1:8001/static/assets/ticket_confirmed.png"
    print(f"🎫 [Attempting Ticket Send] To user_phone='{user_phone}'")

    async def send_ticket():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{BRIDGE_URL}/send-image", json={
                    "to": user_phone,
                    "image_url": ticket_image_url,
                    "caption": "🎉 Aapki booking confirm ho gayi! Yeh hai aapka ticket 🎬"
                })
                print(f"✅ [Ticket Send Response]: {resp.status_code} - {resp.text}")
        except Exception as send_err:
            print(f"❌ [Bridge Send Error]: {type(send_err).__name__}: {send_err}")

    asyncio.create_task(send_ticket())  # fire-and-forget — page returns immediately, ticket sends in background

    return templates.TemplateResponse(request, "payment_success.html", {"booking_id": booking_id})


@app.post("/bridge-webhook")
async def bridge_webhook(payload: dict):
    user_id = payload.get("from")
    user_text = (payload.get("text") or "").strip()

    print(f"📩 [Bridge Incoming Payload] From: {user_id} | Text: '{user_text}'")

    if user_id not in SESSION_LOCKS:
        SESSION_LOCKS[user_id] = asyncio.Lock()

    async with SESSION_LOCKS[user_id]:
        if user_id not in USER_SESSIONS:
            USER_SESSIONS[user_id] = {
                "messages": [],
                "booking_status": "browsing",
                "user_phone": user_id
            }
            print(f"👋 [First Contact via Bridge]: Sending welcome message to {user_id}")
            return {"reply": WELCOME_MESSAGE}

        USER_SESSIONS[user_id]["messages"].append(HumanMessage(content=user_text))

        current_graph_state = {
            "messages": USER_SESSIONS[user_id]["messages"],
            "booking_status": USER_SESSIONS[user_id]["booking_status"],
            "user_phone": USER_SESSIONS[user_id]["user_phone"]
        }

        try:
            updated_graph_state = cinema_app.invoke(current_graph_state)
            USER_SESSIONS[user_id]["messages"] = updated_graph_state["messages"]
            USER_SESSIONS[user_id]["booking_status"] = updated_graph_state["booking_status"]
            bot_response = updated_graph_state["messages"][-1].content
        except Exception as graph_err:
            print(f"❌ [Logic Error]: {graph_err}")
            bot_response = "System update in progress. Let's restart. How can I help you?"

        print(f"📤 [Bridge Reply]: {bot_response}")
        return {"reply": bot_response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)