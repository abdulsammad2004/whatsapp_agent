import os
import sqlite3
import random
from langchain_core.tools import tool

DB_PATH = "database/cinema_ops.db"

@tool
def fetch_movies() -> str:
    """Fetch the complete catalog of all movies currently playing at Cue Cinema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id, title, duration, rating FROM movies")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "There are currently no movies listed in the schedule database."
        
    result = "🎬 **Currently Playing at Cue Cinema:**\n"
    for m_id, title, duration, rating in rows:
        result += f"• **{title}** | ID: `{m_id}` | Duration: {duration} | Rating: {rating}\n"
    return result

@tool
def fetch_showtimes(movie_title_query: str) -> str:
    """
    Look up all upcoming showtimes, ticket prices, and open seating for a specific movie title.
    Pass a clear text query like 'Spider-Man' or 'The Odyssey'.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id, title FROM movies")
    all_movies = cursor.fetchall()
    
    def normalize(text: str) -> str:
        return "".join(char for char in text.lower() if char.isalnum())
        
    clean_query = normalize(movie_title_query)
    matched_movie = None
    
    for movie_id, title in all_movies:
        clean_title = normalize(title)
        if clean_query in clean_title or clean_title in clean_query:
            matched_movie = (movie_id, title)
            break
            
    if not matched_movie:
        conn.close()
        return f"❌ Sorry, I couldn't find any movie matching '{movie_title_query}' in our database schedule."
        
    movie_id, full_title = matched_movie
    cursor.execute(
        "SELECT showtime_id, show_time, screen, price_pkr, available_seats FROM showtimes WHERE movie_id = ?",
        (movie_id,)
    )
    shows = cursor.fetchall()
    conn.close()
    
    if not shows:
        return f"Looks like there are no upcoming scheduled showtimes right now for **{full_title}**."
        
    result = f"🕒 **Available Showtimes for {full_title}:**\n"
    for s_id, time, screen, price, seats in shows:
        result += f"• Code: `{s_id}` | Time: **{time}** | Location: {screen} | Price: {price} PKR | Available Seats: {seats}\n"
    return result

@tool
def reserve_seats(showtime_id: str, num_tickets: int, user_phone: str) -> dict:
    """
    Creates a real booking hold in the database and returns booking details
    plus a real payment link. Never fabricate a confirmation without calling this.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT s.available_seats, s.price_pkr, m.title, s.show_time, s.screen
        FROM showtimes s
        JOIN movies m ON s.movie_id = m.movie_id
        WHERE s.showtime_id = ?''', (showtime_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {"success": False, "message": f"Showtime code `{showtime_id}` doesn't exist."}

    available_seats, price_pkr, movie_title, show_time, screen = row

    if available_seats == 0:
        conn.close()
        return {"success": False, "message": f"Sold out for showtime `{showtime_id}`."}
    elif available_seats < num_tickets:
        conn.close()
        return {"success": False, "message": f"Only {available_seats} seats left, you asked for {num_tickets}."}

    new_seat_count = available_seats - num_tickets
    cursor.execute("UPDATE showtimes SET available_seats = ? WHERE showtime_id = ?", (new_seat_count, showtime_id))

    booking_id = f"bk_{random.randint(100000, 999999)}"
    total_amount = price_pkr * num_tickets

    cursor.execute(
        "INSERT INTO bookings (booking_id, showtime_id, user_phone, selected_seats, total_amount, payment_status) VALUES (?, ?, ?, ?, ?, 'hold')",
        (booking_id, showtime_id, user_phone, str(num_tickets), total_amount)
    )
    conn.commit()
    conn.close()

    base_url = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")
    payment_link = f"{base_url}/pay/{booking_id}"

    return {
        "success": True,
        "booking_id": booking_id,
        "movie_title": movie_title,
        "show_time": show_time,
        "screen": screen,
        "num_tickets": num_tickets,
        "total_amount": total_amount,
        "payment_link": payment_link
    }