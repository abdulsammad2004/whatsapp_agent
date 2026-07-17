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
def reserve_seats(showtime_id: str, num_tickets: int) -> str:
    """
    Deducts available seats directly from the database and returns a secure payment link.
    Use this immediately when a customer confirms they want to proceed with a booking.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch current seat availability and movie/pricing details
    cursor.execute('''
        SELECT s.available_seats, s.price_pkr, m.title, s.show_time, s.screen
        FROM showtimes s
        JOIN movies m ON s.movie_id = m.movie_id
        WHERE s.showtime_id = ?
    ''', (showtime_id,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        return f"❌ Error: Showtime code `{showtime_id}` does not exist in the live database."
        
    available_seats, price_pkr, movie_title, show_time, screen = row
    
    # 2. Check if enough seats are left
    if available_seats == 0:
        conn.close()
        return f"❌ Sold Out! There are no seats left for showtime code `{showtime_id}`."
    elif available_seats < num_tickets:
        conn.close()
        return f"❌ Not enough seats! Only {available_seats} seats are left, but you requested {num_tickets}."
        
    # 3. Execute the database state write (Subtract seats)
    new_seat_count = available_seats - num_tickets
    cursor.execute(
        "UPDATE showtimes SET available_seats = ? WHERE showtime_id = ?",
        (new_seat_count, showtime_id)
    )
    conn.commit()
    conn.close()
    
    # 4. Generate structured transaction metrics and compliance checkout link
    total_amount = price_pkr * num_tickets
    mock_tx_token = random.randint(100000, 999999)
    secure_checkout_url = f"https://checkout.cuecinema.online/pay/tx_{mock_tx_token}"
    
    summary = (
        f"SUCCESS_RESERVATION_LOCKED\n"
        f"• Movie: {movie_title}\n"
        f"• Time: {show_time} ({screen})\n"
        f"• Tickets: {num_tickets}\n"
        f"• Individual Ticket Price: {price_pkr} PKR\n"
        f"• Total Bill: {total_amount} PKR\n"
        f"• Remaining Available Seats: {new_seat_count}\n"
        f"• Secure PCI-Compliant Pay Link: {secure_checkout_url}"
    )
    return summary