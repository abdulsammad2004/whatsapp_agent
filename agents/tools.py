import sqlite3
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
    
    # Fetch all movies so we can do a robust fuzzy match in Python
    cursor.execute("SELECT movie_id, title FROM movies")
    all_movies = cursor.fetchall()
    
    # Normalizer: strips spaces, hyphens, and punctuation
    def normalize(text: str) -> str:
        return "".join(char for char in text.lower() if char.isalnum())
        
    clean_query = normalize(movie_title_query)
    matched_movie = None
    
    for movie_id, title in all_movies:
        clean_title = normalize(title)
        # Match if the query is inside the title, or vice-versa (handles "spider man" vs "spider-man")
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