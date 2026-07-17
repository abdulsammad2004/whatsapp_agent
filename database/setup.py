import sqlite3

def initialize_db():
    # Connects to the database file cleanly inside your database folder
    conn = sqlite3.connect('database/cinema_ops.db')
    cursor = conn.cursor()

    # Enable foreign keys to protect relational integrity
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. MOVIES TABLE
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movie_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            duration TEXT,
            rating TEXT
        )
    ''')

    # 2. SHOWTIMES TABLE (Tied directly to the movie catalog)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS showtimes (
            showtime_id TEXT PRIMARY KEY,
            movie_id TEXT,
            show_time TEXT NOT NULL,
            screen TEXT NOT NULL,
            price_pkr INTEGER NOT NULL,
            available_seats INTEGER NOT NULL,
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
        )
    ''')

    # 3. BOOKINGS TABLE (Manages temporary holds and finalized checkouts)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id TEXT PRIMARY KEY,
            showtime_id TEXT,
            user_phone TEXT NOT NULL,
            selected_seats TEXT NOT NULL,
            total_amount INTEGER NOT NULL,
            payment_status TEXT DEFAULT 'hold', -- 'hold', 'confirmed', 'expired'
            payment_ref TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (showtime_id) REFERENCES showtimes(showtime_id)
        )
    ''')

    # Wipe tables clean on every execution to ensure clean, predictable test data
    cursor.execute("DELETE FROM bookings")
    cursor.execute("DELETE FROM showtimes")
    cursor.execute("DELETE FROM movies")

    # Real movie catalog parameters for local cinema listings
    movies_data = [
        ('m_01', 'The Odyssey', '150 mins', 'PG-13'),
        ('m_02', 'Spider-Man: Brand New Day', '135 mins', 'PG'),
        ('m_03', 'Aag Lagay Basti Mein', '142 mins', '18+')
    ]
    cursor.executemany("INSERT INTO movies VALUES (?, ?, ?, ?)", movies_data)

    # Scheduled mock showtimes
    showtimes_data = [
        ('st_101', 'm_01', '06:00 PM', 'Gold Screen 1', 1500, 40),
        ('st_102', 'm_01', '09:30 PM', 'Premium Screen 2', 1200, 65),
        ('st_201', 'm_02', '04:00 PM', 'Gold Screen 1', 1500, 10),
        ('st_202', 'm_02', '08:30 PM', 'Premium Screen 1', 1200, 55),
        ('st_301', 'm_03', '07:00 PM', 'Gold Screen 2', 1500, 32)
    ]
    cursor.executemany("INSERT INTO showtimes VALUES (?, ?, ?, ?, ?, ?)", showtimes_data)

    conn.commit()
    conn.close()
    print("⚡ Staging database successfully initialized and seeded inside database/ directory!")

if __name__ == "__main__":
    initialize_db()