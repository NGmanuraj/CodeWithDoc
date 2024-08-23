# database.py
import sqlite3

def connect_db():
    conn = sqlite3.connect('extracted_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS information (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL,
            result TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn, cursor

def insert_data(cursor, query, result):
    try:
        # Execute the SQL statement with the string parameter
        cursor.execute(query, (query, result))
        print("Search results saved to database.")
    except Exception as e:
        print(f"Error inserting data: {e}")
