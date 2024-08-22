import sqlite3
import json

def get_connection():
    return sqlite3.connect('documents.db')

def create_table():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        conn.commit()

def save_document(content, metadata):
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO documents (content, metadata) VALUES (?, ?)', (content, json.dumps(metadata)))
        conn.commit()

def get_documents():
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM documents')
        return cursor.fetchall()