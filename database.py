import sqlite3
import bcrypt
from datetime import datetime


def init_db():
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    # ----- USERS TABLE -----
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # ----- HISTORY TABLE -----
    c.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT,
        answer TEXT,
        time TEXT
    )
    """)

    # ----- CREATE ADMIN (ONLY ONCE) -----
    try:
        hashed = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()

        c.execute("INSERT INTO users VALUES (NULL,?,?,?)",
                  ("admin", hashed, "admin"))
    except:
        pass

    conn.commit()
    conn.close()



def register_user(username, password):
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        c.execute("INSERT INTO users VALUES (NULL,?,?,?)",
                  (username, hashed, "user"))
        conn.commit()
        return True
    except:
        return False



def login_user(username, password):
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    c.execute("SELECT password, role FROM users WHERE username=?",
              (username,))
    data = c.fetchone()

    if not data:
        return None

    stored = data[0]          # string from DB
    stored = stored.encode()  # convert to bytes

    if bcrypt.checkpw(password.encode(), stored):
        return data[1]

    return None



def save_history(username, question, answer):
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    c.execute("INSERT INTO history VALUES (NULL,?,?,?,?)",
              (username, question, answer,
               datetime.now().strftime("%d-%m-%Y %H:%M")))

    conn.commit()



def get_user_history(username):
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    c.execute("SELECT id, question, answer, time FROM history WHERE username=?",
              (username,))
    return c.fetchall()


def get_all_history():
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    c.execute("SELECT id, username, question, time FROM history")
    return c.fetchall()

def delete_history(id):
    conn = sqlite3.connect("analytics.db")
    c = conn.cursor()

    c.execute("DELETE FROM history WHERE id=?", (id,))
    conn.commit()
