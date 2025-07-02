import sqlite3
import hashlib

DB_PATH = "database/app.db"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, role):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Crée la table users si elle n'existe pas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Insère l'utilisateur
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                       (username, hash_password(password), role))
        conn.commit()
        print(f"✅ Utilisateur '{username}' créé avec succès en tant que {role}.")
    except sqlite3.IntegrityError:
        print(f"⚠️ L'utilisateur '{username}' existe déjà.")
    finally:
        conn.close()

# Exemple : créer un admin
create_user("admin", "admin123", "admin")
