import sqlite3

with open("database/schema.sql", "r") as f:
    schema = f.read()

conn = sqlite3.connect("database/app.db")
cursor = conn.cursor()
cursor.executescript(schema)

    # Create users table with department column
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL,
    manager_id INTEGER,
    department TEXT
)
''')

    # Create performance history table
cursor.execute('''
CREATE TABLE IF NOT EXISTS performance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    prediction_date TEXT NOT NULL,
    performance_score REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES users(id)
)
''')

conn.commit()
conn.close()
print("✅ Base de données initialisée.")