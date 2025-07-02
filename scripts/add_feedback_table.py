import sqlite3

def add_feedback_table():
    conn = sqlite3.connect('database/app.db')
    cursor = conn.cursor()
    
    # Create feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER,
        feedback_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        feedback_text TEXT,
        category TEXT,
        satisfaction_level INTEGER,
        FOREIGN KEY (employee_id) REFERENCES users(id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Feedback table created successfully!")

if __name__ == '__main__':
    add_feedback_table()
