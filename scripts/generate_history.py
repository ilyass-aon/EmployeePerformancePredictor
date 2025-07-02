import sqlite3
from datetime import datetime, timedelta
import random

def generate_test_history():
    conn = sqlite3.connect('database/app.db')
    cursor = conn.cursor()
    
    # Get user ID
    cursor.execute('SELECT id FROM users WHERE username = ?', ('user',))
    user_id = cursor.fetchone()[0]
    
    # Generate 10 days of history
    for days_ago in range(10):
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        score = round(random.uniform(3.5, 4.8), 2)
        
        cursor.execute(
            'INSERT INTO performance_history (employee_id, prediction_date, performance_score) VALUES (?, ?, ?)',
            (user_id, date, score)
        )
    
    conn.commit()
    conn.close()
    print('Added 10 days of performance history')

if __name__ == '__main__':
    generate_test_history()
