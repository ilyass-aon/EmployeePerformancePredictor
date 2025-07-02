import sqlite3
from datetime import datetime, timedelta
import random

def main():
    conn = sqlite3.connect('database/app.db')
    cursor = conn.cursor()
    
    # Create test employees if they don't exist
    cursor.executemany(
        "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
        [('emp1', 'hash1', 'employee'), 
         ('emp2', 'hash2', 'employee'),
         ('emp3', 'hash3', 'employee')]
    )
    conn.commit()
    
    # Get all employee IDs
    employees = cursor.execute("SELECT id FROM users WHERE role='employee'").fetchall()
    
    # Generate 30 days of random performance data for each employee
    for emp in employees:
        for days in range(30):
            score = round(random.uniform(3.0, 5.0), 1)
            date = (datetime.now() - timedelta(days=30-days)).strftime('%Y-%m-%d')
            cursor.execute(
                "INSERT INTO performance_history (employee_id, prediction_date, performance_score) VALUES (?, ?, ?)",
                (emp[0], date, score)
            )
    conn.commit()
    conn.close()
    print("Generated test data for", len(employees), "employees")

if __name__ == "__main__":
    main()
