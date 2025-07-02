import sqlite3
from datetime import datetime, timedelta
import random
import hashlib

def create_team_members():
    conn = sqlite3.connect('database/app.db')
    cursor = conn.cursor()
    
    # Create 3 more team members
    team_members = [
        ('team_member1', 'password123', 'employee'),
        ('team_member2', 'password123', 'employee'),
        ('team_member3', 'password123', 'employee')
    ]
    
    for username, password, role in team_members:
        # Create user
        cursor.execute(
            'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
            (username, hashlib.sha256(password.encode()).hexdigest(), role)
        )
        user_id = cursor.lastrowid
        
        # Add to employees table
        cursor.execute(
            'INSERT INTO employees (user_id, department) VALUES (?, ?)',
            (user_id, 'Sales')
        )
    
    conn.commit()
    
    # Generate 30 days of performance history for all employees
    cursor.execute("SELECT u.id FROM users u JOIN employees e ON u.id = e.user_id WHERE e.department = 'Sales'")
    employee_ids = cursor.fetchall()
    
    for emp_id in employee_ids:
        for days_ago in range(30):
            date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            # Generate slightly different patterns for each employee
            base_score = 4.0
            variation = random.uniform(-0.5, 0.5)
            trend = days_ago * 0.02  # Slight improvement trend over time
            score = round(min(5.0, max(1.0, base_score + variation + trend)), 2)
            
            cursor.execute(
                'INSERT INTO performance_history (employee_id, prediction_date, performance_score) VALUES (?, ?, ?)',
                (emp_id[0], date, score)
            )
    
    conn.commit()
    conn.close()
    print('Created team members and generated performance history')

if __name__ == '__main__':
    create_team_members()
