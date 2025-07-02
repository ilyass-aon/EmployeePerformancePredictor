import sqlite3
import hashlib

def create_admin():
    conn = sqlite3.connect('database/app.db')
    cursor = conn.cursor()
    
    # Hash the password
    password = 'admin123'
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        # Create admin user
        cursor.execute("""
            INSERT INTO users (username, password, role)
            VALUES (?, ?, ?)
        """, ('admin', hashed_password, 'admin'))
        
        # Get the user id
        user_id = cursor.lastrowid
        
        # Add to employees table with department
        cursor.execute("""
            INSERT INTO employees (user_id, department)
            VALUES (?, ?)
        """, (user_id, 'Management'))
        
        conn.commit()
        print("Admin user created successfully!")
        print("Username: admin")
        print("Password: admin123")
        
    except sqlite3.IntegrityError:
        print("Admin user already exists!")
    finally:
        conn.close()

if __name__ == '__main__':
    create_admin()
