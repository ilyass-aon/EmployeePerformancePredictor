import streamlit as st
from admin import admin_dashboard, manager_dashboard
from employee import employee_dashboard
import sqlite3
import hashlib

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def check_password(username, password):
    conn = sqlite3.connect('database/app.db')
    cursor = conn.cursor()
    
    # Hash the password for comparison
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    cursor.execute("""
        SELECT id, role FROM users 
        WHERE username = ? AND password = ?
    """, (username, hashed_password))
    
    result = cursor.fetchone()
    conn.close()
    
    return result

def login():
    st.title("🔐 Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        result = check_password(username, password)
        if result:
            user_id, role = result
            st.session_state['logged_in'] = True
            st.session_state['user'] = (user_id, role)
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def main():
    if not st.session_state['logged_in']:
        login()
    else:
        # Show logout button in sidebar
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
        
        # Get user role
        role = st.session_state['user'][1]
        
        if role == 'admin':
            # Admin can switch between Admin and Manager dashboards
            dashboard = st.selectbox("Select Dashboard", ["Admin Dashboard", "Manager Dashboard"])
            if dashboard == "Admin Dashboard":
                admin_dashboard()
            else:
                manager_dashboard()
        elif role == 'employee':
            employee_dashboard()
        elif role == 'manager':
            manager_dashboard()

if __name__ == "__main__":
    main()
