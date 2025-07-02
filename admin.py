import streamlit as st
import pandas as pd
import joblib
from sqlite3 import connect
import sqlite3
import hashlib  # Import pour le hachage des mots de passe
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

DB_PATH = "database/app.db"


# Fonction de hachage du mot de passe
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Charger les modèles enregistrés
linear_model = joblib.load('ml_models/linear_model.pkl')
rf_model = joblib.load('ml_models/rf_model.pkl')
xgb_model = joblib.load('ml_models/xgb_model.pkl')


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM performance_data", conn)
    return df


def predict_performance(model, tasks_completed, hours_worked, feedback_score, punctuality_rate, employee_id):
    input_data = pd.DataFrame({
        "tasks_completed": [tasks_completed],
        "hours_worked": [hours_worked],
        "feedback_score": [feedback_score],
        "punctuality_rate": [punctuality_rate]
    })
    prediction = model.predict(input_data)
    
    # Store prediction in history
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO performance_history (employee_id, prediction_date, performance_score) VALUES (?, datetime('now'), ?)",
        (employee_id, prediction[0])
    )
    conn.commit()
    conn.close()
    
    return prediction[0]


def add_user(username, password, role, manager_id=None, name=None, department=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Vérifier si l'utilisateur existe déjà
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            print(f"User {username} already exists.")
            return False

        # Hacher le mot de passe avant de l'insérer dans la base de données
        hashed_password = hash_password(password)

        # Insérer un utilisateur dans la table 'users'
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                       (username, hashed_password, role))
        conn.commit()

        # Si l'utilisateur est un employé, l'ajouter à la table 'employees'
        if role == "employee":
            user_id = cursor.lastrowid  # Obtenir l'ID du nouvel utilisateur
            if name and department and manager_id:
                cursor.execute("INSERT INTO employees (user_id, name, department, manager_id) VALUES (?, ?, ?, ?)",
                               (user_id, name, department, manager_id))
                conn.commit()
            else:
                raise ValueError("Name, department, and manager are required for an employee.")

        return True
    except sqlite3.IntegrityError as e:
        print(f"IntegrityError: {e}")
        return False
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        conn.close()  # Toujours fermer la connexion à la base de données


def load_team_data():
    """Load all team data from database, including performance metrics."""
    conn = connect(DB_PATH)
    query = """
    SELECT 
        e.user_id,
        e.department,
        u.username,
        pd.tasks_completed,
        pd.hours_worked,
        pd.feedback_score,
        pd.punctuality_rate,
        pd.performance_score
    FROM employees e
    JOIN users u ON e.user_id = u.id
    LEFT JOIN performance_data pd ON pd.employee_id = e.user_id
    WHERE u.role = 'employee'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def predict_future_performance(employee_id, history_days=30, future_days=7):
    """Predict future performance using ML model"""
    conn = connect(DB_PATH)
    
    # Get historical data
    query = """
    SELECT 
        prediction_date,
        performance_score,
        CAST(
            (JulianDay(prediction_date) - JulianDay(
                (SELECT MIN(prediction_date) FROM performance_history WHERE employee_id = ?)
            )) AS INTEGER
        ) as day_number
    FROM performance_history 
    WHERE employee_id = ?
    ORDER BY prediction_date
    """
    df = pd.read_sql_query(query, conn, params=(employee_id, employee_id))
    conn.close()
    
    if len(df) < 5:  # Need minimum data points
        return None
        
    # Prepare features
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    df['dayofweek'] = df['prediction_date'].dt.dayofweek
    df['month'] = df['prediction_date'].dt.month
    
    # Create training data
    X = df[['day_number', 'dayofweek', 'month']].values
    y = df['performance_score'].values
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Prepare future dates
    last_date = df['prediction_date'].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)
    future_X = pd.DataFrame({
        'day_number': range(len(df), len(df) + future_days),
        'dayofweek': future_dates.dayofweek,
        'month': future_dates.month
    })
    
    # Make predictions
    predictions = model.predict(future_X)
    
    return pd.DataFrame({
        'date': future_dates,
        'predicted_score': predictions
    })


def get_team_insights(df):
    """Generate team performance insights"""
    insights = []
    
    # Top performers
    top_performers = df.groupby('username')['performance_score'].mean().nlargest(3)
    insights.append(("🌟 Top Performers", top_performers))
    
    # Most improved
    df['date'] = pd.to_datetime(df['prediction_date'])
    recent_scores = df[df['date'] >= df['date'].max() - timedelta(days=7)]
    improved = recent_scores.groupby('username')['performance_score'].mean() - \
              df.groupby('username')['performance_score'].mean()
    most_improved = improved.nlargest(3)
    insights.append(("📈 Most Improved (Last 7 Days)", most_improved))
    
    # Department performance
    dept_perf = df.groupby('department')['performance_score'].agg(['mean', 'std']).round(2)
    insights.append(("🏢 Department Performance", dept_perf))
    
    return insights


def admin_dashboard():
    st.subheader("📊 Gestion des utilisateurs")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # --- Filtrage ---
    st.markdown("### 🎯 Filtres")
    role_filter = st.selectbox("Filtrer par rôle", ["Tous", "admin", "manager", "employee"])

    cursor.execute(
        "SELECT DISTINCT m.username FROM users u JOIN employees e ON u.id = e.user_id JOIN users m ON e.manager_id = m.id WHERE u.role = 'employee'")
    managers_list = [row[0] for row in cursor.fetchall()]
    manager_filter = st.selectbox("Filtrer par manager", ["Tous"] + managers_list)

    # --- Requête principale ---
    query = """
    SELECT u.id, u.username, u.role, e.name, e.department, m.username AS manager_name
    FROM users u
    LEFT JOIN employees e ON u.id = e.user_id
    LEFT JOIN users m ON e.manager_id = m.id
    """
    df_users = pd.read_sql_query(query, conn)

    if role_filter != "Tous":
        df_users = df_users[df_users["role"] == role_filter]
    if manager_filter != "Tous":
        df_users = df_users[df_users["manager_name"] == manager_filter]

    st.dataframe(df_users)
    # --- Export CSV ---
    csv = df_users.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exporter les utilisateurs en CSV",
        data=csv,
        file_name='utilisateurs.csv',
        mime='text/csv'
    )
    # --- Modification et suppression ---
    st.subheader("✏️ Modifier / Supprimer un utilisateur")
    user_ids = df_users['id'].tolist()
    if not user_ids:
        st.info("Aucun utilisateur trouvé avec ces filtres.")
        return

    selected_user_id = st.selectbox("Sélectionner un utilisateur", user_ids)
    selected_user = df_users[df_users['id'] == selected_user_id].iloc[0]

    with st.form("edit_user_form"):
        username = st.text_input("Nom d'utilisateur", value=selected_user["username"])
        role = st.selectbox("Rôle", ["admin", "manager", "employee"],
                            index=["admin", "manager", "employee"].index(selected_user["role"]))
        new_password = st.text_input("Nouveau mot de passe (laisse vide pour ne pas changer)", type="password")

        name, department, manager_id = None, None, None
        if role == "employee":
            name = st.text_input("Nom complet", value=selected_user["name"] or "")
            department = st.text_input("Département", value=selected_user["department"] or "")
            cursor.execute("SELECT id, username FROM users WHERE role = 'manager'")
            managers = cursor.fetchall()
            manager_id = st.selectbox("Manager", [m[0] for m in managers],
                                      format_func=lambda x: [m[1] for m in managers if m[0] == x][0],
                                      index=next(
                                          (i for i, m in enumerate(managers) if m[1] == selected_user["manager_name"]),
                                          0) if managers else 0)

        col1, col2 = st.columns(2)
        with col1:
            update_btn = st.form_submit_button("Mettre à jour")
        with col2:
            delete_btn = st.form_submit_button("Supprimer")

        if update_btn:
            try:
                cursor.execute("UPDATE users SET username = ?, role = ? WHERE id = ?",
                               (username, role, selected_user_id))
                if new_password:
                    hashed_pwd = hash_password(new_password)
                    cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_pwd, selected_user_id))

                if role == "employee":
                    cursor.execute("SELECT * FROM employees WHERE user_id = ?", (selected_user_id,))
                    if cursor.fetchone():
                        cursor.execute("""
                            UPDATE employees
                            SET name = ?, department = ?, manager_id = ?
                            WHERE user_id = ?
                        """, (name, department, manager_id, selected_user_id))
                    else:
                        cursor.execute("""
                            INSERT INTO employees (user_id, name, department, manager_id)
                            VALUES (?, ?, ?, ?)
                        """, (selected_user_id, name, department, manager_id))
                else:
                    cursor.execute("DELETE FROM employees WHERE user_id = ?", (selected_user_id,))
                conn.commit()
                st.success("✅ Utilisateur mis à jour.")
            except Exception as e:
                st.error(f"⚠️ Erreur lors de la mise à jour : {e}")

        if delete_btn:
            try:
                cursor.execute("DELETE FROM employees WHERE user_id = ?", (selected_user_id,))
                cursor.execute("DELETE FROM users WHERE id = ?", (selected_user_id,))
                conn.commit()
                st.success("🗑️ Utilisateur supprimé.")
            except Exception as e:
                st.error(f"⚠️ Erreur lors de la suppression : {e}")

    # --- Ajout d’un utilisateur ---
    st.subheader("➕ Ajouter un utilisateur")
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["admin", "manager", "employee"])

        name, department, manager_id = None, None, None
        if new_role == "employee":
            name = st.text_input("Nom de l'employé")
            department = st.text_input("Département")
            cursor.execute("SELECT id, username FROM users WHERE role = 'manager'")
            managers = cursor.fetchall()
            if managers:
                manager_id = st.selectbox("Manager", [m[0] for m in managers],
                                          format_func=lambda x: [m[1] for m in managers if m[0] == x][0])
            else:
                st.warning("Aucun manager trouvé. Créez un manager d'abord.")

        create_btn = st.form_submit_button("Créer l'utilisateur")
        if create_btn:
            if new_username and new_password:
                if new_role == "employee":
                    if not name or not department or not manager_id:
                        st.warning("Remplissez tous les champs pour l'employé.")
                    else:
                        if add_user(new_username, new_password, new_role, manager_id, name, department):
                            st.success(f"✅ Utilisateur '{new_username}' créé.")
                        else:
                            st.error("⚠️ Erreur lors de la création.")
                else:
                    if add_user(new_username, new_password, new_role):
                        st.success(f"✅ Utilisateur '{new_username}' créé.")
                    else:
                        st.error("⚠️ Erreur lors de la création.")
            else:
                st.warning("Tous les champs sont requis.")

    conn.close()


def get_departments():
    """Retrieve all unique departments from the database."""
    conn = connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT department FROM employees WHERE department IS NOT NULL")
    departments = [row[0] for row in cursor.fetchall()]
    conn.close()
    return departments


def get_team_feedback(department=None):
    conn = sqlite3.connect('database/app.db')
    
    query = """
    SELECT 
        u.username,
        f.feedback_date,
        f.feedback_text,
        f.category,
        f.satisfaction_level,
        e.department
    FROM feedback f
    JOIN users u ON f.employee_id = u.id
    JOIN employees e ON u.id = e.user_id
    """
    
    if department:
        query += " WHERE e.department = ?"
        df = pd.read_sql_query(query, conn, params=(department,))
    else:
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

def display_feedback_analytics(feedback_df):
    if feedback_df.empty:
        st.info("Aucun feedback disponible pour le moment.")
        return
    
    st.subheader("📊 Analyse des Feedbacks")
    
    # Convert dates
    feedback_df['feedback_date'] = pd.to_datetime(feedback_df['feedback_date'])
    
    # Satisfaction moyenne par département
    col1, col2 = st.columns(2)
    
    with col1:
        avg_satisfaction = feedback_df.groupby('department')['satisfaction_level'].mean()
        fig = px.bar(
            avg_satisfaction,
            title="Satisfaction Moyenne par Département",
            labels={'value': 'Niveau de Satisfaction', 'department': 'Département'}
        )
        st.plotly_chart(fig)
    
    with col2:
        category_counts = feedback_df['category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution des Catégories de Feedback"
        )
        st.plotly_chart(fig)
    
    # Évolution temporelle
    st.subheader("📈 Évolution de la Satisfaction")
    time_trend = feedback_df.groupby([pd.Grouper(key='feedback_date', freq='W')])['satisfaction_level'].mean()
    fig = px.line(
        x=time_trend.index,
        y=time_trend.values,
        labels={'x': 'Date', 'y': 'Satisfaction Moyenne'},
        title="Tendance de Satisfaction (Moyenne Hebdomadaire)"
    )
    st.plotly_chart(fig)
    
    # Feedbacks récents
    st.subheader("📝 Feedbacks Récents")
    recent_feedback = feedback_df.sort_values('feedback_date', ascending=False).head(5)
    for _, row in recent_feedback.iterrows():
        with st.expander(f"{row['username']} - {row['category']} - {row['feedback_date'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(row['feedback_text'])
            st.write(f"Satisfaction: {'⭐' * row['satisfaction_level']}")

def display_team_performance(team_data):
    """Display team performance metrics and visualizations."""
    if team_data.empty:
        st.warning("Aucune donnée d'équipe disponible pour ce département.")
        return

    # Overall team metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_tasks = team_data['tasks_completed'].mean()
        st.metric("Tâches Moyennes", f"{avg_tasks:.1f}")
    
    with col2:
        avg_hours = team_data['hours_worked'].mean()
        st.metric("Heures Moyennes", f"{avg_hours:.1f}")
    
    with col3:
        avg_feedback = team_data['feedback_score'].mean()
        st.metric("Score Feedback Moyen", f"{avg_feedback:.1f}/5")
    
    with col4:
        avg_punctuality = team_data['punctuality_rate'].mean()
        st.metric("Taux Ponctualité Moyen", f"{avg_punctuality:.1%}")

    # Performance distribution
    st.subheader("Distribution des Performances")
    fig_perf = px.histogram(team_data, x='performance_score',
                           title="Distribution des Scores de Performance",
                           labels={'performance_score': 'Score de Performance'})
    st.plotly_chart(fig_perf)

    # Task completion vs Hours worked scatter plot
    st.subheader("Tâches vs Heures Travaillées")
    fig_scatter = px.scatter(team_data, x='hours_worked', y='tasks_completed',
                            title="Relation entre Heures Travaillées et Tâches Complétées",
                            labels={'hours_worked': 'Heures Travaillées',
                                   'tasks_completed': 'Tâches Complétées'})
    st.plotly_chart(fig_scatter)

def display_predictions(department):
    """Display predictions for the selected department (placeholder)."""
    st.info(f"Fonctionnalité de prédiction à venir pour le département : {department if department else 'Tous'}.")

def manager_dashboard():
    st.title("👥 Tableau de Bord Manager")
    
    # Get department selection
    departments = get_departments()
    selected_department = st.selectbox(
        "Sélectionner un département",
        ["Tous"] + departments
    )
    
    department = None if selected_department == "Tous" else selected_department
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Performance", "Feedback", "Prédictions"])
    
    with tab1:
        st.subheader("📊 Performance de l'Équipe")
        team_data = load_team_data()
        display_team_performance(team_data)
    
    with tab2:
        st.subheader("💭 Feedback de l'Équipe")
        feedback_data = get_team_feedback(department)
        display_feedback_analytics(feedback_data)
    
    with tab3:
        st.subheader("🔮 Prédictions")
        display_predictions(department)

def main():
    st.title("👋 Bienvenue dans l'application de gestion des performances")
    st.write("Cette application permet aux administrateurs de gérer les performances des employés.")
    
    # Afficher les dashboards
    dashboard = st.selectbox("Sélectionner un dashboard", ["Administrateur", "Manager"])
    
    if dashboard == "Administrateur":
        admin_dashboard()
    elif dashboard == "Manager":
        manager_dashboard()


if __name__ == "__main__":
    main()
