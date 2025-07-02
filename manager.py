import streamlit as st
import pandas as pd
import joblib
import sqlite3

# === Chemin de la base de données
DB_PATH = "database/app.db"

# === Charger les modèles
linear_model = joblib.load('ml_models/linear_model.pkl')
rf_model = joblib.load('ml_models/rf_model.pkl')
xgb_model = joblib.load('ml_models/xgb_model.pkl')

# === Charger les noms de colonnes et métriques
model_columns = joblib.load('ml_models/column_names.pkl')
model_metrics = {
    "linear_model": {"RMSE": 3.5, "MAE": 2.8, "R²": 0.85},
    "rf_model": {"RMSE": 3.2, "MAE": 2.5, "R²": 0.88},
    "xgb_model": {"RMSE": 3.0, "MAE": 2.3, "R²": 0.90}
}

# === Fonction de prédiction
def predict_performance(model, absences, age, diploma, experience, hours_per_week, peer_feedback,
                        punctuality_rate, tasks_completed, team_project_score):
    input_data = pd.DataFrame({
        "AbsencesCount": [absences],
        "Age": [age],
        "Diploma": [diploma],
        "Experience": [experience],
        "HoursWorkedPerWeek": [hours_per_week],
        "PeerFeedbackScore": [peer_feedback],
        "PunctualityRate": [punctuality_rate],
        "TasksCompleted": [tasks_completed],
        "TeamProjectScore": [team_project_score]
    })

    input_data = input_data[model_columns]
    prediction = model.predict(input_data)
    return prediction[0]

# === Dashboard Manager
def manager_dashboard():
    st.subheader("👨‍💼 Manager Dashboard")

    manager_id = st.session_state["user"][0]  # ID manager connecté

    conn = sqlite3.connect(DB_PATH)

    # Récupération des employés supervisés
    query = """
        SELECT e.id AS employee_id, u.username, e.name, e.department
        FROM employees e
        JOIN users u ON e.user_id = u.id
        WHERE e.manager_id = ?
    """
    team_df = pd.read_sql_query(query, conn, params=(manager_id,))

    if team_df.empty:
        st.warning("Aucun employé ne vous est actuellement assigné.")
    else:
        st.markdown("### 👥 Équipe sous votre responsabilité")
        st.dataframe(team_df)

        # Bouton d'export CSV
        csv = team_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter l’équipe en CSV",
            data=csv,
            file_name='equipe_manager.csv',
            mime='text/csv'
        )

    # === Inputs pour prédiction
    st.markdown("### 🔍 Estimer la performance d’un employé")

    absences = st.number_input("Absences", min_value=0)
    age = st.number_input("Âge", min_value=18, max_value=65)
    diploma = st.selectbox("Diplôme", [1, 2, 3])
    experience = st.number_input("Expérience (années)", min_value=0)
    hours_per_week = st.number_input("Heures travaillées par semaine", min_value=0)
    peer_feedback = st.slider("Score de feedback des pairs", 0, 10)
    punctuality_rate = st.slider("Taux de ponctualité", 0.0, 1.0)
    tasks_completed = st.number_input("Tâches complétées", min_value=0)
    team_project_score = st.slider("Score de projet d’équipe", 0, 10)

    model_choice = st.selectbox("Modèle de prédiction", ["Linear Regression", "Random Forest", "XGBoost"])

    if st.button("Prédire la performance"):
        model_map = {
            "Linear Regression": linear_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model
        }
        model = model_map[model_choice]
        prediction = predict_performance(model, absences, age, diploma, experience, hours_per_week,
                                         peer_feedback, punctuality_rate, tasks_completed, team_project_score)
        st.success(f"Score de performance prédit avec {model_choice} : **{prediction:.2f}**")

    # === Métriques des modèles
    st.markdown("### 📊 Performances des modèles")
    selected_model = st.selectbox("Choisir un modèle", ["Linear Regression", "Random Forest", "XGBoost"])
    model_key_map = {
        "Linear Regression": "linear_model",
        "Random Forest": "rf_model",
        "XGBoost": "xgb_model"
    }

    model_key = model_key_map[selected_model]
    metrics = model_metrics[model_key]

    st.write(f"**RMSE**: {metrics['RMSE']:.2f}")
    st.write(f"**MAE**: {metrics['MAE']:.2f}")
    st.write(f"**R²**: {metrics['R²']:.2f}")

    conn.close()
