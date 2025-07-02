import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

DB_PATH = "database/app.db"

def connect_db():
    return sqlite3.connect(DB_PATH)

def load_data():
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM performance_data", conn)
    return df


def view_performance_history(employee_id):
    conn = connect_db()
    df = pd.read_sql_query(
        """
        SELECT prediction_date, performance_score 
        FROM performance_history 
        WHERE employee_id = ?
        ORDER BY prediction_date DESC
        """, 
        conn, 
        params=(employee_id,)
    )
    conn.close()
    
    if not df.empty:
        st.line_chart(df.set_index('prediction_date'))
    else:
        st.write("No performance history available")
    
    return df


def get_team_average(employee_id):
    conn = connect_db()
    
    # Get employee's department
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.department 
        FROM employees e 
        WHERE e.user_id = ?
    """, (employee_id,))
    department = cursor.fetchone()[0]
    
    # Get team members in same department
    team_df = pd.read_sql_query(
        """
        SELECT ph.* 
        FROM performance_history ph
        JOIN users u ON ph.employee_id = u.id
        JOIN employees e ON e.user_id = u.id
        WHERE e.department = ?
        ORDER BY ph.prediction_date
        """,
        conn,
        params=(department,)
    )
    conn.close()
    
    if not team_df.empty:
        return team_df.groupby('prediction_date')['performance_score'].mean().reset_index()
    return pd.DataFrame()


def calculate_performance_stats(scores):
    """Calculate detailed performance statistics."""
    if len(scores) < 2:
        return None
        
    stats_dict = {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'trend': np.polyfit(range(len(scores)), scores, 1)[0],
        'consistency': 1 - (np.std(scores) / np.mean(scores))  # coefficient of variation
    }
    
    # Calculate percentile ranks
    stats_dict['percentile_25'] = np.percentile(scores, 25)
    stats_dict['percentile_75'] = np.percentile(scores, 75)
    
    return stats_dict

def analyze_performance_trend(scores):
    """Analyze the trend in performance scores."""
    if len(scores) < 2:
        return None
        
    # Calculate moving average
    window = min(5, len(scores) // 2)
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    
    # Calculate trend
    trend = np.polyfit(range(len(scores)), scores, 1)[0]
    
    # Perform statistical test for trend significance
    correlation = stats.pearsonr(range(len(scores)), scores)[0]
    
    return {
        'trend': trend,
        'correlation': correlation,
        'moving_avg': moving_avg
    }

def get_performance_insights(individual_scores, team_scores):
    """Generate insights by comparing individual and team performance."""
    if len(individual_scores) < 2 or len(team_scores) < 2:
        return None
        
    insights = []
    
    # Compare means
    ind_mean = np.mean(individual_scores)
    team_mean = np.mean(team_scores)
    diff = ind_mean - team_mean
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(individual_scores, team_scores)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(individual_scores) + np.var(team_scores)) / 2)
    cohens_d = diff / pooled_std
    
    return {
        'mean_difference': diff,
        'significance': p_value < 0.05,
        'effect_size': cohens_d,
        'p_value': p_value
    }

def submit_feedback(employee_id, feedback_text, category, satisfaction_level):
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO feedback (employee_id, feedback_text, category, satisfaction_level)
    VALUES (?, ?, ?, ?)
    """, (employee_id, feedback_text, category, satisfaction_level))
    
    conn.commit()
    conn.close()

def get_employee_feedback_history(employee_id):
    conn = connect_db()
    query = """
    SELECT feedback_date, feedback_text, category, satisfaction_level
    FROM feedback
    WHERE employee_id = ?
    ORDER BY feedback_date DESC
    """
    df = pd.read_sql_query(query, conn, params=(employee_id,))
    conn.close()
    return df

def feedback_section(employee_id):
    st.subheader("📝 Feedback")
    
    # Feedback form
    with st.form("feedback_form"):
        feedback_text = st.text_area("Votre feedback:", height=100)
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox(
                "Catégorie:",
                ["Général", "Environnement de Travail", "Management", "Projets", "Formation", "Autre"]
            )
        
        with col2:
            satisfaction = st.slider(
                "Niveau de satisfaction:",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = Très insatisfait, 5 = Très satisfait"
            )
        
        submit = st.form_submit_button("Soumettre le Feedback")
        
        if submit and feedback_text:
            submit_feedback(employee_id, feedback_text, category, satisfaction)
            st.success("Feedback soumis avec succès!")
            st.rerun()
    
    # Show feedback history
    st.subheader("Historique des Feedbacks")
    feedback_history = get_employee_feedback_history(employee_id)
    
    if not feedback_history.empty:
        # Convert feedback_date to datetime if it's not already
        feedback_history['feedback_date'] = pd.to_datetime(feedback_history['feedback_date'])
        
        # Display satisfaction trend
        fig = px.line(
            feedback_history,
            x='feedback_date',
            y='satisfaction_level',
            title='Évolution de la Satisfaction'
        )
        st.plotly_chart(fig)
        
        # Display feedback history in a table
        for _, row in feedback_history.iterrows():
            with st.expander(f"{row['category']} - {row['feedback_date'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(row['feedback_text'])
                st.write(f"Satisfaction: {'⭐' * row['satisfaction_level']}")
    else:
        st.info("Aucun feedback soumis pour le moment.")

def employee_dashboard():
    st.subheader("🙋‍♂️ Employee Dashboard")
    
    # Get employee data
    df = load_data()
    employee_data = df[df["employee_id"] == st.session_state["user"][0]]
    
    # Performance Overview
    st.subheader("📊 Performance Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        if not employee_data.empty:
            latest_score = employee_data['performance_score'].iloc[-1]
            st.metric("Latest Performance Score", f"{latest_score:.2f}/5.0")
    
    # Performance History Section
    st.subheader("📈 Your Performance History")
    history = view_performance_history(st.session_state["user"][0])
    
    if not history.empty:
        # Convert dates to datetime
        history['prediction_date'] = pd.to_datetime(history['prediction_date'])
        
        # Add date filter
        min_date = history['prediction_date'].min().date()
        max_date = history['prediction_date'].max().date()
        
        dates = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(dates) == 2:
            start_date, end_date = dates
            filtered_history = history[
                (history['prediction_date'].dt.date >= start_date) & 
                (history['prediction_date'].dt.date <= end_date)
            ]
            
            # Calculate detailed statistics
            scores = filtered_history['performance_score'].values
            stats_dict = calculate_performance_stats(scores)
            trend_analysis = analyze_performance_trend(scores)
            
            if stats_dict and trend_analysis:
                # Display Statistics
                st.subheader("📊 Performance Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Score", f"{stats_dict['mean']:.2f}")
                    st.metric("Consistency", f"{stats_dict['consistency']*100:.1f}%")
                
                with col2:
                    st.metric("Highest Score", f"{stats_dict['max']:.2f}")
                    st.metric("Lowest Score", f"{stats_dict['min']:.2f}")
                
                with col3:
                    trend_icon = "↗️" if trend_analysis['trend'] > 0 else "↘️"
                    st.metric("Score Trend", f"{trend_icon} {abs(trend_analysis['trend']):.3f}/day")
                    st.metric("Variability", f"±{stats_dict['std']:.2f}")
                
                # Performance Distribution
                st.subheader("📊 Score Distribution")
                fig_col1, fig_col2 = st.columns(2)
                
                with fig_col1:
                    # Create histogram
                    hist_data = [scores]
                    group_labels = ['Scores']
                    import plotly.figure_factory as ff
                    fig1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
                    fig1.update_layout(title="Score Distribution")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with fig_col2:
                    # Create box plot
                    import plotly.graph_objects as go
                    fig2 = go.Figure()
                    fig2.add_trace(go.Box(y=scores, name="Performance Scores"))
                    fig2.update_layout(title="Score Range Analysis")
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Team Comparison with Enhanced Analysis
            st.subheader("👥 Team Comparison")
            team_avg = get_team_average(st.session_state["user"][0])
            
            if not team_avg.empty:
                # Convert team dates to datetime
                team_avg['prediction_date'] = pd.to_datetime(team_avg['prediction_date'])
                filtered_team = team_avg[
                    (team_avg['prediction_date'].dt.date >= start_date) & 
                    (team_avg['prediction_date'].dt.date <= end_date)
                ]
                
                # Analyze individual vs team performance
                team_scores = filtered_team['performance_score'].values
                performance_insights = get_performance_insights(scores, team_scores)
                
                if performance_insights:
                    # Display comparison insights
                    st.subheader("🎯 Performance Insights")
                    
                    # Effect size interpretation
                    effect_size = abs(performance_insights['effect_size'])
                    if effect_size < 0.2:
                        effect_text = "minimal"
                    elif effect_size < 0.5:
                        effect_text = "moderate"
                    else:
                        effect_text = "substantial"
                    
                    # Display insights based on statistical analysis
                    if performance_insights['significance']:
                        if performance_insights['mean_difference'] > 0:
                            st.success(f"🌟 Your performance is significantly higher than the team average (p < {performance_insights['p_value']:.3f})")
                            st.info(f"The difference is {effect_text} (Cohen's d = {effect_size:.2f})")
                        else:
                            st.warning(f"📈 Your performance is significantly lower than the team average (p < {performance_insights['p_value']:.3f})")
                            st.info(f"The gap is {effect_text} (Cohen's d = {effect_size:.2f})")
                    else:
                        st.info("Your performance is statistically similar to the team average")
                
                # Plot comparison
                comparison = filtered_history.merge(
                    filtered_team,
                    on='prediction_date',
                    suffixes=('_individual', '_team')
                )
                
                st.line_chart(
                    comparison.set_index('prediction_date')[
                        ['performance_score_individual', 'performance_score_team']
                    ].rename(columns={
                        'performance_score_individual': 'Your Score',
                        'performance_score_team': 'Team Average'
                    }),
                    use_container_width=True
                )
    else:
        st.info("No performance history available yet. Check back after your first evaluation!")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Performance", "Feedback", "Statistiques"])
    
    with tab1:
        st.subheader("📈 Vos Performances")
        view_performance_history(st.session_state["user"][0])
    
    with tab2:
        feedback_section(st.session_state["user"][0])
    
    with tab3:
        st.subheader("📊 Statistiques")
        # display_employee_stats(st.session_state["user"][0])
