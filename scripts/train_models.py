import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Créer les dossiers si inexistant
os.makedirs("data", exist_ok=True)
os.makedirs("ml_models", exist_ok=True)

# Génération de données synthétiques
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'Age': np.random.randint(22, 60, n),
    'Experience': np.random.randint(1, 35, n),
    'Diploma': np.random.choice([1, 2, 3], n, p=[0.3, 0.4, 0.3]),  # 1: Bac, 2: Bac+3, 3: Bac+5+
    'Industry': np.random.choice(['Tech', 'Finance', 'Health', 'Education'], n),
    'HoursWorkedPerWeek': np.random.randint(30, 60, n),
    'AbsencesCount': np.random.poisson(3, n),
    'TeamProjectScore': np.random.randint(0, 11, n),
    'PeerFeedbackScore': np.random.randint(0, 11, n),
    'PunctualityRate': np.round(np.random.uniform(0.0, 1.0, n), 2),
    'TasksCompleted': np.random.randint(0, 100, n),
    'Performance': np.random.randint(60, 100, n).astype(float)
})

# Introduire 20% de valeurs manquantes dans 'Performance'
missing_indices = df.sample(frac=0.2).index
df.loc[missing_indices, 'Performance'] = np.nan

# Sauvegarde
df.to_csv("data/generated_employees.csv", index=False)
print("✅ Données synthétiques générées et sauvegardées.")

# 🔁 Supprimer la colonne Industry (on ne l'utilise pas dans l'interface manager)
df_encoded = df.drop(columns=["Industry"])

# Séparation
df_train = df_encoded[df_encoded["Performance"].notna()]  # Entraîner avec les lignes où 'Performance' n'est pas manquant
df_test = df_encoded[df_encoded["Performance"].isna()]  # Lignes où 'Performance' est manquant (pour prédiction plus tard)
features = [col for col in df_encoded.columns if col != "Performance"]

# Sauvegarder l'ordre des colonnes utilisées pour l'entraînement
model_columns = features  # Liste des colonnes utilisées pour l'entraînement
joblib.dump(model_columns, 'ml_models/column_names.pkl')

# Finaliser les données
X = df_train[features]
y = df_train["Performance"]

# Entraînement des modèles
models = {
    "linear_model": LinearRegression(),
    "rf_model": RandomForestRegressor(n_estimators=100, random_state=42),
    "xgb_model": XGBRegressor(n_estimators=100, random_state=42)
}

# Fonction d'évaluation du modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Calculer les métriques d'évaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Erreur quadratique moyenne
    mae = mean_absolute_error(y_test, y_pred)  # Erreur absolue moyenne
    r2 = r2_score(y_test, y_pred)  # Coefficient de détermination
    return rmse, mae, r2

# Entraînement et évaluation des modèles
for name, model in models.items():
    print(f"🔧 Entraînement du modèle {name}...")
    model.fit(X, y)
    joblib.dump(model, f"ml_models/{name}.pkl")
    print(f"✅ Modèle {name} sauvegardé.")

    # Évaluation du modèle sur les données de test (les lignes où 'Performance' est manquant)
    if df_test.shape[0] > 0:
        # Prédire les valeurs de Performance pour les lignes avec des valeurs manquantes
        X_test = df_test[features]
        predicted_performance = model.predict(X_test)
        df_test["Performance"] = predicted_performance  # Ajouter les prédictions dans la colonne 'Performance'

        # Évaluation sur les données de test
        rmse, mae, r2 = evaluate_model(model, df_test[features], df_test["Performance"])
        print(f"🎯 Évaluation du modèle {name}: RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")
    else:
        print(f"⚠️ Aucun échantillon de test disponible pour {name}.")

print("🎉 Tous les modèles sont entraînés et sauvegardés dans le dossier 'ml_models/'.")
