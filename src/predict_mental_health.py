import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import shap
import joblib

warnings.filterwarnings("ignore")

# ----------------------------
# 1. Data Preparation
# ----------------------------

data_path = os.path.join("data/raw/depression_anxiety_data.csv")
data = pd.read_csv(data_path)

# --- Data Cleaning & Preprocessing ---
data.drop_duplicates(inplace=True)
# Fill numeric columns with median
data.fillna(data.median(numeric_only=True), inplace=True)
# Fill categorical columns with mode
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical variables using LabelEncoder
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# --- Exploratory Data Analysis (EDA) ---
# Plot distribution of target variable
plt.figure(figsize=(10, 6))
sns.countplot(x='suicidal', data=data)
plt.title("Distribution of Suicidal Cases")
plt.savefig("models/eda_countplot.png")
plt.close()

# Plot heatmap of feature correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.savefig("models/eda_heatmap.png")
plt.close()

# --- Feature Engineering & Selection ---
X = data.drop(columns=['suicidal', 'id'], errors='ignore')
y = data['suicidal']

# Use Random Forest to rank features and select the top 10 features
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
selected_features = feature_importances["Feature"].values[:10]
X = X[selected_features]

# ----------------------------
# 2. Model Development
# ----------------------------

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Model Training with Hyperparameter Tuning ---
rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6]}

# Grid Search for RandomForest
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                       rf_params, cv=3, scoring='accuracy')
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# Grid Search for XGBoost
xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss'), 
                        xgb_params, cv=3, scoring='accuracy')
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    try:
        # For binary classification; adjust if multiclass
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    except Exception:
        roc_auc = 'N/A'
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc}")
    print(classification_report(y_test, y_pred))
    
evaluate_model(rf_best, X_test, y_test, "Random Forest")
evaluate_model(xgb_best, X_test, y_test, "XGBoost")

# --- Model Interpretation using SHAP ---
# Create a TreeExplainer for the Random Forest model
explainer = shap.TreeExplainer(rf_best, X_train)
shap_values = explainer.shap_values(X_test, check_additivity=False)
if isinstance(shap_values, list):
    shap_vals_to_plot = shap_values[1]
else:
    shap_vals_to_plot = shap_values

# Save the SHAP summary plot
shap.summary_plot(shap_vals_to_plot, X_test, show=False)
plt.savefig('models/shap_summary.png')
plt.close()

# ----------------------------
# Save Trained Models and Artifacts
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf_best, os.path.join("models/random_forest_model.pkl"))
joblib.dump(xgb_best, os.path.join("models/xgboost_model.pkl"))
joblib.dump(scaler, os.path.join("models/scaler.pkl"))
joblib.dump(selected_features, os.path.join("models/selected_features.pkl"))

print("Data preprocessing, model training, and evaluation complete. Models and artifacts saved.")