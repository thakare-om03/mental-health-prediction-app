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
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import shap
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

###############################################
# 1. Common Data Loading and Preprocessing
###############################################
# Read the CSV file into a DataFrame.
data_path = os.path.join("data", "raw", "depression_anxiety_data.csv")
data = pd.read_csv(data_path)

# Remove duplicate rows to ensure data quality.
data.drop_duplicates(inplace=True)

# Fill missing numeric values with the median and categorical values with the mode.
data.fillna(data.median(numeric_only=True), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode all categorical variables using LabelEncoder.
# This converts text labels into numerical values.
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

###############################################
# 2. Define an Evaluation Function
###############################################
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on test data and print overall performance metrics.
    
    Parameters:
      model: The trained model to be evaluated.
      X_test: Test feature matrix.
      y_test: Test target vector.
      model_name: Name of the model for display.
    
    Returns:
      A dictionary containing Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        roc = np.nan
    # Print detailed model performance
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC-AUC': roc}

###############################################
# 3. OLD PIPELINE: Target = 'suicidal'
###############################################
# This section implements the original pipeline using 'suicidal' as the target.

# Separate features and target. Remove the 'suicidal' column and 'id' from features.
X_old = data.drop(columns=['suicidal', 'id'], errors='ignore')
y_old = data['suicidal']

# Use a Random Forest to determine feature importance and select the top 10 features.
rf_fs_old = RandomForestClassifier(random_state=42)
rf_fs_old.fit(X_old, y_old)
fi_old = pd.DataFrame({'Feature': X_old.columns, 'Importance': rf_fs_old.feature_importances_})
fi_old.sort_values(by='Importance', ascending=False, inplace=True)
top_features_old = fi_old["Feature"].values[:10]
X_old = X_old[top_features_old]

# Split the data into training and testing sets.
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X_old, y_old, test_size=0.2, random_state=42
)

# Standardize features using StandardScaler.
scaler_old = StandardScaler()
X_train_old = scaler_old.fit_transform(X_train_old)
X_test_old = scaler_old.transform(X_test_old)

# Define hyperparameter grids for the old models.
rf_params_old = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
xgb_params_old = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Perform Grid Search with Cross-Validation for Random Forest using 'accuracy' as scoring.
rf_grid_old = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params_old, cv=3, scoring='accuracy', n_jobs=-1
)
rf_grid_old.fit(X_train_old, y_train_old)
rf_best_old = rf_grid_old.best_estimator_

# Perform Grid Search for XGBoost using 'accuracy' as scoring.
xgb_grid_old = GridSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    xgb_params_old, cv=3, scoring='accuracy', n_jobs=-1
)
xgb_grid_old.fit(X_train_old, y_train_old)
xgb_best_old = xgb_grid_old.best_estimator_

# Evaluate the old models and store their metrics.
metrics_rf_old = evaluate_model(rf_best_old, X_test_old, y_test_old, "Old Random Forest (suicidal)")
metrics_xgb_old = evaluate_model(xgb_best_old, X_test_old, y_test_old, "Old XGBoost (suicidal)")

###############################################
# 4. NEW PIPELINE: Target = 'anxiety_diagnosis'
###############################################
# This section implements the improved pipeline using 'anxiety_diagnosis' as the target.

# Separate features and target. Remove 'anxiety_diagnosis' and 'id' from features.
X_new = data.drop(columns=['anxiety_diagnosis', 'id'], errors='ignore')
y_new = data['anxiety_diagnosis']

# Use a Random Forest to determine feature importance and select the top 10 features.
rf_fs_new = RandomForestClassifier(random_state=42)
rf_fs_new.fit(X_new, y_new)
fi_new = pd.DataFrame({'Feature': X_new.columns, 'Importance': rf_fs_new.feature_importances_})
fi_new.sort_values(by='Importance', ascending=False, inplace=True)
top_features_new = fi_new["Feature"].values[:10]
X_new = X_new[top_features_new]

# Split the new data into training and testing sets.
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# Standardize features for the new pipeline.
scaler_new = StandardScaler()
X_train_new = scaler_new.fit_transform(X_train_new)
X_test_new = scaler_new.transform(X_test_new)

# Compute the ratio of negative to positive cases for XGBoost to handle class imbalance.
neg_count_new = sum(y_train_new == 0)
pos_count_new = sum(y_train_new == 1)
scale_pos_weight_new = neg_count_new / pos_count_new if pos_count_new != 0 else 1

# Define hyperparameter grids for the new models.
rf_params_new = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
xgb_params_new = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [scale_pos_weight_new]  # This parameter addresses class imbalance.
}

# Perform Grid Search for the new Random Forest using 'f1' as scoring and class balancing.
rf_grid_new = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    rf_params_new, cv=3, scoring='f1', n_jobs=-1
)
rf_grid_new.fit(X_train_new, y_train_new)
rf_best_new = rf_grid_new.best_estimator_

# Perform Grid Search for the new XGBoost using 'f1' as scoring.
xgb_grid_new = GridSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    xgb_params_new, cv=3, scoring='f1', n_jobs=-1
)
xgb_grid_new.fit(X_train_new, y_train_new)
xgb_best_new = xgb_grid_new.best_estimator_

# Evaluate the new models and store their metrics.
metrics_rf_new = evaluate_model(rf_best_new, X_test_new, y_test_new, "New Random Forest (anxiety_diagnosis)")
metrics_xgb_new = evaluate_model(xgb_best_new, X_test_new, y_test_new, "New XGBoost (anxiety_diagnosis)")

###############################################
# 5. Compare Old vs. New Models and Calculate Average Accuracy
###############################################
# Combine the metrics from all four models into a single DataFrame.
comparison_data = {
    "Old RF (suicidal)": metrics_rf_old,
    "Old XGB (suicidal)": metrics_xgb_old,
    "New RF (anxiety_diagnosis)": metrics_rf_new,
    "New XGB (anxiety_diagnosis)": metrics_xgb_new,
}
comparison_df = pd.DataFrame(comparison_data).T
print("Overall Model Comparison:")
print(comparison_df)

# Calculate the average accuracy across all models.
avg_accuracy_suicidal = ( metrics_rf_old['Accuracy'] + metrics_xgb_old['Accuracy'] ) / 2
avg_accuracy_anxiety = ( metrics_rf_new['Accuracy'] + metrics_xgb_new['Accuracy'] ) / 2
print("Average Accuracy across all models: {:.4f}%".format(avg_accuracy_suicidal*100))
print("Average Accuracy across all models: {:.4f}%".format(avg_accuracy_anxiety*100))

# Plot a grouped bar chart for the performance metrics.
metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
# Convert all metric values to numeric type.
for metric in metrics_list:
    comparison_df[metric] = pd.to_numeric(comparison_df[metric], errors='coerce')

ax = comparison_df[metrics_list].plot(kind='bar', figsize=(12, 8))
plt.title("Model Performance Comparison: Old vs New Pipelines")
plt.ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("models/model_comparison.png")
plt.show()

###############################################
# 6. SHAP Interpretation for New Random Forest Model
###############################################
# Create a SHAP explainer object for the new Random Forest model.
explainer = shap.TreeExplainer(rf_best_new, X_train_new)
shap_values = explainer.shap_values(X_test_new, check_additivity=False)
# For binary classification, select the SHAP values corresponding to the positive class.
if isinstance(shap_values, list):
    shap_vals_to_plot = shap_values[1]
else:
    shap_vals_to_plot = shap_values

# Generate and save a SHAP summary plot.
shap.summary_plot(shap_vals_to_plot, X_test_new, show=False)
plt.savefig('models/shap_summary.png')
plt.close()

###############################################
# 7. Save New Models and Artifacts
###############################################
# Save the new trained models, scaler, and selected features using joblib.
os.makedirs("models", exist_ok=True)
joblib.dump(rf_best_new, os.path.join("models", "new_random_forest_model.pkl"))
joblib.dump(xgb_best_new, os.path.join("models", "new_xgboost_model.pkl"))
joblib.dump(scaler_new, os.path.join("models", "new_scaler.pkl"))
joblib.dump(top_features_new, os.path.join("models", "new_selected_features.pkl"))

print("Data preprocessing, model training, and evaluation complete. Models and artifacts saved.")