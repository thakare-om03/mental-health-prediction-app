# temp/src/predict_mental_health.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import xgboost as xgb
import joblib
import yaml # Add pyyaml to requirements.txt
import logging
from pathlib import Path
import warnings
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# --- Data Loading and Cleaning ---
def load_and_clean_data(file_path):
    """Loads and performs initial cleaning on the dataset."""
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")

        # Basic Cleaning Example (Expand as needed)
        # Handle specific known issues like BMI=0 meaning "Not Available"
        df['bmi'] = df['bmi'].replace(0, np.nan) # Replace 0 BMI with NaN

        # Impute missing numerical values (e.g., with median) - apply to relevant columns
        for col in ['age', 'bmi', 'phq_score', 'gad_score', 'epworth_score']:
             if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logging.info(f"Imputed NaNs in '{col}' with median ({median_val}).")

        # Drop rows where the target variable is missing (if applicable)
        target_col = 'depressiveness' # Adjust if target changes
        if target_col in df.columns and df[target_col].isnull().any():
             initial_rows = len(df)
             df.dropna(subset=[target_col], inplace=True)
             logging.warning(f"Dropped {initial_rows - len(df)} rows due to missing target ('{target_col}').")

        # Handle inconsistent categorical data (Example for Gender if using survey.csv)
        # if 'Gender' in df.columns:
        #     df['Gender'] = df['Gender'].str.lower().str.strip()
        #     gender_map = {
        #         'male': 'Male', 'm': 'Male', 'maile': 'Male', 'cis male': 'Male', 'male-ish': 'Other', # Example mapping
        #         'female': 'Female', 'f': 'Female', 'cis female': 'Female', 'trans-female': 'Female', # Example mapping
        #         # Add mappings for all variations encountered
        #     }
        #     df['Gender'] = df['Gender'].map(gender_map).fillna('Other/Prefer not to say') # Map and handle unmapped
        #     logging.info("Normalized 'Gender' column.")

        logging.info("Initial data cleaning finished.")
        return df

    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error during data loading/cleaning: {e}")
        raise

# --- Feature Engineering and Preprocessing ---
def preprocess_features(df, target_col='depressiveness', test_size=0.2, random_state=42, models_dir='models'):
    """Encodes categorical features, scales numerical features, and splits data."""
    logging.info("Starting feature preprocessing...")

    # --- Drop leaky/irrelevant columns (Keep this part) ---
    columns_to_drop = [
        'id', 'depression_severity', 'depression_diagnosis',
        'depression_treatment', 'suicidal'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df_processed = df.drop(columns=columns_to_drop)
        logging.info(f"Dropped potentially leaky/irrelevant columns: {columns_to_drop}")
    else:
        df_processed = df.copy()

    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    logging.info(f"Remaining Categorical features: {list(categorical_features)}")
    logging.info(f"Remaining Numerical features: {list(numerical_features)}")

    # --- Label Encoding for Categorical Features (Keep this part) ---
    encoders = {}
    X_encoded = X.copy()
    for col in categorical_features:
        # Handle potential NaNs before encoding if necessary, e.g., fill with a placeholder string
        # X_encoded[col].fillna('Missing', inplace=True) # Example placeholder
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
        logging.info(f"Label encoded '{col}'.")

    # --- Target Encoding (Keep this part) ---
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    logging.info(f"Target variable '{target_col}' encoded.")
    logging.info(f"Target classes: {target_encoder.classes_}")

    # --- Data Splitting (Keep this part) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    logging.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # --- Scaling Numerical Features (Keep this part) ---
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    # Ensure numerical_features list is accurate *after* potential drops
    numerical_features_in_train = [col for col in numerical_features if col in X_train.columns]
    if numerical_features_in_train: # Check if there are any numerical features left
        X_train_scaled[numerical_features_in_train] = scaler.fit_transform(X_train[numerical_features_in_train])
        logging.info("Scaler fitted on training data.")
        X_test_scaled[numerical_features_in_train] = scaler.transform(X_test[numerical_features_in_train])
        logging.info("Training and testing numerical features scaled.")
    else:
        logging.warning("No numerical features found to scale.")


    # *** ADD THIS: Ensure all columns are float type before returning ***
    try:
        X_train_scaled = X_train_scaled.astype(float)
        X_test_scaled = X_test_scaled.astype(float)
        logging.info("Converted scaled training and testing DataFrames to float type.")
    except ValueError as e:
        logging.error(f"Could not convert DataFrame to float after scaling/encoding. Error: {e}")
        # Optionally: Log the dtypes to debug which column is failing
        logging.error(f"X_train_scaled dtypes before conversion attempt:\n{X_train_scaled.dtypes}")
        raise # Re-raise the error after logging


    # --- Save artifacts (Keep this part) ---
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(encoders, os.path.join(models_dir, 'label_encoders.joblib'))
    joblib.dump(target_encoder, os.path.join(models_dir, 'target_encoder.joblib'))
    # Save the final feature list (columns of X_train_scaled)
    joblib.dump(list(X_train_scaled.columns), os.path.join(models_dir, 'features.joblib'))
    # Update saved numerical/categorical lists based on final columns if needed by UI
    # It might be simpler just to save the final columns list ('features.joblib')
    # and have the UI determine num/cat based on the loaded data's dtypes before scaling.
    # For now, let's keep saving the original lists identified before split/scale.
    joblib.dump(numerical_features.tolist(), os.path.join(models_dir, 'numerical_features.joblib'))
    joblib.dump(categorical_features.tolist(), os.path.join(models_dir, 'categorical_features.joblib'))
    logging.info(f"Scaler, encoders, and updated feature lists saved to {models_dir}.")


    return X_train_scaled, X_test_scaled, y_train, y_test, target_encoder

# --- Model Training ---
def train_evaluate_model(X_train, y_train, X_test, y_test, model_type='random_forest', config=None, target_encoder=None):
    """Trains and evaluates a specified model type."""
    logging.info(f"Starting training for {model_type}...")

    # --- Model Selection and Training (Random Forest part shown) ---
    if model_type == 'random_forest':
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=1)
        grid_search_rf.fit(X_train, y_train)
        best_rf = grid_search_rf.best_estimator_
        model = best_rf
        logging.info(f"Random Forest best params: {grid_search_rf.best_params_}")
        model_filename = config['models']['random_forest_file']

    elif model_type == 'xgboost':
         # (Keep XGBoost logic as before)
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        # Ensure num_class is correctly set for multi-class if needed, for binary it's usually handled
        num_classes = len(target_encoder.classes_)
        xgb_clf = xgb.XGBClassifier(objective='binary:logistic' if num_classes == 2 else 'multi:softmax',
                                    # num_class=num_classes if num_classes > 2 else None, # Usually not needed for binary
                                    random_state=42, use_label_encoder=False, eval_metric='logloss' if num_classes == 2 else 'mlogloss')
        grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=1)
        grid_search_xgb.fit(X_train, y_train)
        best_xgb = grid_search_xgb.best_estimator_
        model = best_xgb
        logging.info(f"XGBoost best params: {grid_search_xgb.best_params_}")
        model_filename = config['models']['xgboost_file']
    else:
        logging.error(f"Unsupported model type: {model_type}")
        return None

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted') # Use 'binary' if it's strictly binary or 'weighted'/'macro'

    # *** FIX HERE: Convert boolean class names to strings ***
    target_names_str = [str(cls) for cls in target_encoder.classes_]

    # Now use the string version for the report
    report = classification_report(y_test, y_pred, target_names=target_names_str)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"--- {model_type} Evaluation ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score (Weighted): {f1:.4f}")
    logging.info(f"Classification Report:\n{report}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")

    # Save the trained model
    model_path = os.path.join(config['models']['output_dir'], model_filename)
    joblib.dump(model, model_path)
    logging.info(f"{model_type} model saved to {model_path}")

    return model

# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning) # Suppress XGBoost warning if needed

    try:
        config = load_config() # Load config from default path "config.yaml"
        models_dir = Path(config['models']['output_dir'])
        models_dir.mkdir(parents=True, exist_ok=True) # Ensure model dir exists

        data_path = config['data']['raw_depression_anxiety']
        df_cleaned = load_and_clean_data(data_path)

        if df_cleaned is not None and not df_cleaned.empty:
            X_train, X_test, y_train, y_test, target_encoder = preprocess_features(
                df_cleaned,
                models_dir=str(models_dir) # Pass models_dir as string
                # target_col='depressiveness' # Keep default or specify if different
            )

            # Train and evaluate models
            logging.info("\n--- Training Random Forest ---")
            rf_model = train_evaluate_model(X_train, y_train, X_test, y_test, 'random_forest', config, target_encoder)

            logging.info("\n--- Training XGBoost ---")
            xgb_model = train_evaluate_model(X_train, y_train, X_test, y_test, 'xgboost', config, target_encoder)

            logging.info("\nScript finished successfully.")

    except Exception as e:
        logging.critical(f"Script failed: {e}", exc_info=True) # Log full traceback