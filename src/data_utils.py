"""
Utility functions for data loading, preprocessing, and validation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_validate_data(file_path):
    """
    Load data from a CSV file and perform basic validation.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame: Validated and initial-cleaned pandas DataFrame
    """
    try:
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError(f"The file {file_path} is empty")
            
        # Log basic info about the dataset
        logging.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        logging.info(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logging.warning(f"Missing values detected: \n{missing_values[missing_values > 0]}")
            
        return df
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df, is_training=True, label_encoder_path=None, scaler_path=None):
    """
    Preprocess data by handling missing values, encoding categorical features,
    and scaling numeric features.
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data or prediction data
        label_encoder_path: Path to save/load the label encoder
        scaler_path: Path to save/load the scaler
        
    Returns:
        X: Feature matrix
        y: Target vector (if training data)
        label_encoder: Fitted label encoder (if training data)
        scaler: Fitted scaler (if training data)
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            # For categorical columns, replace with mode
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # For numerical columns, replace with median
            data[col] = data[col].fillna(data[col].median())
    
    # Handle specific data issues
    if 'BMI' in data.columns:
        # Replace invalid BMI values (e.g., 0) with median of valid values
        valid_bmi = data[data['BMI'] > 10]['BMI']  # Assuming BMI > 10 is valid
        if not valid_bmi.empty:
            median_bmi = valid_bmi.median()
            data.loc[data['BMI'] <= 10, 'BMI'] = median_bmi
            logging.info(f"Replaced invalid BMI values with median: {median_bmi}")
    
    if 'Gender' in data.columns:
        # Normalize gender values
        data['Gender'] = data['Gender'].str.lower()
        data.loc[data['Gender'].str.contains('female|f', na=False), 'Gender'] = 'female'
        data.loc[data['Gender'].str.contains('male|m', na=False), 'Gender'] = 'male'
        data.loc[~data['Gender'].isin(['male', 'female']), 'Gender'] = 'other'
        logging.info("Normalized gender values to 'male', 'female', and 'other'")
    
    # For categorical features
    for feature in [f for f in CATEGORICAL_FEATURES if f in data.columns]:
        # Convert to string type to ensure consistent handling
        data[feature] = data[feature].astype(str)
    
    # Prepare features
    X = data.drop(columns=['mental_health_disorder'], errors='ignore')
    
    # Encode categorical features
    if is_training:
        label_encoder = LabelEncoder()
        for col in [c for c in CATEGORICAL_FEATURES if c in X.columns]:
            X[col] = label_encoder.fit_transform(X[col])
        
        # Scale numeric features
        scaler = StandardScaler()
        numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
        if numeric_cols:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        
        # Save encoders if paths provided
        if label_encoder_path:
            joblib.dump(label_encoder, label_encoder_path)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            
        # Prepare target if available
        if 'mental_health_disorder' in data.columns:
            y = data['mental_health_disorder']
            return X, y, label_encoder, scaler
        else:
            return X, None, label_encoder, scaler
    else:
        # For prediction data, load pre-fitted transformers
        if label_encoder_path:
            label_encoder = joblib.load(label_encoder_path)
            for col in [c for c in CATEGORICAL_FEATURES if c in X.columns]:
                # Handle unseen categories
                unique_values = set(X[col].unique())
                known_values = set(label_encoder.classes_)
                unknown_values = unique_values - known_values
                
                if unknown_values:
                    logging.warning(f"Found unknown categories in {col}: {unknown_values}")
                    # Map unknown values to the most common class
                    most_common = X[col].value_counts().index[0]
                    for val in unknown_values:
                        X.loc[X[col] == val, col] = most_common
                        
                X[col] = label_encoder.transform(X[col])
        
        if scaler_path:
            scaler = joblib.load(scaler_path)
            numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
            if numeric_cols:
                X[numeric_cols] = scaler.transform(X[numeric_cols])
        
        return X
