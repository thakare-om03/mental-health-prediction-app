import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load datasets
    tech_survey = pd.read_csv('data/raw/survey.csv')
    depression_data = pd.read_csv('data/raw/depression_anxiety_data.csv')
    
    # Merge datasets
    combined = pd.concat([tech_survey, depression_data], axis=0)
    
    # Handle missing values
    combined.fillna({
        'symptoms': 'unknown',
        'severity': combined['severity'].median(),
        'age': combined['age'].mean()
    }, inplace=True)
    
    # Feature engineering
    combined['symptom_count'] = combined['symptoms'].apply(lambda x: len(x.split(',')))
    
    # Encode categorical features
    le = LabelEncoder()
    combined['diagnosis'] = le.fit_transform(combined['diagnosis'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['age', 'severity', 'symptom_count']
    combined[numerical_features] = scaler.fit_transform(combined[numerical_features])
    
    # Split data
    X = combined.drop('diagnosis', axis=1)
    y = combined['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test, le.classes_