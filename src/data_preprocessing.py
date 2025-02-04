# src/data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import shap
import os

sns.set_theme(style="whitegrid")

def load_data(file_path):
    """Load and validate raw data"""
    df = pd.read_csv(file_path)
    
    # Age validation with outlier handling
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[df['Age'].between(15, 100)].copy()
    
    return df

def clean_gender(df):
    """Standardize gender entries with expanded mapping"""
    gender_map = {
        r'^male$|^m$|^man$|^mail$|^cis male$|^mal$|^maile$': 'Male',
        r'^female$|^f$|^woman$|^fem$|^cis female$': 'Female',
        r'trans|non-binary|queer|enby|genderqueer': 'Non-binary',
        r'other|androgyne|something kinda male\?': 'Other'
    }
    df['Gender'] = df['Gender'].str.lower().replace(gender_map, regex=True)
    return df

def perform_eda(df):
    """Automated Exploratory Data Analysis"""
    print("\n=== EDA Report ===")
    
    # Demographic Summary
    print("\nDemographic Summary:")
    print(df[['Age', 'Gender', 'Country']].describe(include='all'))
    
    # Treatment Distribution
    plt.figure(figsize=(10,6))
    sns.countplot(x='treatment', data=df)
    plt.title('Treatment Seeking Distribution')
    plt.savefig('reports/treatment_distribution.png')
    plt.close()
    
    # Correlation Matrix
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('reports/correlation_matrix.png')
    plt.close()

def engineer_features(df):
    """Advanced feature engineering pipeline"""
    # Mental health severity score
    mh_columns = ['work_interfere', 'anonymity', 'leave', 'mental_health_consequence']
    encoder = OrdinalEncoder(categories=[['Never', 'Rarely', 'Sometimes', 'Often', 'Don\'t know']]*4,
                           handle_unknown='use_encoded_value', unknown_value=np.nan)
    df['mh_severity'] = encoder.fit_transform(df[mh_columns]).mean(axis=1)
    
    # Temporal features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['survey_hour'] = df['Timestamp'].dt.hour
    df['is_weekend'] = df['Timestamp'].dt.weekday >= 5
    
    # Text processing
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    comment_features = tfidf.fit_transform(df['comments'].fillna(''))
    comment_df = pd.DataFrame(comment_features.toarray(), 
                            columns=[f"comment_{x}" for x in tfidf.get_feature_names_out()])
    
    return pd.concat([df, comment_df], axis=1)

def feature_selection(df, target='treatment'):
    """Hybrid feature selection strategy"""
    X = df.drop(columns=[target])
    y = df[target]
    
    # Mutual Information
    selector = SelectKBest(mutual_info_classif, k=20)
    selector.fit(X, y)
    mi_features = X.columns[selector.get_support()]
    
    # Tree-based Importance
    rf = RandomForestClassifier()
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
    rf_features = rf_importances.nlargest(15).index
    
    # Combine selected features
    selected_features = list(set(mi_features).union(set(rf_features)))
    
    # SHAP Analysis
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('reports/feature_importance_shap.png')
    plt.close()
    
    return selected_features

def preprocess_pipeline(input_path, output_path):
    """End-to-end preprocessing workflow"""
    os.makedirs('reports', exist_ok=True)
    
    # Load and clean data
    df = load_data(input_path)
    df = clean_gender(df)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Perform EDA
    perform_eda(df)
    
    # Feature selection
    features = feature_selection(df)
    df = df[features + ['treatment']]
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(df.drop('treatment', axis=1))
    df = pd.concat([pd.DataFrame(reduced_features), df['treatment']], axis=1)
    
    # Save processed data
    df.to_pickle(output_path)
    print(f"\nProcessed data shape: {df.shape}")
    return df

if __name__ == "__main__":
    raw_path = r'data\raw\survey.csv'
    processed_path = r'data\processed\processed_data.pkl'
    
    processed_data = preprocess_pipeline(raw_path, processed_path)
    print(f"Preprocessing complete. Data saved to {processed_path}")