import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configure paths
input_path = Path(r"C:\Users\Om Thakare\OneDrive\Desktop\arogo-ai-assignment\data\raw\depression_anxiety_data.csv")
output_path = Path(r"C:\Users\Om Thakare\OneDrive\Desktop\arogo-ai-assignment\data\processed\processed_data.pkl")

# Ensure output directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load data with error handling
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    raise

# 1. Data Cleaning
# Convert boolean columns properly
bool_cols = ['depressiveness', 'suicidal', 'depression_diagnosis', 
             'depression_treatment', 'anxiousness', 'anxiety_diagnosis',
             'anxiety_treatment', 'sleepiness']

df[bool_cols] = df[bool_cols].replace({'TRUE': 1, 'FALSE': 0, 'NA': np.nan})

# Handle missing values
df.replace({'Not Availble': np.nan, 'NA': np.nan, 'none': np.nan}, inplace=True)

# Convert BMI from string to numeric
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# 2. Feature Engineering
# Create composite scores
df['mental_health_score'] = (df['phq_score'] * 0.6 + 
                             df['gad_score'] * 0.4)

# BMI categories based on WHO classification
bmi_bins = [0, 18.5, 25, 30, 35, 40, np.inf]
bmi_labels = ['Underweight', 'Normal', 'Overweight',
              'Obese I', 'Obese II', 'Obese III']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)

# 3. Data Normalization
# Encode categorical variables
cat_cols = ['gender', 'who_bmi', 'depression_severity', 'anxiety_severity']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Scale numerical features
num_cols = ['age', 'bmi', 'phq_score', 'gad_score', 'epworth_score']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. Handle Missing Data
# Impute numerical columns with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Impute categorical columns with mode
for col in cat_cols + ['bmi_category']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 5. Save Processed Data
df.to_pickle(output_path)
print(f"Data successfully processed and saved to {output_path}")
