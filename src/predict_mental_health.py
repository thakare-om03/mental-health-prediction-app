# mental_health_model.py
import pandas as pd
import numpy as np
import shap
import gradio as gr
from transformers import pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lime.lime_tabular import LimeTabularExplainer
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================
# Data Preparation
# ======================
def load_and_preprocess_data():
    df = pd.read_csv('data/raw/depression_anxiety_data.csv')
    
    # Clean and transform data
    bool_cols = ['depressiveness', 'suicidal', 'depression_diagnosis',
                'depression_treatment', 'anxiousness', 'anxiety_diagnosis',
                'anxiety_treatment', 'sleepiness']
    
    df[bool_cols] = df[bool_cols].replace({'TRUE': 1, 'FALSE': 0, 'NA': np.nan})
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    # Feature engineering
    df['mental_health_score'] = 0.6*df['phq_score'] + 0.4*df['gad_score']
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 40, np.inf],
                               labels=['Underweight', 'Normal', 'Overweight',
                                       'Obese I', 'Obese II', 'Obese III'])
    
    # Handle missing values
    num_cols = ['age', 'bmi', 'phq_score', 'gad_score', 'epworth_score']
    cat_cols = ['gender', 'who_bmi', 'depression_severity', 'anxiety_severity']
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols + ['bmi_category']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

# ======================
# Model Development
# ======================
class MentalHealthClassifier:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.features = ['age', 'gender', 'bmi', 'phq_score', 
                        'gad_score', 'epworth_score', 'bmi_category']
        
    def train(self, df):
        X = df[self.features]
        y = df['depression_diagnosis']
        
        # Preprocessing pipeline
        numeric_features = ['age', 'bmi', 'phq_score', 'gad_score', 'epworth_score']
        categorical_features = ['gender', 'bmi_category']
        
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Model pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced'))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10]
        }
        
        self.model = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1')
        self.model.fit(X, y)
        
        # Create explainers
        self.create_explainers(X)
        
    def create_explainers(self, X):
        # SHAP explainer
        processed_data = self.preprocessor.transform(X)
        self.shap_explainer = shap.TreeExplainer(self.model.best_estimator_.named_steps['classifier'])
        
        # LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.preprocessor.transform(X),
            feature_names=self.preprocessor.get_feature_names_out(),
            class_names=['No Depression', 'Depression'],
            mode='classification'
        )
    
    def predict(self, input_data):
        processed_input = self.preprocessor.transform(input_data)
        prediction = self.model.predict(input_data)
        probability = self.model.predict_proba(input_data)
        return prediction[0], probability[0][1]
    
    def explain_prediction(self, input_data):
        # SHAP explanation
        processed_input = self.preprocessor.transform(input_data)
        shap_values = self.shap_explainer.shap_values(processed_input)
        shap_plot = shap.force_plot(self.shap_explainer.expected_value[1],
                                  shap_values[1][0],
                                  processed_input[0],
                                  feature_names=self.preprocessor.get_feature_names_out())
        
        # LIME explanation
        lime_exp = self.lime_explainer.explain_instance(processed_input[0],
                                                      self.model.best_estimator_.predict_proba,
                                                      num_features=10)
        return shap_plot, lime_exp.as_list()

# ======================
# LLM Integration
# ======================
class MentalHealthAdvisor:
    def __init__(self):
        self.llm = pipeline('text-generation', model='gpt2')
        
    def generate_advice(self, prediction, probability):
        prompt = f"""Based on the prediction of {'Depression' if prediction else 'No Depression'} 
        with {probability:.2f} confidence, provide 3 coping strategies:"""
        return self.llm(prompt, max_length=200)[0]['generated_text']

# ======================
# Gradio Interface
# ======================
def create_gradio_interface(model, advisor):
    def predict_health(age, gender, bmi, phq, gad, epworth):
        input_df = pd.DataFrame([[age, gender, bmi, phq, gad, epworth, 'Normal']],
                              columns=model.features)
        prediction, probability = model.predict(input_df)
        advice = advisor.generate_advice(prediction, probability)
        
        # Generate explanations
        shap_plot, lime_exp = model.explain_prediction(input_df)
        
        return {
            'diagnosis': 'Depression' if prediction else 'No Depression',
            'probability': f'{probability:.2%}',
            'advice': advice,
            'shap_plot': shap_plot,
            'lime_explanation': lime_exp
        }
    
    interface = gr.Interface(
        fn=predict_health,
        inputs=[
            gr.Number(label="Age"),
            gr.Dropdown(["male", "female"], label="Gender"),
            gr.Number(label="BMI"),
            gr.Slider(0, 27, step=1, label="PHQ-9 Score"),
            gr.Slider(0, 21, step=1, label="GAD-7 Score"),
            gr.Slider(0, 24, step=1, label="Epworth Sleepiness Score")
        ],
        outputs=[
            gr.Textbox(label="Diagnosis"),
            gr.Textbox(label="Probability"),
            gr.Textbox(label="Coping Strategies"),
            gr.Plot(label="SHAP Explanation"),
            gr.JSON(label="LIME Explanation")
        ],
        title="Mental Health Analysis System",
        description="Predict mental health conditions based on symptoms"
    )
    
    return interface

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train model
    classifier = MentalHealthClassifier()
    classifier.train(df)
    joblib.dump(classifier, 'mental_health_model.pkl')
    
    # Initialize advisor
    advisor = MentalHealthAdvisor()
    
    # Create and launch interface
    interface = create_gradio_interface(classifier, advisor)
    interface.launch()
