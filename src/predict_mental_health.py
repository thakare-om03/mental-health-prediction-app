import joblib
import numpy as np
from llm_explanations import MentalHealthExplainer

class MentalHealthPredictor:
    def __init__(self, model_path='models/mental_health_model.pkl'):
        self.model = joblib.load(model_path)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
    
    def preprocess_input(self, symptoms, age, severity):
        # Implement your preprocessing logic
        processed = {
            'symptom_count': len(symptoms.split(',')),
            'age': age,
            'severity': severity
        }
        return self.scaler.transform([list(processed.values())])
    
    def predict(self, input_data):
        processed = self.preprocess_input(**input_data)
        return self.model.predict(processed)
    
    def explain_prediction(self, input_data):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.preprocess_input(**input_data))
        return shap.force_plot(explainer.expected_value[0], shap_values[0], input_data)
    def predict_with_explanation(self, input_data):
        prediction = self.model.predict(input_data)
        explainer = MentalHealthExplainer()
        explanation = explainer.generate_explanation(
            self.le.inverse_transform(prediction),
            input_data['symptoms']
        )
        return prediction, explanation