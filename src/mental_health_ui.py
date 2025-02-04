import gradio as gr
from predict_mental_health import MentalHealthPredictor

predictor = MentalHealthPredictor()

def predict_health(symptoms, age, severity):
    input_data = {'symptoms': symptoms, 'age': float(age), 'severity': float(severity)}
    prediction = predictor.predict(input_data)
    explanation = predictor.explain_prediction(input_data)
    return {
        "diagnosis": prediction[0],
        "explanation": explanation,
        "recommendations": "Consider consulting a professional and practicing mindfulness exercises."
    }

iface = gr.Interface(
    fn=predict_health,
    inputs=[
        gr.Textbox(label="Symptoms (comma-separated)"),
        gr.Number(label="Age"),
        gr.Slider(1, 10, label="Severity")
    ],
    outputs=[
        gr.Label(label="Diagnosis"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Recommendations")
    ],
    title="Mental Health Analysis"
)

if __name__ == "__main__":
    iface.launch()
