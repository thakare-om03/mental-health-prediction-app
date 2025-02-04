from transformers import pipeline

class MentalHealthExplainer:
    def __init__(self):
        self.explainer = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
        
    def generate_explanation(self, condition, symptoms):
        prompt = f"""As a mental health professional, explain {condition} 
        based on these symptoms: {', '.join(symptoms)}. 
        Provide 3 coping strategies in simple terms."""
        
        response = self.explainer(
            prompt,
            max_length=300,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return self._clean_response(response[0]['generated_text'])
    
    def _clean_response(self, text):
        # Remove prompt repetition
        return text.split("Provide 3 coping strategies")[-1].strip()
