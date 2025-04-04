# temp/src/llm_explanations.py
import os
import logging
from groq import Groq, GroqError # Make sure groq is in requirements.txt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExplanationGenerator:
    """Generates explanations for predictions using the Groq API."""

    def __init__(self, api_key=None, model="deepseek-r1-distill-llama-70b"):
        """
        Initializes the Groq client.

        Args:
            api_key (str, optional): Groq API key. Defaults to None (reads from GROQ_API_KEY env var).
            model (str, optional): The Groq model to use. Defaults to "llama3-8b-8192".
        """
        try:
            resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
            if not resolved_api_key:
                raise ValueError("GROQ_API_KEY environment variable not set and no api_key provided.")
            self.client = Groq(api_key=resolved_api_key)
            self.model = model
            logging.info(f"Groq client initialized successfully for model {self.model}.")
        except GroqError as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            raise
        except ValueError as e:
            logging.error(e)
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during initialization: {e}")
            raise


    def generate_explanation(self, features, prediction, prediction_proba=None):
        """
        Generates a user-friendly explanation for a given prediction.

        Args:
            features (dict): Dictionary of feature names and their values for the instance.
            prediction (str): The predicted class label (e.g., "High Depression Risk").
            prediction_proba (float, optional): The probability associated with the prediction. Defaults to None.

        Returns:
            str: A textual explanation, or an error message if generation fails.
        """
        logging.info(f"Generating explanation for prediction: {prediction}")
        feature_string = ", ".join([f"{k}: {v}" for k, v in features.items()])
        prompt = (
            f"Explain in simple terms why a person with the following characteristics might have a prediction of '{prediction}'. "
            f"Characteristics: {feature_string}. "
            f"Focus on the most likely contributing factors based on typical associations in mental health data, "
            f"but state clearly this is a general explanation based on patterns and not a specific diagnosis for this individual. "
            f"Keep the explanation concise and easy to understand (2-3 sentences)."
        )
        if prediction_proba is not None:
             prompt += f" The model confidence for this prediction was {prediction_proba*100:.1f}%."


        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0.7, # Adjust creativity vs factualness
                max_tokens=150,
            )
            explanation = chat_completion.choices[0].message.content
            logging.info("Explanation generated successfully.")
            return explanation.strip()
        except GroqError as e:
            logging.error(f"Groq API error during explanation generation: {e}")
            return f"Error: Could not generate explanation due to an API issue ({e.status_code})."
        except Exception as e:
            logging.error(f"Unexpected error during explanation generation: {e}")
            return "Error: An unexpected error occurred while generating the explanation."

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Ensure GROQ_API_KEY is set as an environment variable before running this
    try:
        explainer = ExplanationGenerator()
        sample_features = {'age': 30, 'gender': 'Female', 'bmi': 22, 'phq_score': 15, 'gad_score': 12, 'epworth_score': 8}
        sample_prediction = "Moderate Depression"
        explanation = explainer.generate_explanation(sample_features, sample_prediction, prediction_proba=0.75)
        print("--- Example Explanation ---")
        print(explanation)
    except Exception as e:
        print(f"Failed to run example: {e}")