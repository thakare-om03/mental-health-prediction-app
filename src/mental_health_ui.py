# temp/src/mental_health_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml # Add pyyaml to requirements.txt
import logging
import os

# Import the explanation class correctly
try:
    from llm_explanations import ExplanationGenerator
except ImportError:
    st.error("Could not import ExplanationGenerator. Make sure llm_explanations.py is in the src directory.")
    st.stop() # Stop execution if import fails

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
@st.cache_resource # Cache config loading
def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("UI: Configuration loaded successfully.")
        return config
    except Exception as e:
        st.error(f"Failed to load configuration file ({config_path}): {e}")
        logging.error(f"UI: Error loading configuration: {e}")
        st.stop() # Stop if config fails to load

# --- Load Models and Preprocessors ---
@st.cache_resource # Cache resource loading
def load_artifacts(models_dir):
    """Loads the ML model, scaler, encoders, and feature lists."""
    artifacts = {}
    required_files = {
        'model': 'random_forest_model.joblib', # Or 'xgboost_model.joblib' - choose default or let user select
        'scaler': 'scaler.joblib',
        'label_encoders': 'label_encoders.joblib', # Dictionary of encoders
        'target_encoder': 'target_encoder.joblib',
        'features': 'features.joblib',
        'numerical_features': 'numerical_features.joblib',
        'categorical_features': 'categorical_features.joblib'
    }
    try:
        models_path = Path(models_dir)
        for name, filename in required_files.items():
            file_path = models_path / filename
            if not file_path.exists():
                 raise FileNotFoundError(f"Required artifact '{filename}' not found in {models_dir}")
            artifacts[name] = joblib.load(file_path)
        logging.info("UI: All artifacts loaded successfully.")
        return artifacts
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}")
        logging.error(f"UI: Error loading artifacts: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        logging.error(f"UI: An unexpected error occurred while loading artifacts: {e}")
        st.stop()


# --- Initialize Explanation Generator ---
# Use API Key from environment variable by default
# Cache the generator instance
@st.cache_resource
def get_explanation_generator():
    try:
        # Assumes GROQ_API_KEY is set in the environment where Streamlit runs
        generator = ExplanationGenerator()
        logging.info("UI: ExplanationGenerator initialized.")
        return generator
    except Exception as e:
        st.error(f"Failed to initialize the Explanation Generator: {e}. Ensure GROQ_API_KEY is set.")
        logging.error(f"UI: Failed to initialize ExplanationGenerator: {e}")
        # Don't stop, allow prediction without explanation if generator fails
        return None

# --- Main UI Application ---
st.set_page_config(page_title="Mental Health Prediction", layout="wide")
st.title("Mental Health Prediction Tool")
st.markdown("Enter the details below to predict the mental health status.")

# Load config and artifacts
config = load_config() # Assumes config.yaml is in the root
models_dir = config['models']['output_dir']
artifacts = load_artifacts(models_dir)
explainer = get_explanation_generator()

# Extract loaded components
model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders'] # Dict: {'col_name': LabelEncoder}
target_encoder = artifacts['target_encoder']
features_order = artifacts['features'] # Ensure input order matches training
numerical_features = artifacts['numerical_features']
categorical_features = artifacts['categorical_features']


# --- User Input Section ---
st.sidebar.header("User Input Features")
input_data = {}

# Create input fields dynamically based on features
for feature in features_order:
    if feature in numerical_features:
        # *** FIX HERE: Ensure default_val is always float ***
        default_val = 25.0 if feature == 'bmi' else (30.0 if feature == 'age' else 5.0) # Use 30.0 and 5.0
        try:
             # Use format to prevent potential issues with very small/large steps later if changed
             input_data[feature] = st.sidebar.number_input(
                 f"Enter {feature.replace('_', ' ').title()}",
                 value=default_val,
                 step=1.0,
                 format="%.1f" # Optional: format display to one decimal place
             )
        except Exception as e:
             st.sidebar.error(f"Error creating input for {feature}: {e}")
             input_data[feature] = default_val # Fallback

    elif feature in categorical_features:
        encoder = label_encoders.get(feature)
        if encoder:
            options = list(encoder.classes_)
            # Ensure options are strings for display, though they usually are from LabelEncoder.classes_
            options_str = [str(opt) for opt in options]
            input_data[feature] = st.sidebar.selectbox(f"Select {feature.replace('_', ' ').title()}", options=options_str, index=0)
        else:
            input_data[feature] = st.sidebar.text_input(f"Enter {feature.replace('_', ' ').title()}", value="DefaultCategory")
            st.warning(f"Could not find encoder for '{feature}'. Using text input.")

# Create a DataFrame from input
input_df = pd.DataFrame([input_data])
# Reorder columns to match training features order
input_df = input_df[features_order]

st.subheader("Input Data Overview")
st.dataframe(input_df)

# --- Prediction Logic ---
if st.sidebar.button("Predict"):
    try:
        # Preprocess input data
        processed_df = input_df.copy()

        # 1. Encode Categorical Features
        for col in categorical_features:
            encoder = label_encoders.get(col)
            if encoder:
                 # Use transform, handle potential errors if category wasn't seen during fit
                 try:
                     processed_df[col] = encoder.transform(processed_df[col].astype(str))
                 except ValueError as e:
                     # Handle unseen category - Option 1: Error out
                     # st.error(f"Error encoding '{col}': Category '{processed_df[col].iloc[0]}' was not seen during training.")
                     # st.stop()
                     # Option 2: Assign a default/unknown category code (e.g., -1 or len(classes))
                     # This requires the model to potentially handle this code, or careful consideration.
                     # For simplicity, we'll error out here, but a production system needs a strategy.
                     st.error(f"Unseen category '{processed_df[col].iloc[0]}' for feature '{col}'. Prediction may be unreliable.")
                     # Fallback: encode as NaN or a special code if model handles it, here we use 0 as placeholder
                     processed_df[col] = -1 # Assign a code indicating 'unknown'
                     # Or just log and continue, model might handle it surprisingly well or poorly
                     logging.warning(f"Unseen category '{processed_df[col].iloc[0]}' for feature '{col}'. Encoded as -1.")

            else:
                 st.error(f"Encoder for categorical feature '{col}' not found. Cannot proceed.")
                 st.stop()


        # 2. Scale Numerical Features
        processed_df[numerical_features] = scaler.transform(processed_df[numerical_features])
        logging.info("UI: Input data processed successfully.")

        st.subheader("Processed Data (Scaled & Encoded)")
        st.dataframe(processed_df)


        # Make Prediction
        prediction_encoded = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)

        # Decode Prediction
        prediction_label = target_encoder.inverse_transform(prediction_encoded)[0]
        prediction_confidence = np.max(prediction_proba) * 100

        st.subheader("Prediction Result")
        st.success(f"Predicted Status: **{prediction_label}**")
        st.info(f"Confidence: {prediction_confidence:.2f}%")


        # Generate Explanation
        if explainer:
            st.subheader("Explanation")
            with st.spinner("Generating explanation..."):
                # Pass original input_data (human-readable) for the prompt
                explanation = explainer.generate_explanation(input_data, prediction_label, prediction_confidence / 100.0)
                st.markdown(explanation)
        else:
            st.warning("Explanation generator is not available.")


    except FileNotFoundError as e:
         st.error(f"Prediction failed: Required model file not found. {e}")
         logging.error(f"UI: Prediction failed - file not found: {e}")
    except KeyError as e:
         st.error(f"Prediction failed: Feature mismatch or missing artifact component. Check feature names. Error: {e}")
         logging.error(f"UI: Prediction failed - KeyError: {e}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        logging.error(f"UI: An unexpected error occurred during prediction: {e}", exc_info=True)