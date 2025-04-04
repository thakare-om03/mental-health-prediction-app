"""
Configuration settings for the Mental Health Prediction App.
Centralizing configuration helps with maintenance and flexibility.
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Data paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DEPRESSION_ANXIETY_DATA = os.path.join(RAW_DATA_DIR, "depression_anxiety_data.csv")
SURVEY_DATA = os.path.join(RAW_DATA_DIR, "survey.csv")

# Model paths
RANDOM_FOREST_MODEL = os.path.join(MODEL_DIR, "random_forest_model.pkl")
XGBOOST_MODEL = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LABEL_ENCODER = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER = os.path.join(MODEL_DIR, "scaler.pkl")

# LLM settings
DEFAULT_LLM_MODEL = "llama3-8b-8192"
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.3

# Feature settings
CATEGORICAL_FEATURES = ["Gender", "family_history", "treatment"]
NUMERIC_FEATURES = ["Age", "self_employed", "work_interfere", "no_employees", 
                    "remote_work", "tech_company", "benefits", "care_options", 
                    "wellness_program", "seek_help", "anonymity", "leave", 
                    "mental_health_consequence", "phys_health_consequence", 
                    "coworkers", "supervisor", "mental_health_interview", 
                    "phys_health_interview", "mental_vs_physical", "obs_consequence"]
TARGET_COLUMN = "mental_health_disorder"

# Model evaluation settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
