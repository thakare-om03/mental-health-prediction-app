"""
Utility functions for model training, evaluation, and prediction.
"""
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from config import RANDOM_STATE, TEST_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate_models(X, y, random_forest_path=None, xgboost_path=None):
    """
    Train and evaluate Random Forest and XGBoost models.
    
    Args:
        X: Feature matrix
        y: Target vector
        random_forest_path: Path to save Random Forest model
        xgboost_path: Path to save XGBoost model
        
    Returns:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        results: Dictionary with evaluation results
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    logging.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost model
    logging.info("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    results = {}
    
    # Random Forest evaluation
    rf_preds = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_preds),
        'precision': precision_score(y_test, rf_preds),
        'recall': recall_score(y_test, rf_preds),
        'f1': f1_score(y_test, rf_preds),
        'roc_auc': roc_auc_score(y_test, rf_proba),
        'confusion_matrix': confusion_matrix(y_test, rf_preds)
    }
    
    # XGBoost evaluation
    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    results['xgboost'] = {
        'accuracy': accuracy_score(y_test, xgb_preds),
        'precision': precision_score(y_test, xgb_preds),
        'recall': recall_score(y_test, xgb_preds),
        'f1': f1_score(y_test, xgb_preds),
        'roc_auc': roc_auc_score(y_test, xgb_proba),
        'confusion_matrix': confusion_matrix(y_test, xgb_preds)
    }
    
    # Log results
    logging.info("Model Evaluation Results:")
    logging.info(f"Random Forest - Accuracy: {results['random_forest']['accuracy']:.4f}, F1: {results['random_forest']['f1']:.4f}")
    logging.info(f"XGBoost - Accuracy: {results['xgboost']['accuracy']:.4f}, F1: {results['xgboost']['f1']:.4f}")
    
    # Save models if paths provided
    if random_forest_path:
        joblib.dump(rf_model, random_forest_path)
        logging.info(f"Random Forest model saved to {random_forest_path}")
    
    if xgboost_path:
        joblib.dump(xgb_model, xgboost_path)
        logging.info(f"XGBoost model saved to {xgboost_path}")
    
    return rf_model, xgb_model, results

def plot_confusion_matrix(results, save_path=None):
    """
    Plot confusion matrices for both models.
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Random Forest confusion matrix
    cm_rf = results['random_forest']['confusion_matrix']
    axes[0].imshow(cm_rf, cmap='Blues')
    axes[0].set_title('Random Forest Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    for i in range(cm_rf.shape[0]):
        for j in range(cm_rf.shape[1]):
            axes[0].text(j, i, str(cm_rf[i, j]), 
                     ha='center', va='center', color='black')
    
    # Plot XGBoost confusion matrix
    cm_xgb = results['xgboost']['confusion_matrix']
    axes[1].imshow(cm_xgb, cmap='Blues')
    axes[1].set_title('XGBoost Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    for i in range(cm_xgb.shape[0]):
        for j in range(cm_xgb.shape[1]):
            axes[1].text(j, i, str(cm_xgb[i, j]), 
                     ha='center', va='center', color='black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()

def load_model(model_path):
    """
    Load a saved model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        The loaded model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        raise
