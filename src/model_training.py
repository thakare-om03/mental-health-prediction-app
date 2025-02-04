from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)
import shap
import joblib

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=150),
        'XGBoost': GradientBoostingClassifier(n_estimators=150)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(f"Precision: {precision_score(y_test, preds, average='weighted'):.3f}")
        print(f"Recall: {recall_score(y_test, preds, average='weighted'):.3f}") 
        print(f"F1-Score: {f1_score(y_test, preds, average='weighted'):.3f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, probs, multi_class='ovo'):.3f}")
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        
    joblib.dump(models['XGBoost'], '../models/best_model.pkl')
    return models['XGBoost']