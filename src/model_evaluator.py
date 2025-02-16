"""
    Handles model evaluation and visualization of results.
    
    Methods:
        evaluate_models(models, X_test, y_test):
            Evaluates all models and computes performance metrics
            Returns: Dictionary containing for each model:
                - Predictions
                - Probabilities
                - Confusion Matrix
                - Classification Report
                - ROC AUC Score
            
        plot_results(results, y_test):
            Creates visualization of model performance
            - ROC curves for all models
            - Confusion matrices
            - Performance comparison plots
    
    Usage:
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_models(trained_models, X_test, y_test)
        evaluator.plot_results(results, y_test)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    roc_auc_score, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def evaluate_models(self, models, X_test, y_test):
        results = {}
        best_model = None
        best_auc = 0  # Track best AUC Score

        for name, model in models.items():
            y_pred = model.predict(X_test)

            # Check if model supports predict_proba
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities
                auc = roc_auc_score(y_test, y_prob)  # Compute AUC
            else:
                y_prob = None  # No probabilities available
                auc = roc_auc_score(y_test, y_pred)  # Use predictions instead
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': auc,  # Store AUC
                'y_prob': y_prob,  # Store probabilities (if available)
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Update best model if AUC is higher
            if auc > best_auc:
                best_auc = auc
                best_model = name

        print(f"\n📌 **Best Model:** {best_model} with AUC Score: {best_auc:.4f}")
        return results, best_model

    def plot_results(self, results, X_test, y_test, best_model_Used, features):
        
        # feature importance
        if hasattr(best_model_Used, 'feature_importances_'):
            # Get feature importance from RandomForest or XGBoost
            importances = best_model_Used.feature_importances_
            # Convert to DataFrame for better visualization
            feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            print(feature_importance_df)

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance')
            plt.gca().invert_yaxis()  # Invert y-axis to show top features first
            plt.show()

        # Confusion Matrices
        for name, result in results.items():
            cm = result['confusion_matrix']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: {name}')
            plt.show()
        
        # ROC Curves
        for name, result in results.items():
            if result['y_prob'] is not None:  # Only plot ROC if y_prob exists
                fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.show()
        
        # Identify False Positives and False Negatives
        predictions = best_model_Used.predict(X_test)
        false_positives = (predictions == 1) & (y_test == 0)
        false_negatives = (predictions == 0) & (y_test == 1)

        print(f'False Positives: {np.sum(false_positives)}')
        print(f'False Negatives: {np.sum(false_negatives)}')
    
        # Plot calibration curve
        if hasattr(best_model_Used, 'predict_proba'):
            prob_true, prob_pred = calibration_curve(y_test, best_model_Used.predict_proba(X_test)[:, 1], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
            plt.xlabel('Predicted Probability')
            plt.ylabel('True Probability')
            plt.title('Calibration Curve')
            plt.legend()
            plt.show()