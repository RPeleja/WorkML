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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, ConfusionMatrixDisplay, accuracy_score
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
            accuracy_test = accuracy_score(y_test, y_pred)

            # Check if model supports predict_proba
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)  # Get probabilities
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')  # Compute AUC
            else:
                y_prob = None  # No probabilities available
                roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro')  # Use predictions instead

            results[name] = {
                'accuracy': accuracy_test,
                'predictions': y_pred,
                'y_prob': y_prob,  # Store probabilities (if available)
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Update best model if AUC is higher
            if accuracy_test > best_auc:
                best_auc = accuracy_test
                best_model = name

        print(f"\n📌 **Best Model:** {best_model} with accuracy Score: {best_auc:.4f}")
        return results, best_model

    def plot_hist_show(self, df):
        df.hist(bins=20, figsize=(10, 10))
        plt.show()

    def plot_results(self, results, X_test, y_test, best_model_Used, features, best_model):
        
        # Convert the classification report dictionary into a DataFrame
        report_df = pd.DataFrame(results[best_model]['classification_report']).T

        # Print the formatted DataFrame
        print(f"\n📊 Classification Test Report {best_model}:\n")
        print(report_df.round(4))  # Round numbers for better readability
        
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
        
        # # ROC Curves
        # Convert y_test into binary labels
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Converts (n_samples,) → (n_samples, n_classes)

        plt.figure(figsize=(8, 6))

        for name, result in results.items():
            # Loop through each class (0, 1, 2)
            for i, class_label in enumerate([0, 1, 2]):
                y_prob_class = result['y_prob'][:, i]  # Probabilities for class i
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_class)  # Compute ROC curve
                roc_auc = auc(fpr, tpr)  # Compute AUC

                plt.plot(fpr, tpr, label=f'{name} class {class_label} (AUC = {roc_auc:.2f})')

        # Plot baseline (random chance)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve')
        plt.legend()
        plt.show()

        # Identify False Positives and False Negatives
        predictions = result['predictions']
        false_positives = (predictions == 1) & (y_test == 0)
        false_negatives = (predictions == 0) & (y_test == 1)

        print(f'False Positives: {np.sum(false_positives)}')
        print(f'False Negatives: {np.sum(false_negatives)}')
        
        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(X_test).corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()

        # Class Distribution in Test Data
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y_test, palette='viridis')
        plt.title("Class Distribution (Wine quality) in Test Data")
        plt.xlabel("0-5: low, 6: Medium, 7-10: High")
        plt.ylabel("Count")
        plt.show()