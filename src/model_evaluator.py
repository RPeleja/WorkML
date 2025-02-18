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
    confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, roc_curve,
    roc_auc_score, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def evaluate_models(self, models, X_test, y_test):
        results = {}
        best_model = None
        best_r2 = -np.inf # Best model was the best R2 score

        for name, model in models.items():
            y_pred = model.predict(X_test)

            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'y_pred': y_pred 
            }

            # Update best model if AUC is higher
            if r2 > best_r2:
                best_r2 = r2
                best_model = name

        print(f"\n📌 **Best Model:** {best_model} with R2 Score: {best_r2:.4f}")
        return results, best_model

    def before_trainning_plot(self, merged_df):
        plt.figure(figsize=(10, 6))
        sns.heatmap(merged_df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.show()

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

        for name, result in results.items():
            y_pred = result['y_pred']
            errors = y_test - y_pred  # Calculate errors

            plt.figure(figsize=(12, 5))
            # Histogram of Errors
            plt.subplot(1, 2, 1)
            sns.histplot(errors, bins=20, kde=True, color='blue')
            plt.axvline(0, color='red', linestyle='dashed')
            plt.title(f'Histogram of Errors - {name}')
            plt.xlabel('Error (y_real - y_pred)')
            plt.ylabel('Frequency')

            # Scatter Plot (y_test vs y_pred)
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.title(f'Prediction vs Actual - {name}')
            plt.xlabel('Actual Value (y_test)')
            plt.ylabel('Predicted Value (y_pred)')
            plt.show()