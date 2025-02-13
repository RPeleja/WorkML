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

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def evaluate_models(self, models, X_test, y_test):
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_prob,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
        return results

    def plot_results(self, results, y_test):
        # ROC curves
        plt.figure(figsize=(10, 6))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.show()
