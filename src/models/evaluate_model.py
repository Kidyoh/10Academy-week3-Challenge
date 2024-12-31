from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_regression(self, y_true, y_pred):
        """Evaluate regression model performance."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_true, y_pred)
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

    def evaluate_classification(self, y_true, y_pred):
        """Evaluate classification model performance."""
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return {
            'Accuracy': accuracy,
            'Confusion Matrix': cm
        } 