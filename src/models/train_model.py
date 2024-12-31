from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        }
        self.trained_models = {}
        self.model_scores = {}

    def prepare_data(self, df: pd.DataFrame, target: str = 'TotalClaims'):
        """Prepare data for modeling."""
        X = df.drop([target], axis=1)
        y = df[target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and evaluate their performance."""
        for name, model in self.models.items():
            # Train the model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.model_scores[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }

    def get_feature_importance(self, model_name: str, feature_names):
        """Get feature importance for the specified model."""
        model = self.trained_models[model_name]
        
        if model_name == 'linear':
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': abs(model.coef_)
            })
        else:
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            
        return importance.sort_values('importance', ascending=False) 