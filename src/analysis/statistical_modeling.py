import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import json

class StatisticalModeling:
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.reports_dir = Path('reports/modeling')
        self.models_dir = Path('models')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_modeling(self, df):
        """Run complete modeling pipeline"""
        # Zipcode-based modeling
        self._fit_zipcode_models(df)
        
        # Premium prediction models
        self._fit_premium_models(df)
        
        # Save results and visualizations
        self._save_results()

    def _fit_zipcode_models(self, df):
        """Fit linear regression model for each zipcode"""
        zipcodes = df['PostalCode'].unique()
        zipcode_results = {}

        for zipcode in zipcodes:
            zipcode_data = df[df['PostalCode'] == zipcode]
            if len(zipcode_data) < 10:  # Skip if too few samples
                continue

            X = zipcode_data[['SumInsured', 'TermFrequency']]
            y = zipcode_data['TotalClaims']

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            zipcode_results[str(zipcode)] = {
                'r2_score': r2,
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            }

        self.results['zipcode_models'] = zipcode_results

    def _fit_premium_models(self, df):
        """Fit models to predict premium values"""
        # Prepare features
        feature_cols = [
            'SumInsured', 'TermFrequency', 'Cylinders', 'Cubiccapacity',
            'Kilowatts', 'NumberOfDoors', 'NumberOfVehiclesInFleet'
        ]
        X = df[feature_cols]
        y = df['TotalPremium']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train and evaluate models
        models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        }

        model_results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            model_results[name] = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'r2': float(r2)
            }

            # Feature importance
            if name == 'linear':
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': np.abs(model.coef_)
                })
            else:
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                })
            
            model_results[f"{name}_feature_importance"] = importance.to_dict()

            # Save model
            if name == 'xgboost':
                model.save_model(str(self.models_dir / f"{name}_model.json"))

        self.results['premium_models'] = model_results

    def _save_results(self):
        """Save modeling results and create visualizations"""
        # Save results to JSON
        with open(self.reports_dir / 'modeling_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)

        # Create performance comparison plot
        model_names = []
        r2_scores = []
        for name, results in self.results['premium_models'].items():
            if not name.endswith('_feature_importance'):
                model_names.append(name)
                r2_scores.append(results['r2'])

        plt.figure(figsize=(10, 6))
        plt.bar(model_names, r2_scores)
        plt.title('Model Performance Comparison')
        plt.ylabel('R² Score')
        plt.savefig(self.reports_dir / 'model_performance.png')
        plt.close()

def main():
    logging.basicConfig(level=logging.INFO)
    data_path = 'Data/MachineLearningRating_v3/MachineLearningRating_v3.txt'
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Run modeling
    modeler = StatisticalModeling(data_path)
    modeler.run_modeling(df)

if __name__ == "__main__":
    main() 