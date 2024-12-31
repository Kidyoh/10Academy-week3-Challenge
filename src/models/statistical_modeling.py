import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class StatisticalModeler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        self.reports_dir = Path('reports/modeling')
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def train_zipcode_models(self, df):
        """Train separate models for each zipcode"""
        zipcodes = df['PostalCode'].unique()
        
        for zipcode in zipcodes:
            zipcode_data = df[df['PostalCode'] == zipcode]
            if len(zipcode_data) < 50:  # Skip if insufficient data
                continue
                
            X = zipcode_data[['VehicleType', 'TermFrequency', 'SumInsured']]
            y = zipcode_data['TotalClaims']
            
            # Handle categorical variables
            X = pd.get_dummies(X, columns=['VehicleType'])
            
            # Train models
            models = {
                'linear': LinearRegression(),
                'rf': RandomForestRegressor(n_estimators=100)
            }
            
            model_results = {}
            for name, model in models.items():
                # Cross validation
                cv_scores = cross_val_score(model, X, y, cv=5)
                model.fit(X, y)
                
                model_results[name] = {
                    'cv_scores': cv_scores,
                    'mean_cv_score': cv_scores.mean(),
                    'model': model
                }
            
            self.models[zipcode] = model_results

    def train_premium_models(self, df):
        """Train models for premium prediction"""
        # Prepare features
        feature_cols = [
            'VehicleType', 'TermFrequency', 'SumInsured', 
            'Cylinders', 'Province', 'Gender'
        ]
        
        # Create feature matrix
        X = pd.get_dummies(df[feature_cols])
        y = df['TotalPremium']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100),
            'xgb': xgb.XGBRegressor(objective='reg:squarederror')
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.results[name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'feature_importance': self._get_feature_importance(model, X.columns, name)
            }

    def analyze_risk_factors(self, df):
        """Analyze risk factors affecting claims"""
        # Create binary target for high/low risk
        df['high_risk'] = df['TotalClaims'] > df['TotalClaims'].median()
        
        # Prepare features
        X = pd.get_dummies(df[[
            'VehicleType', 'Province', 'Gender', 
            'TermFrequency', 'SumInsured'
        ]])
        y = df['high_risk']
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.results['risk_factors'] = importance

    def _get_feature_importance(self, model, feature_names, model_type):
        """Extract feature importance from model"""
        if model_type == 'linear':
            importance = np.abs(model.coef_)
        else:
            importance = model.feature_importances_
            
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def plot_results(self):
        """Create visualizations of model results"""
        # Model performance comparison
        plt.figure(figsize=(10, 6))
        r2_scores = [results['r2'] for results in self.results.values() 
                    if 'r2' in results]
        plt.bar(self.results.keys(), r2_scores)
        plt.title('Model Performance Comparison')
        plt.ylabel('R² Score')
        plt.savefig(self.reports_dir / 'model_performance.png')
        plt.close()
        
        # Feature importance
        for name, results in self.results.items():
            if 'feature_importance' in results:
                plt.figure(figsize=(12, 6))
                importance = results['feature_importance']
                sns.barplot(data=importance.head(10), x='importance', y='feature')
                plt.title(f'Top 10 Important Features - {name}')
                plt.tight_layout()
                plt.savefig(self.reports_dir / f'feature_importance_{name}.png')
                plt.close()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    data_path = 'Data/MachineLearningRating_v3/MachineLearningRating_v3.txt'
    df = pd.read_csv(data_path)
    
    # Initialize modeler
    modeler = StatisticalModeler()
    
    # Run analysis
    logger.info("Training zipcode-specific models...")
    modeler.train_zipcode_models(df)
    
    logger.info("Training premium prediction models...")
    modeler.train_premium_models(df)
    
    logger.info("Analyzing risk factors...")
    modeler.analyze_risk_factors(df)
    
    logger.info("Creating visualizations...")
    modeler.plot_results()

if __name__ == "__main__":
    main() 