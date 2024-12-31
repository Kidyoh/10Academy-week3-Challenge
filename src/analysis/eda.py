import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from src.data.data_loader import DataLoader
from src.visualization.visualize import Visualizer

class ExploratoryAnalysis:
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader(data_path)
        self.visualizer = Visualizer()
        self.reports_dir = Path('reports/figures')
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self):
        """Run complete EDA pipeline"""
        # Load data
        df = self.data_loader.load_data()
        if df is None:
            self.logger.error("Failed to load data")
            return

        # Data quality analysis
        self._analyze_data_quality(df)
        
        # Statistical analysis
        self._perform_statistical_analysis(df)
        
        # Risk analysis
        self._analyze_risk_factors(df)
        
        # Save results
        self._save_analysis_results(df)

    def _analyze_data_quality(self, df):
        """Analyze data quality and missing values"""
        missing = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing = missing[missing['Missing Values'] > 0]
        
        self.logger.info("\nMissing Values Analysis:")
        self.logger.info(missing)
        
        # Save missing values plot
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
        plt.title('Missing Values Heatmap')
        plt.savefig(self.reports_dir / 'missing_values.png')
        plt.close()

    def _perform_statistical_analysis(self, df):
        """Perform statistical analysis on numerical and categorical variables"""
        # Numerical statistics
        num_stats = df.describe()
        self.logger.info("\nNumerical Statistics:")
        self.logger.info(num_stats)

        # Categorical analysis
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.logger.info(f"\n{col} Distribution:")
            self.logger.info(df[col].value_counts().head())

    def _analyze_risk_factors(self, df):
        """Analyze risk factors and their relationships"""
        # Claims by province
        self.visualizer.plot_claims_by_category(df, 'Province')
        plt.savefig(self.reports_dir / 'claims_by_province.png')
        plt.close()

        # Claims by vehicle type
        self.visualizer.plot_claims_by_category(df, 'VehicleType')
        plt.savefig(self.reports_dir / 'claims_by_vehicle.png')
        plt.close()

        # Premium vs Claims
        self.visualizer.plot_premium_claims_scatter(df)
        plt.savefig(self.reports_dir / 'premium_vs_claims.png')
        plt.close()

    def _save_analysis_results(self, df):
        """Save analysis results and summary"""
        summary = {
            'total_records': len(df),
            'total_features': df.shape[1],
            'numerical_features': len(df.select_dtypes(include=['float64', 'int64']).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'total_claims': df['TotalClaims'].sum(),
            'average_premium': df['TotalPremium'].mean()
        }
        
        # Save summary to file
        with open(self.reports_dir / 'eda_summary.txt', 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

def main():
    logging.basicConfig(level=logging.INFO)
    data_path = 'Data/MachineLearningRating_v3/MachineLearningRating_v3.txt'
    
    analyzer = ExploratoryAnalysis(data_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 