import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from scipy import stats
from src.data.data_loader import DataLoader

class ABTesting:
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader(data_path)
        self.reports_dir = Path('reports/ab_testing')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_analysis(self):
        """Run all A/B tests"""
        # Load data
        df = self.data_loader.load_data()
        if df is None:
            self.logger.error("Failed to load data")
            return

        # Run tests
        self._test_risk_by_province(df)
        self._test_risk_by_zipcode(df)
        self._test_margin_by_zipcode(df)
        self._test_risk_by_gender(df)
        
        # Save results
        self._save_results()
        self._create_visualizations(df)

    def _test_risk_by_province(self, df):
        """Test if there are risk differences across provinces"""
        provinces = df['Province'].unique()
        claims_by_province = [df[df['Province'] == p]['TotalClaims'] for p in provinces]
        
        f_stat, p_value = stats.f_oneway(*claims_by_province)
        
        self.results['province_risk'] = {
            'test_name': 'Risk Differences Across Provinces',
            'statistic': f_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def _test_risk_by_zipcode(self, df):
        """Test if there are risk differences between zipcodes"""
        zipcodes = df['PostalCode'].unique()
        claims_by_zipcode = [df[df['PostalCode'] == z]['TotalClaims'] for z in zipcodes[:2]]
        
        t_stat, p_value = stats.ttest_ind(*claims_by_zipcode)
        
        self.results['zipcode_risk'] = {
            'test_name': 'Risk Differences Between Zipcodes',
            'statistic': t_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def _test_margin_by_zipcode(self, df):
        """Test if there are significant margin differences between zipcodes"""
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        zipcodes = df['PostalCode'].unique()
        margins_by_zipcode = [df[df['PostalCode'] == z]['Margin'] for z in zipcodes[:2]]
        
        t_stat, p_value = stats.ttest_ind(*margins_by_zipcode)
        
        self.results['zipcode_margin'] = {
            'test_name': 'Margin Differences Between Zipcodes',
            'statistic': t_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def _test_risk_by_gender(self, df):
        """Test if there are significant risk differences between genders"""
        male_claims = df[df['Gender'] == 'M']['TotalClaims']
        female_claims = df[df['Gender'] == 'F']['TotalClaims']
        
        t_stat, p_value = stats.ttest_ind(male_claims, female_claims)
        
        self.results['gender_risk'] = {
            'test_name': 'Risk Differences Between Genders',
            'statistic': t_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def _create_visualizations(self, df):
        """Create visualizations for A/B test results"""
        # Claims distribution by province
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Province', y='TotalClaims')
        plt.title('Claims Distribution by Province')
        plt.xticks(rotation=45)
        plt.savefig(self.reports_dir / 'claims_by_province_box.png')
        plt.close()

        # Claims distribution by gender
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Gender', y='TotalClaims')
        plt.title('Claims Distribution by Gender')
        plt.savefig(self.reports_dir / 'claims_by_gender_box.png')
        plt.close()

    def _save_results(self):
        """Save test results to file"""
        with open(self.reports_dir / 'ab_test_results.txt', 'w') as f:
            for test_name, result in self.results.items():
                f.write(f"\n{result['test_name']}:\n")
                f.write(f"Statistic: {result['statistic']:.4f}\n")
                f.write(f"P-value: {result['p_value']:.4f}\n")
                f.write(f"Conclusion: {result['conclusion']}\n")
                f.write("-" * 50 + "\n")

def main():
    logging.basicConfig(level=logging.INFO)
    data_path = 'Data/MachineLearningRating_v3/MachineLearningRating_v3.txt'
    
    ab_tester = ABTesting(data_path)
    ab_tester.run_analysis()

if __name__ == "__main__":
    main() 