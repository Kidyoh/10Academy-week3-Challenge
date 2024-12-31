import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict

class HypothesisTesting:
    def __init__(self):
        self.test_results = {}

    def test_risk_by_province(self, df: pd.DataFrame) -> Dict:
        """Test if there are risk differences across provinces."""
        provinces = df['Province'].unique()
        f_stat, p_value = stats.f_oneway(*[
            df[df['Province'] == province]['TotalClaims']
            for province in provinces
        ])
        
        return {
            'test_name': 'Risk Differences Across Provinces',
            'statistic': f_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def test_risk_by_zipcode(self, df: pd.DataFrame) -> Dict:
        """Test if there are risk differences between zipcodes."""
        zipcodes = df['PostalCode'].unique()
        f_stat, p_value = stats.f_oneway(*[
            df[df['PostalCode'] == zipcode]['TotalClaims']
            for zipcode in zipcodes[:2]  # Compare first two zipcodes
        ])
        
        return {
            'test_name': 'Risk Differences Between Zipcodes',
            'statistic': f_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def test_margin_by_zipcode(self, df: pd.DataFrame) -> Dict:
        """Test if there are significant margin differences between zipcodes."""
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        zipcodes = df['PostalCode'].unique()
        f_stat, p_value = stats.f_oneway(*[
            df[df['PostalCode'] == zipcode]['Margin']
            for zipcode in zipcodes[:2]  # Compare first two zipcodes
        ])
        
        return {
            'test_name': 'Margin Differences Between Zipcodes',
            'statistic': f_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def test_risk_by_gender(self, df: pd.DataFrame) -> Dict:
        """Test if there are significant risk differences between genders."""
        t_stat, p_value = stats.ttest_ind(
            df[df['Gender'] == 'M']['TotalClaims'],
            df[df['Gender'] == 'F']['TotalClaims']
        )
        
        return {
            'test_name': 'Risk Differences Between Genders',
            'statistic': t_stat,
            'p_value': p_value,
            'conclusion': 'Reject H0' if p_value < 0.05 else 'Fail to reject H0'
        }

    def run_all_tests(self, df: pd.DataFrame) -> Dict:
        """Run all hypothesis tests and store results."""
        self.test_results = {
            'province_risk': self.test_risk_by_province(df),
            'zipcode_risk': self.test_risk_by_zipcode(df),
            'zipcode_margin': self.test_margin_by_zipcode(df),
            'gender_risk': self.test_risk_by_gender(df)
        }
        return self.test_results 