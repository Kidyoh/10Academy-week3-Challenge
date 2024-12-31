import json
import markdown
from pathlib import Path
import yaml
import logging
from datetime import datetime

class BlogPostGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path('reports/blog')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_blog_post(self):
        """Generate complete blog post with all sections"""
        content = self._create_header()
        content += self._create_executive_summary()
        content += self._create_introduction()
        content += self._create_project_overview()
        content += self._create_methodology()
        content += self._create_findings()
        content += self._create_technical_details()
        content += self._create_recommendations()
        content += self._create_conclusion()
        content += self._create_footer()
        
        self._save_blog_post(content)
        self._create_html_version(content)

    def _create_header(self):
        """Create blog post header with metadata"""
        return f"""# Optimizing Insurance Risk Analytics: A Deep Dive into AlphaCare's Data-Driven Transformation

**Date**: {datetime.now().strftime('%B %d, %Y')}
**Tags**: #DataScience #InsuranceAnalytics #MachineLearning #RiskAnalysis #Python

"""

    def _create_executive_summary(self):
        """Create executive summary section"""
        return """## Executive Summary

This technical case study explores how AlphaCare Insurance Solutions leveraged data analytics to optimize their risk assessment and marketing strategies. Through comprehensive analysis of historical insurance claim data from 2014-2015, we developed a sophisticated analytics pipeline that revealed significant insights into risk patterns and premium optimization opportunities.

"""

    def _create_introduction(self):
        """Create introduction and business context section"""
        return """## 1. Introduction & Business Context

### The Role of Data Analytics in Insurance
The insurance industry is undergoing a dramatic transformation driven by data analytics. Traditional actuarial methods are being enhanced with machine learning and advanced statistical techniques, enabling more precise risk assessment and personalized premium pricing.

### AlphaCare's Objectives
- Develop cutting-edge risk and predictive analytics
- Optimize marketing strategies
- Identify low-risk customer segments
- Understand geographic risk patterns

### Key Business Questions
1. Risk differences across provinces and zip codes
2. Demographic factors influencing insurance claims
3. Optimal premium levels for different segments
4. Key factors affecting claim likelihood

"""

    def _create_project_overview(self):
        """Create project overview section"""
        with open('requirements/requirements.txt', 'r') as f:
            requirements = f.read()

        return f"""## 2. Project Overview

### Data Description
- Timeframe: February 2014 to August 2015
- Source: Historical insurance claim data
- Key components: Policy information, client demographics, vehicle details

### Technical Stack
```python
{requirements}
```

### Project Structure
```
insurance-analytics/
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
├── src/
│   ├── data/
│   ├── analysis/
│   └── visualization/
├── reports/
└── models/
```

"""

    def _create_methodology(self):
        """Create methodology section with code examples"""
        return """## 3. Data Analysis & Methodology

### a) Exploratory Data Analysis
```python
def analyze_data_quality(df):
    missing = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    return missing[missing['Missing Values'] > 0]
```

### b) A/B Testing Results
```python
def test_risk_by_province(df):
    provinces = df['Province'].unique()
    claims_by_province = [df[df['Province'] == p]['TotalClaims'] 
                         for p in provinces]
    f_stat, p_value = stats.f_oneway(*claims_by_province)
    return {'statistic': f_stat, 'p_value': p_value}
```

### c) Statistical Modeling
```python
def fit_premium_models(df, feature_cols):
    X = df[feature_cols]
    y = df['TotalPremium']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    models = {
        'linear': LinearRegression(),
        'random_forest': RandomForestRegressor(),
        'xgboost': xgb.XGBRegressor()
    }
    return train_evaluate_models(models, X_train, X_test, y_train, y_test)
```

"""

    def _create_findings(self):
        """Create findings and insights section"""
        return """## 4. Key Findings & Business Insights

### Risk Patterns
- Significant variations in risk profiles across provinces
- Gender-based risk differences identified
- Vehicle type correlations with claim frequency

### Premium Optimization Opportunities
- Identified segments for potential premium adjustments
- Geographic-based pricing optimization
- Risk-based premium structuring

### Customer Segmentation
- Low-risk customer profiles identified
- High-value customer characteristics
- Geographic risk clusters

"""

    def _create_technical_details(self):
        """Create technical implementation details section"""
        return """## 5. Technical Implementation Details

### Data Pipeline Architecture
```python
# DVC pipeline configuration
stages:
  prepare:
    cmd: python src/data/data_preprocessor.py
    deps:
      - data/raw/insurance_data.csv
    outs:
      - data/processed/preprocessed_data.csv

  analyze:
    cmd: python src/analysis/eda.py
    deps:
      - data/processed/preprocessed_data.csv
    outs:
      - reports/figures/
```

### Model Training Approach
- Feature engineering pipeline
- Cross-validation strategy
- Hyperparameter optimization
- Model evaluation metrics

"""

    def _create_recommendations(self):
        """Create recommendations section"""
        return """## 6. Recommendations

### Business Strategy
1. Implement risk-based pricing strategy
2. Develop targeted marketing campaigns
3. Optimize customer acquisition costs

### Risk Management
1. Enhanced monitoring of high-risk segments
2. Proactive risk mitigation strategies
3. Regular model retraining and validation

"""

    def _create_conclusion(self):
        """Create conclusion section"""
        return """## 7. Conclusion

### Project Impact
- Improved risk assessment accuracy
- Data-driven decision making framework
- Enhanced customer segmentation

### Future Work
1. Real-time risk assessment implementation
2. Integration with marketing automation
3. Enhanced feature engineering

"""

    def _create_footer(self):
        """Create footer with author info and references"""
        return """---

### Author Information
[Your Name]
Data Scientist at AlphaCare Insurance Solutions
[Contact Information]

### References
1. Insurance Analytics Best Practices
2. Statistical Modeling in Python
3. Machine Learning for Risk Assessment

"""

    def _save_blog_post(self, content):
        """Save blog post as markdown"""
        output_file = self.output_dir / 'technical_blog_post.md'
        with open(output_file, 'w') as f:
            f.write(content)
        self.logger.info(f"Blog post saved to {output_file}")

    def _create_html_version(self, content):
        """Create HTML version of the blog post"""
        html = markdown.markdown(content)
        output_file = self.output_dir / 'technical_blog_post.html'
        with open(output_file, 'w') as f:
            f.write(html)
        self.logger.info(f"HTML version saved to {output_file}")

def main():
    logging.basicConfig(level=logging.INFO)
    generator = BlogPostGenerator()
    generator.generate_blog_post()

if __name__ == "__main__":
    main() 