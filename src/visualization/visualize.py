import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional

class Visualizer:
    def __init__(self):
        self.style = 'seaborn'
        plt.style.use(self.style)

    def plot_numerical_distribution(self, df: pd.DataFrame, columns: List[str], 
                                  figsize: tuple = (12, 6)):
        """Plot distribution of numerical variables."""
        plt.figure(figsize=figsize)
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                               columns: Optional[List[str]] = None):
        """Plot correlation heatmap for numerical variables."""
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        corr = df[columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_claims_by_category(self, df: pd.DataFrame, category: str):
        """Plot average claims by category."""
        plt.figure(figsize=(12, 6))
        avg_claims = df.groupby(category)['TotalClaims'].mean().sort_values(ascending=False)
        avg_claims.plot(kind='bar')
        plt.title(f'Average Claims by {category}')
        plt.xlabel(category)
        plt.ylabel('Average Claims')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_premium_claims_scatter(self, df: pd.DataFrame):
        """Plot scatter plot of premium vs claims."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', alpha=0.5)
        plt.title('Premium vs Claims')
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        plt.tight_layout()
        plt.show() 