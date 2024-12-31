from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.hypothesis_testing import HypothesisTesting
from src.models.train_model import ModelTrainer
from src.visualization.visualize import Visualizer
import logging
import pandas as pd
import os

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize data loader
    data_path = 'Data/MachineLearningRating_v3/MachineLearningRating_v3.txt'
    data_loader = DataLoader(data_path)
    
    # Load the data
    df = data_loader.load_data()
    
    if df is not None and data_loader.validate_data(df):
        logger.info("Data validation successful")
        
        # Get feature groups
        feature_groups = data_loader.get_feature_groups(df)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Prepare features
        df_processed = preprocessor.prepare_features(df)
        
        # Create time features
        df_processed = preprocessor.create_time_features(df_processed)
        
        # Initialize visualizer
        visualizer = Visualizer()
        
        # Run hypothesis tests
        hypothesis_tester = HypothesisTesting()
        test_results = hypothesis_tester.run_all_tests(df_processed)
        
        # Train models
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df_processed)
        trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Print results
        logger.info("\nHypothesis Testing Results:")
        for test_name, result in test_results.items():
            logger.info(f"\n{result['test_name']}:")
            logger.info(f"p-value: {result['p_value']:.4f}")
            logger.info(f"Conclusion: {result['conclusion']}")
        
        logger.info("\nModel Performance Results:")
        for model_name, scores in trainer.model_scores.items():
            logger.info(f"\n{model_name} Model Scores:")
            logger.info(f"MSE: {scores['mse']:.4f}")
            logger.info(f"RMSE: {scores['rmse']:.4f}")
            logger.info(f"R2: {scores['r2']:.4f}")
    else:
        logger.error("Data validation failed")

if __name__ == "__main__":
    main() 