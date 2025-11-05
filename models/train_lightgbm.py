"""
LightGBM Model Training for Financial Distress Prediction
Trains a binary classifier to predict financial distress probability.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)
import sys
sys.path.append('..')

from data.feature_engineering import FinancialFeatureEngineer


class LightGBMTrainer:
    """
    Train and evaluate LightGBM model for financial distress prediction.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['lightgbm']
        self.save_path = self.model_config['save_path']
        
    def load_and_prepare_data(self) -> tuple:
        """
        Load synthetic data, engineer features, and prepare for training.
        
        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        # Load data
        data_path = self.config['data']['output_path']
        
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Engineer features
        print("Engineering features...")
        engineer = FinancialFeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Get feature columns
        feature_cols = engineer.get_feature_columns(df_features)
        X = df_features[feature_cols]
        y = df_features['distress_label']
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")
        print(f"   Distress rate: {y.mean():.2%}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train(self) -> lgb.Booster:
        """
        Train LightGBM model on financial distress data.
        
        Returns:
            Trained LightGBM model
        """
        X_train, X_test, y_train, y_test, feature_names = self.load_and_prepare_data()
        
        print("\n" + "="*50)
        print("Training LightGBM Model")
        print("="*50)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            self.model_config['params'],
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'eval'],
            num_boost_round=self.model_config['params']['n_estimators'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=True),
                lgb.log_evaluation(period=20)
            ]
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("Model Performance")
        print("="*50)
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save model
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.save_path)
        print(f"\n✅ Model saved to {self.save_path}")
        
        # Save feature list
        feature_list_path = "models/feature_list.txt"
        with open(feature_list_path, 'w') as f:
            f.write('\n'.join(feature_names))
        
        return model
    
    def load_model(self):
        """Load trained model from disk."""
        return joblib.load(self.save_path)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        model = self.load_model()
        return model.predict(X)


def main():
    """Train LightGBM model."""
    trainer = LightGBMTrainer()
    model = trainer.train()
    return model


if __name__ == "__main__":
    model = main()

