"""
IsolationForest Anomaly Detection for Behavioral Drift
Trains an anomaly detector to identify users deviating from their normal patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append('..')

from data.feature_engineering import FinancialFeatureEngineer


class IsolationForestTrainer:
    """
    Train IsolationForest to detect anomalous financial behaviors.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['isoforest']
        self.save_path = self.model_config['save_path']
        self.contamination = self.model_config['contamination']
        
    def get_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features that capture behavioral changes for anomaly detection.
        
        Args:
            df: Feature-engineered DataFrame
            
        Returns:
            DataFrame with behavioral features
        """
        behavioral_cols = [
            # Income stability
            'income_deviation',
            'income_trend_3m',
            'income_pct_change_3m',
            'income_cv_6',
            'income_drop_30pct',
            'income_drop_50pct',
            
            # Spending behavior
            'spend_deviation',
            'spend_trend_3m',
            'spend_income_ratio',
            'essential_luxury_ratio',
            'spending_behavior_change',
            
            # Credit behavior
            'credit_trend',
            'credit_util_ma_3',
            'cumulative_missed',
            
            # Momentum and trends
            'income_momentum',
            'spend_momentum',
        ]
        
        # Only select columns that exist
        available_cols = [col for col in behavioral_cols if col in df.columns]
        
        return df[available_cols]
    
    def train(self):
        """
        Train IsolationForest model on behavioral features.
        
        Returns:
            Trained model and scaler
        """
        # Load data
        data_path = self.config['data']['output_path']
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Engineer features
        print("Engineering features...")
        engineer = FinancialFeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Get behavioral features
        X = self.get_behavioral_features(df_features)
        
        # Handle NaN values
        X = X.fillna(X.median())
        
        print(f"   Behavioral features: {len(X.columns)}")
        print(f"   Samples: {len(X)}")
        
        # Train-test split
        X_train, X_test, _, _ = train_test_split(
            X, df_features['distress_label'], 
            test_size=0.2, 
            random_state=42
        )
        
        print("\n" + "="*50)
        print("Training IsolationForest Model")
        print("="*50)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.model_config['random_state'],
            n_estimators=200
        )
        
        model.fit(X_train_scaled)
        
        # Evaluate on test set (using distress label as proxy for anomaly)
        anomaly_scores_train = model.score_samples(X_train_scaled)
        anomaly_scores_test = model.score_samples(X_test_scaled)
        
        # Convert to anomaly labels (-1 = anomaly, 1 = normal)
        predictions_train = model.predict(X_train_scaled)
        predictions_test = model.predict(X_test_scaled)
        
        print("\n" + "="*50)
        print("Model Evaluation")
        print("="*50)
        
        print(f"\nAnomaly Score Statistics:")
        print(f"   Train - Mean: {anomaly_scores_train.mean():.4f}, Std: {anomaly_scores_train.std():.4f}")
        print(f"   Test  - Mean: {anomaly_scores_test.mean():.4f}, Std: {anomaly_scores_test.std():.4f}")
        
        print(f"\nAnomaly Detection Rate:")
        print(f"   Train: {(predictions_train == -1).mean():.2%}")
        print(f"   Test:  {(predictions_test == -1).mean():.2%}")
        
        # Save model and scaler
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.save_path)
        
        scaler_path = "models/isoforest_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        print(f"\n✅ Model saved to {self.save_path}")
        print(f"✅ Scaler saved to {scaler_path}")
        
        return model, scaler
    
    def load_model_and_scaler(self):
        """Load trained model and scaler from disk."""
        model = joblib.load(self.save_path)
        scaler = joblib.load("models/isoforest_scaler.pkl")
        return model, scaler
    
    def predict(self, X: pd.DataFrame) -> tuple:
        """
        Get anomaly scores and predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            (anomaly_scores, predictions) where:
            - anomaly_scores: negative values = more anomalous
            - predictions: -1 = anomaly, 1 = normal
        """
        model, scaler = self.load_model_and_scaler()
        X_scaled = scaler.transform(X.fillna(X.median()))
        
        anomaly_scores = model.score_samples(X_scaled)
        predictions = model.predict(X_scaled)
        
        return anomaly_scores, predictions


def main():
    """Train IsolationForest model."""
    trainer = IsolationForestTrainer()
    model, scaler = trainer.train()
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()

