"""
Financial Stability Index (FSI) Calculator
Combines risk probability and anomaly score into unified stability metric.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple
import sys
sys.path.append('..')

from models.train_lightgbm import LightGBMTrainer
from models.train_isoforest import IsolationForestTrainer
from data.feature_engineering import FinancialFeatureEngineer


class FSICalculator:
    """
    Compute Financial Stability Index (FSI) for users.
    
    FSI = (1 - risk_prob) * 0.7 + (1 - anomaly_score_normalized) * 0.3
    
    Where:
    - risk_prob: LightGBM prediction (0-1)
    - anomaly_score_normalized: IsolationForest anomaly score normalized to (0-1)
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fsi_config = self.config['fsi']
        self.thresholds = self.fsi_config['thresholds']
        
        # Dynamic threshold sensitivity factor (alpha)
        # Higher alpha = more sensitive (lower threshold)
        self.alpha = 1.5
        
    def normalize_anomaly_scores(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Normalize IsolationForest anomaly scores to 0-1 range.
        
        IsolationForest scores are typically negative (more negative = more anomalous).
        We need to convert them to 0-1 where 1 = most stable.
        
        Args:
            anomaly_scores: Raw IsolationForest scores
            
        Returns:
            Normalized scores in [0, 1] range
        """
        # Anomaly scores are typically in range [-0.5, 0.5]
        # More negative = more anomalous
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        
        # Normalize to 0-1, then invert so high = stable
        normalized = (anomaly_scores - min_score) / (max_score - min_score + 1e-8)
        normalized = 1 - normalized  # Invert: high anomaly = low stability
        
        return normalized
    
    def compute_fsi(self,
                    risk_prob: np.ndarray,
                    anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Compute Financial Stability Index.
        
        Args:
            risk_prob: LightGBM risk probabilities (0-1)
            anomaly_scores: IsolationForest anomaly scores
            
        Returns:
            FSI scores in [0, 100] range
        """
        # Normalize anomaly scores
        anomaly_normalized = self.normalize_anomaly_scores(anomaly_scores)
        
        # Compute FSI
        stability = (1 - risk_prob) * self.fsi_config['risk_weight'] + \
                   anomaly_normalized * self.fsi_config['anomaly_weight']
        
        # Scale to 0-100
        fsi = stability * 100
        
        return fsi
    
    def classify_fsi(self, fsi_scores: np.ndarray) -> np.ndarray:
        """
        Classify FSI scores into risk categories.
        
        Args:
            fsi_scores: FSI values
            
        Returns:
            Array of risk labels
        """
        labels = []
        
        for score in fsi_scores:
            if score >= self.thresholds['stable']:
                labels.append(('🟢 Stable', 'stable'))
            elif score >= self.thresholds['watchlist']:
                labels.append(('🟡 Watchlist', 'watchlist'))
            elif score >= self.thresholds['early_distress']:
                labels.append(('🟠 Early Distress', 'early_distress'))
            else:
                labels.append(('🔴 High Risk', 'high_risk'))
        
        return labels
    
    def compute_dynamic_thresholds(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute dynamic, personalized risk thresholds based on user's historical FSI volatility.
        
        For each user, the threshold adapts to their typical financial stability range.
        Users with naturally erratic patterns get more tolerant thresholds.
        Users with steady patterns get stricter thresholds.
        
        Args:
            results: DataFrame with FSI scores by user and month
            
        Returns:
            DataFrame with additional columns: fsi_mean, fsi_std, dynamic_threshold, risk_flag
        """
        # Calculate user-level statistics
        user_stats = results.groupby('user_id')['FSI'].agg(['mean', 'std']).reset_index()
        user_stats.columns = ['user_id', 'fsi_mean', 'fsi_std']
        
        # Fill NaN std (users with only 1 month) with median std
        user_stats['fsi_std'] = user_stats['fsi_std'].fillna(results.groupby('user_id')['FSI'].std().median())
        
        # Merge back to results
        results = results.merge(user_stats, on='user_id', how='left')
        
        # Compute dynamic threshold
        # Formula: threshold = mean_FSI - alpha * std_FSI
        # Where alpha controls sensitivity (default 1.5)
        results['dynamic_threshold'] = results['fsi_mean'] - self.alpha * results['fsi_std']
        
        # Ensure threshold is within reasonable bounds (20-80)
        results['dynamic_threshold'] = results['dynamic_threshold'].clip(lower=20, upper=80)
        
        # Flag users who fall below their personalized threshold
        results['risk_flag'] = (results['FSI'] < results['dynamic_threshold']).astype(int)
        
        # Add personalized risk classification
        def classify_personalized(row):
            current_fsi = row['FSI']
            threshold = row['dynamic_threshold']
            mean_fsi = row['fsi_mean']
            
            if current_fsi >= mean_fsi + row['fsi_std']:
                return '🟢 Stable', 'stable'
            elif current_fsi >= threshold:
                return '🟡 Watchlist', 'watchlist'
            elif current_fsi >= threshold - 10:
                return '🟠 Early Distress', 'early_distress'
            else:
                return '🔴 High Risk', 'high_risk'
        
        results[['personalized_risk_label', 'personalized_risk_category']] = \
            results.apply(classify_personalized, axis=1, result_type='expand')
        
        return results
    
    def compute_all_fsi(self) -> pd.DataFrame:
        """
        Compute FSI for all users in the dataset.
        
        Returns:
            DataFrame with FSI scores and classifications
        """
        # Load data
        data_path = self.config['data']['output_path']
        df = pd.read_csv(data_path)
        
        # Engineer features
        engineer = FinancialFeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Get feature columns
        feature_cols = engineer.get_feature_columns(df_features)
        
        # Split data by model type
        behavioral_cols = [
            'income_deviation', 'income_trend_3m', 'income_pct_change_3m',
            'income_cv_6', 'income_drop_30pct', 'income_drop_50pct',
            'spend_deviation', 'spend_trend_3m', 'spend_income_ratio',
            'essential_luxury_ratio', 'spending_behavior_change',
            'credit_trend', 'credit_util_ma_3', 'cumulative_missed',
            'income_momentum', 'spend_momentum'
        ]
        behavioral_cols = [col for col in behavioral_cols if col in df_features.columns]
        
        # Get predictions from both models
        # LightGBM
        lgbm_trainer = LightGBMTrainer()
        risk_prob = lgbm_trainer.predict(df_features[feature_cols].fillna(df_features[feature_cols].median()))
        
        # IsolationForest
        iso_trainer = IsolationForestTrainer()
        anomaly_scores, _ = iso_trainer.predict(df_features[behavioral_cols])
        
        # Compute FSI
        fsi_scores = self.compute_fsi(risk_prob, anomaly_scores)
        
        # Classify
        classifications = self.classify_fsi(fsi_scores)
        labels, categories = zip(*classifications)
        
        # Create results DataFrame
        results = df_features[['user_id', 'month', 'month_label']].copy()
        results['risk_probability'] = risk_prob
        results['anomaly_score'] = anomaly_scores
        results['normalized_anomaly'] = self.normalize_anomaly_scores(anomaly_scores)
        results['FSI'] = fsi_scores
        results['risk_label'] = labels
        results['risk_category'] = categories
        
        # Compute dynamic personalized thresholds
        results = self.compute_dynamic_thresholds(results)
        
        return results


def main():
    """Test FSI calculation."""
    calculator = FSICalculator()
    results = calculator.compute_all_fsi()
    
    print("="*50)
    print("Financial Stability Index Results")
    print("="*50)
    print(f"\nTotal records: {len(results)}")
    print(f"\nFSI Statistics:")
    print(results['FSI'].describe())
    print(f"\nRisk Distribution:")
    print(results['risk_label'].value_counts())
    
    return results


if __name__ == "__main__":
    results = main()

