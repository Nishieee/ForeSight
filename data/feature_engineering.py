"""
Feature Engineering for Financial Distress Detection
Creates rolling averages, ratios, trends, and behavioral indicators.
"""

import pandas as pd
import numpy as np
from typing import List
import yaml

class FinancialFeatureEngineer:
    """
    Engineering features for financial distress detection:
    - Rolling averages for income and spending
    - Trend indicators
    - Ratio features (essential/luxury, spending/income)
    - Volatility metrics
    - Behavioral change indicators
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.window_sizes = self.config['features']['window_sizes']
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the dataset.
        
        Args:
            df: Raw financial data with columns:
                user_id, month, income, total_spend, essential_spend,
                luxury_spend, credit_utilization, missed_payments
        
        Returns:
            DataFrame with engineered features added
        """
        df = df.sort_values(['user_id', 'month']).reset_index(drop=True)
        
        df = self._add_income_features(df)
        df = self._add_spending_features(df)
        df = self._add_ratio_features(df)
        df = self._add_trend_features(df)
        df = self._add_volatility_features(df)
        df = self._add_credit_features(df)
        df = self._add_behavioral_features(df)
        
        # Drop rows with NaN (caused by rolling windows at the start)
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _add_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add income-related features."""
        for window in self.window_sizes:
            df[f'income_ma_{window}'] = df.groupby('user_id')['income'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'income_std_{window}'] = df.groupby('user_id')['income'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Income trend (current vs previous periods)
        df['income_trend_3m'] = df.groupby('user_id')['income'].transform(
            lambda x: x - x.shift(1)
        )
        df['income_pct_change_3m'] = df.groupby('user_id')['income'].transform(
            lambda x: x.pct_change()
        )
        
        return df
    
    def _add_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spending-related features."""
        for window in self.window_sizes:
            df[f'spend_ma_{window}'] = df.groupby('user_id')['total_spend'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            df[f'essential_ma_{window}'] = df.groupby('user_id')['essential_spend'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            df[f'luxury_ma_{window}'] = df.groupby('user_id')['luxury_spend'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Spending trend
        df['spend_trend_3m'] = df.groupby('user_id')['total_spend'].transform(
            lambda x: x - x.shift(1)
        )
        
        return df
    
    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ratio-based features."""
        # Spending to income ratio
        df['spend_income_ratio'] = df['total_spend'] / (df['income'] + 1e-8)
        
        # Essential to luxury spending ratio
        df['essential_luxury_ratio'] = df['essential_spend'] / (df['luxury_spend'] + 1e-8)
        
        # Current vs historical ratios
        df['spend_ratio_vs_avg'] = df['spend_income_ratio'] / (
            df['total_spend'] / (df['income_ma_6'] + 1e-8)
        )
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Rolling momentum
        df['income_momentum'] = df.groupby('user_id')['income'].transform(
            lambda x: x.rolling(window=3, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / (y.mean() + 1e-8)
            )
        )
        
        df['spend_momentum'] = df.groupby('user_id')['total_spend'].transform(
            lambda x: x.rolling(window=3, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / (y.mean() + 1e-8)
            )
        )
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility and stability metrics."""
        for window in self.window_sizes:
            df[f'income_cv_{window}'] = (
                df[f'income_std_{window}'] / (df[f'income_ma_{window}'] + 1e-8)
            )
            df[f'spend_cv_{window}'] = (
                df.groupby('user_id')['total_spend'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ) / (df[f'spend_ma_{window}'] + 1e-8)
            )
        
        # Stability score (inverse of volatility)
        df['income_stability'] = 1 / (df['income_cv_6'] + 0.1)
        
        return df
    
    def _add_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add credit utilization and payment features."""
        for window in self.window_sizes:
            df[f'credit_util_ma_{window}'] = df.groupby('user_id')['credit_utilization'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Credit trend
        df['credit_trend'] = df.groupby('user_id')['credit_utilization'].transform(
            lambda x: x - x.shift(1)
        )
        
        # Cumulative missed payments
        df['cumulative_missed'] = df.groupby('user_id')['missed_payments'].transform(
            lambda x: x.cumsum()
        )
        
        # Missed payment streak
        df['missed_streak'] = df.groupby('user_id').apply(
            lambda x: (x['missed_payments'] == 1).groupby(
                (x['missed_payments'] != x['missed_payments'].shift()).cumsum()
            ).cumsum()
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral change indicators."""
        # Deviation from personal baseline
        df['income_deviation'] = (df['income'] - df['income_ma_6']) / (df['income_ma_6'] + 1e-8)
        df['spend_deviation'] = (df['total_spend'] - df['spend_ma_6']) / (df['spend_ma_6'] + 1e-8)
        
        # Spending behavior change
        df['spending_behavior_change'] = np.where(
            df['essential_luxury_ratio'] > df.groupby('user_id')['essential_luxury_ratio'].transform('median') * 1.2,
            1, 0
        )
        
        # Income drops
        df['income_drop_30pct'] = (df['income_deviation'] < -0.3).astype(int)
        df['income_drop_50pct'] = (df['income_deviation'] < -0.5).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
        """
        Get list of engineered feature columns.
        
        Args:
            df: DataFrame with engineered features
            exclude: Columns to exclude from feature list
        
        Returns:
            List of feature column names
        """
        exclude = exclude or ['user_id', 'month', 'distress_label', 'month_label']
        
        feature_cols = [col for col in df.columns if col not in exclude]
        return feature_cols


def main():
    """Test feature engineering."""
    import synthetic_data
    
    # Generate test data
    generator = synthetic_data.SigmaFinancialDataGenerator(config_path="config/settings.yaml")
    data, _ = generator.generate()
    
    # Engineer features
    engineer = FinancialFeatureEngineer()
    df_features = engineer.engineer_features(data)
    
    print(f"✅ Feature engineering complete")
    print(f"   Original columns: {len(data.columns)}")
    print(f"   Engineered columns: {len(df_features.columns)}")
    print(f"\nFeature columns:")
    feature_cols = engineer.get_feature_columns(df_features)
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:10]}")
    
    return df_features


if __name__ == "__main__":
    df_features = main()

