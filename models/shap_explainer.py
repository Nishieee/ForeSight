"""
SHAP Explainer for Model Interpretability
Computes global and per-user explanations for LightGBM predictions.
"""

import pandas as pd
import numpy as np
import shap
import joblib
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append('..')

from data.feature_engineering import FinancialFeatureEngineer


class SHAPExplainer:
    """
    Generate SHAP explanations for LightGBM model predictions.
    Provides both global feature importance and individual user explanations.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize explainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = self.config['models']['lightgbm']['save_path']
        
    def load_model_and_data(self) -> Tuple:
        """
        Load trained model and prepare data for SHAP computation.
        
        Returns:
            (model, X_train, X_test, y_test, feature_names)
        """
        # Load model
        model = joblib.load(self.model_path)
        
        # Load data
        data_path = self.config['data']['output_path']
        df = pd.read_csv(data_path)
        
        # Engineer features
        engineer = FinancialFeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Get feature columns
        feature_cols = engineer.get_feature_columns(df_features)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X = df_features[feature_cols]
        y = df_features['distress_label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Loaded model and data:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return model, X_train, X_test, y_test, feature_cols
    
    def compute_global_explanations(self, 
                                    model, 
                                    X_train: pd.DataFrame,
                                    max_samples: int = 1000) -> shap.TreeExplainer:
        """
        Compute global SHAP explanations using TreeExplainer.
        
        Args:
            model: Trained LightGBM model
            X_train: Training data
            max_samples: Maximum samples for explanation (for speed)
            
        Returns:
            TreeExplainer with computed SHAP values
        """
        print("\nComputing global SHAP explanations...")
        
        # Sample for speed if needed
        X_sample = X_train.sample(min(max_samples, len(X_train)), random_state=42)
        
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        print(f"   Computing SHAP values for {len(X_sample)} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        print("✅ Global explanations computed")
        
        # Save global feature importance
        global_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'shap_importance': np.abs(shap_values[1] if isinstance(shap_values, list) else shap_values).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        importance_path = "models/shap_global_importance.csv"
        global_importance.to_csv(importance_path, index=False)
        print(f"✅ Global importance saved to {importance_path}")
        
        return explainer
    
    def compute_user_explanation(self,
                                explainer: shap.TreeExplainer,
                                user_features: pd.Series) -> Dict:
        """
        Compute SHAP explanation for a single user.
        
        Args:
            explainer: Trained SHAP explainer
            user_features: Feature vector for one user
            
        Returns:
            Dictionary with user explanation
        """
        # Ensure user_features is a DataFrame
        if isinstance(user_features, pd.Series):
            user_features = user_features.to_frame().T
        
        # Compute SHAP values
        shap_values = explainer.shap_values(user_features)[1] if isinstance(
            explainer.shap_values(user_features), list
        ) else explainer.shap_values(user_features)
        
        # Get prediction
        prediction = explainer.expected_value[1] if isinstance(
            explainer.expected_value, list
        ) else explainer.expected_value
        prediction += shap_values.sum()
        
        # Prepare explanation
        explanation = {
            'expected_value': explainer.expected_value[1] if isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value,
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'features': user_features.columns.tolist(),
            'prediction': float(prediction),
            'feature_contributions': dict(zip(
                user_features.columns,
                shap_values.flatten()
            ))
        }
        
        return explanation
    
    def save_user_explanations(self,
                               explainer: shap.TreeExplainer,
                               df_features: pd.DataFrame,
                               feature_cols: List[str],
                               save_path: str = "data/user_explanations.json"):
        """
        Compute and save SHAP explanations for all users.
        
        Args:
            explainer: Trained SHAP explainer
            df_features: Feature-engineered DataFrame with all users
            feature_cols: List of feature column names
            save_path: Path to save explanations
        """
        print(f"\nComputing per-user SHAP explanations...")
        
        user_explanations = {}
        
        # Process users in batches
        batch_size = 100
        X = df_features[feature_cols]
        
        for i in range(0, len(df_features), batch_size):
            batch = X.iloc[i:i+batch_size]
            user_ids = df_features.iloc[i:i+batch_size]['user_id'].values
            months = df_features.iloc[i:i+batch_size]['month'].values
            
            # Compute SHAP for batch
            shap_values = explainer.shap_values(batch)[1] if isinstance(
                explainer.shap_values(batch), list
            ) else explainer.shap_values(batch)
            
            # Get base prediction
            base_value = explainer.expected_value[1] if isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value
            
            for j, (user_id, month) in enumerate(zip(user_ids, months)):
                key = f"{user_id}_m{month}"
                
                contrib = dict(zip(
                    feature_cols,
                    shap_values[j]
                ))
                
                # Get top 5 driving features
                top_features = sorted(
                    contrib.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
                
                user_explanations[key] = {
                    'user_id': int(user_id),
                    'month': int(month),
                    'base_value': float(base_value),
                    'prediction': float(base_value + shap_values[j].sum()),
                    'top_features': [
                        {
                            'feature': feat,
                            'shap_value': float(val),
                            'direction': 'increases_risk' if val > 0 else 'decreases_risk'
                        }
                        for feat, val in top_features
                    ]
                }
            
            if (i // batch_size) % 10 == 0:
                print(f"   Processed {i}/{len(df_features)} users...")
        
        # Save to JSON
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(user_explanations, f, indent=2)
        
        print(f"✅ User explanations saved to {save_path}")
        print(f"   Total users: {len(user_explanations)}")
        
        return user_explanations
    
    def generate_explanations(self):
        """
        Main method to generate all SHAP explanations.
        """
        print("="*50)
        print("Generating SHAP Explanations")
        print("="*50)
        
        # Load model and data
        model, X_train, X_test, y_test, feature_cols = self.load_model_and_data()
        
        # Compute global explanations
        explainer = self.compute_global_explanations(model, X_train)
        
        # Load full data for user explanations
        data_path = self.config['data']['output_path']
        df = pd.read_csv(data_path)
        engineer = FinancialFeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Save user explanations
        user_explanations = self.save_user_explanations(
            explainer, df_features, feature_cols
        )
        
        return explainer, user_explanations


def main():
    """Generate SHAP explanations."""
    explainer = SHAPExplainer()
    shap_explainer, user_explanations = explainer.generate_explanations()
    return shap_explainer, user_explanations


if __name__ == "__main__":
    shap_explainer, user_explanations = main()

