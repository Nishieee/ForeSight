"""
Example Usage of ForeSight Components
Demonstrates how to use the system programmatically.
"""

import sys
from pathlib import Path

# Add paths
sys.path.append('data')
sys.path.append('models')
sys.path.append('app')


def example_1_generate_data():
    """Example: Generate synthetic financial data."""
    print("Example 1: Generating Synthetic Data")
    print("-" * 50)
    
    from data.synthetic_data import SigmaFinancialDataGenerator
    
    # Create generator
    generator = SigmaFinancialDataGenerator(config_path="config/settings.yaml")
    
    # Generate data
    data, metadata = generator.generate()
    
    print(f"Generated {len(data)} rows for {metadata['user_id'].nunique()} users")
    print("\nSample data:")
    print(data.head())
    return data, metadata


def example_2_engineer_features():
    """Example: Engineer features from raw data."""
    print("\nExample 2: Feature Engineering")
    print("-" * 50)
    
    from data.feature_engineering import FinancialFeatureEngineer
    import pandas as pd
    
    # Load data
    df = pd.read_csv("data/synthetic_financial_data.csv")
    
    # Engineer features
    engineer = FinancialFeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    feature_cols = engineer.get_feature_columns(df_features)
    print(f"Created {len(feature_cols)} engineered features")
    print(f"\nSample features: {feature_cols[:10]}")
    
    return df_features


def example_3_predict_risk():
    """Example: Predict financial distress risk for a user."""
    print("\nExample 3: Risk Prediction")
    print("-" * 50)
    
    from models.train_lightgbm import LightGBMTrainer
    from data.feature_engineering import FinancialFeatureEngineer
    import pandas as pd
    
    # Load model (assumes it's already trained)
    if not Path("models/lightgbm_model.pkl").exists():
        print("Model not found. Please run train_pipeline.py first.")
        return None
    
    # Load data
    df = pd.read_csv("data/synthetic_financial_data.csv")
    engineer = FinancialFeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    # Get features for user 1, month 1
    user_data = df_features[(df_features['user_id'] == 1) & (df_features['month'] == 1)]
    feature_cols = engineer.get_feature_columns(df_features)
    X = user_data[feature_cols].fillna(user_data[feature_cols].median())
    
    # Predict
    trainer = LightGBMTrainer()
    risk_prob = trainer.predict(X)
    
    print(f"User 1, Month 1 - Risk Probability: {risk_prob[0]:.2%}")
    return risk_prob


def example_4_compute_fsi():
    """Example: Compute Financial Stability Index."""
    print("\nExample 4: Financial Stability Index")
    print("-" * 50)
    
    from app.fsi_calculator import FSICalculator
    import pandas as pd
    
    # Load FSI results (or compute)
    if Path("data/fsi_results.csv").exists():
        results = pd.read_csv("data/fsi_results.csv")
    else:
        calculator = FSICalculator()
        results = calculator.compute_all_fsi()
    
    print(f"Computed FSI for {len(results)} records")
    print("\nFSI Statistics:")
    print(results['FSI'].describe())
    
    print("\nRisk Distribution:")
    print(results['risk_label'].value_counts())
    
    return results


def example_5_get_explanation():
    """Example: Get SHAP explanation for a user."""
    print("\nExample 5: SHAP Explanation")
    print("-" * 50)
    
    import json
    
    # Load explanations
    if not Path("data/user_explanations.json").exists():
        print("SHAP explanations not found. Please run train_pipeline.py first.")
        return None
    
    with open("data/user_explanations.json", 'r') as f:
        explanations = json.load(f)
    
    # Get explanation for user 1, month 1
    key = "1_m1"
    if key in explanations:
        expl = explanations[key]
        print(f"User {expl['user_id']}, Month {expl['month']}")
        print(f"Prediction: {expl['prediction']:.2%}")
        print("\nTop Risk Factors:")
        
        for i, feat in enumerate(expl['top_features'][:5], 1):
            direction = "increases" if feat['direction'] == 'increases_risk' else "decreases"
            print(f"{i}. {feat['feature']}: {direction} risk (SHAP = {feat['shap_value']:.4f})")
    
    return explanations.get(key)


def example_6_generate_insight():
    """Example: Generate AI insight (requires OpenAI API key)."""
    print("\nExample 6: AI-Generated Insight")
    print("-" * 50)
    
    import os
    
    if not os.getenv('OPENAI_API_KEY'):
        print("OPENAI_API_KEY not set. Skipping...")
        return None
    
    from app.insights import OpenAIInsightGenerator
    
    generator = OpenAIInsightGenerator(config_path="config/settings.yaml")
    insight = generator.generate_insight(user_id=1, month=1)
    
    print("Generated Insight:")
    print("-" * 50)
    print(insight)
    
    return insight


def main():
    """Run all examples."""
    print("=" * 70)
    print("ForeSight - Example Usage")
    print("=" * 70)
    
    try:
        # Example 1: Generate data
        data, metadata = example_1_generate_data()
        
        # Example 2: Engineer features
        df_features = example_2_engineer_features()
        
        # Example 3: Predict risk (requires trained model)
        if Path("models/lightgbm_model.pkl").exists():
            risk_prob = example_3_predict_risk()
        
        # Example 4: Compute FSI
        if Path("models/lightgbm_model.pkl").exists() and Path("models/isoforest_model.pkl").exists():
            fsi_results = example_4_compute_fsi()
        
        # Example 5: Get SHAP explanation
        if Path("data/user_explanations.json").exists():
            explanation = example_5_get_explanation()
        
        # Example 6: Generate AI insight
        example_6_generate_insight()
        
        print("\n" + "=" * 70)
        print("✅ Examples Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you've run train_pipeline.py first to create models and data.")


if __name__ == "__main__":
    main()

