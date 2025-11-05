"""
Complete Training Pipeline for ForeSight
Runs all steps: data generation → feature engineering → model training → SHAP explanations
"""

import os
import sys
from pathlib import Path

def main():
    """Run complete training pipeline."""
    
    print("="*70)
    print("ForeSight - AI-Powered Early Warning Advisor")
    print("Complete Training Pipeline")
    print("="*70)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating Synthetic Financial Data...")
    print("-" * 70)
    sys.path.insert(0, 'data')
    from data.synthetic_data import SigmaFinancialDataGenerator
    generator = SigmaFinancialDataGenerator()
    generator.generate()
    
    # Step 2: No separate feature engineering step needed (built into models)
    
    # Step 3: Train LightGBM
    print("\n[2/5] Training LightGBM Risk Classifier...")
    print("-" * 70)
    sys.path.insert(0, 'models')
    from models.train_lightgbm import LightGBMTrainer
    lgbm_trainer = LightGBMTrainer()
    lgbm_trainer.train()
    
    # Step 4: Train IsolationForest
    print("\n[3/5] Training IsolationForest Anomaly Detector...")
    print("-" * 70)
    from models.train_isoforest import IsolationForestTrainer
    iso_trainer = IsolationForestTrainer()
    iso_trainer.train()
    
    # Step 5: Generate SHAP explanations
    print("\n[4/5] Generating SHAP Explanations...")
    print("-" * 70)
    from models.shap_explainer import SHAPExplainer
    shap_explainer = SHAPExplainer()
    shap_explainer.generate_explanations()
    
    # Step 6: Compute FSI scores (optional)
    print("\n[5/5] Computing Financial Stability Index (FSI)...")
    print("-" * 70)
    # Import FSI calculator directly to avoid circular imports
    import importlib.util
    spec = importlib.util.spec_from_file_location("fsi_calculator", "app/fsi_calculator.py")
    fsi_calculator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fsi_calculator)
    
    fsi_calc = fsi_calculator.FSICalculator()
    results = fsi_calc.compute_all_fsi()
    
    # Save FSI results
    fsi_path = Path("data/fsi_results.csv")
    fsi_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(fsi_path, index=False)
    print(f"✅ FSI results saved to {fsi_path}")
    
    print("\n" + "="*70)
    print("✅ Pipeline Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY in .env file (optional, for AI insights)")
    print("  2. Run dashboard: cd app && streamlit run app.py")
    print("\n")


if __name__ == "__main__":
    main()

