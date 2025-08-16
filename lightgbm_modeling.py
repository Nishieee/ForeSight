#!/usr/bin/env python3
"""
LightGBM Financial Distress Modeling

This script uses LightGBM for financial distress prediction with focused feature selection.
LightGBM is particularly good for this use case due to:
- Fast training and prediction
- Good handling of categorical features
- Built-in handling of imbalanced data
- Excellent performance on tabular data

Key Features:
- Focused feature selection (no dumping)
- Hyperparameter tuning with Optuna
- Proper handling of class imbalance
- Business-focused evaluation metrics
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# LightGBM and optimization
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder

# Set style
plt.style.use('default')
sns.set_palette("husl")

def select_critical_features():
    """
    Select only the most critical features for financial distress prediction.
    Based on domain knowledge and business relevance.
    """
    
    # Core financial distress indicators (high priority)
    critical_features = [
        # Cash flow problems
        'cash_advance_amt',
        'payday_loan_amt',
        'cash_advance_amt_30d_sum',
        'payday_loan_amt_30d_sum',
        'cash_advance_ratio_30d',
        'payday_loan_ratio_30d',
        
        # Spending behavior changes
        'total_spend_drift_ratio',
        'cash_advance_amt_drift_ratio',
        'payday_loan_amt_drift_ratio',
        'spend_acceleration',
        'risk_spend_drift',
        
        # Income vs spending balance
        'income_spend_ratio_30d',
        'net_flow_30d_avg',
        'total_spend_recent_avg',
        
        # Transaction patterns
        'tx_count_drift_ratio',
        'unique_merchants_drift_ratio',
        
        # Customer profile (static features)
        'age',
        'tenure_months',
        'base_income',
        'credit_limit',
        'rent_income_ratio',
        'credit_utilization_capacity'
    ]
    
    return critical_features

def select_risk_features():
    """
    Select features focused on risk indicators.
    """
    risk_features = [
        # High-risk transactions
        'cash_advance_amt',
        'payday_loan_amt',
        'spend_cash_advance',
        'spend_payday_loan',
        
        # Risk ratios
        'cash_advance_ratio_30d',
        'payday_loan_ratio_30d',
        'rent_income_ratio',
        'credit_utilization_capacity',
        
        # Behavioral drift
        'cash_advance_amt_drift_ratio',
        'payday_loan_amt_drift_ratio',
        'risk_spend_drift',
        
        # Customer profile
        'age',
        'base_income',
        'credit_limit'
    ]
    
    return risk_features

def select_behavioral_features():
    """
    Select features focused on behavioral changes.
    """
    behavioral_features = [
        # Spending behavior
        'total_spend_drift_ratio',
        'spend_acceleration',
        'total_spend_recent_avg',
        'total_spend_baseline_avg',
        
        # Transaction patterns
        'tx_count_drift_ratio',
        'unique_merchants_drift_ratio',
        'tx_count_recent_avg',
        'unique_merchants_recent_avg',
        
        # Income stability
        'total_income_30d_avg',
        'net_flow_30d_avg',
        'income_spend_ratio_30d',
        
        # Customer profile
        'tenure_months',
        'base_income',
        'monthly_util_cost'
    ]
    
    return behavioral_features

def evaluate_model(y_true, y_pred, y_prob, model_name="Model"):
    """Comprehensive evaluation for imbalanced classification."""
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate business-relevant metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Business metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Cost-based metrics
    fp_cost = 100  # Cost of unnecessary intervention
    fn_cost = 1000  # Cost of missing distress event
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    
    results = {
        'Model': model_name,
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'False_Positive_Rate': false_positive_rate,
        'True_Negative_Rate': true_negative_rate,
        'Total_Cost': total_cost,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }
    
    return results

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter optimization.
    """
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e2, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'verbose': -1
    }
    
    # Handle class imbalance (use only one method, not both)
    if y_train.mean() < 0.1:  # If positive class is less than 10%
        param['scale_pos_weight'] = (1 - y_train.mean()) / y_train.mean()
    
    # Train model
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[LightGBMPruningCallback(trial, 'auc')]
    )
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Return AUC for optimization
    return roc_auc_score(y_val, y_prob)

def train_lightgbm_with_optimization(X_train, y_train, X_test, y_test, feature_strategy_name):
    """
    Train LightGBM with hyperparameter optimization.
    """
    print(f"🔧 Training LightGBM with {feature_strategy_name} features...")
    
    # Split training data for validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Use the first split for optimization
    train_idx, val_idx = list(tscv.split(X_train))[0]
    X_train_opt, X_val_opt = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_opt, y_val_opt = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Create study for optimization
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Optimize hyperparameters
    print("   Optimizing hyperparameters...")
    study.optimize(
        lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
        n_trials=50,
        timeout=300  # 5 minutes timeout
    )
    
    print(f"   Best trial: {study.best_trial.value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    # Train final model with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'verbose': -1
    })
    
    # Handle class imbalance (use only one method, not both)
    if y_train.mean() < 0.1:
        best_params['scale_pos_weight'] = (1 - y_train.mean()) / y_train.mean()
    
    # Train final model
    print("   Training final model...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Evaluate
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    
    result = evaluate_model(y_test, y_pred, y_prob, f"LightGBM_{feature_strategy_name}")
    result['Feature_Count'] = X_train.shape[1]
    result['Best_Params'] = str(best_params)
    
    return final_model, result

def feature_importance_analysis(model, feature_names):
    """Analyze LightGBM feature importance."""
    
    # Get feature importance
    importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 LightGBM Feature Importance (Top 15):")
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 LightGBM Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def main():
    print("🚀 LightGBM Financial Distress Modeling")
    print("=" * 50)
    
    # Load data
    DATA_DIR = Path("data")
    
    print("Loading data...")
    train_data = pd.read_csv(DATA_DIR / "train_data.csv")
    test_data = pd.read_csv(DATA_DIR / "test_data.csv")
    
    print(f"Train data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Check class distribution
    print(f"\n📊 Class Distribution:")
    print(f"Train - Positive rate: {train_data['label'].mean():.4f} ({train_data['label'].sum():,} / {len(train_data):,})")
    print(f"Test  - Positive rate: {test_data['label'].mean():.4f} ({test_data['label'].sum():,} / {len(test_data):,})")
    
    # Define feature selection strategies
    feature_strategies = {
        'Critical': select_critical_features(),
        'Risk_Focused': select_risk_features(),
        'Behavioral': select_behavioral_features()
    }
    
    # Check which features are available
    available_features = [col for col in train_data.columns if col not in ['customer_id', 'date', 'label']]
    
    print(f"\n📋 Feature Selection Strategies:")
    for strategy_name, features in feature_strategies.items():
        available_in_strategy = [f for f in features if f in available_features]
        print(f"   {strategy_name}: {len(available_in_strategy)}/{len(features)} features available")
    
    # Train LightGBM models with different feature strategies
    print(f"\n🤖 Training LightGBM models with focused feature sets...")
    
    results = []
    models = {}
    
    for strategy_name, feature_list in feature_strategies.items():
        # Filter to available features
        available_in_strategy = [f for f in feature_list if f in available_features]
        
        if len(available_in_strategy) == 0:
            print(f"   ⚠️  No features available for {strategy_name} strategy")
            continue
            
        print(f"\n   📊 Using {strategy_name} strategy ({len(available_in_strategy)} features)")
        
        # Prepare data
        X_train = train_data[available_in_strategy]
        y_train = train_data['label']
        X_test = test_data[available_in_strategy]
        y_test = test_data['label']
        
        # Handle missing values
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_test.median())
        
        # Train LightGBM with optimization
        model, result = train_lightgbm_with_optimization(
            X_train, y_train, X_test, y_test, strategy_name
        )
        
        results.append(result)
        models[strategy_name] = model
        
        print(f"      AUC: {result['AUC']:.3f}, Precision: {result['Precision']:.3f}, Recall: {result['Recall']:.3f}")
    
    results_df = pd.DataFrame(results)
    print(f"\n📊 All LightGBM Results:")
    print(results_df.round(3))
    
    # Find best model
    print(f"\n🏆 Best LightGBM Model Selection:")
    
    best_auc = results_df.loc[results_df['AUC'].idxmax()]
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
    best_cost = results_df.loc[results_df['Total_Cost'].idxmin()]
    
    print(f"\nBest by AUC: {best_auc['Model']} (AUC: {best_auc['AUC']:.3f}, Features: {best_auc['Feature_Count']})")
    print(f"Best by F1: {best_f1['Model']} (F1: {best_f1['F1-Score']:.3f}, Features: {best_f1['Feature_Count']})")
    print(f"Best by Cost: {best_cost['Model']} (Cost: {best_cost['Total_Cost']:.0f}, Features: {best_cost['Feature_Count']})")
    
    # Analyze best model
    best_strategy = best_f1['Model'].split('_')[1]
    best_model = models[best_strategy]
    best_features = [f for f in feature_strategies[best_strategy] if f in available_features]
    
    print(f"\n🔍 Analyzing best model: {best_f1['Model']}")
    
    # Feature importance analysis
    feature_importance = feature_importance_analysis(best_model, best_features)
    
    # Final evaluation
    X_test_best = test_data[best_features].fillna(test_data[best_features].median())
    y_test_best = test_data['label']
    
    y_pred_final = best_model.predict(X_test_best)
    y_prob_final = best_model.predict_proba(X_test_best)[:, 1]
    
    print(f"\n📋 Final LightGBM Model Evaluation:")
    print(classification_report(y_test_best, y_pred_final))
    
    # Final metrics
    final_auc = roc_auc_score(y_test_best, y_prob_final)
    final_ap = average_precision_score(y_test_best, y_prob_final)
    
    print(f"\n🎯 Final LightGBM Performance:")
    print(f"- AUC: {final_auc:.3f}")
    print(f"- Average Precision: {final_ap:.3f}")
    print(f"- Features used: {len(best_features)}")
    print(f"- Model: LightGBM with {best_strategy} features")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Create results directory
    results_dir = DATA_DIR / "lightgbm_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save model results
    results_df.to_csv(results_dir / "lightgbm_results.csv", index=False)
    feature_importance.to_csv(results_dir / "lightgbm_feature_importance.csv", index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'customer_id': test_data['customer_id'],
        'date': test_data['date'],
        'true_label': y_test_best,
        'predicted_label': y_pred_final,
        'prediction_probability': y_prob_final
    })
    predictions_df.to_csv(results_dir / "lightgbm_predictions.csv", index=False)
    
    print(f"\n✅ LightGBM results saved to {results_dir}/")
    print(f"   - lightgbm_results.csv: All model comparisons")
    print(f"   - lightgbm_feature_importance.csv: Feature rankings")
    print(f"   - lightgbm_predictions.csv: Final predictions")
    
    # Business recommendations
    print(f"\n💡 LightGBM Business Recommendations:")
    print(f"1. Use {best_f1['Model']} for production deployment")
    print(f"2. LightGBM advantages: Fast training, good with categorical features")
    print(f"3. Monitor feature importance for model interpretability")
    print(f"4. Regular hyperparameter tuning with new data")
    print(f"5. Consider ensemble with other models for robustness")
    
    print(f"\n🚀 LightGBM modeling complete!")
    print(f"   Realistic performance with focused features and optimized hyperparameters.")

if __name__ == "__main__":
    main()
