"""
Synthetic Financial Data Generator with Sigma Layer
Creates realistic financial data for 8000+ users over 12 months
with diverse income brackets, spending habits, and distress patterns.
"""

import numpy as np
import pandas as pd
from faker import Faker
import yaml
from typing import Tuple, Dict
from pathlib import Path

fake = Faker()

class SigmaFinancialDataGenerator:
    """
    Sigma layer implementation for generating realistic financial data
    with proper correlations, seasonality, and distress patterns.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_users = self.config['data']['n_users']
        self.n_months = self.config['data']['n_months']
        self.seed = 42
        np.random.seed(self.seed)
        
    def generate_user_profiles(self) -> pd.DataFrame:
        """
        Generate diverse user profiles with demographics.
        Returns DataFrame with user_id, age, income_bracket, region, etc.
        """
        profiles = []
        
        # Define user risk categories for better distribution
        risk_categories = ['very_safe', 'safe', 'moderate', 'risky', 'very_risky']
        risk_weights = [0.25, 0.30, 0.25, 0.15, 0.05]  # Better distribution
        
        for i in range(self.n_users):
            # Income brackets with realistic distribution
            income_bracket = np.random.choice(
                ['low', 'medium', 'high', 'very_high'],
                p=[0.35, 0.40, 0.20, 0.05]
            )
            
            # Base annual income by bracket (in thousands)
            base_income_map = {
                'low': (30, 55),
                'medium': (55, 90),
                'high': (90, 150),
                'very_high': (150, 300)
            }
            
            base_income = np.random.uniform(*base_income_map[income_bracket])
            
            # Regional and demographic factors
            region = fake.state()
            
            # Assign risk category
            risk_category = np.random.choice(risk_categories, p=risk_weights)
            
            # Generate attributes based on risk category
            if risk_category == 'very_safe':
                career_stability = np.random.beta(8, 2)  # Very stable
                spending_propensity = np.random.beta(2, 3)  # Conservative
                credit_skill = np.random.beta(8, 2)  # Excellent
                chronic_issues = 0  # No chronic issues
                
            elif risk_category == 'safe':
                career_stability = np.random.beta(5, 3)  # Stable
                spending_propensity = np.random.beta(3, 4)  # Moderate
                credit_skill = np.random.beta(6, 3)  # Good
                chronic_issues = 0
                
            elif risk_category == 'moderate':
                career_stability = np.random.beta(3, 3)  # Moderate
                spending_propensity = np.random.beta(4, 4)  # Variable
                credit_skill = np.random.beta(4, 4)  # Average
                chronic_issues = np.random.choice([0, 1], p=[0.7, 0.3])  # Some issues
                
            elif risk_category == 'risky':
                career_stability = np.random.beta(2, 5)  # Unstable
                spending_propensity = np.random.beta(6, 3)  # High spending
                credit_skill = np.random.beta(3, 6)  # Poor
                chronic_issues = np.random.choice([0, 1], p=[0.4, 0.6])  # Often has issues
                
            else:  # very_risky
                career_stability = np.random.beta(1, 8)  # Very unstable
                spending_propensity = np.random.beta(8, 2)  # Very high spending
                credit_skill = np.random.beta(2, 8)  # Very poor
                chronic_issues = 1  # Always has chronic issues
            
            profiles.append({
                'user_id': i + 1,
                'income_bracket': income_bracket,
                'base_annual_income': base_income,
                'region': region,
                'career_stability': career_stability,
                'spending_propensity': spending_propensity,
                'credit_skill': credit_skill,
                'risk_category': risk_category,
                'chronic_issues': chronic_issues,
                'age': np.random.randint(25, 65)
            })
        
        return pd.DataFrame(profiles)
    
    def generate_temporal_data(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 12 months of financial data per user with:
        - Income trends and seasonality
        - Spending patterns tied to income
        - Occasional distress events (gradual, sudden, chronic, recovery)
        - Realistic correlations and cascading problems
        """
        all_data = []
        
        for _, user in profiles.iterrows():
            user_id = user['user_id']
            base_income = user['base_annual_income']
            career_stability = user['career_stability']
            spending_prop = user['spending_propensity']
            credit_skill = user['credit_skill']
            risk_category = user['risk_category']
            chronic_issues = user['chronic_issues']
            
            # Monthly income with volatility
            monthly_income = base_income / 12
            
            # Track state for complex scenarios
            credit_balance = 0  # Track cumulative credit issues
            consecutive_missed = 0  # Track missed payments streak
            gradual_decline = False
            if risk_category in ['risky', 'very_risky']:
                # Create gradual decline patterns for risky users
                decline_start = np.random.randint(3, 9)  # Start decline mid-year
                gradual_decline = np.random.choice([True, False], p=[0.6, 0.4])
            
            # Generate monthly data
            for month in range(1, self.n_months + 1):
                # Income calculation with seasonality
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                income_noise = np.random.normal(0, 0.15 if risk_category in ['risky', 'very_risky'] else 0.1)
                income = monthly_income * (1 + income_noise) * seasonal_factor
                
                # Create diverse distress scenarios
                # 1. Gradual decline (for risky users)
                if gradual_decline and month >= decline_start:
                    decline_factor = 0.95 ** (month - decline_start + 1)  # Exponential decay
                    income = income * decline_factor
                
                # 2. Sudden income shocks (job loss, medical leave, etc.)
                shock_probability = (1 - career_stability) * 0.20
                if np.random.random() < shock_probability and not gradual_decline:
                    shock_severity = np.random.uniform(0.2, 0.9)
                    shock_type = np.random.choice(['job_loss', 'reduced_hours', 'medical'])
                    if shock_type == 'job_loss':
                        income_shock_severity = np.random.uniform(0.6, 0.9)
                    elif shock_type == 'reduced_hours':
                        income_shock_severity = np.random.uniform(0.3, 0.6)
                    else:  # medical
                        income_shock_severity = np.random.uniform(0.4, 0.8)
                    income = income * (1 - income_shock_severity)
                
                # Total spending with diverse patterns
                base_spend_ratio = 0.7 + spending_prop * 0.3
                
                # Chronic spenders burn through money even when income drops
                if chronic_issues and income < monthly_income * 0.8:
                    spending = income * base_spend_ratio * np.random.uniform(1.0, 1.5)
                else:
                    spending = income * base_spend_ratio * np.random.uniform(0.8, 1.2)
                
                # Essential vs luxury spending
                essential_ratio = 0.6 + (1 - spending_prop) * 0.25
                essential_spend = spending * essential_ratio * np.random.uniform(0.9, 1.1)
                luxury_spend = spending * (1 - essential_ratio) * np.random.uniform(0.7, 1.3)
                
                # Credit utilization with cascading problems
                target_utilization = 0.3 + (1 - credit_skill) * 0.5
                
                # If spending exceeds income, credit utilization goes up
                if spending > income:
                    deficit = spending - income
                    credit_balance += deficit
                    credit_utilization = min(1, target_utilization + (credit_balance / (monthly_income * 12)))
                else:
                    credit_balance = max(0, credit_balance - (income - spending) * 0.3)
                    credit_utilization = max(0, min(1, target_utilization * np.random.uniform(0.7, 1.3)))
                
                # Missed payments with cascading effects
                missed_payment_prob = (1 - credit_skill) * 0.12
                
                if consecutive_missed > 0:
                    missed_payment_prob *= 1.5
                if income < monthly_income * 0.7:
                    missed_payment_prob *= 2
                
                missed_payments = np.random.binomial(1, min(0.8, missed_payment_prob))
                
                if missed_payments:
                    consecutive_missed += 1
                    credit_balance += (income * 0.05)
                else:
                    consecutive_missed = max(0, consecutive_missed - 1)
                
                # Determine distress label with more nuanced scoring
                distress_score = 0
                
                # Income distress (more severe for larger drops)
                if income < monthly_income * 0.5:
                    distress_score += 0.6  # Severe income drop
                elif income < monthly_income * 0.7:
                    distress_score += 0.4  # Moderate drop
                elif income < monthly_income * 0.85:
                    distress_score += 0.2  # Mild drop
                
                # Credit distress
                if credit_utilization > 0.95:
                    distress_score += 0.4  # Maxed out
                elif credit_utilization > 0.8:
                    distress_score += 0.3  # High utilization
                elif credit_utilization > 0.6:
                    distress_score += 0.1  # Elevated
                
                # Payment history distress
                if consecutive_missed >= 3:
                    distress_score += 0.5  # Chronic missed payments
                elif consecutive_missed >= 2:
                    distress_score += 0.3
                elif missed_payments:
                    distress_score += 0.2
                
                # Spending distress
                if spending > income * 1.3:
                    distress_score += 0.4  # Severe overspending
                elif spending > income * 1.1:
                    distress_score += 0.2
                elif chronic_issues and spending > income * 0.95:
                    distress_score += 0.1  # Chronic spender barely covering
                
                # Cascading problems (multiple issues compound)
                issue_count = sum([
                    income < monthly_income * 0.8,
                    credit_utilization > 0.7,
                    missed_payments,
                    spending > income * 1.05
                ])
                if issue_count >= 3:
                    distress_score += 0.3  # Multiple problems compound
                elif issue_count >= 2:
                    distress_score += 0.1
                
                # Cap at 1.0 and set distress
                distress_score = min(1.0, distress_score)
                distress_label = 1 if distress_score > 0.45 else 0
                
                all_data.append({
                    'user_id': user_id,
                    'month': month,
                    'income': round(income, 2),
                    'total_spend': round(spending, 2),
                    'essential_spend': round(essential_spend, 2),
                    'luxury_spend': round(luxury_spend, 2),
                    'credit_utilization': round(credit_utilization, 3),
                    'missed_payments': missed_payments,
                    'distress_label': distress_label,
                    'month_label': f"{month:02d}/2023"
                })
        
        return pd.DataFrame(all_data)
    
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with user profiles and temporal data.
        Returns (temporal_data, user_metadata)
        """
        print(f"Generating profiles for {self.n_users} users...")
        profiles = self.generate_user_profiles()
        
        print(f"Generating {self.n_months} months of financial data...")
        temporal_data = self.generate_temporal_data(profiles)
        
        # Save outputs
        output_path = Path(self.config['data']['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_path = Path(self.config['data']['output_metadata_path'])
        
        print(f"Saving data to {output_path}...")
        temporal_data.to_csv(output_path, index=False)
        profiles.to_csv(metadata_path, index=False)
        
        print(f"✅ Generated {len(temporal_data)} rows across {self.n_users} users")
        print(f"   Distress rate: {temporal_data['distress_label'].mean():.2%}")
        
        return temporal_data, profiles


def main():
    """Generate synthetic financial data."""
    generator = SigmaFinancialDataGenerator()
    data, metadata = generator.generate()
    return data, metadata


if __name__ == "__main__":
    data, metadata = main()

