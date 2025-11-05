"""
OpenAI Insight Generation Layer
Converts SHAP explanations into human-readable financial advice.
"""

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIInsightGenerator:
    """
    Generate natural-language financial insights using OpenAI API.
    Converts SHAP feature explanations into empathetic, actionable advice.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize with configuration and OpenAI client."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = self.config['openai']['model']
        self.temperature = self.config['openai']['temperature']
        self.max_tokens = self.config['openai']['max_tokens']
        
    def load_user_explanations(self, user_id: int, month: int) -> Optional[Dict]:
        """
        Load SHAP explanation for a specific user and month.
        
        Args:
            user_id: User ID
            month: Month number
            
        Returns:
            User explanation dict or None if not found
        """
        explanations_path = "data/user_explanations.json"
        
        if not Path(explanations_path).exists():
            return None
        
        with open(explanations_path, 'r') as f:
            explanations = json.load(f)
        
        key = f"{user_id}_m{month}"
        return explanations.get(key)
    
    def format_shap_summary(self, explanation: Dict) -> str:
        """
        Format SHAP explanation into a summary string for OpenAI.
        
        Args:
            explanation: SHAP explanation dictionary
            
        Returns:
            Formatted summary string
        """
        prediction = explanation.get('prediction', 0)
        risk_level = "high" if prediction > 0.7 else "moderate" if prediction > 0.4 else "low"
        
        summary = f"Risk Probability: {prediction:.2%} ({risk_level} risk)\n\n"
        summary += "Key Factors:\n"
        
        for idx, feature_info in enumerate(explanation.get('top_features', [])[:3], 1):
            feature = feature_info['feature']
            direction = feature_info['direction']
            impact = "increases" if direction == 'increases_risk' else "decreases"
            
            # Human-readable feature names
            readable_names = {
                'income_drop_30pct': 'significant income drop',
                'income_drop_50pct': 'major income drop',
                'spend_income_ratio': 'spending exceeding income',
                'credit_util_ma_3': 'high credit utilization',
                'missed_streak': 'missed payment pattern',
                'essential_luxury_ratio': 'shift to essential spending',
                'income_deviation': 'income instability',
                'spend_deviation': 'spending deviation',
            }
            
            feature_readable = readable_names.get(feature, feature.replace('_', ' '))
            summary += f"{idx}. {feature_readable.title()} - {impact} financial risk\n"
        
        return summary
    
    def generate_insight(self,
                        user_id: int,
                        month: int,
                        user_data: Optional[Dict] = None) -> str:
        """
        Generate personalized financial insight using OpenAI.
        
        Args:
            user_id: User ID
            month: Month number
            user_data: Optional additional user context
            
        Returns:
            Generated financial advice text
        """
        # Load SHAP explanation
        explanation = self.load_user_explanations(user_id, month)
        
        if not explanation:
            return "No explanation available for this user."
        
        # Format summary
        summary = self.format_shap_summary(explanation)
        
        # Prepare prompt for OpenAI
        prompt = f"""You are a compassionate financial advisor for a fintech platform.

A user is showing early signs of financial distress. Here's what the AI detected:
{summary}

Please provide:
1. A brief explanation (1-2 sentences) of why this might be happening
2. Practical, actionable advice to help them prevent issues
3. Keep the tone empathetic and supportive
4. Maximum 100 words total

Format your response as a clear, concise paragraph."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial advisor helping users avoid financial distress. Be empathetic, practical, and supportive."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            insight = response.choices[0].message.content
            return insight
            
        except Exception as e:
            return f"Error generating insight: {str(e)}"
    
    def batch_generate_insights(self,
                               user_months: List[tuple],
                               save_path: str = "data/openai_insights.json"):
        """
        Generate insights for multiple users and save to file.
        
        Args:
            user_months: List of (user_id, month) tuples
            save_path: Path to save insights
        """
        insights = {}
        
        for user_id, month in user_months:
            print(f"Generating insight for user {user_id}, month {month}...")
            insight = self.generate_insight(user_id, month)
            key = f"{user_id}_m{month}"
            insights[key] = insight
        
        # Save to file
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"✅ Saved {len(insights)} insights to {save_path}")
        return insights


def main():
    """Test OpenAI insight generation."""
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set. Please set it in your .env file.")
        print("   Skipping OpenAI insight generation.")
        return
    
    generator = OpenAIInsightGenerator()
    
    # Test with a sample user
    insight = generator.generate_insight(user_id=1, month=1)
    print("\n" + "="*50)
    print("Sample Generated Insight")
    print("="*50)
    print(insight)


if __name__ == "__main__":
    main()

