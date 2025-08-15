# ForeSight
An AI-powered early warning system for fintechs that detects subtle, personalized signs of financial distress before they turn into missed payments or defaults. Instead of reacting after the damage is done, our system gives lenders and BNPL providers the lead time to act, protect revenue, and support customers.

### Data Generation & Infrastructure
- ✅ Created synthetic data generator (`make_synth.py`) using Faker
- ✅ Generates realistic financial transaction data with distress patterns
- ✅ Set up automated daily data generation via cron job (runs at 2 AM)
- ✅ Created project structure with organized directories:
  - `data/` - Generated CSV files (customers, transactions, outcomes, daily_labels)
  - `src/foresight/` - Main application code with modules for models, API, preprocessing, and utilities

### Generated Data Files
- `customers.csv` - 2,500 synthetic customer profiles with income, credit limits, demographics
- `transactions.csv` - 1M+ realistic transactions with spending patterns and distress signals
- `outcomes.csv` - Ground truth distress events and timing for model training
- `daily_labels.csv` - 450K+ daily binary labels for supervised learning

The synthetic dataset models realistic retail banking patterns with ~18% distress rate, including subtle early warning signals like increased cash advances and payday loans before financial events.
