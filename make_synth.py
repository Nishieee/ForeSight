#!/usr/bin/env python3
"""
make_synth.py — realistic drifty synthetic fintech dataset

Creates:
  data/customers.csv
  data/transactions.csv
  data/outcomes.csv
  data/daily_labels.csv

What it models (enhanced):
- Customers with realistic income volatility, life events, and behavioral drift
- Complex spending patterns with merchant loyalty, seasonal effects, and gradual changes
- Subtle financial distress progression with early warning signals
- Income drift: job changes, promotions, seasonal work, unemployment periods
- Realistic bill payment patterns with occasional delays and missed payments
- Merchant switching patterns and loyalty decay over time
- Labels: 1 during the LABEL_WINDOW days before event_date; else 0.
"""

import os
import random
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
from faker import Faker

# ---------------- Defaults (tweak as needed) ----------------
SEED = 7
N_CUSTOMERS = 5000       # Increased from 2500 to 5000 customers
DAYS = 365               # Full year of data instead of 6 months
DISTRESS_SHARE = 0.08    # ~8% at-risk (more realistic for real-world)
LEAD_DAYS = 45           # longer drift-to-event lead time for subtlety
LABEL_WINDOW = 14        # label=1 in the 14 days pre-event (wider window)
OUTDIR = "data"

# Enhanced distress progression parameters (more subtle and realistic)
EARLY_DISTRESS_PROB = 0.08         # much more subtle early signs
DISTRESS_CASH_ADV_PROB = 0.25      # reduced for more realistic signal
DISTRESS_PAYDAY_PROB_BASE = 0.05   # much lower base probability
DISTRESS_PAYDAY_PROB_PEAK = 0.20   # lower peak probability
DISTRESS_UTIL_DELAY_PROB = 0.15    # more realistic delayed payments
DISTRESS_BILL_MISS_PROB = 0.08     # more realistic missed payments

# Income volatility and drift parameters
INCOME_VOLATILITY = 0.15           # monthly income can vary by ±15%
JOB_CHANGE_PROB = 0.08             # 8% chance of job change over period
PROMOTION_PROB = 0.03              # 3% chance of promotion
PAY_CUT_PROB = 0.04                # 4% chance of pay reduction
UNEMPLOYMENT_PROB = 0.02           # 2% chance of temporary unemployment

# Behavioral drift parameters
MERCHANT_LOYALTY_DECAY = 0.98      # merchant preference decay per transaction
SPENDING_HABIT_DRIFT = 0.002       # gradual drift in spending categories
SEASONAL_STRENGTH = 0.25           # stronger seasonal effects

# Income (monthly take-home USD)
INCOME_MEAN = 4200
INCOME_STD = 1200
INCOME_MIN, INCOME_MAX = 1200, 15000

# Rent as fraction of income (keeps things realistic)
RENT_MEAN, RENT_STD = 0.32, 0.07
RENT_MIN, RENT_MAX = 0.15, 0.55

# Utilities (monthly), split weekly-ish
UTIL_MEAN, UTIL_STD = 220, 80
UTIL_MIN, UTIL_MAX = 80, 600

# Credit limit correlated with income + tenure
CL_BASE = 1.2            # multiplier on income
CL_TENURE_BONUS = 50     # per month of tenure
CL_MIN, CL_MAX = 1000, 30000

# Category baseline monthly frequencies (approximate) - more realistic
CAT_FREQ_PM = {
    "groceries": 16,
    "transport": 20,
    "dining": 10,
    "entertainment": 6,
    "ecommerce": 8,
    "cash_advance": 0.8,   # even rarer in normal circumstances
    "payday_loan": 0.2,    # very rare in normal circumstances
}

# Typical per-transaction amount ranges (USD)
CAT_AMOUNTS = {
    "groceries": (-140, -25),
    "transport": (-60, -8),
    "dining": (-95, -12),
    "entertainment": (-90, -8),
    "ecommerce": (-180, -15),
    "cash_advance": (-300, -60),
    "payday_loan": (-600, -150),
}

MERCHANTS = {
    "groceries": ["FreshMart", "GroceryHub", "DailyFoods", "MarketPlace", "OrganicFresh", "ValueGrocer", "CornerstoneMarket", "FreshChoice"],
    "transport": ["CityTransit", "RideHail", "FuelStop", "MetroCard", "BikeShare", "ParkingPlus", "TaxiCorp", "BusPass"],
    "dining": ["CafeCorner", "BistroBox", "FoodTruck", "DeliTime", "QuickBite", "FastEats", "FamilyDiner", "TakeoutKing", "DriveThru"],
    "entertainment": ["StreamPlus", "CinemaCity", "GameHub", "SportsTix", "ConcertVenue", "BookStore", "ArcadeZone", "LiveMusic"],
    "ecommerce": ["ShopOnline", "MegaMart", "QuickBuy", "SuperDeals", "FashionHub", "TechStore", "HomeGoods", "BargainBin", "PremiumShop"],
    "cash_advance": ["ATMWithdrawal", "CashPoint", "QuickCash", "InstantATM", "FastCash"],
    "payday_loan": ["QuickCash", "InstantLoan", "PayDayPlus", "CashNow", "EasyMoney"],
    "rent": ["RentCo", "AptMgmt", "PropertyPlus", "LandlordPay", "RentPortal"],
    "utilities": ["PowerGrid", "WaterWorks", "TeleComms", "Heat&Gas", "Electric Co", "Internet Plus", "CableTV", "MobileCarrier"],
    "salary_income": ["EmployerPay", "PayrollDirect", "CompanyPay", "WorkInc"],
}


def pick_merchant(cat: str, customer_preferences: dict = None, day: int = 0) -> str:
    """Enhanced merchant selection with loyalty patterns and drift."""
    base = MERCHANTS.get(cat, ["Misc"])
    
    # Initialize customer preferences if not provided
    if customer_preferences is None:
        customer_preferences = {}
    
    # Get or initialize preferences for this category
    if cat not in customer_preferences:
        # Initial preference distribution - favor first few merchants
        prefs = {merchant: len(base) - i + random.uniform(-1, 1) 
                for i, merchant in enumerate(base)}
        customer_preferences[cat] = prefs
    
    prefs = customer_preferences[cat]
    
    # Apply loyalty decay over time (preferences shift slowly)
    if day > 0 and random.random() < SPENDING_HABIT_DRIFT:
        # Randomly adjust one preference
        merchant = random.choice(base)
        prefs[merchant] *= random.uniform(0.9, 1.1)
    
    # Seasonal/trendy merchant boosts
    if random.random() < 0.05:  # 5% chance of trying trendy new place
        new_merchant = f"{cat.capitalize()}_{random.randint(1000,9999)}"
        if random.random() < 0.3:  # 30% chance it becomes a new favorite
            prefs[new_merchant] = max(prefs.values()) * random.uniform(0.8, 1.2)
        return new_merchant
    
    # Weight selection by preferences with some randomness
    weights = [max(0.1, prefs.get(merchant, 1.0)) for merchant in base]
    
    # Add randomness - sometimes ignore preferences
    if random.random() < 0.15:  # 15% completely random choice
        return random.choice(base)
    
    # Weighted selection based on customer loyalty
    return random.choices(base, weights=weights)[0]


def make_customers(n: int, start: date, fake: Faker) -> pd.DataFrame:
    rows = []
    for i in range(n):
        cid = f"C{str(i).zfill(6)}"
        
        # More diverse pay cycles
        pay_cycle = np.random.choice(["monthly", "biweekly", "weekly"], p=[0.55, 0.35, 0.10])
        
        # Income with job type influence
        job_type = np.random.choice(["salary", "hourly", "gig", "seasonal"], p=[0.6, 0.25, 0.10, 0.05])
        
        if job_type == "salary":
            income = float(np.clip(np.random.normal(INCOME_MEAN * 1.1, INCOME_STD), INCOME_MIN, INCOME_MAX))
            income_stability = 0.95  # very stable
        elif job_type == "hourly":
            income = float(np.clip(np.random.normal(INCOME_MEAN * 0.9, INCOME_STD), INCOME_MIN, INCOME_MAX))
            income_stability = 0.85  # somewhat stable
        elif job_type == "gig":
            income = float(np.clip(np.random.normal(INCOME_MEAN * 0.7, INCOME_STD * 1.5), INCOME_MIN, INCOME_MAX))
            income_stability = 0.65  # quite variable
        else:  # seasonal
            income = float(np.clip(np.random.normal(INCOME_MEAN * 0.8, INCOME_STD * 2), INCOME_MIN, INCOME_MAX))
            income_stability = 0.45  # very variable

        tenure = int(np.clip(np.random.normal(24, 18), 1, 120))
        age = int(np.clip(np.random.normal(36, 12), 19, 75))

        # Age-based spending personality
        if age < 30:
            spending_personality = np.random.choice(["impulsive", "social", "tech_savvy"], p=[0.4, 0.4, 0.2])
        elif age < 50:
            spending_personality = np.random.choice(["balanced", "family_focused", "career_focused"], p=[0.5, 0.3, 0.2])
        else:
            spending_personality = np.random.choice(["conservative", "comfort_focused", "health_focused"], p=[0.5, 0.3, 0.2])

        # Rent varies by age and income
        if age < 25:
            rent_ratio = float(np.clip(np.random.normal(RENT_MEAN + 0.05, RENT_STD), RENT_MIN, RENT_MAX))  # often pay more
        else:
            rent_ratio = float(np.clip(np.random.normal(RENT_MEAN, RENT_STD), RENT_MIN, RENT_MAX))
        rent_amount = -round(rent_ratio * income, 2)

        # Utilities vary by age and lifestyle
        util_base = UTIL_MEAN if age < 40 else UTIL_MEAN * 1.2  # older people tend to have higher utilities
        util_month = float(np.clip(np.random.normal(util_base, UTIL_STD), UTIL_MIN, UTIL_MAX))
        util_weekly = -round(util_month / 4.3, 2)

        credit_limit = float(np.clip(CL_BASE * income + CL_TENURE_BONUS * tenure, CL_MIN, CL_MAX))

        rows.append(dict(
            customer_id=cid,
            name=fake.name(),
            age=age,
            tenure_months=tenure,
            pay_cycle=pay_cycle,
            job_type=job_type,
            income_stability=round(income_stability, 3),
            spending_personality=spending_personality,
            base_income=round(income, 2),
            credit_limit=round(credit_limit, 2),
            city=fake.city(),
            signup_date=(start - timedelta(days=np.random.randint(60, 720))),
            rent_amount=rent_amount,
            util_weekly=util_weekly,
        ))
    return pd.DataFrame(rows)


def gen_transactions(customers: pd.DataFrame, days: int, start: date,
                     flags: np.ndarray, start_offsets: np.ndarray, event_offsets: np.ndarray) -> pd.DataFrame:
    all_tx = []
    
    for i, row in customers.reset_index(drop=True).iterrows():
        cid = row.customer_id
        pay = row.pay_cycle
        base_income = row.base_income
        rent_amt = row.rent_amount
        util_week = row.util_weekly
        job_type = row.job_type
        income_stability = row.income_stability
        personality = row.spending_personality
        age = row.age

        # Initialize customer state
        current_income = base_income
        customer_preferences = {}
        
        # Personality-based spending multipliers
        if personality in ["impulsive", "social"]:
            mult = {k: np.random.uniform(1.1, 1.4) for k in CAT_FREQ_PM.keys()}
            mult["dining"] *= 1.3
            mult["entertainment"] *= 1.2
        elif personality in ["conservative", "health_focused"]:
            mult = {k: np.random.uniform(0.6, 0.9) for k in CAT_FREQ_PM.keys()}
            mult["groceries"] *= 1.1
        elif personality == "tech_savvy":
            mult = {k: np.random.uniform(0.9, 1.1) for k in CAT_FREQ_PM.keys()}
            mult["ecommerce"] *= 1.5
        else:  # balanced, family_focused, career_focused, comfort_focused
            mult = {k: np.random.uniform(0.8, 1.2) for k in CAT_FREQ_PM.keys()}

        ds_flag = bool(flags[i])
        ds_start = int(start_offsets[i])
        ds_event = int(event_offsets[i])
        
        # Track financial state over time
        monthly_spending = []
        missed_bills = []

        for d in range(days):
            dt = start + timedelta(days=d)
            month = dt.month

            # Income volatility and life events
            if d > 0 and d % 30 == 0:  # monthly income adjustments
                # Job changes and income volatility
                if random.random() < JOB_CHANGE_PROB / 6:  # per month probability
                    if random.random() < 0.3:  # promotion
                        current_income *= random.uniform(1.1, 1.4)
                    elif random.random() < 0.5:  # pay cut or demotion
                        current_income *= random.uniform(0.7, 0.9)
                    else:  # job change (lateral)
                        current_income *= random.uniform(0.9, 1.1)
                
                # Regular income volatility based on job type
                if job_type == "gig":
                    current_income = base_income * random.uniform(0.4, 1.6)
                elif job_type == "seasonal":
                    # Seasonal pattern
                    seasonal_factor = 1.0 + 0.4 * math.sin((month - 1) * math.pi / 6)
                    current_income = base_income * seasonal_factor * random.uniform(0.8, 1.2)
                elif job_type == "hourly":
                    current_income = base_income * random.uniform(0.85, 1.15)
                else:  # salary - more stable
                    current_income = base_income * random.uniform(0.95, 1.05)

            # Income deposits with realistic timing and amounts
            income_deposited = False
            if pay == "monthly" and dt.day == 1:
                variance = current_income * (1 - income_stability) * 0.5
                amt = abs(np.random.normal(current_income, variance))
                all_tx.append([cid, dt, "salary_income", pick_merchant("salary_income", customer_preferences, d), round(amt, 2), True, False])
                income_deposited = True
            elif pay == "biweekly" and (dt.toordinal() + i) % 14 == 0:
                variance = current_income * (1 - income_stability) * 0.25
                amt = abs(current_income / 2 + np.random.normal(0, variance))
                all_tx.append([cid, dt, "salary_income", pick_merchant("salary_income", customer_preferences, d), round(amt, 2), True, False])
                income_deposited = True
            elif pay == "weekly" and dt.weekday() == 4:  # Friday
                variance = current_income * (1 - income_stability) * 0.15
                amt = abs(current_income / 4.3 + np.random.normal(0, variance))
                all_tx.append([cid, dt, "salary_income", pick_merchant("salary_income", customer_preferences, d), round(amt, 2), True, False])
                income_deposited = True

            # Bill payments with realistic delays and missed payments
            rent_due = dt.day == (1 if i % 2 == 0 else 3)
            util_due = d % 7 == (i % 7)
            
            # Financial stress affects bill payment reliability
            stress_factor = 0.0
            if ds_flag and d >= ds_start:
                days_to_event = ds_event - d if ds_event >= 0 else float('inf')
                stress_factor = max(0.0, 1.0 - (days_to_event / LEAD_DAYS)) if days_to_event < LEAD_DAYS else 0.3

            # Rent payment
            if rent_due:
                if random.random() < 0.05 + stress_factor * 0.2:  # late payment
                    delay = np.random.randint(1, 8)
                    all_tx.append([cid, dt + timedelta(days=delay), "rent", pick_merchant("rent", customer_preferences, d), rent_amt, False, True])
                elif random.random() < DISTRESS_BILL_MISS_PROB * stress_factor:  # missed payment
                    missed_bills.append(("rent", dt))
                else:  # on time
                    all_tx.append([cid, dt, "rent", pick_merchant("rent", customer_preferences, d), rent_amt, False, True])

            # Utility payment
            if util_due:
                if random.random() < 0.08 + stress_factor * 0.15:  # late payment
                    delay = np.random.randint(1, 5)
                    all_tx.append([cid, dt + timedelta(days=delay), "utilities", pick_merchant("utilities", customer_preferences, d), util_week, False, True])
                elif random.random() < DISTRESS_BILL_MISS_PROB * stress_factor:  # missed payment
                    missed_bills.append(("utilities", dt))
                else:  # on time
                    all_tx.append([cid, dt, "utilities", pick_merchant("utilities", customer_preferences, d), util_week, False, True])

            # Seasonal and life event effects
            weekend = (dt.weekday() >= 5)
            # Enhanced seasonality with holidays and events
            base_season = 1.0 + SEASONAL_STRENGTH * math.sin(d / 28.0)  # monthly cycle
            holiday_boost = 1.0
            
            # Holiday spending spikes
            if month == 12 or month == 11:  # Holiday season
                holiday_boost = 1.4
            elif month in [6, 7, 8]:  # Summer
                holiday_boost = 1.15
            elif dt.weekday() == 4:  # Friday boost
                holiday_boost = 1.1

            season = base_season * holiday_boost

            # Regular spending with drift and personality effects
            for cat, freq_pm in CAT_FREQ_PM.items():
                # Gradual drift in spending habits
                if random.random() < SPENDING_HABIT_DRIFT:
                    mult[cat] *= random.uniform(0.98, 1.02)
                
                # Calculate lambda with all modifiers
                base_lambda = (freq_pm / 30.0) * mult[cat] * season
                
                # Weekend effects vary by category and personality
                if weekend:
                    if cat in ["dining", "entertainment"]:
                        if personality == "social":
                            base_lambda *= 1.8
                        else:
                            base_lambda *= 1.3
                    elif cat == "ecommerce":
                        base_lambda *= 1.2

                # Age effects
                if age < 25 and cat in ["dining", "entertainment"]:
                    base_lambda *= 1.3
                elif age > 55 and cat == "groceries":
                    base_lambda *= 1.2

                # Generate transactions
                n = np.random.poisson(base_lambda)
                lo, hi = CAT_AMOUNTS[cat]
                
                for _ in range(n):
                    amt = np.random.uniform(lo, hi)
                    
                    # Occasional refunds/returns
                    if random.random() < 0.015:
                        amt = -amt * random.uniform(0.7, 0.95)
                    
                    # Apply personality-based amount modifiers
                    if personality == "impulsive" and random.random() < 0.3:
                        amt *= random.uniform(1.2, 2.0)  # impulse purchases
                    elif personality == "conservative":
                        amt *= random.uniform(0.8, 0.95)  # frugal spending
                    
                    all_tx.append([cid, dt, cat, pick_merchant(cat, customer_preferences, d), round(amt, 2), False, (cat == "payday_loan")])

            # Enhanced distress behavior - more realistic with noise and false signals
            if ds_flag:
                days_to_event = ds_event - d if ds_event >= 0 else float('inf')
                
                # Early distress signs (much more subtle and noisy)
                if d >= ds_start - 21:  # 3 weeks before official distress start
                    early_intensity = max(0.05, 0.15 - (ds_start - d) / 21.0) if d < ds_start else 0.15
                    
                    # Very subtle cash advance increase (with noise)
                    if random.random() < EARLY_DISTRESS_PROB * early_intensity * random.uniform(0.5, 1.5):
                        amt = np.random.uniform(-150, -30)
                        all_tx.append([cid, dt, "cash_advance", pick_merchant("cash_advance", customer_preferences, d), round(amt, 2), False, False])
                    
                    # Slightly reduced discretionary spending (not always)
                    if random.random() < 0.1 * early_intensity:
                        # Skip some entertainment/dining occasionally
                        continue

                # Main distress period (more gradual and noisy)
                if d >= ds_start:
                    distress_intensity = max(0.2, 0.8 - (days_to_event / LEAD_DAYS)) if days_to_event < LEAD_DAYS else 0.2
                    
                    # Cash advances with more noise and randomness
                    if random.random() < DISTRESS_CASH_ADV_PROB * distress_intensity * random.uniform(0.3, 1.7):
                        base_min, base_max = -250, -40
                        amt_min = base_min * (1 + 0.4 * distress_intensity)
                        amt_max = base_max * (1 + 0.3 * distress_intensity)
                        amt = np.random.uniform(amt_min, amt_max)
                        all_tx.append([cid, dt, "cash_advance", pick_merchant("cash_advance", customer_preferences, d), round(amt, 2), False, False])
                    
                    # Payday loans near the end (much rarer and more random)
                    if days_to_event <= 25 and ds_event >= 0:
                        if days_to_event <= 5:
                            payday_prob = DISTRESS_PAYDAY_PROB_PEAK * random.uniform(0.5, 1.0)
                        elif days_to_event <= 10:
                            payday_prob = DISTRESS_PAYDAY_PROB_BASE * 1.5 * random.uniform(0.3, 1.0)
                        elif days_to_event <= 20:
                            payday_prob = DISTRESS_PAYDAY_PROB_BASE * random.uniform(0.5, 1.0)
                        else:
                            payday_prob = DISTRESS_PAYDAY_PROB_BASE * 0.5
                        
                        if random.random() < payday_prob:
                            base_min, base_max = -600, -150
                            amt_min = base_min * (1 + 0.3 * distress_intensity)
                            amt_max = base_max * (1 + 0.2 * distress_intensity)
                            amt = np.random.uniform(amt_min, amt_max)
                            all_tx.append([cid, dt, "payday_loan", pick_merchant("payday_loan", customer_preferences, d), round(amt, 2), False, True])
            
            # Add false positive signals (non-distressed customers showing distress-like behavior)
            if not ds_flag and random.random() < 0.02:  # 2% of non-distressed customers show distress signals
                # Random cash advances (not correlated with distress)
                if random.random() < 0.1:
                    amt = np.random.uniform(-200, -30)
                    all_tx.append([cid, dt, "cash_advance", pick_merchant("cash_advance", customer_preferences, d), round(amt, 2), False, False])
                
                # Random payday loans (not correlated with distress)
                if random.random() < 0.02:
                    amt = np.random.uniform(-500, -100)
                    all_tx.append([cid, dt, "payday_loan", pick_merchant("payday_loan", customer_preferences, d), round(amt, 2), False, True])

        # Add any makeup payments for missed bills
        for bill_type, missed_date in missed_bills:
            if random.random() < 0.6:  # 60% chance to eventually pay
                delay = np.random.randint(7, 30)
                amt = rent_amt if bill_type == "rent" else util_week
                # Late fees
                amt *= random.uniform(1.05, 1.25)
                all_tx.append([cid, missed_date + timedelta(days=delay), bill_type, pick_merchant(bill_type, customer_preferences, days), round(amt, 2), False, True])

    tx = pd.DataFrame(all_tx, columns=["customer_id", "date", "category", "merchant", "amount", "is_income", "is_bill"])
    tx.sort_values(["customer_id", "date"], inplace=True, kind="mergesort")
    tx.reset_index(drop=True, inplace=True)
    return tx


def build_daily_labels(outcomes: pd.DataFrame, days: int, start: date, window: int) -> pd.DataFrame:
    rec = []
    for cid, ev in outcomes[["customer_id", "event_date"]].itertuples(index=False):
        for d in range(days):
            dt = start + timedelta(days=d)
            y = 1 if (pd.notna(ev) and (ev - timedelta(days=window) <= dt <= ev)) else 0
            rec.append((cid, dt, y))
    return pd.DataFrame(rec, columns=["customer_id", "date", "label"])


def main():
    # seeds
    np.random.seed(SEED); random.seed(SEED)
    fake = Faker("en_US")

    start = date.today() - timedelta(days=DAYS)

    # customers
    customers = make_customers(N_CUSTOMERS, start, fake)

    # distress flags and timing (with more realistic variability)
    flags = (np.random.rand(len(customers)) < DISTRESS_SHARE).astype(int)
    start_offsets = np.where(flags == 1, np.random.randint(DAYS // 4, DAYS - 60, len(customers)), -1)
    
    # Add variability to event timing (not all events happen exactly LEAD_DAYS after start)
    event_offsets = np.where(flags == 1, 
                            start_offsets + LEAD_DAYS + np.random.randint(-10, 15, len(customers)), -1)
    
    # Ensure events don't go beyond the data period
    event_offsets = np.where(event_offsets > DAYS - 7, DAYS - 7, event_offsets)

    # transactions
    transactions = gen_transactions(customers, DAYS, start, flags, start_offsets, event_offsets)

    # outcomes & labels
    outcomes = pd.DataFrame({
        "customer_id": customers["customer_id"],
        "is_distressed": flags,
        "distress_start_date": [(start + timedelta(days=int(o))) if o >= 0 else pd.NaT for o in start_offsets],
        "event_date": [(start + timedelta(days=int(o))) if o >= 0 else pd.NaT for o in event_offsets],
    })
    daily_labels = build_daily_labels(outcomes, DAYS, start, LABEL_WINDOW)

    # save
    os.makedirs(OUTDIR, exist_ok=True)
    customers.to_csv(f"{OUTDIR}/customers.csv", index=False)
    transactions.to_csv(f"{OUTDIR}/transactions.csv", index=False)
    outcomes.to_csv(f"{OUTDIR}/outcomes.csv", index=False)
    daily_labels.to_csv(f"{OUTDIR}/daily_labels.csv", index=False)

    # quick print
    print({
        "customers": len(customers),
        "transactions": len(transactions),
        "distress_rate": float(outcomes["is_distressed"].mean()),
        "avg_income": float(customers["base_income"].mean()),
        "avg_rent_ratio": float((-customers["rent_amount"] / customers["base_income"]).mean()),
        "avg_util_month": float((-customers["util_weekly"] * 4.3).mean()),
    })


if __name__ == "__main__":
    main()
