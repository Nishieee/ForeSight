#!/usr/bin/env python3
"""
make_synth.py — realistic-but-pragmatic synthetic fintech dataset

Creates:
  data/customers.csv
  data/transactions.csv
  data/outcomes.csv
  data/daily_labels.csv

What it models (brief):
- Customers with calibrated income, rent (as share of income), utilities, credit limits ~ income + tenure.
- Transactions over D days with weekend + mild monthly seasonality.
- Distress drift for a subset: more cash advances, payday loans near event, occasional delayed utilities.
- Labels: 1 during the LABEL_WINDOW days before event_date; else 0.
"""

import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ---------------- Defaults (tweak as needed) ----------------
SEED = 7
N_CUSTOMERS = 2500
DAYS = 180
DISTRESS_SHARE = 0.18    # ~18% at-risk
LEAD_DAYS = 21           # drift-to-event lead time
LABEL_WINDOW = 7         # label=1 in the 7 days pre-event
OUTDIR = "data"

# Distress drift intensity parameters
DISTRESS_CASH_ADV_PROB = 0.75      # increased from 0.5
DISTRESS_PAYDAY_PROB_BASE = 0.35   # increased from 0.22
DISTRESS_PAYDAY_PROB_PEAK = 0.65   # higher probability very close to event
DISTRESS_UTIL_DELAY_PROB = 0.4     # increased from 0.25

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

# Category baseline monthly frequencies (approximate)
CAT_FREQ_PM = {
    "groceries": 16,
    "transport": 20,
    "dining": 10,
    "entertainment": 6,
    "ecommerce": 8,
    "cash_advance": 1.2,   # rare
    "payday_loan": 0.4,    # very rare
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
    "groceries": ["FreshMart", "GroceryHub", "DailyFoods", "MarketPlace"],
    "transport": ["CityTransit", "RideHail", "FuelStop"],
    "dining": ["CafeCorner", "BistroBox", "FoodTruck", "DeliTime"],
    "entertainment": ["StreamPlus", "CinemaCity", "GameHub"],
    "ecommerce": ["ShopOnline", "MegaMart", "QuickBuy", "SuperDeals"],
    "cash_advance": ["ATMWithdrawal", "CashPoint"],
    "payday_loan": ["QuickCash", "InstantLoan"],
    "rent": ["RentCo", "AptMgmt"],
    "utilities": ["PowerGrid", "WaterWorks", "TeleComms", "Heat&Gas"],
    "salary_income": ["EmployerPay"],
}


def pick_merchant(cat: str) -> str:
    """Zipf-ish selection with occasional 'novel' merchant label."""
    base = MERCHANTS.get(cat, ["Misc"])
    r = random.random()
    if r < 0.83:
        # skew toward earlier items
        weights = [len(base) - i for i in range(len(base))]
        return random.choices(base, weights=weights)[0]
    elif r < 0.97:
        return random.choice(base)
    return f"{cat.capitalize()}_{random.randint(1000,9999)}"


def make_customers(n: int, start: date, fake: Faker) -> pd.DataFrame:
    rows = []
    for i in range(n):
        cid = f"C{str(i).zfill(6)}"
        pay_cycle = np.random.choice(["monthly", "biweekly"], p=[0.68, 0.32])

        income = float(np.clip(np.random.normal(INCOME_MEAN, INCOME_STD), INCOME_MIN, INCOME_MAX))
        tenure = int(np.clip(np.random.normal(24, 18), 1, 120))

        rent_ratio = float(np.clip(np.random.normal(RENT_MEAN, RENT_STD), RENT_MIN, RENT_MAX))
        rent_amount = -round(rent_ratio * income, 2)

        util_month = float(np.clip(np.random.normal(UTIL_MEAN, UTIL_STD), UTIL_MIN, UTIL_MAX))
        util_weekly = -round(util_month / 4.3, 2)

        credit_limit = float(np.clip(CL_BASE * income + CL_TENURE_BONUS * tenure, CL_MIN, CL_MAX))

        rows.append(dict(
            customer_id=cid,
            name=fake.name(),
            age=int(np.clip(np.random.normal(36, 10), 19, 75)),
            tenure_months=tenure,
            pay_cycle=pay_cycle,
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
        income = row.base_income
        rent_amt = row.rent_amount
        util_week = row.util_weekly

        # per-user multipliers to create personality
        mult = {k: np.random.uniform(0.8, 1.2) for k in CAT_FREQ_PM.keys()}

        ds_flag = bool(flags[i])
        ds_start = int(start_offsets[i])
        ds_event = int(event_offsets[i])

        for d in range(days):
            dt = start + timedelta(days=d)

            # income deposits
            if pay == "monthly" and dt.day == 1:
                amt = abs(np.random.normal(income, max(80, income * 0.07)))
                all_tx.append([cid, dt, "salary_income", pick_merchant("salary_income"), round(amt, 2), True, False])
            elif pay == "biweekly" and (dt.toordinal() + i) % 14 == 0:
                amt = abs(income / 2 + np.random.normal(0, income * 0.04))
                all_tx.append([cid, dt, "salary_income", pick_merchant("salary_income"), round(amt, 2), True, False])

            # rent monthly (1st or 3rd)
            if dt.day == (1 if i % 2 == 0 else 3):
                all_tx.append([cid, dt, "rent", pick_merchant("rent"), rent_amt, False, True])

            # utilities weekly aligned per-customer
            if d % 7 == (i % 7):
                all_tx.append([cid, dt, "utilities", pick_merchant("utilities"), util_week, False, True])

            # mild seasonality & weekend effect
            weekend = (dt.weekday() >= 5)
            season = 1.0 + 0.10 * np.sin(d / 28.0)  # ~monthly cycle

            # stochastic daily spending
            for cat, freq_pm in CAT_FREQ_PM.items():
                lam = (freq_pm / 30.0) * mult[cat] * season * (1.15 if weekend and cat in ["dining", "entertainment", "ecommerce"] else 1.0)
                n = np.random.poisson(lam)
                lo, hi = CAT_AMOUNTS[cat]
                for _ in range(n):
                    amt = np.random.uniform(lo, hi)
                    # occasional refunds/voids to add realism
                    if random.random() < 0.01:
                        amt = -amt * 0.9
                    all_tx.append([cid, dt, cat, pick_merchant(cat), round(amt, 2), False, (cat == "payday_loan")])

            # distress drift after start
            if ds_flag and d >= ds_start:
                # escalating cash advances as distress increases
                days_to_event = ds_event - d if ds_event >= 0 else float('inf')
                distress_intensity = max(0.5, 1.0 - (days_to_event / LEAD_DAYS)) if days_to_event < LEAD_DAYS else 0.5
                
                # more frequent cash advances with increasing intensity
                if random.random() < DISTRESS_CASH_ADV_PROB * distress_intensity:
                    # larger amounts when closer to event
                    base_min, base_max = -300, -60
                    amt_min = base_min * (1 + 0.5 * distress_intensity)  # up to 50% larger
                    amt_max = base_max * (1 + 0.3 * distress_intensity)  # up to 30% larger
                    amt = np.random.uniform(amt_min, amt_max)
                    all_tx.append([cid, dt, "cash_advance", pick_merchant("cash_advance"), round(amt, 2), False, False])
                
                # payday loans with escalating probability near event
                if ds_event >= 0 and d >= ds_event - 15:  # extend window to 15 days
                    # probability increases dramatically in final days
                    if days_to_event <= 5:
                        payday_prob = DISTRESS_PAYDAY_PROB_PEAK
                    elif days_to_event <= 10:
                        payday_prob = DISTRESS_PAYDAY_PROB_BASE * 1.5
                    else:
                        payday_prob = DISTRESS_PAYDAY_PROB_BASE
                    
                    if random.random() < payday_prob:
                        # larger payday loans when more desperate
                        base_min, base_max = -600, -150
                        amt_min = base_min * (1 + 0.4 * distress_intensity)
                        amt_max = base_max * (1 + 0.2 * distress_intensity)
                        amt = np.random.uniform(amt_min, amt_max)
                        all_tx.append([cid, dt, "payday_loan", pick_merchant("payday_loan"), round(amt, 2), False, True])
                
                # more frequent delayed utilities reflecting financial chaos
                if random.random() < DISTRESS_UTIL_DELAY_PROB * distress_intensity:
                    delay = np.random.randint(2, 7)  # longer delays possible
                    all_tx.append([cid, dt + timedelta(days=delay), "utilities", pick_merchant("utilities"), util_week, False, True])

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

    # distress flags and timing
    flags = (np.random.rand(len(customers)) < DISTRESS_SHARE).astype(int)
    start_offsets = np.where(flags == 1, np.random.randint(DAYS // 3, DAYS - 30, len(customers)), -1)
    event_offsets = np.where(flags == 1, start_offsets + LEAD_DAYS, -1)

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
