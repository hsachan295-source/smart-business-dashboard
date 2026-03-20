"""
generate_data.py
Generates realistic synthetic business data for the Smart Business Dashboard.
Run once to create: sales_data.csv, customer_data.csv, churn_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.dirname(__file__)

REGIONS      = ["North", "South", "East", "West", "Central"]
PRODUCTS     = ["Product A", "Product B", "Product C", "Product D", "Product E"]
SEGMENTS     = ["Enterprise", "SMB", "Startup", "Government"]
SALES_REPS   = [f"Rep_{i}" for i in range(1, 11)]

# ─── 1. SALES DATA (2 years, daily) ──────────────────────────────────────────
def generate_sales():
    rows = []
    start = datetime(2023, 1, 1)
    for d in range(730):
        date = start + timedelta(days=d)
        month = date.month
        # Seasonal multiplier
        season = 1 + 0.3 * np.sin((month - 3) * np.pi / 6)
        # Inject a drop in East region during mid 2024
        for _ in range(random.randint(8, 18)):
            region  = random.choice(REGIONS)
            product = random.choice(PRODUCTS)
            segment = random.choice(SEGMENTS)
            rep     = random.choice(SALES_REPS)
            base    = random.uniform(2000, 15000)
            # Regional trend: East drops in 2024-Q2
            if region == "East" and date >= datetime(2024, 4, 1) and date <= datetime(2024, 9, 30):
                base *= 0.55
            # Product C gaining traction in 2024
            if product == "Product C" and date >= datetime(2024, 1, 1):
                base *= 1.35
            revenue = round(base * season * random.uniform(0.85, 1.15), 2)
            units   = max(1, int(revenue / random.uniform(150, 600)))
            rows.append({
                "date":       date.strftime("%Y-%m-%d"),
                "region":     region,
                "product":    product,
                "segment":    segment,
                "sales_rep":  rep,
                "revenue":    revenue,
                "units_sold": units,
                "discount_pct": round(random.uniform(0, 25), 1),
                "cost":       round(revenue * random.uniform(0.45, 0.68), 2),
            })
    df = pd.DataFrame(rows)
    df["profit"]       = (df["revenue"] - df["cost"]).round(2)
    df["profit_margin"] = ((df["profit"] / df["revenue"]) * 100).round(2)
    path = os.path.join(OUTPUT_DIR, "sales_data.csv")
    df.to_csv(path, index=False)
    print(f"✅  sales_data.csv  → {len(df):,} rows")
    return df


# ─── 2. CUSTOMER DATA ────────────────────────────────────────────────────────
def generate_customers():
    n = 2000
    start = datetime(2022, 1, 1)
    rows = []
    for cid in range(1, n + 1):
        join_date = start + timedelta(days=random.randint(0, 700))
        region    = random.choice(REGIONS)
        segment   = random.choice(SEGMENTS)
        base_spend = {"Enterprise": 50000, "SMB": 15000, "Startup": 8000, "Government": 35000}[segment]
        rows.append({
            "customer_id":     cid,
            "join_date":       join_date.strftime("%Y-%m-%d"),
            "region":          region,
            "segment":         segment,
            "age":             random.randint(25, 65),
            "annual_spend":    round(base_spend * random.uniform(0.6, 1.8), 2),
            "support_tickets": random.randint(0, 20),
            "nps_score":       random.randint(1, 10),
            "products_owned":  random.randint(1, 5),
            "last_login_days": random.randint(1, 180),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "customer_data.csv")
    df.to_csv(path, index=False)
    print(f"✅  customer_data.csv → {len(df):,} rows")
    return df


# ─── 3. CHURN DATA ───────────────────────────────────────────────────────────
def generate_churn(customers_df):
    df = customers_df.copy()
    # Churn probability based on real signals
    df["churn_prob"] = (
        0.05
        + (df["last_login_days"] / 180) * 0.30
        + ((10 - df["nps_score"]) / 10) * 0.25
        + (df["support_tickets"] / 20) * 0.20
        - (df["products_owned"] / 5) * 0.10
    ).clip(0.02, 0.92)

    # East region churn spike (mirrors sales drop)
    df.loc[df["region"] == "East", "churn_prob"] = (
        df.loc[df["region"] == "East", "churn_prob"] * 1.45
    ).clip(upper=0.95)

    df["churned"] = (np.random.rand(len(df)) < df["churn_prob"]).astype(int)

    keep_cols = [
        "customer_id", "region", "segment", "age", "annual_spend",
        "support_tickets", "nps_score", "products_owned",
        "last_login_days", "churn_prob", "churned"
    ]
    out = df[keep_cols].copy()
    out["churn_prob"] = out["churn_prob"].round(4)
    path = os.path.join(OUTPUT_DIR, "churn_data.csv")
    out.to_csv(path, index=False)
    print(f"✅  churn_data.csv  → {len(out):,} rows  |  churn rate: {out['churned'].mean():.1%}")
    return out


if __name__ == "__main__":
    print("Generating data …")
    sales_df     = generate_sales()
    customers_df = generate_customers()
    churn_df     = generate_churn(customers_df)
    print("\nAll datasets generated successfully!")
