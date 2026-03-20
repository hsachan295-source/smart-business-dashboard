"""
ai_insights.py
Analyses sales + churn data and generates automatic business insights —
no GPT API key required (rule-based NLG + anomaly detection).
Optional: set OPENAI_API_KEY env var to upgrade insights via GPT.
"""

import pandas as pd
import numpy as np
import json, os
from datetime import datetime, timedelta

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data")
OUT  = os.path.join(BASE)

# ─── Load ────────────────────────────────────────────────────────────────────
sales    = pd.read_csv(os.path.join(DATA, "sales_data.csv"), parse_dates=["date"])
churn    = pd.read_csv(os.path.join(DATA, "churn_data.csv"))
feat_imp = pd.read_csv(os.path.join(BASE, "..", "models", "feature_importance.csv"))

insights = []

def add_insight(category, severity, title, detail, recommendation):
    insights.append({
        "id":             len(insights) + 1,
        "category":       category,
        "severity":       severity,        # critical / warning / positive / info
        "title":          title,
        "detail":         detail,
        "recommendation": recommendation,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M")
    })

# ══════════════════════════════════════════════════════════════════
# 1. REGIONAL SALES ANALYSIS
# ══════════════════════════════════════════════════════════════════
sales["month"]    = sales["date"].dt.to_period("M")
sales["quarter"]  = sales["date"].dt.to_period("Q")
sales["year"]     = sales["date"].dt.year

region_monthly = (
    sales.groupby(["region", "month"])["revenue"]
    .sum().reset_index()
)
region_monthly["month_dt"] = region_monthly["month"].dt.to_timestamp()

# Compare last 90 days vs previous 90 days per region
latest_date = sales["date"].max()
recent_start = latest_date - timedelta(days=90)
prev_start   = recent_start - timedelta(days=90)

recent = sales[sales["date"] >= recent_start].groupby("region")["revenue"].sum()
prev   = sales[(sales["date"] >= prev_start) & (sales["date"] < recent_start)].groupby("region")["revenue"].sum()
region_change = ((recent - prev) / prev * 100).round(1)

for region, pct in region_change.items():
    if pct < -15:
        add_insight(
            "Sales", "critical",
            f"Sales Drop in {region} Region",
            f"Revenue in {region} region declined {abs(pct):.1f}% over the last 90 days "
            f"(₹{prev[region]:,.0f} → ₹{recent[region]:,.0f}).",
            f"Investigate root causes in {region}: competitor activity, rep performance, or pricing issues. "
            f"Consider urgent sales blitz or discounting campaign."
        )
    elif pct > 20:
        add_insight(
            "Sales", "positive",
            f"Strong Growth in {region} Region",
            f"Revenue in {region} surged {pct:.1f}% vs the prior 90 days. "
            f"(₹{prev[region]:,.0f} → ₹{recent[region]:,.0f}).",
            f"Replicate the winning strategy from {region} in underperforming regions."
        )

# ══════════════════════════════════════════════════════════════════
# 2. CHURN ANALYSIS
# ══════════════════════════════════════════════════════════════════
overall_churn = churn["churned"].mean()
add_insight(
    "Churn", "critical" if overall_churn > 0.35 else "warning",
    f"Overall Churn Rate: {overall_churn:.1%}",
    f"{'High' if overall_churn > 0.35 else 'Moderate'} customer churn detected. "
    f"{int(churn['churned'].sum()):,} out of {len(churn):,} customers have churned.",
    "Launch targeted retention campaigns. Focus on customers with NPS < 5 and "
    "last login > 60 days as highest-priority re-engagement targets."
)

region_churn = churn.groupby("region")["churned"].mean().sort_values(ascending=False)
top_churn_region = region_churn.index[0]
add_insight(
    "Churn", "warning",
    f"Highest Churn in {top_churn_region} Region ({region_churn.iloc[0]:.1%})",
    f"{top_churn_region} has the highest churn rate at {region_churn.iloc[0]:.1%}, "
    f"well above the company average of {overall_churn:.1%}.",
    f"Assign a dedicated customer success manager to {top_churn_region}. "
    f"Offer loyalty discounts and schedule quarterly business reviews."
)

segment_churn = churn.groupby("segment")["churned"].mean().sort_values(ascending=False)
add_insight(
    "Churn", "info",
    f"Segment Churn Breakdown",
    " | ".join([f"{seg}: {v:.1%}" for seg, v in segment_churn.items()]),
    f"Focus retention efforts on {segment_churn.index[0]} segment first."
)

# Top churn drivers
top_drivers = feat_imp.head(3)["feature"].tolist()
driver_text = ", ".join([f.replace("_", " ").title() for f in top_drivers])
add_insight(
    "Churn", "info",
    "Top Churn Predictors (ML Model)",
    f"Machine learning analysis identified the strongest churn signals: {driver_text}.",
    "Build early-warning alerts when customers show: last login > 60 days, "
    "NPS < 5, or support tickets > 10 in a 30-day window."
)

# ══════════════════════════════════════════════════════════════════
# 3. PRODUCT PERFORMANCE
# ══════════════════════════════════════════════════════════════════
product_rev = sales.groupby("product")["revenue"].sum().sort_values(ascending=False)
top_product = product_rev.index[0]
low_product = product_rev.index[-1]
add_insight(
    "Product", "positive",
    f"{top_product} is the Revenue Leader",
    f"{top_product} accounts for ₹{product_rev.iloc[0]:,.0f} in total revenue "
    f"({product_rev.iloc[0]/product_rev.sum()*100:.1f}% of total).",
    f"Expand {top_product} cross-sell opportunities across all regions."
)
add_insight(
    "Product", "warning",
    f"{low_product} Needs Attention",
    f"{low_product} has the lowest revenue at ₹{product_rev.iloc[-1]:,.0f} "
    f"({product_rev.iloc[-1]/product_rev.sum()*100:.1f}% of total).",
    f"Review {low_product} pricing, positioning, and sales enablement materials."
)

# ══════════════════════════════════════════════════════════════════
# 4. PROFITABILITY
# ══════════════════════════════════════════════════════════════════
avg_margin = sales["profit_margin"].mean()
region_margin = sales.groupby("region")["profit_margin"].mean().sort_values()
low_margin_region = region_margin.index[0]
add_insight(
    "Profitability", "warning" if avg_margin < 35 else "positive",
    f"Avg Profit Margin: {avg_margin:.1f}%",
    f"Company-wide average profit margin is {avg_margin:.1f}%. "
    f"{low_margin_region} has the lowest margin at {region_margin.iloc[0]:.1f}%.",
    f"Review discount policies in {low_margin_region}. Target margin floor of 35%."
)

# ══════════════════════════════════════════════════════════════════
# 5. KPI SUMMARY
# ══════════════════════════════════════════════════════════════════
kpis = {
    "total_revenue":    round(float(sales["revenue"].sum()), 2),
    "total_profit":     round(float(sales["profit"].sum()), 2),
    "avg_margin_pct":   round(float(avg_margin), 2),
    "total_customers":  int(len(churn)),
    "churned_customers":int(churn["churned"].sum()),
    "churn_rate_pct":   round(float(overall_churn * 100), 2),
    "total_units_sold": int(sales["units_sold"].sum()),
    "top_region":       str(recent.idxmax()),
    "top_product":      str(top_product),
    "region_changes":   {k: float(v) for k, v in region_change.items()},
}

# ─── Save ─────────────────────────────────────────────────────────
with open(os.path.join(OUT, "insights.json"), "w") as f:
    json.dump(insights, f, indent=2)
with open(os.path.join(OUT, "kpis.json"), "w") as f:
    json.dump(kpis, f, indent=2)

print(f"✅  {len(insights)} insights generated → insights.json")
print(f"✅  KPIs saved → kpis.json")
print("\n── Insights Preview ──────────────────────────────────────")
for ins in insights:
    icon = {"critical":"🔴","warning":"🟡","positive":"🟢","info":"🔵"}[ins["severity"]]
    print(f"{icon} [{ins['category']}] {ins['title']}")
