# 📊 Smart Business Dashboard with AI Insights

> **An end-to-end Data Science project** combining Machine Learning, automated AI insights, and an interactive web dashboard — built by **Harsh Sachan** (Oracle Certified Data Scientist)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Chart.js](https://img.shields.io/badge/Chart.js-Dashboard-pink?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Project Overview

This project simulates a real-world business analytics platform that:
- 📈 **Tracks sales performance** across regions, products, and reps
- 🤖 **Auto-generates AI insights** like *"Sales drop in North region (-22%)"*
- 👥 **Predicts customer churn** using a Random Forest ML model
- 📊 **Visualizes everything** in an interactive web dashboard

---

## 🏗 Architecture

```
smart-business-dashboard/
├── data/
│   ├── generate_data.py        # Synthetic data generator (9,451 sales rows)
│   ├── sales_data.csv          # 2 years of daily sales transactions
│   ├── customer_data.csv       # 2,000 customer profiles
│   └── churn_data.csv          # Churn labels + features for ML
│
├── models/
│   ├── churn_model.py          # Trains Random Forest, GBM, Logistic Regression
│   ├── churn_model.pkl         # Saved best model (Random Forest)
│   ├── feature_importance.csv  # Top churn predictors
│   └── model_results.json      # AUC scores per model
│
├── insights/
│   ├── ai_insights.py          # Auto-generates business insights (no GPT needed)
│   ├── insights.json           # 12 structured insights with recommendations
│   └── kpis.json               # KPI summary for dashboard
│
├── dashboard/
│   └── index.html              # Interactive dashboard (Chart.js, zero dependencies)
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis notebook
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/hsachan295-source/smart-business-dashboard.git
cd smart-business-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data
```bash
python data/generate_data.py
```

### 4. Train ML model
```bash
python models/churn_model.py
```

### 5. Generate AI insights
```bash
python insights/ai_insights.py
```

### 6. Open dashboard
```bash
# Just open in browser — no server needed!
open dashboard/index.html
# or on Windows:
start dashboard/index.html
```

---

## 🤖 AI Insights Engine

The `ai_insights.py` script automatically detects and narrates business problems:

| Severity | Example Insight |
|----------|----------------|
| 🔴 Critical | "Sales Drop in North Region (-22.4%)" |
| 🔴 Critical | "Overall Churn Rate: 39.5%" |
| 🟡 Warning  | "Highest Churn in East Region (50.5%)" |
| 🟢 Positive | "Strong Growth in East Region (+28.3%)" |
| 🟢 Positive | "Product C is the Revenue Leader" |

Each insight includes:
- **Detail**: What exactly is happening and by how much
- **Recommendation**: Concrete next steps for the business

---

## 📈 ML Model Performance

| Model | Accuracy | AUC | CV-AUC |
|-------|----------|-----|--------|
| Random Forest ⭐ | 67% | 0.72 | 0.71 |
| Gradient Boosting | 66% | 0.71 | 0.70 |
| Logistic Regression | 63% | 0.68 | 0.67 |

**Top Churn Predictors:**
1. Last Login Days (24.0%)
2. Annual Spend (17.4%)
3. Support Tickets (14.7%)
4. Age (13.7%)
5. NPS Score (12.2%)

---

## 📊 Dashboard Features

| Page | What it shows |
|------|--------------|
| **Overview** | KPIs, Revenue trend, Region/Product breakdown |
| **Sales Analysis** | Timeline, Units sold, Rep performance, Discount analysis |
| **Churn Prediction** | At-risk customers, Feature importance, Risk table |
| **AI Insights** | 12 auto-generated insights with recommendations |

---

## 🔧 Optional: GPT Integration

To upgrade insights with GPT-4/Claude:
```python
# In insights/ai_insights.py, uncomment the LLM section:
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# Each insight gets enriched with LLM-generated narrative
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Python, Pandas, NumPy |
| ML | Scikit-learn (Random Forest, GBM, Logistic) |
| Insights | Custom NLG engine (rule-based) |
| Visualization | Chart.js (no backend needed) |
| Optional AI | OpenAI GPT / LangChain |

---

## 👤 Author

**Harsh Sachan**
- 🏅 Oracle Cloud Infrastructure 2025 Certified Data Science Professional
- 🏅 Oracle Cloud Infrastructure 2025 Certified AI Foundations Associate
- 📧 hsachan295@gmail.com
- 🐙 [GitHub](https://github.com/hsachan295-source)

---

## 📄 License

MIT License — free to use, modify, and distribute.
