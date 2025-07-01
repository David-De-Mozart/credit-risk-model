[![CI Pipeline](https://github.com/david-de-mozart/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/david-de-mozart/credit-risk-model/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end implementation for building, deploying, and automating a credit risk model using alternative behavioral data.

# 📊 Credit Scoring with RFMS Behavioral Features

This project explores the use of RFMS (Recency, Frequency, Monetary, and Standard Deviation) variables in credit scoring. We apply logistic regression to evaluate user credit risk based on transactional behavior, improving upon traditional credit scoring approaches that rely solely on basic demographic and summary statistics.

---

## 📁 Table of Contents

- [Background](#background)
- [Objective](#objective)
- [Dataset](#dataset)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
- [Feature Engineering](#feature-engineering)
  - [Basic Variables](#basic-variables)
  - [RFMS Variables](#rfms-variables)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Business Insights](#business-insights)
- [Deployment](#-deployment)
- [Getting Started](#-getting-started)
- [Usage](#usage)
- [License](#license)

---

## 📌 Background

Microcredit companies often face the challenge of accurately evaluating customer creditworthiness in the absence of detailed financial histories. Traditional scoring systems are limited in their ability to capture nuanced behavioral patterns, especially for underbanked populations.

This project uses **RFMS modeling** on user transaction data to derive behavior-based features, which are then used in a logistic regression model to predict default probability.

---

## 🎯 Objective

- Construct new credit scoring features using **Recency**, **Frequency**, **Monetary**, and **Standard Deviation** of transactions across multiple behavior categories.
- Compare traditional and RFMS-based logistic regression models.
- Improve predictive accuracy for credit default detection.
- Provide actionable insights to microcredit lenders.

---


## 📊 Data and Features

Dataset Overview

- Source: Xente Challenge (Kaggle)
- Size: 95,662 transactions
- Time Period: 3 months
- Customers: 3,742 unique users

The dataset includes:
- Basic demographic and transaction summary information
- Detailed behavior logs across 10 transaction categories

---
Key Features Engineered
Feature Category	Variables	Transformation
RFM Metrics	Recency, Frequency, Monetary	Days since last transaction, Count, Sum
Temporal Features	Transaction hour, day, month	DateTime extraction
Behavioral Features	Product category mode, Channel mode	Aggregation
Statistical Features	Amount mean/std, Value sum	Statistical aggregation

---

Feature Engineering Pipeline

pipeline = Pipeline([
    ('rfm', RFMTransformer()),
    ('aggregator', FeatureAggregator()),
    ('preprocessor', ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]))
])

---

# Credit Scoring Business Understanding

This section addresses key business considerations underpinning credit scoring model development, especially in the context of regulatory compliance and practical challenges in risk prediction.

---

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord mandates that financial institutions rigorously measure and manage credit risk to maintain capital adequacy. This regulatory framework requires credit scoring models to be **transparent, interpretable, and auditable** to ensure that risk assessments are reliable and justifiable. An interpretable model enables stakeholders, including regulators and internal risk managers, to understand the drivers of credit decisions, verify compliance, and identify potential biases or errors. Well-documented models facilitate ongoing monitoring, validation, and adaptation, thereby supporting responsible risk management aligned with Basel II’s objectives.

---

🤖 Modeling Approach
Model Development Process
  1. Proxy Target Creation:

      - K-Means clustering on RFM features
      - High-risk cluster identified (38.7% of customers)

  2. Model Selection:
      models = {
           "LogisticRegression": LogisticRegression(),
           "RandomForest": RandomForestClassifier(),
           "GradientBoosting": GradientBoostingClassifier(),
           "XGBoost": XGBClassifier()
                }
            
  3. Evaluation Metrics:

        -  Accuracy, Precision, Recall
        -  F1 Score, ROC AUC

Hyperparameter Tuning

  - GridSearchCV for optimal parameters
  - Stratified sampling for class imbalance
  - MLflow for experiment tracking

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many real-world datasets, especially in emerging credit markets, a direct label indicating borrower default may be unavailable or incomplete. Creating a **proxy variable**—such as late payment indicators, non-performing loan status, or other financial distress signals—is essential to approximate default behavior for model training.

However, relying on proxies introduces risks:
- The proxy may imperfectly represent true default, causing **label noise** that can degrade model accuracy.
- It may fail to capture all dimensions of credit risk, leading to **misclassification** and suboptimal lending decisions.
- Business risks include **increased financial losses** due to incorrect approvals or denials, as well as **regulatory scrutiny** if model performance is inconsistent with actual risk outcomes.

Hence, the proxy's definition and quality must be carefully validated to mitigate these risks.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Aspect                  | Simple Model (Logistic Regression with WoE)                     | Complex Model (Gradient Boosting)                          |
|-------------------------|-----------------------------------------------------------------|------------------------------------------------------------|
| **Interpretability**    | High: coefficients directly explain feature impact; easy to audit | Low: ensemble structure is a "black box"; hard to explain decisions |
| **Regulatory Compliance** | Easier to justify to regulators; preferred in conservative contexts | Harder to validate; may require additional explainability tools |
| **Performance**          | Generally robust but may have lower predictive accuracy          | Often achieves higher accuracy and captures nonlinearities  |
| **Model Maintenance**    | Simpler to maintain, update, and monitor                         | More complex maintenance; prone to overfitting if not carefully managed |
| **Business Impact**      | Transparent decisions build trust with stakeholders              | Better performance can reduce losses but risk opacity       |

In regulated environments, the trade-off favors **model transparency and explainability** to ensure compliance and trust, even if it means sacrificing some predictive power. Complex models may be deployed with caution, supported by robust validation and explanation frameworks.

---

*This understanding guides our approach to credit scoring, balancing predictive accuracy with regulatory and operational requirements.*


## 🔧 Feature Engineering

### 🔹 Basic Variables (7 total)

| Variable | Description |
|----------|-------------|
| `basic_score` | Platform-generated traditional credit score |
| `reg_days` | Days since user registered |
| `trans_count` | Number of transactions |
| `avg_transaction` | Mean transaction amount |
| `max_transaction` | Largest single transaction |
| `credit_debit_ratio` | Ratio of credit card transactions |
| `card_count` | Number of bank cards owned |

---

### 🔹 RFMS Variables (40 total)

RFMS features are generated across **10 behavior categories**, each with:

- **R** (Recency): Days since last activity
- **F** (Frequency): Number of transactions
- **M** (Monetary): Average transaction amount
- **S** (Std Dev): Standard deviation of transaction amounts

| Categories |
|------------|
| Debit, Consumption, Loan, Transfer, Phone Bill, Utility Bill, Game, State-owned Bank, Medium Bank, VIP Card |

---

## 🤖 Modeling Approach

- **Model**: Logistic Regression
- **Feature Selection**: BIC-based stepwise selection
- **Data Prep**:
  - Log-transformations on monetary features
  - Standardization of numerical variables

### 🔬 Models Compared

| Model | Description |
|-------|-------------|
| Model A | Baseline using only `basic_score` |
| Model B | All 47 variables (basic + RFMS) |
| Model C | Selected features via BIC |

---

## 📈 Results

- **Model A AUC**: Baseline
- **Model B AUC**: Improved by ~13.6%
- **Model C AUC**: Similar to Model B with fewer variables

### 🔍 Key Predictors

✅ **Non-default indicators**:
- High `basic_score`
- High `credit_debit_ratio`
- More `debit frequency`
- Higher `avg_transaction`

❌ **Default risk indicators**:
- Long recency in `loan` category
- Many bank cards
- Low activity in `utility`, `transfer`, `state-owned` bank categories
- Long registration duration

---

## 💼 Business Insights

- Traditional scoring is inadequate alone.
- Behavioral features (RFMS) offer deeper risk insight.
- RFMS scoring can help micro-lenders:
  - Reject high-risk users early
  - Approve users with lower traditional scores but good behavior
- Credit score can be derived using:
  
Credit_Score = 400 + 400 × (1 - Default_Probability)

---
Project Structure
credit-risk-model/
├── .github/workflows/       # CI/CD pipelines
├── data/                    # Raw and processed data
├── models/                  # Trained models
├── notebooks/               # Exploratory analysis
├── reports/                 # Visualizations and outputs
├── src/                     # Source code
│   ├── data_processing.py   # Feature engineering
│   ├── train.py             # Model training
│   └── api/                 # FastAPI implementation
├── tests/                   # Unit tests
├── Dockerfile               # Containerization
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

----


📈 Results

Model Performance Comparison

Model	                | Accuracy	    | Precision	   | Recall	   | F1	     | ROC AUC
Logistic Regression	  | 0.9960	      | 1.0000	     | 0.9897	   | 0.9948	 | 1.0000
Random   Forest	      | 0.9973	      | 0.9932	     | 1.0000	   | 0.9966	 | 1.0000
Gradient Boosting	    | 0.9987	      | 0.9966	     | 1.0000	   | 0.9983	 | 1.0000
XGBoost	              | 0.9987	      | 0.9966	     | 1.0000	   | 0.9983	 | 1.0000

---

Key Insights

1. High-risk indicators:

    - Long transaction recency (>30 days)

    - Low transaction frequency (<2 transactions/month)

    - Small transaction amounts (<$10 average)

2. Credit score calculation:

credit_score = 300 + (1 - risk_prob) * 550

---

🚀 Deployment

### API Endpoints

python

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Returns:
    #   risk_probability: 0-1
    #   risk_category: Low/Medium/High
    #   credit_score: 300-850
    #   recommended_limit: USD


### CI/CD Pipeline

yaml

name: CI Pipeline
on: [push]
jobs:
  lint-and-test:
    steps:
      - run: flake8 src
      - run: pytest tests/


### Deployment Options

- Local:
bash
    - uvicorn src.api.main:app --reload

- Docker:
bash
    - docker build -t credit-risk-api .
    - docker run -p 8000:80 credit-risk-api

---

💻 Getting Started

# Prerequisites
      - Python 3.9+
      - Docker
      - MLflow

# Installation
bash
# Clone repository
git clone https://github.com/david-de-mozart/credit-risk-model.git
cd credit-risk-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
Running the Pipeline
bash
# 1. Data processing
python src/data_processing.py

# 2. Model training
python src/train.py

# 3. Start API
uvicorn src.api.main:app --reload
Testing
bash
# Run unit tests
pytest tests/

# Example API request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

"The key innovation lies in transforming behavioral data into predictive risk signals." - Bati Bank Analytics Team

