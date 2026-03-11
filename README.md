# AI-Powered Loan Application Processor (ML Approach)

## Overview
This repository contains a machine learning approach to evaluating loan applications using a synthetic dataset of 2,000 historical applications. The goal of the project is to compare a **data-driven ML model** with an existing **rule-based scoring system**, analyze their performance, and examine fairness and explainability considerations.

The project includes data generation, exploratory analysis, feature engineering, model training, evaluation against the baseline system, and interpretability analysis.

---

## Repository Structure
├── generate_dataset.py
├── data/
│ └── loan_applications.csv
├── loan_default_prediction.ipynb
├── requirements.txt
└── README.md

### `generate_dataset.py`
Script provided in the assignment to generate the synthetic dataset of loan applications.

Running this script creates `loan_applications.csv`, which simulates:

- applicant financial information
- rule-based loan decisions
- historical repayment outcomes

The generated CSV is already committed in the repository under the `data/` directory so reviewers do not need to regenerate it.

---

### `data/loan_applications.csv`
The dataset used for the project.

Contains ~2,000 loan applications with fields such as:

- income information
- loan amount
- banking activity
- employment status
- rule-based decision
- actual repayment outcome

---

### `loan_default_prediction.ipynb`
Main notebook containing the full machine learning workflow.

The notebook is organized into sections that mirror the assignment requirements:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and optimization
- Evaluation against the rule-based baseline
- Fairness analysis
- Explainability using SHAP
- Production considerations

All analysis, model training, and results are contained within this notebook.

---

### `requirements.txt`
Lists the Python dependencies required to reproduce the environment used in this project.

Key libraries include:

- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn

---

## Running the Project

1. Install dependencies

```bash
pip install -r requirements.txt
