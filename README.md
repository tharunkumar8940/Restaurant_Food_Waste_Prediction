## 🚀 Live Application

[Open the Restaurant Food Waste & Demand Prediction Dashboard](https://restaurantfoodwasteprediction-gkrmea6sgr9alf3ru8than.streamlit.app/)
# 🍽️ Restaurant Food Waste & Demand Prediction System

## 📌 Project Overview

The Restaurant Food Waste & Demand Prediction System is a machine learning and data analytics project designed to help restaurants understand food demand, analyze food waste, predict future demand, and make data-driven food preparation decisions.

The system analyzes historical restaurant order data to identify:

- Most-ordered food items
- Least-ordered food items
- High-waste food items
- Low-waste food items
- Chicken food demand
- Monthly and weekly demand patterns
- Expected future food demand
- Food preparation recommendations

The project combines **Data Analysis, Machine Learning, and Business Intelligence** in a single Streamlit application.

> **Note:** The current dataset is synthetic demonstration data created for portfolio and learning purposes.

---

## 🎯 Project Objectives

The main objectives are:

1. Analyze restaurant food-order patterns.
2. Identify high-demand and low-demand food items.
3. Measure food preparation and waste.
4. Analyze chicken-based food demand separately.
5. Predict expected food demand using machine learning.
6. Compare multiple regression models.
7. Generate preparation recommendations.
8. Provide an interactive business dashboard.
9. Help restaurants reduce unnecessary food preparation and waste.

---

## 🏗️ Project Architecture

```text
Restaurant Order Data
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Exploratory Data Analysis
        │
        ├──────────────► Food Demand Analysis
        │
        ├──────────────► Food Waste Analysis
        │
        └──────────────► Chicken Food Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ▼
Machine Learning
        │
        ├── Linear Regression
        ├── Decision Tree
        ├── Random Forest
        └── Gradient Boosting
        │
        ▼
Model Evaluation
        │
        ▼
Best Demand Prediction Model
        │
        ▼
Recommendation Engine
        │
        ▼
Streamlit Dashboard
