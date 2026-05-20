# Review Intelligence and Suspicious Pattern Analytics System

An NLP and Machine Learning based analytics platform for identifying suspicious review behavior, coordinated review patterns, and marketplace manipulation signals using rule-based analytics, TF-IDF vectorization, cosine similarity analysis, and Linear SVM classification.

---

# Project Overview

This project is designed as a **Review Intelligence and Suspicious Pattern Analytics System** rather than a guaranteed fake-review detector.

The system analyzes product review datasets to identify:

- suspicious linguistic patterns
- duplicate or coordinated review behavior
- reviewer-level anomalies
- potential rating manipulation
- promotional review activity

The platform combines:

- Natural Language Processing (NLP)
- Machine Learning
- Rule-Based Analytics
- Similarity Detection
- Interactive Dashboard Visualization

to support review moderation and marketplace risk analysis.

---

# Key Features

# Review Analytics

- Rating distribution analysis
- Review length analysis
- Reviewer activity analysis
- Product-level review analysis
- Interactive visual analytics dashboard

---

# Suspicious Pattern Detection

The system identifies potentially suspicious review behavior using:

- duplicate review detection
- near-duplicate similarity analysis
- excessive promotional wording
- excessive capitalization
- reviewer activity anomalies
- spam-like linguistic patterns

The system does **not** claim to prove reviews are fake with certainty.

Instead, it highlights reviews that exhibit suspicious behavioral or linguistic characteristics for further inspection.

---

# Machine Learning Pipeline

The project uses:

- TF-IDF Vectorization
- Feature Engineering
- Linear Support Vector Machine (SVM)

to classify reviews as:

- Lower Risk Reviews
- Higher Risk / Suspicious Reviews

---

# Coordinated Review Pattern Analysis

Using cosine similarity analysis, the system identifies:

- highly similar review pairs
- repetitive review templates
- potential coordinated review campaigns

This helps detect suspicious review groups rather than only individual suspicious reviews.

---

# Reviewer Trust Analytics

The system generates reviewer-level trust analysis using:

- duplicate review behavior
- promotional wording
- reviewer activity frequency
- suspicious review indicators

This provides reviewer risk insights for moderation purposes.

---

# Adjusted Product Rating Analysis

The dashboard estimates:

- original product ratings
- adjusted ratings after excluding suspicious reviews

This helps analyze the potential impact of suspicious reviews on marketplace ratings.

---

# Explainable Suspicious Indicators

Each suspicious review can include explainable indicators such as:

- Promotional Language
- Duplicate Review Pattern
- Excessive Capitalization
- High Reviewer Activity
- General Suspicious ML Pattern

This improves interpretability and moderation transparency.

---

# Interactive Dashboard Features

The Streamlit dashboard provides:

- CSV upload support
- Suspicious review analytics
- Interactive charts and metrics
- Suspicious review moderation table
- Reviewer trust analysis
- Product-level analytics
- Coordinated review pattern analysis
- Adjusted rating visualization

---

# Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK |
| Machine Learning | Scikit-learn |
| Similarity Analysis | Cosine Similarity |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Version Control | Git & GitHub |

---

# System Architecture

```text
Dataset Upload
        ↓
Text Preprocessing
        ↓
Feature Engineering
        ↓
TF-IDF Vectorization
        ↓
Rule-Based Suspicious Detection
        ↓
Linear SVM Classification
        ↓
Similarity Analysis
        ↓
Reviewer Trust Analytics
        ↓
Interactive Dashboard Visualization


review-intelligence-system/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│   ├── linear_svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   ├── similarity_detection.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── similarity.py
│   ├── model.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
---

## Workflow

1. Load product review dataset
2. Perform data cleaning and preprocessing
3. Conduct exploratory data analysis
4. Extract TF-IDF features
5. Apply rule-based suspicious pattern detection
6. Train Linear SVM classifier
7. Display insights and predictions in dashboard

---

## Dataset

The system accepts CSV datasets containing:
- Review text
- Ratings
- Product ID
- Reviewer ID
- Review date

Example datasets:
- Amazon Reviews Dataset
- Yelp Reviews Dataset
- Kaggle Review Datasets

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/review-analytics-system.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

## Future Improvements

- Deep learning based classification
- Sentiment analysis integration
- Real-time review monitoring
- Reviewer behavior analysis
- Advanced anomaly detection

---

## Author

Aron Varghese John
