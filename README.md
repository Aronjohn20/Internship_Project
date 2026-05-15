# Review Analytics and Suspicious Review Detection System

An NLP and Machine Learning based system for analyzing product reviews and detecting potentially suspicious or spam-like reviews using rule-based analytics and Linear SVM classification.

---

## Project Overview

This project analyzes product review datasets to identify suspicious review patterns and provide interactive analytics through a dashboard interface.

The system performs:
- Exploratory Data Analysis (EDA)
- Text preprocessing
- TF-IDF feature extraction
- Suspicious pattern detection
- Machine learning classification using Linear SVM
- Interactive dashboard visualization

The goal is not to guarantee whether a review is fake, but to highlight suspicious patterns for further manual inspection.

---

## Features

### Review Analytics
- Rating distribution analysis
- Review length analysis
- Frequent word visualization
- Time-based review trends

### Suspicious Review Detection
- Very short review detection
- Duplicate and near-duplicate review detection
- Excessive punctuation detection
- Spam-like wording detection

### Machine Learning
- TF-IDF vectorization
- Linear Support Vector Machine (SVM) classifier
- Genuine vs Suspicious review prediction

### Interactive Dashboard
- CSV upload support
- Suspicious review percentage
- Visual analytics and charts
- Suspicious review table
- Similar review identification

---

## Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Version Control | Git & GitHub |

---

## Project Structure

```bash
review-analytics-system/
│
├── data/
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── rules.py
│   ├── similarity.py
│   ├── model.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
├── requirements.txt
└── README.md
```

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
