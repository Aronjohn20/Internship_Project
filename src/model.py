# =========================================================
# IMPORTS
# =========================================================

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import (
    hstack,
    csr_matrix
)

from src.preprocessing import (
    preprocess_text
)

from src.features import (
    count_exclamations,
    capital_ratio,
    promo_word_count,
    reviewer_review_counts,
    product_review_counts,
    duplicate_review_feature
)


# =========================================================
# LOAD TRAINED ARTIFACTS
# =========================================================

svm_model = joblib.load(
    "models/linear_svm_model.pkl"
)

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# =========================================================
# DATASET PREDICTION PIPELINE
# =========================================================

def predict_dataset(df):

    # -----------------------------------------------------
    # PREPROCESS TEXT
    # -----------------------------------------------------

    df['processed_review'] = (
        df['reviewText']
        .apply(preprocess_text)
    )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    df['exclamation_count'] = (
        df['reviewText']
        .apply(count_exclamations)
    )

    df['capital_ratio'] = (
        df['reviewText']
        .apply(capital_ratio)
    )

    df['promo_word_count'] = (
        df['processed_review']
        .apply(promo_word_count)
    )

    # -----------------------------------------------------
    # DATASET LEVEL FEATURES
    # -----------------------------------------------------

    df['reviewer_review_count'] = (
        reviewer_review_counts(df)
    )

    df['product_review_count'] = (
        product_review_counts(df)
    )

    df['is_duplicate_review'] = (
        duplicate_review_feature(df)
    )

    # -----------------------------------------------------
    # TF-IDF TRANSFORMATION
    # -----------------------------------------------------

    X_text = tfidf_vectorizer.transform(
        df['processed_review']
    )

    # -----------------------------------------------------
    # NUMERICAL FEATURES
    # -----------------------------------------------------

    numerical_features = df[
        [
            'review_length',
            'exclamation_count',
            'capital_ratio',
            'reviewer_review_count',
            'product_review_count',
            'is_duplicate_review',
            'promo_word_count'
        ]
    ]

    numerical_features = csr_matrix(
        numerical_features.values
    )

    # -----------------------------------------------------
    # COMBINE FEATURES
    # -----------------------------------------------------

    X_final = hstack([
        X_text,
        numerical_features
    ])

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    predictions = svm_model.predict(
        X_final
    )

    # -----------------------------------------------------
    # ADD PREDICTIONS TO DATAFRAME
    # -----------------------------------------------------

    df['prediction'] = predictions

    return df