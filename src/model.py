# =========================================================
# IMPORTS
# =========================================================

import joblib
import numpy as np

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from src.preprocessing import preprocess_text

from src.features import (
    count_exclamations,
    capital_ratio,
    promo_word_count
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
# PREDICT REVIEW FUNCTION
# =========================================================

def predict_review(review_text):

    # -----------------------------------------------------
    # TEXT PREPROCESSING
    # -----------------------------------------------------

    processed_review = preprocess_text(
        review_text
    )

    # -----------------------------------------------------
    # TF-IDF TRANSFORMATION
    # -----------------------------------------------------

    text_vector = tfidf_vectorizer.transform(
        [processed_review]
    )

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    exclamation_feature = count_exclamations(
        review_text
    )

    capital_feature = capital_ratio(
        review_text
    )

    promo_feature = promo_word_count(
        processed_review
    )

    # -----------------------------------------------------
    # NUMERICAL FEATURE MATRIX
    # -----------------------------------------------------

    numerical_features = np.array([
        [
            exclamation_feature,
            capital_feature,
            promo_feature
        ]
    ])

    numerical_features = csr_matrix(
        numerical_features
    )

    # -----------------------------------------------------
    # COMBINE FEATURES
    # -----------------------------------------------------

    final_features = hstack([
        text_vector,
        numerical_features
    ])

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = svm_model.predict(
        final_features
    )[0]

    return prediction