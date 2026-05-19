# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.model import predict_dataset

from src.similarity import (
    clean_similarity_data,
    create_similarity_vectors,
    compute_similarity_matrix,
    find_similar_reviews
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Review Analytics System",
    layout="wide"
)


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title(
    "Review Analytics and Suspicious Review Detection System"
)

st.markdown("""
This system analyzes uploaded product review datasets
and identifies suspicious review patterns using:

- NLP preprocessing
- Feature engineering
- Cosine similarity analysis
- Linear SVM classification
""")


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Review Dataset CSV",
    type=["csv"]
)


# =========================================================
# MAIN PIPELINE
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    # -----------------------------------------------------
    # DATA PREVIEW
    # -----------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # -----------------------------------------------------
    # RUN ML PREDICTION PIPELINE
    # -----------------------------------------------------

    st.subheader("Running Prediction Pipeline...")

    predicted_df = predict_dataset(df)

    st.success("Prediction Completed")

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------

    suspicious_count = (
        predicted_df['prediction'] == 0
    ).sum()

    genuine_count = (
        predicted_df['prediction'] == 1
    ).sum()

    total_reviews = len(predicted_df)

    suspicious_percentage = (
        suspicious_count / total_reviews
    ) * 100

    # -----------------------------------------------------
    # DISPLAY METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Suspicious Reviews",
        suspicious_count
    )

    col3.metric(
        "Genuine Reviews",
        genuine_count
    )

    col4.metric(
        "Suspicious %",
        f"{suspicious_percentage:.2f}%"
    )

    # -----------------------------------------------------
    # PREDICTION TABLE
    # -----------------------------------------------------

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        predicted_df[
            [
                'reviewText',
                'prediction',
                'promo_word_count',
                'is_duplicate_review'
            ]
        ].head(50)
    )

    # =====================================================
    # RATING DISTRIBUTION
    # =====================================================

    st.subheader(
        "Rating Distribution"
    )

    fig, ax = plt.subplots()

    predicted_df['overall'].value_counts().sort_index().plot(
        kind='bar',
        ax=ax
    )

    ax.set_xlabel("Ratings")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # =====================================================
    # SUSPICIOUS VS GENUINE
    # =====================================================

    st.subheader(
        "Prediction Distribution"
    )

    fig2, ax2 = plt.subplots()

    predicted_df['prediction'].value_counts().plot(
        kind='bar',
        ax=ax2
    )

    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)

    # =====================================================
    # SIMILARITY DETECTION
    # =====================================================

    st.subheader(
        "Similarity Detection"
    )

    similarity_df = clean_similarity_data(
        predicted_df.copy()
    )

    # -----------------------------------------------------
    # SAMPLE SMALLER DATA
    # -----------------------------------------------------

    sample_size = min(
        1000,
        len(similarity_df)
    )

    sample_df = similarity_df.sample(
        n=sample_size,
        random_state=42
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # TF-IDF FOR SIMILARITY
    # -----------------------------------------------------

    X_similarity = create_similarity_vectors(
        sample_df['processed_review']
    )

    # -----------------------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------------------

    similarity_matrix = compute_similarity_matrix(
        X_similarity
    )

    # -----------------------------------------------------
    # FIND SIMILAR REVIEWS
    # -----------------------------------------------------

    similarity_results = find_similar_reviews(
        sample_df,
        similarity_matrix,
        threshold=0.65
    )

    st.write(
        "Highly Similar Review Pairs:"
    )

    st.write(
        len(similarity_results)
    )

    # -----------------------------------------------------
    # SHOW RESULTS
    # -----------------------------------------------------

    if len(similarity_results) > 0:

        st.dataframe(
            similarity_results.head(20)
        )

    else:

        st.info(
            "No highly similar reviews found."
        )