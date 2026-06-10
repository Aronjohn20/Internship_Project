# =========================================================
# IMPORTS
# =========================================================

import pandas as pd

from src.preprocessing import (
    preprocess_text
)

from src.features import (
    count_exclamations,
    capital_ratio,
    promo_word_count
)

from src.model import (
    predict_dataset
)

from src.similarity import (
    clean_similarity_data,
    create_similarity_vectors,
    compute_similarity_matrix,
    find_similar_reviews
)


# =========================================================
# LOAD SAMPLE DATASET
# =========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    "data/test_sample.csv"
)

print("Dataset Loaded Successfully")

print()

print("Dataset Shape:")
print(df.shape)

print()

print(df.head())


# =========================================================
# TEST PREPROCESSING
# =========================================================

print()
print("=" * 60)
print("TESTING PREPROCESSING")
print("=" * 60)

sample_text = df['reviewText'].iloc[0]

processed = preprocess_text(
    sample_text
)

print("Original Review:")
print(sample_text)

print()

print("Processed Review:")
print(processed)


# =========================================================
# TEST FEATURE ENGINEERING
# =========================================================

print()
print("=" * 60)
print("TESTING FEATURES")
print("=" * 60)

print("Exclamation Count:")
print(
    count_exclamations(sample_text)
)

print()

print("Capital Ratio:")
print(
    capital_ratio(sample_text)
)

print()

print("Promo Word Count:")
print(
    promo_word_count(processed)
)


# =========================================================
# TEST ML PREDICTION PIPELINE
# =========================================================

print()
print("=" * 60)
print("TESTING MODEL PIPELINE")
print("=" * 60)

predicted_df = predict_dataset(
    df.copy()
)

print("Prediction Pipeline Completed")

print()

print(
    predicted_df[
        [
            'reviewText',
            'prediction'
        ]
    ].head()
)


# =========================================================
# SUMMARY STATISTICS
# =========================================================

print()
print("=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

suspicious_count = (
    predicted_df['prediction'] == 0
).sum()

genuine_count = (
    predicted_df['prediction'] == 1
).sum()

print("Suspicious Reviews:")
print(suspicious_count)

print()

print("Genuine Reviews:")
print(genuine_count)


# =========================================================
# TEST SIMILARITY DETECTION
# =========================================================

print()
print("=" * 60)
print("TESTING SIMILARITY DETECTION")
print("=" * 60)

similarity_df = clean_similarity_data(
    predicted_df.copy()
)

sample_size = min(
    500,
    len(similarity_df)
)

sample_df = similarity_df.sample(
    n=sample_size,
    random_state=42
).reset_index(drop=True)

print("Similarity Sample Shape:")
print(sample_df.shape)

print()

print("Creating Similarity Vectors...")

X_similarity = create_similarity_vectors(
    sample_df['processed_review']
)

print("Vector Shape:")
print(X_similarity.shape)

print()

print("Computing Similarity Matrix...")

similarity_matrix = compute_similarity_matrix(
    X_similarity
)

print("Matrix Shape:")
print(similarity_matrix.shape)

print()

print("Finding Similar Reviews...")

similarity_results = find_similar_reviews(
    sample_df,
    similarity_matrix,
    threshold=0.65
)

print()

print("Highly Similar Review Pairs:")
print(len(similarity_results))

print()

if len(similarity_results) > 0:

    print(
        similarity_results[
            [
                'similarity_score',
                'review_1',
                'review_2'
            ]
        ].head()
    )

else:

    print("No Similar Reviews Found")


# =========================================================
# FINAL SUCCESS MESSAGE
# =========================================================

print()
print("=" * 60)
print("BACKEND TEST COMPLETED SUCCESSFULLY")
print("=" * 60)