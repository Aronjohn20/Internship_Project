import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

def clean_similarity_data(df):

    df['processed_review'] = (
        df['processed_review']
        .fillna('')
    )

    return df
def create_similarity_vectors(processed_reviews):
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1,2),
        min_df=2
    )

    X_similarity = tfidf.fit_transform(
        sample_df['processed_review']
    )
    return X_similarity

def compute_similarity_matrix(X_similarity):

    similarity_matrix = cosine_similarity(
        X_similarity
    )

    return similarity_matrix


def find_similar_reviews(
    sample_df,
    similarity_matrix,
    threshold=0.65
):
    threshold = 0.65

    similar_pairs = []

    rows = similarity_matrix.shape[0]

    for i in range(rows):

        for j in range(i + 1, rows):

            similarity_score = similarity_matrix[i][j]

            if similarity_score >= threshold:

                similar_pairs.append({

                    'review_1_index': i,
                    'review_2_index': j,
                    'similarity_score': similarity_score,

                    'review_1': sample_df.iloc[i]['reviewText'],
                    'review_2': sample_df.iloc[j]['reviewText'],

                    'reviewer_1': sample_df.iloc[i]['reviewerID'],
                    'reviewer_2': sample_df.iloc[j]['reviewerID'],


                    'product_1': sample_df.iloc[i]['asin'],
                    'product_2': sample_df.iloc[j]['asin']
                })

        similarity_results = pd.DataFrame(
        similar_pairs
    )

    return similarity_results

def same_product_analysis(similarity_results):
    same_product = similarity_results[
        similarity_results['product_1']
        ==
        similarity_results['product_2']
    ]
    return same_product