import pandas as pd

promo_words = [
    'excellent',
    'perfect',
    'best',
    'amazing',
    'awesome',
    'great',
    'recommend',
    'highly',
    'fantastic',
    'love'
]

def count_exclamations(text):

    if pd.isna(text):
        return 0

    return text.count('!')

def capital_ratio(text):

    if pd.isna(text):
        return 0

    total_chars = len(text)

    if total_chars == 0:
        return 0

    capital_chars = sum(
        1 for c in text
        if c.isupper()
    )

    return capital_chars / total_chars

def promo_word_count(text):

    if pd.isna(text):
        return 0

    words = text.split()

    count = 0

    for word in words:

        if word in promo_words:
            count += 1

    return count

def reviewer_review_counts(df):

    reviewer_counts = (
        df['reviewerID']
        .value_counts()
    )

    return df['reviewerID'].map(
        reviewer_counts
    )


def product_review_counts(df):

    product_counts = (
        df['asin']
        .value_counts()
    )

    return df['asin'].map(
        product_counts
    )


def duplicate_review_feature(df):

    return df.duplicated(
        subset=['processed_review'],
        keep=False
    ).astype(int)
