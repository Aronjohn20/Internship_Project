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

    return text.count('!')

def capital_ratio(text):

    total_chars = len(text)

    if total_chars == 0:
        return 0

    capital_chars = sum(1 for c in text if c.isupper())

    return capital_chars / total_chars

def promo_word_count(text):
   
    words = text.split()

    count = 0

    for word in words:

        if word in promo_words:
            count += 1

    return count

#where does the text appear from? Is it from the review text or the processed review text? I will assume it's from the processed review text for now, but this can be adjusted as needed.
#Also in similarity.py , they are asking for df and sample df where we get that from, will need to sovle the problem
