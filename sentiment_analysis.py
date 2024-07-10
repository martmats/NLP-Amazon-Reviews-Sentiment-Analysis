import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import random

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
# Add TextBlob to spaCy pipeline for sentiment analysis
nlp.add_pipe('spacytextblob')

# Function to clean text data
def clean_text(text):
    doc = nlp(text)
    clean_tokens = [token.text.lower().strip() for token in doc if not token.is_stop and token.text.isalpha()]
    return " ".join(clean_tokens)

# Function for sentiment analysis
def analyze_sentiment(review_text):
    cleaned_text = clean_text(review_text)
    doc = nlp(cleaned_text)
    polarity = doc._.blob.polarity
    if polarity > 0:
        return 'Positive', polarity
    elif polarity < 0:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Load the dataset
df = pd.read_csv('amazon_product_reviews.csv')

# Clean the dataset by removing missing values in 'reviews.text'
clean_data = df.dropna(subset=['reviews.text'])

# Analyze sentiment of the first few reviews
for index, row in clean_data.head().iterrows():
    sentiment, polarity = analyze_sentiment(row['reviews.text'])
    print(f"Review: {row['reviews.text'][:100]}... | Sentiment: {sentiment} | Polarity: {polarity}")

# Compare similarity between two reviews
if len(clean_data) >= 2:
    my_review_of_choice_1 = clean_data['reviews.text'][0]  # First review
    my_review_of_choice_2 = clean_data['reviews.text'][1]  # Second review
    
    # Clean and process the selected reviews
    doc1 = nlp(clean_text(my_review_of_choice_1))
    doc2 = nlp(clean_text(my_review_of_choice_2))
    
    # Calculate similarity
    similarity = doc1.similarity(doc2)
    print(f"Similarity between the selected reviews: {similarity}")
else:
    print("Not enough data in the dataset to perform comparison.")


# Function to test the sentiment analysis model on a random selection of reviews from the dataset
def test_random_reviews(data, num_samples=3):
    sample_reviews = data['reviews.text'].sample(n=num_samples).tolist()
    for review in sample_reviews:
        sentiment, polarity = analyze_sentiment(review)
        print(f"Review: {review[:100]}... | Sentiment: {sentiment} | Polarity: {polarity}")

# Test the model on a random selection of reviews from the dataset
test_random_reviews(clean_data)