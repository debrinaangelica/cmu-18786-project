import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Read in the data
df = pd.read_csv('data/tweets/elon_musk_tweets.csv')
df.head()

# Initialize VADER Sentiment Intensity Analyzer
analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text):
    # Use VADER to get sentiment scores
    sentiment_dict = analyzer.polarity_scores(text)
    
    # VADER returns a dictionary with four values: negative, neutral, positive, and compound.
    # We use the 'compound' score as a general sentiment score
    score = sentiment_dict['compound']
    pos = sentiment_dict['pos']
    neg = sentiment_dict['neg']
    neu = sentiment_dict['neu']
    
    return score, pos, neg, neu

# Run sentiment analysis
df["sentiment_result"] = df["Text"].apply(lambda x: analyzer.polarity_scores(x))
df["sentiment"] = df["sentiment_result"].apply(lambda x: 'positive' if x['compound'] > 0 else ('neutral' if x['compound'] == 0 else 'negative'))
df["confidence"] = df["sentiment_result"].apply(lambda x: x['compound'])

# Run sentiment scoring
scores = []
neg_probs = []
neu_probs = []
pos_probs = []

for tweet in tqdm(df["Text"], desc="Analyzing tweets"):
    score, pos, neg, neu = sentiment_score(tweet)
    scores.append(score)
    neg_probs.append(neg)
    neu_probs.append(neu)
    pos_probs.append(pos)

df["sentiment_score"] = scores
df["prob_negative"] = neg_probs
df["prob_neutral"] = neu_probs
df["prob_positive"] = pos_probs

# Save result
df.to_csv("data/sentiment/tweets_with_vader_sentiment.csv", index=False)
print("Saved to data/sentiment/tweets_with_vader_sentiment.csv")

# See sentiment distribution
print(df["sentiment"].value_counts())

# Get daily summary csv
df["Datetime"] = pd.to_datetime(df["Datetime"])

df["day"] = df["Datetime"].dt.date

# Group by day and compute the required stats
daily_summary = df.groupby("day").agg({
    "sentiment_score": "mean",
    "prob_negative": "mean",
    "prob_neutral": "mean",
    "prob_positive": "mean",
    "Text": "count" # number of tweets
}).reset_index()

daily_summary = daily_summary.rename(columns={"Text": "tweet_count"})
daily_summary.to_csv("daily_sentiment_summary.csv", index=False)

print("Saved daily summary to daily_sentiment_summary.csv")
