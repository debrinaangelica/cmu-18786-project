import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
from tqdm import tqdm

df = pd.read_csv('elon_musk_tweets.csv')
df.head()


# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Create sentiment analysis pipeline
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def directional_score(text):
    try:
        result = finbert(text[:512])[0]  # Truncate long tweets
        label = result["label"]
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0
    except:
        return None

def sentiment_score(text):
  # Bypass finbert pipeline thing
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
  model.to(device)

  with torch.no_grad():
      logits = model(**inputs).logits
      probs = F.softmax(logits, dim=1).squeeze().tolist()

  sentiment_score = -1 * probs[1] + 0 * probs[2] + 1 * probs[0]
  return sentiment_score, probs[0], probs[1], probs[2]

# Run sentiment analysis
df["sentiment_result"] = df["text"].apply(lambda x: finbert(x)[0])
df["sentiment"] = df["sentiment_result"].apply(lambda x: x['label'])
df["confidence"] = df["sentiment_result"].apply(lambda x: x['score'])

# Run sentiment scoring
scores = []
neg_probs = []
neu_probs = []
pos_probs = []

for tweet in tqdm(df["text"], desc="Analyzing tweets"):
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
df.to_csv("tweets_with_finbert_sentiment.csv", index=False)
print("Saved to tweets_with_finbert_sentiment.csv")

# See sentiment distribution
print(df["sentiment"].value_counts())

# Get daily summary csv
df["date"] = pd.to_datetime(df["date"])

df["day"] = df["date"].dt.date

# Group by day and compute the required stats
daily_summary = df.groupby("day").agg({
    "sentiment_score": "mean",
    "prob_negative": "mean",
    "prob_neutral": "mean",
    "prob_positive": "mean",
    "text": "count" # number of tweets
}).reset_index()

daily_summary = daily_summary.rename(columns={"text": "tweet_count"})
daily_summary.to_csv("daily_sentiment_summary.csv", index=False)

print("Saved daily summary to daily_sentiment_summary.csv")
