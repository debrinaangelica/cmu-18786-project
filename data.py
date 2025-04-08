from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def get_tweet_dataset(filename):
    tweets = pd.read_csv(filename)
    tweets = tweets[["date", "text"]]
    tweets = _append_sentiment_analysis(tweets)
    return tweets

def _append_sentiment_analysis(dataset):
    analyzer = SentimentIntensityAnalyzer(lexicon_file="data/sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt")
    for t in dataset["text"]:
        # Hashtags are not taken into consideration by analyzer.polarity_scores
        t = t.replace("#", "")
        dataset["scores_pos"] = analyzer.polarity_scores(t)["pos"]
        dataset["scores_neg"] = analyzer.polarity_scores(t)["neg"]
        dataset["scores_neu"] = analyzer.polarity_scores(t)["neu"]
        dataset["scores_compound"] = analyzer.polarity_scores(t)["compound"]
    return dataset

def main():
    tweets = get_tweet_dataset("data/tweets/elon_musk_tweets.csv")
    print(tweets.head())

if __name__ == "__main__":
    main()