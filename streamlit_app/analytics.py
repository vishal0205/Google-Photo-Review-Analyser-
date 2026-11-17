
from pathlib import Path

import pandas as pd


# Locate labelled_reviews.csv in the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "labelled_reviews.csv"


# Sentiment definitions
GOOD_SENTIMENTS = ["Happy", "Satisfied"]
BAD_SENTIMENTS = ["Negative", "Frustrated", "Angry"]

SENTIMENT_ORDER = [
    "Happy",
    "Satisfied",
    "Neutral",
    "Frustrated",
    "Negative",
    "Angry"
]

CATEGORY_ORDER = [
    "opinion",
    "reflect_rating",
    "bug",
    "feature_request"
]


def load_reviews():
    """Load and prepare the labelled review dataset."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find labelled_reviews.csv at: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "review_id",
        "review_date",
        "rating",
        "review_text",
        "thumbs_up",
        "app_version",
        "developer_replied",
        "sentiment",
        "category",
        "topic"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Clean and prepare fields
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["topic"] = df["topic"].fillna("").astype(str).str.strip()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["thumbs_up"] = pd.to_numeric(
        df["thumbs_up"], errors="coerce"
    ).fillna(0)

    df["review_date"] = pd.to_datetime(
        df["review_date"], errors="coerce"
    )

    # Create broad sentiment groups
    df["broad_sentiment"] = "Unclassified"

    df.loc[
        df["sentiment"].isin(GOOD_SENTIMENTS),
        "broad_sentiment"
    ] = "Good"

    df.loc[
        df["sentiment"].isin(BAD_SENTIMENTS),
        "broad_sentiment"
    ] = "Bad"

    df.loc[
        df["sentiment"] == "Neutral",
        "broad_sentiment"
    ] = "Neutral"

    # Monthly date field for time-series analysis
    df["month"] = df["review_date"].dt.to_period("M").astype(str)

    return df


def get_summary(df):
    """Calculate overall review statistics."""

    total = len(df)

    counts = df["broad_sentiment"].value_counts()

    good = int(counts.get("Good", 0))
    bad = int(counts.get("Bad", 0))
    neutral = int(counts.get("Neutral", 0))

    avg_rating = df["rating"].mean()

    replied = (
        pd.to_numeric(df["developer_replied"], errors="coerce")
        .fillna(0)
    )

    reply_percentage = (
        (replied.eq(1).sum() / total) * 100 if total else 0
    )

    return {
        "total": total,
        "good": good,
        "bad": bad,
        "neutral": neutral,
        "good_pct": good / total * 100 if total else 0,
        "bad_pct": bad / total * 100 if total else 0,
        "neutral_pct": neutral / total * 100 if total else 0,
        "avg_rating": avg_rating if pd.notna(avg_rating) else 0,
        "reply_pct": reply_percentage
    }


def get_sentiment_breakdown(df):
    return (
        df["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )


def get_category_breakdown(df):
    return (
        df["category"]
        .value_counts()
        .reindex(CATEGORY_ORDER, fill_value=0)
    )


def get_topic_breakdown(df):
    """Rank topics by review count and split by broad sentiment."""

    result = pd.crosstab(
        df["topic"],
        df["broad_sentiment"]
    )

    for sentiment in ["Good", "Bad", "Neutral"]:
        if sentiment not in result.columns:
            result[sentiment] = 0

    result["Total"] = result[["Good", "Bad", "Neutral"]].sum(axis=1)

    return result.sort_values("Total", ascending=False)


def get_top_complaints(df, limit=5, examples_per_topic=3):
    """Find the most complained-about topics and example reviews."""

    bad_reviews = df[df["broad_sentiment"] == "Bad"].copy()

    topic_counts = (
        bad_reviews.groupby("topic")
        .size()
        .sort_values(ascending=False)
        .head(limit)
    )

    complaints = []

    for topic, count in topic_counts.items():

        examples = (
            bad_reviews[bad_reviews["topic"] == topic]
            .sort_values(
                ["thumbs_up", "rating"],
                ascending=[False, True]
            )
            .head(examples_per_topic)
        )

        complaints.append({
            "topic": topic,
            "count": int(count),
            "examples": examples
        })

    return complaints


def get_monthly_trends(df):
    """Calculate monthly review counts by broad sentiment."""

    valid_dates = df.dropna(subset=["review_date"]).copy()

    monthly = (
        valid_dates.groupby(["month", "broad_sentiment"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    for sentiment in ["Good", "Bad", "Neutral"]:
        if sentiment not in monthly.columns:
            monthly[sentiment] = 0

    return monthly[["Good", "Bad", "Neutral"]]


def get_rating_sentiment_matrix(df):
    """Cross-tabulate star ratings against the six sentiment levels."""

    return pd.crosstab(
        df["rating"],
        df["sentiment"]
    ).reindex(
        index=[5, 4, 3, 2, 1],
        columns=SENTIMENT_ORDER,
        fill_value=0
    )
