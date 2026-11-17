
import os
import re

import pandas as pd
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# SENTIMENT DEFINITIONS
# --------------------------------------------------

GOOD_SENTIMENTS = ["Happy", "Satisfied"]

BAD_SENTIMENTS = ["Negative", "Frustrated", "Angry"]

NEUTRAL_SENTIMENTS = ["Neutral"]


# --------------------------------------------------
# TOPIC KEYWORD MAPPING
# --------------------------------------------------

TOPIC_KEYWORDS = {
    "Backup_Sync": [
        "backup", "sync", "synchronization", "upload",
        "backing up", "cloud backup", "photos not uploading"
    ],
    "Editing": [
        "edit", "editing", "editor", "filters",
        "crop", "enhance", "photo editing"
    ],
    "Sharing_Albums": [
        "sharing", "share", "album", "shared album",
        "collaborate"
    ],
    "UI_Design": [
        "ui", "interface", "design", "layout",
        "navigation", "user experience", "buttons"
    ],
    "Search": [
        "search", "find photos", "searching",
        "search results"
    ],
    "Memories": [
        "memories", "memory", "highlights",
        "recollections"
    ],
    "Storage_Plans": [
        "storage", "space", "subscription",
        "google one", "storage plan", "payment"
    ],
    "App_Crash": [
        "crash", "crashing", "freeze",
        "freezing", "not opening", "force close"
    ],
    "Performance": [
        "slow", "lag", "performance",
        "loading", "battery", "responsive"
    ],
    "Privacy_Security": [
        "privacy", "security", "permission",
        "personal data", "account security"
    ],
    "Device_Support": [
        "device", "phone", "tablet",
        "compatibility", "android"
    ],
    "AI_Features": [
        "ai", "artificial intelligence", "magic editor",
        "magic eraser", "ai features"
    ]
}


# --------------------------------------------------
# DATASET STATISTICS
# --------------------------------------------------

def get_dataset_statistics(df):
    """
    Calculate exact statistics from the complete labelled dataset.
    Does not use the retrieved review subset.
    """

    total = len(df)

    sentiment = df["sentiment"].astype(str).str.strip().str.lower()

    good = sentiment.isin(
        [s.lower() for s in GOOD_SENTIMENTS]
    ).sum()

    bad = sentiment.isin(
        [s.lower() for s in BAD_SENTIMENTS]
    ).sum()

    neutral = sentiment.isin(
        [s.lower() for s in NEUTRAL_SENTIMENTS]
    ).sum()

    return {
        "total": total,
        "good": int(good),
        "bad": int(bad),
        "neutral": int(neutral),
        "good_pct": good / total * 100 if total else 0,
        "bad_pct": bad / total * 100 if total else 0,
        "neutral_pct": neutral / total * 100 if total else 0
    }


# --------------------------------------------------
# EXACT NUMERICAL QUESTION HANDLER
# --------------------------------------------------

def answer_numerical_question(query, df):
    """
    Handle dataset-level numerical questions using Pandas.
    Returns None when the question requires Gemini analysis.
    """

    q = query.lower().strip()

    stats = get_dataset_statistics(df)

    # Overall review count
    if re.search(r"\b(how many|total number|count of)\b", q):

        if re.search(r"\b(good|positive)\b", q):
            return (
                f"Out of {stats['total']:,} labelled Google Photos reviews, "
                f"**{stats['good']:,} were classified as Good "
                f"({stats['good_pct']:.1f}%).**\n\n"
                f"Good reviews combine the Happy and Satisfied "
                f"sentiment levels.\n\n"
                f"For comparison:\n"
                f"- Good: {stats['good']:,} ({stats['good_pct']:.1f}%)\n"
                f"- Bad: {stats['bad']:,} ({stats['bad_pct']:.1f}%)\n"
                f"- Neutral: {stats['neutral']:,} "
                f"({stats['neutral_pct']:.1f}%)"
            )

        if re.search(r"\b(bad|negative)\b", q):
            return (
                f"Out of {stats['total']:,} labelled reviews, "
                f"**{stats['bad']:,} were classified as Bad "
                f"({stats['bad_pct']:.1f}%).**\n\n"
                f"Bad reviews combine Negative, Frustrated and Angry."
            )

        if re.search(r"\bneutral\b", q):
            return (
                f"Out of {stats['total']:,} labelled reviews, "
                f"**{stats['neutral']:,} were classified as Neutral "
                f"({stats['neutral_pct']:.1f}%).**"
            )

        if re.search(r"\b(review|reviews|user|users)\b", q):
            return (
                f"The labelled dataset contains "
                f"**{stats['total']:,} reviews**."
            )

    # Percentage questions
    if re.search(r"\b(percentage|percent|proportion|what percentage)\b", q):

        if re.search(r"\b(good|positive)\b", q):
            return (
                f"**{stats['good_pct']:.1f}%** of labelled reviews "
                f"were Good ({stats['good']:,} out of "
                f"{stats['total']:,})."
            )

        if re.search(r"\b(bad|negative)\b", q):
            return (
                f"**{stats['bad_pct']:.1f}%** of labelled reviews "
                f"were Bad ({stats['bad']:,} out of "
                f"{stats['total']:,})."
            )

        if re.search(r"\bneutral\b", q):
            return (
                f"**{stats['neutral_pct']:.1f}%** of labelled reviews "
                f"were Neutral ({stats['neutral']:,} out of "
                f"{stats['total']:,})."
            )

    return None


# --------------------------------------------------
# TOPIC DETECTION
# --------------------------------------------------

def detect_topics(query):
    """Identify topics explicitly mentioned in a question."""

    query_lower = query.lower()
    matched_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_topics.append(topic)

    return matched_topics


# --------------------------------------------------
# REVIEW RETRIEVAL
# --------------------------------------------------

def retrieve_reviews(query, df, top_k=15):
    """
    Retrieve relevant reviews from the labelled dataset.
    Uses topic matching and review-text keyword matching.
    """

    query_lower = query.lower().strip()

    matched_topics = detect_topics(query)

    if matched_topics:
        candidates = df[
            df["topic"].isin(matched_topics)
        ].copy()
    else:
        candidates = df.copy()

    stop_words = {
        "what", "why", "how", "does", "do", "the",
        "is", "are", "of", "to", "for", "in", "on",
        "with", "about", "users", "user", "app",
        "google", "photos", "show", "me", "tell",
        "give", "find", "reviews", "review", "issue",
        "issues", "problem", "problems", "complaint",
        "complaints", "and", "or", "a", "an", "this",
        "that", "there", "their", "they", "can"
    }

    terms = [
        word for word in re.findall(
            r"\b[a-zA-Z]{3,}\b", query_lower
        )
        if word not in stop_words
    ]

    if not matched_topics and not terms:
        return pd.DataFrame()

    text = candidates["review_text"].fillna("").str.lower()

    scores = pd.Series(
        0,
        index=candidates.index,
        dtype=int
    )

    for term in terms:
        scores += text.str.contains(
            re.escape(term),
            regex=True,
            na=False
        ).astype(int)

    if matched_topics:
        scores += 3

    candidates["relevance_score"] = scores

    candidates = candidates[
        candidates["relevance_score"] > 0
    ]

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["relevance_score", "thumbs_up"],
        ascending=[False, False]
    )

    return candidates.head(top_k)


# --------------------------------------------------
# GEMINI RESPONSE GENERATION
# --------------------------------------------------

def generate_answer(query, reviews, df):
    """
    Numerical questions use complete dataset statistics.
    Descriptive questions use Gemini and retrieved reviews.
    """

    # First: check whether this is an exact numerical question
    numerical_answer = answer_numerical_question(
        query,
        df
    )

    if numerical_answer is not None:
        return numerical_answer

    # Otherwise: use retrieved reviews for qualitative analysis
    if reviews.empty:
        return (
            "I couldn't find sufficiently relevant reviews "
            "in the labelled dataset for this question. "
            "Try asking about a specific app feature or issue."
        )

    review_context = []

    for _, row in reviews.iterrows():

        review_context.append(
            f"""
Review ID: {row['review_id']}
Rating: {row['rating']} stars
Sentiment: {row['sentiment']}
Category: {row['category']}
Topic: {row['topic']}
Review: {row['review_text']}
Helpful votes: {row['thumbs_up']}
"""
        )

    context = "\n".join(review_context)

    prompt = f"""
You are the Google Photos Review Intelligence Assistant.

Your task is to analyse Google Photos user feedback.

USER QUESTION:
{query}

RETRIEVED REVIEWS:
{context}

IMPORTANT DATASET INFORMATION:

- The complete labelled dataset contains {len(df)} reviews.
- The retrieved reviews above are only a subset.
- Do not use the retrieved subset to calculate
  dataset-wide review counts or percentages.

INSTRUCTIONS:

1. Analyse only the supplied reviews for qualitative findings.
2. Identify recurring problems, positive experiences,
   feature requests or opinions where relevant.
3. Explain findings in clear, simple language.
4. Distinguish actual user feedback from your interpretation.
5. Include actual review IDs when giving examples.
6. Never invent a review, review ID, rating or statistic.
7. Do not claim the retrieved reviews represent all
   7,687 original reviews.
8. If evidence is insufficient, clearly say so.
9. Do not estimate dataset-wide counts from retrieved reviews.
10. For questions requiring exact dataset-wide counts,
    explain that the full dataset statistics should be used.

Organise your answer using headings and bullet points.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
