
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from chatbot import (
    retrieve_reviews,
    generate_answer,
    answer_numerical_question
)

from analytics import (
    load_reviews,
    get_summary,
    get_sentiment_breakdown,
    get_category_breakdown,
    get_topic_breakdown,
    get_top_complaints,
    get_monthly_trends,
    get_rating_sentiment_matrix,
    SENTIMENT_ORDER
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Google Photos Review ",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Google Photos Review Intelligence")

st.caption(
    "Interactive sentiment, topic, complaint and time-series "
    "analytics powered by LLM-labelled reviews."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def get_data():
    return load_reviews()


try:
    df = get_data()

except Exception as e:
    st.error(f"Unable to load review dataset: {e}")
    st.stop()


# ==================================================
# GEMINI-STYLE RIGHT-SIDE AI CHAT DRAWER
# ==================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def process_ai_question(question):

    st.session_state.chat_messages.append({
        "role": "user",
        "content": question
    })

    try:
        with st.spinner("Searching reviews and analysing feedback..."):

            relevant_reviews = retrieve_reviews(
                question,
                df,
                top_k=15
            )

            answer = generate_answer(
                        question,
                        relevant_reviews,
                        df
                    )
            # Check whether this is a numerical dataset-level question
            is_numerical = answer_numerical_question(
                                question,
                                df
                                ) is not None

# Numerical answers don't need supporting review cards
            if is_numerical:
                source_records = []

            else:
                source_records = (
                    relevant_reviews[
                [
                "review_id",
                "review_text",
                "rating",
                "sentiment",
                "category",
                "topic",
                "thumbs_up"
            ]
        ].to_dict("records")
        if not relevant_reviews.empty
        else []
    )
            
            
            
  
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_records
            })

    except Exception as e:

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"Sorry, something went wrong: {str(e)}",
            "sources": []
        })


@st.dialog(
    "🤖 Google Photos AI Assistant",
    width=420,
    position="right",
    dismissible=True
)
def show_ai_chat():

    st.caption(
        "Ask questions about Google Photos issues, "
        "user complaints and feature requests."
    )

    # Suggested questions
    st.markdown("### 💡 Suggested questions")

    col1, col2 = st.columns(2)

    with col1:
        common_problem = st.button(
            "⚠️ Common Problems",
            use_container_width=True
        )

    with col2:
        backup_problem = st.button(
            "☁️ Backup & Sync",
            use_container_width=True
        )

    if common_problem:
        process_ai_question(
            "What are the biggest problems users face "
            "with Google Photos?"
        )
        st.rerun(scope="fragment")

    if backup_problem:
        process_ai_question(
            "Why are users complaining about Backup and Sync?"
        )
        st.rerun(scope="fragment")

    st.divider()

    # Conversation history
    st.markdown("### Conversation")

    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    f"📄 View {len(message['sources'])} supporting reviews"
                ):

                    for review in message["sources"]:

                        st.markdown(
                            f"**{review['topic']} | "
                            f"{review['sentiment']} | "
                            f"{review['rating']} ★**"
                        )

                        st.write(review["review_text"])

                        st.caption(
                            f"Review ID: {review['review_id']} "
                            f"| Helpful votes: {review['thumbs_up']}"
                        )

                        st.divider()

    # Question input
    with st.form("right_drawer_chat_form", clear_on_submit=True):

        question = st.text_input(
            "Ask your question",
            placeholder="What are users saying about photo editing?"
        )

        submitted = st.form_submit_button(
            "➤ Send",
            use_container_width=True
        )

    if submitted and question.strip():

        process_ai_question(question.strip())
        st.rerun(scope="fragment")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.chat_messages = []
        st.rerun(scope="fragment")


# --------------------------------------------------
# CHAT LAUNCHER
# --------------------------------------------------

if st.button(
    "💬 Ask Google Photos AI Assistant",
    key="launch_google_photos_chat",
    type="primary"
):
    show_ai_chat()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔎 Dashboard Filters")


# Reset filters
if st.sidebar.button("Reset Filters"):

    st.session_state["broad_filter"] = "All"
    st.session_state["sentiment_filter"] = "All"
    st.session_state["category_filter"] = "All"
    st.session_state["topic_filter"] = "All"
    st.session_state["rating_filter"] = "All"
    st.session_state["search_query"] = ""

    st.rerun()


broad_options = ["All"] + sorted(
    df["broad_sentiment"].dropna().unique().tolist()
)

sentiment_options = ["All"] + SENTIMENT_ORDER

category_options = ["All"] + sorted(
    df["category"].dropna().unique().tolist()
)

topic_options = ["All"] + sorted(
    df["topic"].dropna().unique().tolist()
)

rating_options = ["All", 5, 4, 3, 2, 1]


selected_broad = st.sidebar.selectbox(
    "Broad Sentiment",
    broad_options,
    key="broad_filter"
)

selected_sentiment = st.sidebar.selectbox(
    "Granular Sentiment",
    sentiment_options,
    key="sentiment_filter"
)

selected_category = st.sidebar.selectbox(
    "Category",
    category_options,
    key="category_filter"
)

selected_topic = st.sidebar.selectbox(
    "Topic",
    topic_options,
    key="topic_filter"
)

selected_rating = st.sidebar.selectbox(
    "Star Rating",
    rating_options,
    key="rating_filter"
)


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df.copy()


if selected_broad != "All":

    filtered_df = filtered_df[
        filtered_df["broad_sentiment"] == selected_broad
    ]


if selected_sentiment != "All":

    filtered_df = filtered_df[
        filtered_df["sentiment"] == selected_sentiment
    ]


if selected_category != "All":

    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]


if selected_topic != "All":

    filtered_df = filtered_df[
        filtered_df["topic"] == selected_topic
    ]


if selected_rating != "All":

    filtered_df = filtered_df[
        filtered_df["rating"] == selected_rating
    ]


# --------------------------------------------------
# SEARCH REVIEWS
# --------------------------------------------------

search_query = st.sidebar.text_input(
    "Search Review Text, Version or ID",
    key="search_query"
)


if search_query:

    query = search_query.strip().lower()

    mask = (
        filtered_df["review_text"].str.lower().str.contains(
            query,
            na=False,
            regex=False
        )
        |
        filtered_df["review_id"].astype(str).str.lower().str.contains(
            query,
            na=False,
            regex=False
        )
        |
        filtered_df["app_version"].astype(str).str.lower().str.contains(
            query,
            na=False,
            regex=False
        )
    )

    filtered_df = filtered_df[mask]


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

summary = get_summary(filtered_df)

st.subheader("📊 Review Overview")

st.caption(
    f"Showing analytics for {len(filtered_df):,} matching reviews "
    f"out of {len(df):,} total labelled reviews."
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Reviews Analysed",
    f"{len(df):,}"
)

col2.metric(
    "Good Reviews",
    f"{summary['good']:,}",
    f"{summary['good_pct']:.1f}%"
)

col3.metric(
    "Bad Reviews",
    f"{summary['bad']:,}",
    f"{summary['bad_pct']:.1f}%",
    delta_color="inverse"
)

col4.metric(
    "Neutral Reviews",
    f"{summary['neutral']:,}",
    f"{summary['neutral_pct']:.1f}%"
)


st.caption(
    f"Average Rating: {summary['avg_rating']:.2f} ★"
    f"  |  Developer Replied: {summary['reply_pct']:.1f}%"
)


st.divider()


# --------------------------------------------------
# SENTIMENT AND CATEGORY BREAKDOWN
# --------------------------------------------------

st.subheader("📈 Sentiment and Category Analysis")

col1, col2 = st.columns(2)


sentiment_counts = get_sentiment_breakdown(filtered_df)


with col1:

    st.markdown("#### Six-Level Sentiment Breakdown")

    sentiment_fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={
            "x": "Sentiment",
            "y": "Number of Reviews"
        },
        color=sentiment_counts.index,
        color_discrete_map={
            "Happy": "#10b981",
            "Satisfied": "#34d399",
            "Neutral": "#fbbf24",
            "Frustrated": "#fb923c",
            "Negative": "#f43f5e",
            "Angry": "#be123c"
        }
    )

    sentiment_fig.update_layout(
        showlegend=False,
        template="plotly_dark"
    )

    st.plotly_chart(
        sentiment_fig,
        use_container_width=True
    )


category_counts = get_category_breakdown(filtered_df)


with col2:

    st.markdown("#### Four-Category Breakdown")

    category_fig = px.pie(
        names=category_counts.index,
        values=category_counts.values,
        hole=0.45,
        color_discrete_sequence=[
            "#38bdf8",
            "#818cf8",
            "#f43f5e",
            "#f59e0b"
        ]
    )

    category_fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        category_fig,
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# TOPIC BREAKDOWN
# --------------------------------------------------

st.subheader("📌 Topics Ranked by Review Volume")

st.caption(
    "Compare Good, Bad and Neutral reviews across topics."
)


topic_data = get_topic_breakdown(filtered_df)


if not topic_data.empty:

    topic_fig = go.Figure()

    for sentiment, color in [
        ("Good", "#10b981"),
        ("Bad", "#f43f5e"),
        ("Neutral", "#fbbf24")
    ]:

        topic_fig.add_trace(
            go.Bar(
                name=sentiment,
                y=topic_data.index,
                x=topic_data[sentiment],
                orientation="h",
                marker_color=color
            )
        )


    topic_fig.update_layout(
        barmode="stack",
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Number of Reviews",
        template="plotly_dark",
        legend_title="Sentiment"
    )

    st.plotly_chart(
        topic_fig,
        use_container_width=True
    )

else:

    st.info(
        "No topic data available for the selected filters."
    )


st.divider()


# --------------------------------------------------
# REVIEWS OVER TIME
# --------------------------------------------------

st.subheader("📅 Reviews Over Time by Sentiment")


time_scope = st.radio(
    "Time Period",
    ["All Time", "2026 Only"],
    horizontal=True
)


monthly_data = get_monthly_trends(filtered_df)


if time_scope == "2026 Only":

    monthly_data = monthly_data[
        monthly_data.index.str.startswith("2026")
    ]


if not monthly_data.empty:

    time_fig = go.Figure()

    for sentiment, color in [
        ("Good", "#10b981"),
        ("Bad", "#f43f5e"),
        ("Neutral", "#fbbf24")
    ]:

        time_fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data[sentiment],
                mode="lines+markers",
                name=sentiment,
                line=dict(color=color, width=3)
            )
        )


    time_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Reviews",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(
        time_fig,
        use_container_width=True
    )

else:

    st.info(
        "No monthly review data available."
    )


st.divider()


# --------------------------------------------------
# TOP FIVE COMPLAINT TOPICS
# --------------------------------------------------

st.subheader("⚠️ Top Five Complaint Topics")

st.caption(
    "Ranked by number of Bad reviews. Example reviews are "
    "selected by helpfulness and lower star rating."
)


complaints = get_top_complaints(filtered_df)


if complaints:

    for index, complaint in enumerate(
        complaints,
        start=1
    ):

        with st.expander(
            f"#{index} {complaint['topic']} — "
            f"{complaint['count']} Bad Reviews",
            expanded=True
        ):

            for _, review in complaint["examples"].iterrows():

                st.markdown(
                    f"**{review['sentiment']}** "
                    f" | {review['rating']} ★ "
                    f" | 👍 {int(review['thumbs_up'])}"
                )

                st.write(review["review_text"])

                st.caption(
                    f"Review ID: {review['review_id']} "
                    f"| Date: {review['review_date']}"
                )

                st.divider()

else:

    st.info(
        "No complaint examples match the selected filters."
    )


st.divider()


# --------------------------------------------------
# RATING VS SENTIMENT MATRIX
# --------------------------------------------------

st.subheader("⭐ Rating vs Sentiment Matrix")


matrix = get_rating_sentiment_matrix(filtered_df)


st.dataframe(
    matrix,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# REVIEW EXPLORER
# --------------------------------------------------

st.subheader("🔍 Review Explorer")


st.caption(
    f"Showing {min(len(filtered_df), 100)} of "
    f"{len(filtered_df):,} matching reviews."
)


display_columns = [
    "review_id",
    "review_date",
    "rating",
    "sentiment",
    "category",
    "topic",
    "thumbs_up",
    "review_text"
]


st.dataframe(
    filtered_df[display_columns].head(100),
    use_container_width=True,
    hide_index=True
)


st.download_button(
    "⬇️ Download Filtered Reviews",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_google_photos_reviews.csv",
    mime="text/csv"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Google Photos Review Intelligence | "
    "Powered by LLM-labelled user reviews"
)
