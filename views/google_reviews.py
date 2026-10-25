import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    MissingSecretError,
    load_google_reviews,
    location_filter_controls,
)


STAR_OPTIONS = [5, 4, 3, 2, 1]
RESPONSE_OPTIONS = (
    "All reviews",
    "Waiting on a response",
    "Has a response",
)


def format_range(start, end) -> str:
    """Label a calendar range by its first and last day."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.normalize() == end_ts.normalize():
        return f"{start_ts:%b %d, %Y}"
    if start_ts.year == end_ts.year:
        return f"{start_ts:%b %d} – {end_ts:%b %d, %Y}"
    return f"{start_ts:%b %d, %Y} – {end_ts:%b %d, %Y}"


def selected_bounds(selected_dates):
    """Start and end from a date-input range, once both are set."""

    if isinstance(selected_dates, dt.date):
        return selected_dates, selected_dates

    dates = list(selected_dates)
    if len(dates) < 2:
        return None

    start, end = dates[0], dates[1]
    if end < start:
        start, end = end, start
    return start, end


def average_rating(frame: pd.DataFrame) -> float:
    """Mean star rating. Blank when no review is scored."""

    if frame.empty or "rating" not in frame.columns:
        return float("nan")
    ratings = pd.to_numeric(frame["rating"], errors="coerce").dropna()
    if ratings.empty:
        return float("nan")
    return float(ratings.mean())


def rating_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Review counts for stars 1 through 5, including empty buckets."""

    working = frame.copy()
    working["stars"] = pd.to_numeric(
        working["rating"],
        errors="coerce",
    ).round()
    counted = (
        working.dropna(subset=["stars"])
        .groupby("stars", as_index=False)
        .size()
        .rename(columns={"size": "reviews"})
    )
    buckets = pd.DataFrame({"stars": STAR_OPTIONS})
    counted = buckets.merge(counted, on="stars", how="left")
    counted["reviews"] = counted["reviews"].fillna(0).astype(int)
    counted["stars"] = counted["stars"].astype(int)
    counted["star_label"] = counted["stars"].map(
        lambda stars: "1 star" if stars == 1 else f"{stars} stars"
    )
    return counted.sort_values("stars")


def rating_chart(counts: pd.DataFrame) -> alt.Chart:
    """Vertical bars for the star mix, 1 star on the left."""

    review_max = int(counts["reviews"].max())
    if review_max <= 0:
        review_max = 1
    return (
        alt.Chart(counts)
        .mark_bar(color="#1f77b4")
        .encode(
            x=alt.X(
                "star_label:N",
                title=None,
                sort=counts["star_label"].tolist(),
            ),
            y=alt.Y(
                "reviews:Q",
                title="Reviews",
                axis=alt.Axis(format=",.0f"),
                scale=alt.Scale(domain=[0, review_max * 1.15]),
            ),
            tooltip=[
                alt.Tooltip("star_label:N", title="Rating"),
                alt.Tooltip("reviews:Q", title="Reviews", format=",.0f"),
            ],
        )
        .properties(height=280)
    )


def calendar_dates(series: pd.Series) -> pd.Series:
    """Calendar dates, with a blank where the timestamp is missing."""

    dates = pd.to_datetime(series, errors="coerce")
    return dates.map(
        lambda value: value.date() if pd.notna(value) else None
    )


def table_height(row_count: int) -> int:
    """Fit a short list, then scroll once it passes 16 rows."""

    visible = min(max(row_count, 1), 16)
    return 48 + 36 * visible


def review_column_config() -> dict:
    """Headers for the review list."""

    return {
        "business_date": st.column_config.DateColumn(
            "Posted",
            format="MMM DD, YYYY",
            help="Day the review was posted.",
        ),
        "location": st.column_config.TextColumn("Store"),
        "reviewer_name": st.column_config.TextColumn("Reviewer"),
        "rating": st.column_config.NumberColumn(
            "Stars",
            format="%.0f",
            help="Star rating from 1 to 5.",
        ),
        "likes_count": st.column_config.NumberColumn(
            "Likes",
            format="%.0f",
        ),
        "owner_response": st.column_config.TextColumn(
            "Owner response",
            width="large",
        ),
        "owner_response_date": st.column_config.DateColumn(
            "Responded",
            format="MMM DD, YYYY",
            help="Day the owner response was posted.",
        ),
    }


st.title("Google Reviews")
st.caption(
    "One row per Google review. City in the source table "
    "is the store name."
)

try:
    review_df = load_google_reviews()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Unable to load Google reviews from Supabase: {error}")
    st.stop()

if review_df.empty:
    st.warning("No Google reviews were returned from google_reviews.")
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(review_df)

if location_df.empty:
    st.warning("No Google reviews match that location.")
    st.stop()

available_dates = location_df["business_date"].dropna()
if available_dates.empty:
    st.warning("No Google reviews have a posted date.")
    st.stop()

min_date = available_dates.min().date()
max_date = available_dates.max().date()
default_start = max(min_date, max_date - dt.timedelta(days=6))

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    key=f"google_reviews_range_{selected_location}",
)
selected_stars = st.sidebar.multiselect(
    "Stars",
    options=STAR_OPTIONS,
    default=STAR_OPTIONS,
    help="Limit the summary and the review list to these ratings.",
)
response_choice = st.sidebar.selectbox(
    "Owner response",
    options=list(RESPONSE_OPTIONS),
    help="An owner response is text stored on the review.",
)

bounds = selected_bounds(selected_dates)
if bounds is None:
    st.info("Select a start and end date.")
    st.stop()
if not selected_stars:
    st.info("Select at least one star rating.")
    st.stop()

start_date, end_date = bounds
range_df = location_df[
    location_df["business_date"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
].copy()
if set(selected_stars) != set(STAR_OPTIONS):
    range_df = range_df[
        pd.to_numeric(range_df["rating"], errors="coerce")
        .round()
        .isin(selected_stars)
    ]
has_response = range_df["owner_response"].str.strip().ne("")
if response_choice == "Waiting on a response":
    range_df = range_df.loc[~has_response].copy()
elif response_choice == "Has a response":
    range_df = range_df.loc[has_response].copy()

range_label = format_range(start_date, end_date)
st.subheader(f"{selected_location} · {range_label}")

if range_df.empty:
    st.info("No Google reviews fall in those filters.")
    st.stop()

review_count = len(range_df)
mean_rating = average_rating(range_df)
replied = int(range_df["owner_response"].str.strip().ne("").sum())
response_rate = replied / review_count if review_count else float("nan")
rating_label = (
    f"{mean_rating:.2f}" if pd.notna(mean_rating) else "—"
)

summary = st.columns(4)
summary[0].metric(
    "Reviews",
    f"{review_count:,}",
    help=f"{selected_location} · {range_label}",
)
summary[1].metric(
    "Average rating",
    rating_label,
    help="Average star rating of the reviews in these filters.",
)
summary[2].metric(
    "Owner responses",
    f"{replied:,}",
    help="Reviews in these filters that have an owner response.",
)
summary[3].metric(
    "Response rate",
    f"{response_rate:.0%}" if pd.notna(response_rate) else "—",
    help="Owner responses divided by reviews in these filters.",
)

st.altair_chart(
    rating_chart(rating_counts(range_df)),
    use_container_width=True,
)

display = range_df[
    [
        "business_date",
        "location",
        "reviewer_name",
        "rating",
        "likes_count",
        "owner_response",
        "owner_response_date",
    ]
].sort_values(
    ["business_date", "rating", "reviewer_name"],
    ascending=[False, False, True],
)
display["business_date"] = calendar_dates(display["business_date"])
display["owner_response_date"] = calendar_dates(
    display["owner_response_date"]
)

st.dataframe(
    display,
    column_config=review_column_config(),
    hide_index=True,
    use_container_width=True,
    height=table_height(len(display)),
)
st.caption(
    f"{review_count:,} reviews in {range_label}. "
    "The table is the newest review first."
)
