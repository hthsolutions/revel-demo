import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    WEEK_START_OPTIONS,
    WEEKDAY_NAMES,
    MissingSecretError,
    add_week_columns,
    load_shift_data,
)


# Hours drawn inside each day. The column always runs
# from 0 on the left to 24 on the right.
DAY_HOUR_SPAN = 24

SEGMENT_ORDER = ["Regular", "Overtime", "Not worked"]
SEGMENT_COLORS = ["#2ca02c", "#d62728", "#e6e6e6"]


def format_week(week_start) -> str:
    """Label a week by its first and last calendar day."""

    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    return f"{start:%b %d} – {end:%b %d, %Y}"


def day_labels(week_start, week_start_weekday: int) -> pd.DataFrame:
    """One row for each day of the selected week, in week order."""

    start = pd.Timestamp(week_start)
    frame = pd.DataFrame({"day_index": range(7)})
    frame["business_date"] = start + pd.to_timedelta(
        frame["day_index"],
        unit="D",
    )
    frame["day_label"] = [
        (
            f"{WEEKDAY_NAMES[(week_start_weekday + index) % 7][:3]}"
            f" {date.day}"
        )
        for index, date in enumerate(frame["business_date"])
    ]
    return frame


st.title("Hourly Employee Hours")
st.caption(
    "One week at a time. Each day runs from 0 to 24 hours. "
    "Green is regular time and red is overtime, using the "
    "split already stored on each shift."
)

try:
    shift_df = load_shift_data()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        f"Unable to load shift data from Supabase: {error} "
        "Confirm SHIFT_TABLE in revel_data.py."
    )
    st.stop()

if shift_df.empty:
    st.warning("No hourly shift records were returned.")
    st.stop()

shift_df = shift_df.copy()
shift_df["employee"] = (
    shift_df["employee"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)
shift_df.loc[shift_df["employee"] == "", "employee"] = "Unknown"


# ---------------------------------------------------------
# Sidebar: one location and one week
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    shift_df["location"].dropna().unique().tolist()
)
selected_location = st.sidebar.selectbox(
    "Location",
    options=["All Locations", *available_locations],
)
selected_week_start_name = st.sidebar.selectbox(
    "Week starts on",
    options=list(WEEK_START_OPTIONS.keys()),
)
week_start_weekday = WEEK_START_OPTIONS[selected_week_start_name]

if selected_location != "All Locations":
    shift_df = shift_df[
        shift_df["location"] == selected_location
    ].copy()

if shift_df.empty:
    st.warning("No hourly shifts match that location.")
    st.stop()

shift_df = add_week_columns(shift_df, week_start_weekday)

week_starts = sorted(
    shift_df["week_start"].dropna().unique(),
    reverse=True,
)
selected_week = st.sidebar.selectbox(
    "Week",
    options=week_starts,
    format_func=format_week,
)

week_df = shift_df[
    shift_df["week_start"] == selected_week
].copy()

if week_df.empty:
    st.info("No hourly shifts fall in that week.")
    st.stop()

daily_df = (
    week_df
    .groupby(["employee", "day_index"], as_index=False)
    .agg(
        regular_hours=("regular_hours", "sum"),
        ot_hours=("ot_hours", "sum"),
    )
)
daily_df["total_hours"] = (
    daily_df["regular_hours"] + daily_df["ot_hours"]
)

employee_totals = (
    daily_df
    .groupby("employee", as_index=False)["total_hours"]
    .sum()
)
employee_order = (
    employee_totals
    .loc[employee_totals["total_hours"] > 0, "employee"]
    .sort_values(key=lambda names: names.str.lower())
    .tolist()
)

if not employee_order:
    st.info("No worked hours were recorded in that week.")
    st.stop()

days_df = day_labels(selected_week, week_start_weekday)

roster = pd.MultiIndex.from_product(
    [employee_order, days_df["day_index"].tolist()],
    names=["employee", "day_index"],
).to_frame(index=False)
roster = roster.merge(days_df, on="day_index", how="left")
roster = roster.merge(
    daily_df,
    on=["employee", "day_index"],
    how="left",
)
roster[["regular_hours", "ot_hours", "total_hours"]] = (
    roster[["regular_hours", "ot_hours", "total_hours"]].fillna(0.0)
)
roster["not_worked"] = (
    DAY_HOUR_SPAN - roster["total_hours"]
).clip(lower=0)
roster["regular_bar"] = roster["regular_hours"]
roster["ot_bar"] = roster["ot_hours"]

hours_long = roster.melt(
    id_vars=[
        "employee",
        "day_label",
        "day_index",
        "business_date",
        "regular_hours",
        "ot_hours",
        "total_hours",
    ],
    value_vars=["regular_bar", "ot_bar", "not_worked"],
    var_name="segment_key",
    value_name="bar_hours",
)
hours_long["segment"] = hours_long["segment_key"].map(
    {
        "regular_bar": "Regular",
        "ot_bar": "Overtime",
        "not_worked": "Not worked",
    }
)
hours_long["segment_order"] = hours_long["segment"].map(
    {name: index for index, name in enumerate(SEGMENT_ORDER)}
)

day_label_order = days_df["day_label"].tolist()
row_height = max(360, 24 * len(employee_order))

hours_chart = (
    alt.Chart(hours_long)
    .mark_bar()
    .encode(
        y=alt.Y(
            "employee:N",
            title=None,
            sort=employee_order,
            axis=alt.Axis(labelLimit=220),
        ),
        x=alt.X(
            "bar_hours:Q",
            title=None,
            stack="zero",
            scale=alt.Scale(domain=[0, DAY_HOUR_SPAN]),
            axis=alt.Axis(
                values=[0, 12, 24],
                labelFlush=True,
            ),
        ),
        color=alt.Color(
            "segment:N",
            title=None,
            sort=SEGMENT_ORDER,
            scale=alt.Scale(
                domain=SEGMENT_ORDER,
                range=SEGMENT_COLORS,
            ),
            legend=alt.Legend(orient="top"),
        ),
        order=alt.Order("segment_order:Q", sort="ascending"),
        column=alt.Column(
            "day_label:N",
            title=None,
            sort=day_label_order,
            header=alt.Header(
                labelOrient="top",
                labelAlign="center",
            ),
            spacing=8,
        ),
        tooltip=[
            alt.Tooltip("employee:N", title="Employee"),
            alt.Tooltip("day_label:N", title="Day"),
            alt.Tooltip(
                "business_date:T",
                title="Date",
                format="%b %d, %Y",
            ),
            alt.Tooltip(
                "regular_hours:Q",
                title="Regular Hours",
                format=".1f",
            ),
            alt.Tooltip(
                "ot_hours:Q",
                title="Overtime Hours",
                format=".1f",
            ),
            alt.Tooltip(
                "total_hours:Q",
                title="Total Hours",
                format=".1f",
            ),
        ],
    )
    .properties(width=128, height=row_height)
)

st.subheader(format_week(selected_week))
st.altair_chart(hours_chart, use_container_width=True)
st.caption(
    f"{len(employee_order):,} hourly employee(s). "
    "A bar fills from the left for the hours worked that "
    "day, green through regular time and red once the "
    "shift is overtime. Gray is the rest of the 24-hour day."
)
