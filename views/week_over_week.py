from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    NOT_AVAILABLE,
    SALES_NUMERIC_COLUMNS,
    WEEK_START_OPTIONS,
    WEEKDAY_NAMES,
    MissingSecretError,
    add_week_columns,
    format_metric_value,
    load_sales_data,
    location_filter_controls,
    percent_change,
    reindex_weekly,
)


# Revenue channels reported alongside the selected metric.
CHANNEL_COLUMNS = {
    "drive_through": "Drive Through",
    "eat_in": "Eat In",
    "to_go": "To Go",
    "pickup": "Pickup",
    "dd_marketplace": "DoorDash Marketplace",
    "uber_eats": "Uber Eats",
}

# A metric with a denominator is a ratio of two summed
# columns. A metric without one is a straight sum.
METRIC_OPTIONS = {
    "Net Sales": {
        "numerator": "net_sales",
        "denominator": None,
        "value_format": "currency",
    },
    "Gross Sales": {
        "numerator": "gross_sales",
        "denominator": None,
        "value_format": "currency",
    },
    "Total Orders": {
        "numerator": "total_orders",
        "denominator": None,
        "value_format": "count",
    },
    "Average Ticket": {
        "numerator": "net_sales",
        "denominator": "total_orders",
        "value_format": "currency",
    },
    "Total Discounts": {
        "numerator": "total_discounts",
        "denominator": None,
        "value_format": "currency",
    },
}


def compute_metric_series(
    frame: pd.DataFrame,
    metric: dict,
) -> pd.Series:
    """Derive metric values from summed source columns."""

    numerator = frame[metric["numerator"]].astype(float)

    if metric["denominator"] is None:
        return numerator

    # A week with no orders has no meaningful ratio, so it
    # stays blank instead of dividing by zero.
    denominator = (
        frame[metric["denominator"]]
        .astype(float)
        .replace(0.0, NOT_AVAILABLE)
    )

    return numerator / denominator


st.title("Revel Week-over-Week Dashboard")
st.caption(
    "Weekly sales performance extracted from Revel and "
    "stored in Supabase."
)

try:
    sales_df = load_sales_data()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Unable to load data from Supabase: {error}")
    st.stop()

if sales_df.empty:
    st.warning(
        "No sales records were returned from Supabase. "
        "Check the table name, credentials, and Supabase "
        "Row Level Security settings."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(
    sales_df
)

selected_metric_name = st.sidebar.selectbox(
    "Metric",
    options=list(METRIC_OPTIONS.keys()),
)

selected_metric = METRIC_OPTIONS[selected_metric_name]

selected_week_start_name = st.sidebar.selectbox(
    "Week starts on",
    options=list(WEEK_START_OPTIONS.keys()),
)

week_start_weekday = WEEK_START_OPTIONS[
    selected_week_start_name
]

weeks_to_display = st.sidebar.slider(
    "Weeks to display",
    min_value=2,
    max_value=26,
    value=8,
)

match_partial_week = st.sidebar.checkbox(
    "Compare week to date",
    value=True,
    help=(
        "Trim every week to the same number of days as the "
        "most recent week so a partial week is not compared "
        "with a full one."
    ),
)

if location_df.empty:
    st.warning("No sales data matches the selected filters.")
    st.stop()


# ---------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------

location_df = add_week_columns(
    location_df,
    week_start_weekday,
)

latest_week_start = location_df["week_start"].max()

latest_week_day_index = int(
    location_df.loc[
        location_df["week_start"] == latest_week_start,
        "day_index",
    ].max()
)

# Streamlit Community Cloud runs in UTC. Use Central Time so
# "in progress" follows the restaurant's local business date.
central_today = datetime.now(
    ZoneInfo("America/Chicago")
).date()

latest_week_end = latest_week_start.date() + timedelta(days=6)
latest_week_is_partial = latest_week_day_index < 6

comparison_df = location_df.copy()
week_to_date_applied = (
    match_partial_week and latest_week_is_partial
)

if week_to_date_applied:
    comparison_df = comparison_df[
        comparison_df["day_index"] <= latest_week_day_index
    ].copy()

weekly_df = (
    comparison_df
    .groupby("week_start", as_index=False)
    .agg(
        **{
            column_name: (column_name, "sum")
            for column_name in SALES_NUMERIC_COLUMNS
        },
        days_counted=("business_date", "nunique"),
    )
    .sort_values("week_start")
    .reset_index(drop=True)
)

weekly_df = reindex_weekly(
    weekly_df,
    location_df["week_start"],
)

weekly_df["days_counted"] = (
    weekly_df["days_counted"].fillna(0).astype(int)
)

weekly_df["metric_value"] = compute_metric_series(
    weekly_df,
    selected_metric,
)

weekly_df["previous_metric_value"] = (
    weekly_df["metric_value"].shift(1)
)

weekly_df["metric_change"] = (
    weekly_df["metric_value"]
    - weekly_df["previous_metric_value"]
)

weekly_df["metric_growth"] = (
    (
        weekly_df["metric_change"]
        / weekly_df["previous_metric_value"].abs()
    )
    * 100
).replace(
    [float("inf"), float("-inf")],
    NOT_AVAILABLE,
)

weekly_df["week_label"] = weekly_df["week_start"].dt.strftime(
    "%b %d"
)

# Weeks missing days have smaller totals for reasons that
# have nothing to do with performance, so they are flagged.
expected_days_per_week = (
    latest_week_day_index + 1 if week_to_date_applied else 7
)

weekly_df["is_complete_week"] = (
    weekly_df["days_counted"] >= expected_days_per_week
)

# Only the most recent weeks are charted and tabulated.
visible_weeks_df = weekly_df.tail(weeks_to_display).copy()

week_label_order = visible_weeks_df["week_label"].tolist()

current_week = weekly_df.iloc[-1]

previous_week = (
    weekly_df.iloc[-2] if len(weekly_df) >= 2 else None
)

metric_value_format = selected_metric["value_format"]

if metric_value_format == "currency":
    metric_axis_format = "$,.0f"
    metric_tooltip_format = "$,.2f"
else:
    metric_axis_format = ",.0f"
    metric_tooltip_format = ",.0f"


# ---------------------------------------------------------
# Week-over-week summary
# ---------------------------------------------------------

current_week_label = (
    f"{current_week['week_start']:%b %d}–"
    f"{current_week['week_start'] + timedelta(days=6):%b %d, %Y}"
)

st.subheader(f"{selected_metric_name}: Week Over Week")

current_week_growth = percent_change(
    current_week["metric_value"],
    current_week["previous_metric_value"],
)

summary_columns = st.columns(4)

summary_columns[0].metric(
    "Current Week",
    format_metric_value(
        current_week["metric_value"],
        metric_value_format,
    ),
    help=f"Week of {current_week_label}",
)

summary_columns[1].metric(
    "Previous Week",
    format_metric_value(
        current_week["previous_metric_value"],
        metric_value_format,
    ),
)

summary_columns[2].metric(
    "Change",
    format_metric_value(
        current_week["metric_change"],
        metric_value_format,
    ),
)

growth_has_value = not pd.isna(current_week_growth)

summary_columns[3].metric(
    "Week-Over-Week Growth",
    (
        f"{current_week_growth:+.2f}%"
        if growth_has_value
        else "—"
    ),
    delta=(
        f"{current_week_growth:+.2f}%"
        if growth_has_value
        else None
    ),
)

if previous_week is None:
    st.info(
        "Only one week of history is available, so there is "
        "nothing to compare the current week against yet."
    )

if latest_week_is_partial:
    days_in_current_week = int(current_week["days_counted"])

    if week_to_date_applied:
        st.caption(
            f"The week of {current_week_label} is still in "
            f"progress with {days_in_current_week} day(s) of "
            "data. Every week below is trimmed to the same "
            "days for a like-for-like comparison."
        )
    else:
        st.caption(
            f"The week of {current_week_label} is still in "
            f"progress with {days_in_current_week} day(s) of "
            "data and is compared against full prior weeks. "
            "Enable \"Compare week to date\" for a "
            "like-for-like comparison."
        )
elif central_today <= latest_week_end:
    st.caption(
        f"The week of {current_week_label} is complete "
        "through its final business date."
    )


# ---------------------------------------------------------
# Weekly trend with week-over-week growth overlay
# ---------------------------------------------------------

st.subheader("Weekly Trend and Growth")

weekly_metric_max = float(
    visible_weeks_df["metric_value"].max()
)

if pd.isna(weekly_metric_max) or weekly_metric_max <= 0:
    weekly_axis_max = 1.0
else:
    weekly_axis_max = weekly_metric_max * 1.05

visible_growth_values = visible_weeks_df[
    "metric_growth"
].dropna()

if visible_growth_values.empty:
    growth_axis_min = -1.0
    growth_axis_max = 1.0
else:
    raw_growth_min = min(
        0.0,
        float(visible_growth_values.min()),
    )
    raw_growth_max = max(
        0.0,
        float(visible_growth_values.max()),
    )
    growth_padding = max(
        (raw_growth_max - raw_growth_min) * 0.10,
        1.0,
    )
    growth_axis_min = raw_growth_min - growth_padding
    growth_axis_max = raw_growth_max + growth_padding

# Weekly totals use bars and the left value axis.
weekly_bars = (
    alt.Chart(visible_weeks_df)
    .mark_bar(color="#1f77b4")
    .encode(
        x=alt.X(
            "week_label:N",
            title="Week Starting",
            sort=week_label_order,
        ),
        y=alt.Y(
            "metric_value:Q",
            title=selected_metric_name,
            axis=alt.Axis(
                format=metric_axis_format,
                titleColor="#1f77b4",
            ),
            scale=alt.Scale(
                domain=[0, weekly_axis_max],
            ),
        ),
        opacity=alt.condition(
            alt.datum.is_complete_week,
            alt.value(0.8),
            alt.value(0.35),
        ),
        tooltip=[
            alt.Tooltip(
                "week_label:N",
                title="Week Starting",
            ),
            alt.Tooltip(
                "metric_value:Q",
                title=selected_metric_name,
                format=metric_tooltip_format,
            ),
            alt.Tooltip(
                "days_counted:Q",
                title="Days Counted",
                format=".0f",
            ),
        ],
    )
)

# Growth uses a line and the right percentage axis.
growth_line = (
    alt.Chart(visible_weeks_df)
    .mark_line(
        color="#ff7f0e",
        strokeWidth=3,
        point=alt.OverlayMarkDef(
            color="#ff7f0e",
            size=55,
        ),
    )
    .encode(
        x=alt.X(
            "week_label:N",
            title="Week Starting",
            sort=week_label_order,
        ),
        y=alt.Y(
            "metric_growth:Q",
            title="Week-Over-Week Growth",
            axis=alt.Axis(
                orient="right",
                format=".1f",
                labelExpr="datum.value + '%'",
                titleColor="#ff7f0e",
            ),
            scale=alt.Scale(
                domain=[growth_axis_min, growth_axis_max],
                zero=False,
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "week_label:N",
                title="Week Starting",
            ),
            alt.Tooltip(
                "metric_growth:Q",
                title="Growth (%)",
                format="+.2f",
            ),
        ],
    )
)

# A 0% reference separates growth from decline.
zero_growth_line = (
    alt.Chart(pd.DataFrame({"zero_growth": [0.0]}))
    .mark_rule(
        color="#6b7280",
        strokeWidth=2,
        strokeDash=[5, 5],
    )
    .encode(
        y=alt.Y(
            "zero_growth:Q",
            axis=None,
            scale=alt.Scale(
                domain=[growth_axis_min, growth_axis_max],
                zero=False,
            ),
        ),
    )
)

weekly_chart = (
    alt.layer(
        weekly_bars,
        growth_line,
        zero_growth_line,
    )
    .resolve_scale(y="independent")
    .properties(height=440)
)

st.altair_chart(
    weekly_chart,
    use_container_width=True,
)

st.caption(
    f"Blue bars show weekly {selected_metric_name.lower()}. "
    "The orange line shows the change against the "
    "immediately preceding week, and the dashed gray line "
    "marks 0% growth."
)

incomplete_week_labels = visible_weeks_df.loc[
    ~visible_weeks_df["is_complete_week"],
    "week_label",
].tolist()

if incomplete_week_labels:
    st.caption(
        "Faded bars have fewer than "
        f"{expected_days_per_week} day(s) of data: "
        f"{', '.join(incomplete_week_labels)}. Their totals "
        "and growth are understated."
    )


# ---------------------------------------------------------
# Day-of-week comparison
# ---------------------------------------------------------

st.subheader("Current Week vs Previous Week by Day")

if previous_week is None:
    st.info(
        "A day-level comparison needs at least two weeks of "
        "history."
    )
else:
    previous_week_start = previous_week["week_start"]

    # Use untrimmed daily data so the chart shows which days
    # of the current week have not been reported yet.
    daily_comparison_df = (
        location_df[
            location_df["week_start"].isin(
                [latest_week_start, previous_week_start]
            )
        ]
        .groupby(
            ["week_start", "day_index"],
            as_index=False,
        )[SALES_NUMERIC_COLUMNS]
        .sum()
    )

    daily_comparison_df["metric_value"] = (
        compute_metric_series(
            daily_comparison_df,
            selected_metric,
        )
    )

    daily_comparison_df["weekday_name"] = (
        daily_comparison_df["day_index"].map(
            lambda day_index: WEEKDAY_NAMES[
                (week_start_weekday + day_index) % 7
            ]
        )
    )

    daily_comparison_df["week_series"] = (
        daily_comparison_df["week_start"]
        .eq(latest_week_start)
        .map({True: "Current week", False: "Previous week"})
    )

    weekday_order = [
        WEEKDAY_NAMES[(week_start_weekday + offset) % 7]
        for offset in range(7)
    ]

    day_comparison_chart = (
        alt.Chart(daily_comparison_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "weekday_name:N",
                title="Day of Week",
                sort=weekday_order,
            ),
            xOffset=alt.XOffset(
                "week_series:N",
                sort=["Previous week", "Current week"],
            ),
            y=alt.Y(
                "metric_value:Q",
                title=selected_metric_name,
                axis=alt.Axis(format=metric_axis_format),
            ),
            color=alt.Color(
                "week_series:N",
                title="Week",
                sort=["Previous week", "Current week"],
                scale=alt.Scale(
                    domain=[
                        "Previous week",
                        "Current week",
                    ],
                    range=["#9ecae1", "#1f77b4"],
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "weekday_name:N",
                    title="Day",
                ),
                alt.Tooltip(
                    "week_series:N",
                    title="Week",
                ),
                alt.Tooltip(
                    "metric_value:Q",
                    title=selected_metric_name,
                    format=metric_tooltip_format,
                ),
            ],
        )
        .properties(height=380)
    )

    st.altair_chart(
        day_comparison_chart,
        use_container_width=True,
    )

    st.caption(
        "Days missing from the current week have not been "
        "reported yet."
    )


# ---------------------------------------------------------
# Channel mix week over week
# ---------------------------------------------------------

st.subheader("Channel Mix Week Over Week")

if previous_week is None:
    st.info(
        "Channel comparison needs at least two weeks of "
        "history."
    )
else:
    channel_rows = []

    for column_name, channel_label in CHANNEL_COLUMNS.items():
        current_channel_value = float(
            current_week[column_name]
        )
        previous_channel_value = float(
            previous_week[column_name]
        )

        current_net_sales = float(current_week["net_sales"])
        previous_net_sales = float(previous_week["net_sales"])

        channel_rows.append(
            {
                "Channel": channel_label,
                "Current Week": current_channel_value,
                "Previous Week": previous_channel_value,
                "Change": (
                    current_channel_value
                    - previous_channel_value
                ),
                "Growth %": percent_change(
                    current_channel_value,
                    previous_channel_value,
                ),
                "Share %": (
                    current_channel_value
                    / current_net_sales
                    * 100
                    if current_net_sales
                    else NOT_AVAILABLE
                ),
                "Prior Share %": (
                    previous_channel_value
                    / previous_net_sales
                    * 100
                    if previous_net_sales
                    else NOT_AVAILABLE
                ),
            }
        )

    channel_df = pd.DataFrame(channel_rows).sort_values(
        "Current Week",
        ascending=False,
    )

    st.dataframe(
        channel_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Week": st.column_config.NumberColumn(
                format="$%.2f",
            ),
            "Previous Week": st.column_config.NumberColumn(
                format="$%.2f",
            ),
            "Change": st.column_config.NumberColumn(
                format="$%.2f",
            ),
            "Growth %": st.column_config.NumberColumn(
                format="%.2f%%",
            ),
            "Share %": st.column_config.NumberColumn(
                format="%.1f%%",
            ),
            "Prior Share %": st.column_config.NumberColumn(
                format="%.1f%%",
            ),
        },
    )

    st.caption(
        "Channel dollars always reflect net sales, "
        "regardless of the metric selected above. Revel "
        "reports channels independently, so shares do not "
        "always total exactly 100%."
    )


# ---------------------------------------------------------
# Weekly breakdown table
# ---------------------------------------------------------

with st.expander("View weekly breakdown"):
    breakdown_df = visible_weeks_df[
        [
            "week_start",
            "days_counted",
            "metric_value",
            "previous_metric_value",
            "metric_change",
            "metric_growth",
        ]
    ].copy()

    breakdown_df["week_start"] = (
        breakdown_df["week_start"].dt.date
    )

    breakdown_df = breakdown_df.sort_values(
        "week_start",
        ascending=False,
    )

    value_column_format = (
        "$%.2f"
        if metric_value_format == "currency"
        else "%.0f"
    )

    st.dataframe(
        breakdown_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "week_start": st.column_config.DateColumn(
                "Week Starting",
            ),
            "days_counted": st.column_config.NumberColumn(
                "Days Counted",
                format="%d",
            ),
            "metric_value": st.column_config.NumberColumn(
                selected_metric_name,
                format=value_column_format,
            ),
            "previous_metric_value": (
                st.column_config.NumberColumn(
                    "Previous Week",
                    format=value_column_format,
                )
            ),
            "metric_change": st.column_config.NumberColumn(
                "Change",
                format=value_column_format,
            ),
            "metric_growth": st.column_config.NumberColumn(
                "Growth %",
                format="%.2f%%",
            ),
        },
    )


# ---------------------------------------------------------
# Source data
# ---------------------------------------------------------

with st.expander("View daily sales records"):
    visible_week_starts = visible_weeks_df["week_start"]

    display_df = location_df[
        location_df["week_start"].isin(visible_week_starts)
    ][
        [
            "business_date",
            "location",
            "net_sales",
            "total_orders",
            "total_discounts",
        ]
    ].copy()

    display_df["business_date"] = (
        display_df["business_date"].dt.date
    )

    display_df = display_df.sort_values(
        "business_date",
        ascending=False,
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "business_date": st.column_config.DateColumn(
                "Business Date",
            ),
            "location": "Location",
            "net_sales": st.column_config.NumberColumn(
                "Net Sales",
                format="$%.2f",
            ),
            "total_orders": st.column_config.NumberColumn(
                "Orders",
                format="%d",
            ),
            "total_discounts": st.column_config.NumberColumn(
                "Discounts",
                format="$%.2f",
            ),
        },
    )


st.caption(
    f"Displaying {len(visible_weeks_df):,} week(s) of "
    f"{selected_metric_name.lower()} for "
    f"{selected_location.lower()}, weeks starting "
    f"{selected_week_start_name}."
)
