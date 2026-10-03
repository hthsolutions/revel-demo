from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    NOT_AVAILABLE,
    WEEK_START_OPTIONS,
    MissingSecretError,
    add_week_columns,
    escape_dollar_signs,
    format_metric_value,
    load_salary_data,
    load_sales_data,
    load_shift_data,
    percent_change,
    reindex_weekly,
)


# Columns summed when rolling daily labor up to a week.
WEEKLY_SUM_COLUMNS = [
    "net_sales",
    "hourly_labor",
    "salary_labor",
    "total_labor",
    "labor_hours",
    "ot_hours",
    "ot_labor",
]


LABOR_TYPE_LABELS = {
    "hourly_labor": "Hourly",
    "salary_labor": "Salaried",
}

LABOR_TYPE_COLORS = alt.Scale(
    domain=list(LABOR_TYPE_LABELS.values()),
    range=["#1f77b4", "#9467bd"],
)


def stack_labor_types(
    frame: pd.DataFrame,
    id_columns: list[str],
) -> pd.DataFrame:
    """Reshape to one row per labor type for stacked bars."""

    long_df = frame.melt(
        id_vars=id_columns,
        value_vars=list(LABOR_TYPE_LABELS.keys()),
        var_name="labor_type",
        value_name="labor_dollars",
    )

    long_df["labor_type"] = long_df["labor_type"].map(
        LABOR_TYPE_LABELS
    )

    return long_df


def percent_of_sales(labor_dollars, net_sales) -> float:
    """Labor dollars as a percent of sales, or NaN."""

    if pd.isna(labor_dollars) or pd.isna(net_sales):
        return NOT_AVAILABLE

    if net_sales == 0:
        return NOT_AVAILABLE

    return labor_dollars / net_sales * 100


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two series, leaving zero denominators blank."""

    return numerator.astype(float) / denominator.astype(
        float
    ).replace(0.0, NOT_AVAILABLE)


st.title("Labor % of Net Sales")
st.caption(
    "Hourly shift wages plus prorated salaried pay, measured "
    "against net sales."
)

try:
    sales_df = load_sales_data()
    shift_df = load_shift_data()
    salary_df = load_salary_data()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        f"Unable to load data from Supabase: {error} "
        "Confirm the table names in revel_data.py "
        "(SALES_TABLE, SHIFT_TABLE, SALARY_TABLE)."
    )
    st.stop()

if sales_df.empty:
    st.warning("No sales records were returned from Supabase.")
    st.stop()

if shift_df.empty and salary_df.empty:
    st.warning(
        "No labor records were returned from Supabase. "
        "Check the shift and salary table names, "
        "credentials, and Row Level Security settings."
    )
    st.stop()


# ---------------------------------------------------------
# Daily labor assembled from three tables
# ---------------------------------------------------------

daily_sales_df = (
    sales_df
    .groupby(["business_date", "location_key"], as_index=False)
    .agg(
        location=("location", "first"),
        net_sales=("net_sales", "sum"),
    )
)

def empty_daily_frame(value_columns: list[str]) -> pd.DataFrame:
    """Build a typed empty frame so merges keep their dtypes."""

    columns = {
        "business_date": pd.Series(dtype="datetime64[ns]"),
        "location_key": pd.Series(dtype="object"),
    }

    for column_name in value_columns:
        columns[column_name] = pd.Series(dtype="float64")

    return pd.DataFrame(columns)


if shift_df.empty:
    daily_hourly_df = empty_daily_frame(
        [
            "hourly_wages",
            "regular_hours",
            "ot_hours",
            "ot_wages",
            "shift_count",
            "employee_count",
        ]
    )
else:
    daily_hourly_df = (
        shift_df
        .groupby(
            ["business_date", "location_key"],
            as_index=False,
        )
        .agg(
            # shift_wages is the per-shift total. The other
            # wage columns in this table are pre-aggregated
            # per employee per day and would double count.
            hourly_wages=("shift_wages", "sum"),
            regular_hours=("regular_hours", "sum"),
            ot_hours=("ot_hours", "sum"),
            ot_wages=("shift_ot_wages", "sum"),
            shift_count=("shift_wages", "size"),
            employee_count=("employee", "nunique"),
        )
    )

if salary_df.empty:
    daily_salary_df = empty_daily_frame(["salary_wages"])
else:
    daily_salary_df = (
        salary_df
        .groupby(
            ["business_date", "location_key"],
            as_index=False,
        )
        .agg(salary_wages=("daily_salary", "sum"))
    )

# Labor percent is only meaningful on days that report both
# sales and hourly labor, so this join is deliberately an
# inner join on the hourly data.
labor_df = daily_sales_df.merge(
    daily_hourly_df,
    on=["business_date", "location_key"],
    how="inner",
).merge(
    daily_salary_df,
    on=["business_date", "location_key"],
    how="left",
)

labor_df["salary_wages"] = (
    labor_df["salary_wages"].fillna(0.0)
)


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    labor_df["location"].dropna().unique().tolist()
)

selected_location = st.sidebar.selectbox(
    "Location",
    options=["All Locations", *available_locations],
)

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

hourly_target_percent = st.sidebar.number_input(
    "Hourly labor %",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=0.5,
    help="Target for hourly shift wages as a percent of net sales.",
)

gm_target_percent = st.sidebar.number_input(
    "GM labor %",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.5,
    help="Target for GM salary as a percent of net sales.",
)

target_labor_percent = hourly_target_percent + gm_target_percent

st.sidebar.metric(
    "Target labor %",
    f"{target_labor_percent:.1f}%",
    help="Hourly labor % plus GM labor %.",
)

if selected_location != "All Locations":
    labor_df = labor_df[
        labor_df["location"] == selected_location
    ].copy()
    location_sales_df = daily_sales_df[
        daily_sales_df["location"] == selected_location
    ].copy()
else:
    labor_df = labor_df.copy()
    location_sales_df = daily_sales_df.copy()

if labor_df.empty:
    st.warning(
        "No days have both sales and labor records for the "
        "selected location. The sales table covers "
        f"{len(location_sales_df):,} day(s), but none of "
        "them line up with a shift record."
    )
    st.stop()


# ---------------------------------------------------------
# Labor dollars, hours, and rate
# ---------------------------------------------------------

labor_df["hourly_labor"] = labor_df["hourly_wages"]
labor_df["salary_labor"] = labor_df["salary_wages"]
labor_df["ot_labor"] = labor_df["ot_wages"]
labor_df["total_labor"] = (
    labor_df["hourly_labor"] + labor_df["salary_labor"]
)
labor_df["labor_hours"] = (
    labor_df["regular_hours"] + labor_df["ot_hours"]
)

labor_df["labor_percent"] = (
    safe_ratio(
        labor_df["total_labor"],
        labor_df["net_sales"],
    )
    * 100
)

labor_df["sales_per_labor_hour"] = safe_ratio(
    labor_df["net_sales"],
    labor_df["labor_hours"],
)

labor_df = add_week_columns(labor_df, week_start_weekday)


# ---------------------------------------------------------
# Coverage: labor data usually lags the sales feed
# ---------------------------------------------------------

labor_day_count = len(labor_df)
sales_day_count = len(location_sales_df)
days_without_labor = sales_day_count - labor_day_count

earliest_labor_date = labor_df["business_date"].min()
latest_labor_date = labor_df["business_date"].max()

coverage_message = (
    f"Labor data covers {labor_day_count:,} day(s) from "
    f"{earliest_labor_date:%b %d, %Y} through "
    f"{latest_labor_date:%b %d, %Y}."
)

if days_without_labor > 0:
    st.info(
        f"{coverage_message} The sales table has "
        f"{days_without_labor:,} additional day(s) with no "
        "matching shift records; those days are excluded so "
        "they do not show as 0% labor."
    )
else:
    st.caption(coverage_message)

days_without_salary = int(
    (labor_df["salary_wages"] == 0).sum()
)

if days_without_salary:
    st.caption(
        f"{days_without_salary:,} day(s) have no salaried "
        "pay recorded and count hourly wages only."
    )


# ---------------------------------------------------------
# Weekly aggregation
# ---------------------------------------------------------

latest_week_start = labor_df["week_start"].max()

latest_week_day_index = int(
    labor_df.loc[
        labor_df["week_start"] == latest_week_start,
        "day_index",
    ].max()
)

latest_week_is_partial = latest_week_day_index < 6

comparison_df = labor_df.copy()
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
            for column_name in WEEKLY_SUM_COLUMNS
        },
        days_counted=("business_date", "nunique"),
    )
    .sort_values("week_start")
    .reset_index(drop=True)
)

weekly_df = reindex_weekly(
    weekly_df,
    labor_df["week_start"],
)

weekly_df["days_counted"] = (
    weekly_df["days_counted"].fillna(0).astype(int)
)

# A weekly rate is total labor over total sales, not the
# average of the daily rates, which would weight a slow day
# the same as a busy one.
weekly_df["labor_percent"] = (
    safe_ratio(
        weekly_df["total_labor"],
        weekly_df["net_sales"],
    )
    * 100
)

weekly_df["sales_per_labor_hour"] = safe_ratio(
    weekly_df["net_sales"],
    weekly_df["labor_hours"],
)

weekly_df["previous_labor_percent"] = weekly_df[
    "labor_percent"
].shift(1)

# Labor percent is already a percentage, so the week-over-
# week move is reported in percentage points.
weekly_df["labor_percent_change"] = (
    weekly_df["labor_percent"]
    - weekly_df["previous_labor_percent"]
)

weekly_df["week_label"] = weekly_df[
    "week_start"
].dt.strftime("%b %d")

expected_days_per_week = (
    latest_week_day_index + 1 if week_to_date_applied else 7
)

weekly_df["is_complete_week"] = (
    weekly_df["days_counted"] >= expected_days_per_week
)

visible_weeks_df = weekly_df.tail(weeks_to_display).copy()

week_label_order = visible_weeks_df["week_label"].tolist()

current_week = weekly_df.iloc[-1]

previous_week = (
    weekly_df.iloc[-2] if len(weekly_df) >= 2 else None
)


# ---------------------------------------------------------
# Week-over-week summary
# ---------------------------------------------------------

current_week_label = (
    f"{current_week['week_start']:%b %d}–"
    f"{current_week['week_start'] + timedelta(days=6):%b %d, %Y}"
)

st.subheader("Current Week vs Previous Week")

labor_percent_change = current_week["labor_percent_change"]

summary_columns = st.columns(4)

summary_columns[0].metric(
    "Labor % of Net Sales",
    format_metric_value(
        current_week["labor_percent"],
        "percent",
    ),
    delta=(
        f"{labor_percent_change:+.2f} pts"
        if not pd.isna(labor_percent_change)
        else None
    ),
    # A falling labor rate is a good outcome.
    delta_color="inverse",
    help=f"Week of {current_week_label}",
)

summary_columns[1].metric(
    "Total Labor",
    format_metric_value(
        current_week["total_labor"],
        "currency",
    ),
    delta=(
        f"{percent_change(current_week['total_labor'], previous_week['total_labor']):+.2f}%"
        if previous_week is not None
        and not pd.isna(
            percent_change(
                current_week["total_labor"],
                previous_week["total_labor"],
            )
        )
        else None
    ),
    delta_color="inverse",
)

summary_columns[2].metric(
    "Net Sales",
    format_metric_value(
        current_week["net_sales"],
        "currency",
    ),
    delta=(
        f"{percent_change(current_week['net_sales'], previous_week['net_sales']):+.2f}%"
        if previous_week is not None
        and not pd.isna(
            percent_change(
                current_week["net_sales"],
                previous_week["net_sales"],
            )
        )
        else None
    ),
)

summary_columns[3].metric(
    "Sales per Labor Hour",
    format_metric_value(
        current_week["sales_per_labor_hour"],
        "currency",
    ),
    delta=(
        f"{percent_change(current_week['sales_per_labor_hour'], previous_week['sales_per_labor_hour']):+.2f}%"
        if previous_week is not None
        and not pd.isna(
            percent_change(
                current_week["sales_per_labor_hour"],
                previous_week["sales_per_labor_hour"],
            )
        )
        else None
    ),
)

previous_label = format_metric_value(
    current_week["previous_labor_percent"],
    "percent",
)

hourly_label = format_metric_value(
    current_week["hourly_labor"],
    "currency",
)

salary_label = format_metric_value(
    current_week["salary_labor"],
    "currency",
)

current_hourly_percent = format_metric_value(
    percent_of_sales(
        current_week["hourly_labor"],
        current_week["net_sales"],
    ),
    "percent",
)

current_gm_percent = format_metric_value(
    percent_of_sales(
        current_week["salary_labor"],
        current_week["net_sales"],
    ),
    "percent",
)

st.caption(
    escape_dollar_signs(
        f"Previous week was {previous_label}. Target is "
        f"{target_labor_percent:.1f}% "
        f"({hourly_target_percent:.1f}% hourly + "
        f"{gm_target_percent:.1f}% GM). Hourly pay was "
        f"{hourly_label} ({current_hourly_percent} of net "
        f"sales) and GM salary was {salary_label} "
        f"({current_gm_percent})."
    )
)

if previous_week is None:
    st.info(
        "Only one week of labor history is available, so "
        "there is nothing to compare against yet."
    )

if latest_week_is_partial:
    days_in_current_week = int(current_week["days_counted"])

    if week_to_date_applied:
        st.caption(
            f"The week of {current_week_label} has "
            f"{days_in_current_week} day(s) of labor data. "
            "Every week below is trimmed to the same days "
            "for a like-for-like comparison."
        )
    else:
        st.caption(
            f"The week of {current_week_label} has only "
            f"{days_in_current_week} day(s) of labor data "
            "and is compared against full prior weeks."
        )

# Trimming can empty a prior week completely when its labor
# records land on different weekdays than the current week.
if (
    week_to_date_applied
    and previous_week is not None
    and int(previous_week["days_counted"]) == 0
    and (
        labor_df["week_start"] == previous_week["week_start"]
    ).any()
):
    st.info(
        "The previous week has labor records, but none on "
        "the same weekdays as the current week, so the "
        "week-to-date comparison is empty. Turn off "
        "\"Compare week to date\" to compare whole weeks."
    )


# ---------------------------------------------------------
# Weekly labor dollars and labor rate
# ---------------------------------------------------------

st.subheader("Weekly Labor Dollars and Labor Rate")

weekly_labor_max = float(
    visible_weeks_df["total_labor"].max()
)

if pd.isna(weekly_labor_max) or weekly_labor_max <= 0:
    labor_axis_max = 1.0
else:
    labor_axis_max = weekly_labor_max * 1.05

visible_percent_values = visible_weeks_df[
    "labor_percent"
].dropna()

if visible_percent_values.empty:
    percent_axis_min = 0.0
    percent_axis_max = max(target_labor_percent * 1.5, 1.0)
else:
    raw_percent_min = min(
        float(visible_percent_values.min()),
        target_labor_percent,
    )
    raw_percent_max = max(
        float(visible_percent_values.max()),
        target_labor_percent,
    )
    percent_padding = max(
        (raw_percent_max - raw_percent_min) * 0.15,
        1.0,
    )
    percent_axis_min = max(
        0.0,
        raw_percent_min - percent_padding,
    )
    percent_axis_max = raw_percent_max + percent_padding

weekly_stacked_df = stack_labor_types(
    visible_weeks_df,
    [
        "week_label",
        "is_complete_week",
        "total_labor",
        "net_sales",
        "days_counted",
    ],
)

labor_dollar_bars = (
    alt.Chart(weekly_stacked_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "week_label:N",
            title="Week Starting",
            sort=week_label_order,
        ),
        y=alt.Y(
            "labor_dollars:Q",
            title="Total Labor",
            stack="zero",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(domain=[0, labor_axis_max]),
        ),
        color=alt.Color(
            "labor_type:N",
            title="Labor Type",
            scale=LABOR_TYPE_COLORS,
        ),
        opacity=alt.condition(
            alt.datum.is_complete_week,
            alt.value(0.85),
            alt.value(0.4),
        ),
        tooltip=[
            alt.Tooltip(
                "week_label:N",
                title="Week Starting",
            ),
            alt.Tooltip("labor_type:N", title="Labor Type"),
            alt.Tooltip(
                "labor_dollars:Q",
                title="Segment",
                format="$,.2f",
            ),
            alt.Tooltip(
                "total_labor:Q",
                title="Total Labor",
                format="$,.2f",
            ),
            alt.Tooltip(
                "net_sales:Q",
                title="Net Sales",
                format="$,.2f",
            ),
            alt.Tooltip(
                "days_counted:Q",
                title="Days Counted",
                format=".0f",
            ),
        ],
    )
)

labor_percent_line = (
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
            "labor_percent:Q",
            title="Labor % of Net Sales",
            axis=alt.Axis(
                orient="right",
                format=".1f",
                labelExpr="datum.value + '%'",
                titleColor="#ff7f0e",
            ),
            scale=alt.Scale(
                domain=[percent_axis_min, percent_axis_max],
                zero=False,
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "week_label:N",
                title="Week Starting",
            ),
            alt.Tooltip(
                "labor_percent:Q",
                title="Labor %",
                format=".2f",
            ),
        ],
    )
)

target_rule = (
    alt.Chart(
        pd.DataFrame({"target": [float(target_labor_percent)]})
    )
    .mark_rule(
        color="#15803d",
        strokeWidth=2,
        strokeDash=[8, 5],
    )
    .encode(
        y=alt.Y(
            "target:Q",
            axis=None,
            scale=alt.Scale(
                domain=[percent_axis_min, percent_axis_max],
                zero=False,
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "target:Q",
                title="Target Labor %",
                format=".1f",
            ),
        ],
    )
)

weekly_labor_chart = (
    alt.layer(
        labor_dollar_bars,
        labor_percent_line,
        target_rule,
    )
    .resolve_scale(y="independent")
    .properties(height=440)
)

st.altair_chart(
    weekly_labor_chart,
    use_container_width=True,
)

st.caption(
    "Bars show labor dollars on the left axis, split into "
    "hourly (blue) and salaried (purple) pay. The orange "
    "line shows labor as a percent of net sales on the "
    "right axis, and the dashed green line is the target "
    "rate (hourly labor % plus GM labor %)."
)


# ---------------------------------------------------------
# Daily detail across the visible weeks
# ---------------------------------------------------------

st.subheader("Daily Labor Dollars and Labor Rate")

visible_week_starts = set(visible_weeks_df["week_start"])

visible_daily_df = labor_df[
    labor_df["week_start"].isin(visible_week_starts)
].copy()

# Bucketing the date to whole days gives the bars a band to
# fill and keeps the axis to one label per day.
daily_axis = alt.X(
    "yearmonthdate(business_date):O",
    title="Business Date",
    axis=alt.Axis(
        format="%b %d",
        labelAngle=-45,
        labelOverlap="greedy",
    ),
)

daily_stacked_df = stack_labor_types(
    visible_daily_df,
    [
        "business_date",
        "net_sales",
        "total_labor",
        "labor_percent",
    ],
)

daily_labor_bars = (
    alt.Chart(daily_stacked_df)
    .mark_bar(opacity=0.85)
    .encode(
        x=daily_axis,
        y=alt.Y(
            "labor_dollars:Q",
            title="Labor Dollars",
            stack="zero",
            axis=alt.Axis(format="$,.0f"),
        ),
        color=alt.Color(
            "labor_type:N",
            title="Labor Type",
            scale=LABOR_TYPE_COLORS,
        ),
        tooltip=[
            alt.Tooltip(
                "business_date:T",
                title="Business Date",
                format="%B %d, %Y",
            ),
            alt.Tooltip("labor_type:N", title="Labor Type"),
            alt.Tooltip(
                "labor_dollars:Q",
                title="Segment",
                format="$,.2f",
            ),
            alt.Tooltip(
                "net_sales:Q",
                title="Net Sales",
                format="$,.2f",
            ),
            alt.Tooltip(
                "total_labor:Q",
                title="Total Labor",
                format="$,.2f",
            ),
            alt.Tooltip(
                "labor_percent:Q",
                title="Labor %",
                format=".2f",
            ),
        ],
    )
)

daily_percent_line = (
    alt.Chart(visible_daily_df)
    .mark_line(
        color="#ff7f0e",
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(
            color="#ff7f0e",
            size=40,
        ),
    )
    .encode(
        x=daily_axis,
        y=alt.Y(
            "labor_percent:Q",
            title="Labor % of Net Sales",
            axis=alt.Axis(
                orient="right",
                format=".1f",
                labelExpr="datum.value + '%'",
                titleColor="#ff7f0e",
            ),
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip(
                "business_date:T",
                title="Business Date",
                format="%B %d, %Y",
            ),
            alt.Tooltip(
                "labor_percent:Q",
                title="Labor %",
                format=".2f",
            ),
        ],
    )
)

daily_chart = (
    alt.layer(daily_labor_bars, daily_percent_line)
    .resolve_scale(y="independent")
    .properties(height=400)
)

st.altair_chart(daily_chart, use_container_width=True)

st.caption(
    "Bars show each day's hourly and salaried labor dollars; "
    "hover a bar for that day's net sales. Low-volume days "
    "carry the same fixed salaried cost as "
    "busy days, so the labor rate usually spikes on the "
    "slowest days."
)


# ---------------------------------------------------------
# Labor dollars by role
# ---------------------------------------------------------

st.subheader("Labor Dollars by Role")

role_frames = []

if not shift_df.empty:
    hourly_roles_df = shift_df.copy()

    if selected_location != "All Locations":
        hourly_roles_df = hourly_roles_df[
            hourly_roles_df["location"] == selected_location
        ]

    hourly_roles_df = add_week_columns(
        hourly_roles_df,
        week_start_weekday,
    )

    hourly_roles_df = hourly_roles_df[
        hourly_roles_df["week_start"].isin(
            visible_week_starts
        )
    ]

    if not hourly_roles_df.empty:
        role_frames.append(
            hourly_roles_df
            .groupby(["week_start", "role"], as_index=False)
            .agg(labor_dollars=("shift_wages", "sum"))
        )

if not salary_df.empty:
    salary_roles_df = salary_df.copy()

    # The salary table stores names in upper case, so it is
    # matched on the shared location key.
    if selected_location != "All Locations":
        selected_key = selected_location.upper()
        salary_roles_df = salary_roles_df[
            salary_roles_df["location_key"] == selected_key
        ]

    salary_roles_df = add_week_columns(
        salary_roles_df,
        week_start_weekday,
    )

    salary_roles_df = salary_roles_df[
        salary_roles_df["week_start"].isin(
            visible_week_starts
        )
    ]

    if not salary_roles_df.empty:
        salary_roles_df = (
            salary_roles_df
            .groupby(["week_start", "role"], as_index=False)
            .agg(labor_dollars=("daily_salary", "sum"))
        )
        salary_roles_df["role"] = (
            salary_roles_df["role"] + " (salary)"
        )
        role_frames.append(salary_roles_df)

if not role_frames:
    st.info(
        "No role-level labor records fall inside the "
        "selected weeks."
    )
else:
    role_df = pd.concat(role_frames, ignore_index=True)

    role_df["week_label"] = role_df[
        "week_start"
    ].dt.strftime("%b %d")

    role_chart = (
        alt.Chart(role_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "week_label:N",
                title="Week Starting",
                sort=week_label_order,
            ),
            y=alt.Y(
                "labor_dollars:Q",
                title="Labor Dollars",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "role:N",
                title="Role",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip(
                    "week_label:N",
                    title="Week Starting",
                ),
                alt.Tooltip("role:N", title="Role"),
                alt.Tooltip(
                    "labor_dollars:Q",
                    title="Labor Dollars",
                    format="$,.2f",
                ),
            ],
        )
        .properties(height=400)
    )

    st.altair_chart(role_chart, use_container_width=True)

    st.caption(
        "Role totals cover every day with a labor record, "
        "including days the sales feed has not reported, so "
        "they can exceed the weekly totals above."
    )


# ---------------------------------------------------------
# Weekly and daily tables
# ---------------------------------------------------------

with st.expander("View weekly labor breakdown"):
    weekly_table_df = visible_weeks_df[
        [
            "week_start",
            "days_counted",
            "net_sales",
            "hourly_labor",
            "salary_labor",
            "total_labor",
            "labor_percent",
            "previous_labor_percent",
            "labor_percent_change",
            "labor_hours",
            "sales_per_labor_hour",
        ]
    ].copy()

    weekly_table_df["week_start"] = weekly_table_df[
        "week_start"
    ].dt.date

    weekly_table_df = weekly_table_df.sort_values(
        "week_start",
        ascending=False,
    )

    st.dataframe(
        weekly_table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "week_start": st.column_config.DateColumn(
                "Week Starting",
            ),
            "days_counted": st.column_config.NumberColumn(
                "Days",
                format="%d",
            ),
            "net_sales": st.column_config.NumberColumn(
                "Net Sales",
                format="$%.2f",
            ),
            "hourly_labor": st.column_config.NumberColumn(
                "Hourly",
                format="$%.2f",
            ),
            "salary_labor": st.column_config.NumberColumn(
                "Salaried",
                format="$%.2f",
            ),
            "total_labor": st.column_config.NumberColumn(
                "Total Labor",
                format="$%.2f",
            ),
            "labor_percent": st.column_config.NumberColumn(
                "Labor %",
                format="%.2f%%",
            ),
            "previous_labor_percent": (
                st.column_config.NumberColumn(
                    "Prior Labor %",
                    format="%.2f%%",
                )
            ),
            "labor_percent_change": (
                st.column_config.NumberColumn(
                    "Change (pts)",
                    format="%.2f",
                )
            ),
            "labor_hours": st.column_config.NumberColumn(
                "Labor Hours",
                format="%.1f",
            ),
            "sales_per_labor_hour": (
                st.column_config.NumberColumn(
                    "Sales / Hour",
                    format="$%.2f",
                )
            ),
        },
    )

with st.expander("View daily labor records"):
    daily_table_df = visible_daily_df[
        [
            "business_date",
            "location",
            "net_sales",
            "hourly_labor",
            "salary_labor",
            "total_labor",
            "labor_percent",
            "labor_hours",
            "ot_hours",
            "sales_per_labor_hour",
            "employee_count",
        ]
    ].copy()

    daily_table_df["business_date"] = daily_table_df[
        "business_date"
    ].dt.date

    daily_table_df = daily_table_df.sort_values(
        "business_date",
        ascending=False,
    )

    st.dataframe(
        daily_table_df,
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
            "hourly_labor": st.column_config.NumberColumn(
                "Hourly",
                format="$%.2f",
            ),
            "salary_labor": st.column_config.NumberColumn(
                "Salaried",
                format="$%.2f",
            ),
            "total_labor": st.column_config.NumberColumn(
                "Total Labor",
                format="$%.2f",
            ),
            "labor_percent": st.column_config.NumberColumn(
                "Labor %",
                format="%.2f%%",
            ),
            "labor_hours": st.column_config.NumberColumn(
                "Hours",
                format="%.1f",
            ),
            "ot_hours": st.column_config.NumberColumn(
                "OT Hours",
                format="%.1f",
            ),
            "sales_per_labor_hour": (
                st.column_config.NumberColumn(
                    "Sales / Hour",
                    format="$%.2f",
                )
            ),
            "employee_count": st.column_config.NumberColumn(
                "Employees",
                format="%d",
            ),
        },
    )


st.caption(
    f"Displaying {len(visible_weeks_df):,} week(s) of labor "
    f"for {selected_location.lower()}, weeks starting "
    f"{selected_week_start_name}."
)
