import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    NOT_AVAILABLE,
    SALES_NUMERIC_COLUMNS,
    MissingSecretError,
    load_sales_data,
    location_filter_controls,
    percent_change,
)


# Channel dollars reported with net sales. "Other" is added
# later when these columns add up to less than net sales.
CHANNEL_COLUMNS = {
    "drive_through": "Drive Through",
    "eat_in": "Eat In",
    "to_go": "To Go",
    "pickup": "Pickup",
    "dd_marketplace": "DoorDash Marketplace",
    "uber_eats": "Uber Eats",
}

CHANNEL_COLORS = {
    "Drive Through": "#1f77b4",
    "Eat In": "#2ca02c",
    "To Go": "#17becf",
    "Pickup": "#9467bd",
    "DoorDash Marketplace": "#d62728",
    "Uber Eats": "#e377c2",
    "Other": "#c7c7c7",
}

GROWTH_COLOR = "#ff7f0e"

# Fixed-length buckets counted backward from the date the
# user picks. A month is 30 days so February and March are
# the same length.
INTERVALS = {
    "Day over day": {
        "days": 1,
        "max_duration": 60,
        "default_duration": 30,
        "unit": "days",
        "growth_title": "Day-over-day growth",
        "x_title": "Business date",
        "duration_label": "Duration (days)",
    },
    "Week over week": {
        "days": 7,
        "max_duration": 52,
        "default_duration": 8,
        "unit": "weeks",
        "growth_title": "Week-over-week growth",
        "x_title": "Week",
        "duration_label": "Duration (weeks)",
    },
    "Month over month": {
        "days": 30,
        "max_duration": 12,
        "default_duration": 6,
        "unit": "months",
        "growth_title": "Month-over-month growth",
        "x_title": "30-day period",
        "duration_label": "Duration (months)",
    },
}


def format_dollars(value) -> str:
    """Nearest dollar, with a thousands separator."""

    if value is None or pd.isna(value):
        return "—"

    rounded = int(round(float(value)))
    sign = "-" if rounded < 0 else ""

    return f"{sign}${abs(rounded):,}"


def format_period_label(start, end, period_days: int, include_year: bool) -> str:
    """Short axis label for one bucket."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if period_days == 1:
        if include_year:
            return f"{end:%m/%d/%y}"
        return f"{end:%m/%d}"

    if include_year:
        return f"{start:%m/%d/%y}–{end:%m/%d/%y}"

    return f"{start:%m/%d}–{end:%m/%d}"


def format_period_range(start, end, period_days: int) -> str:
    """Full date range shown in tooltips and metric help."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if period_days == 1:
        return f"{end:%b %d, %Y}"

    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def build_period_frame(
    frame: pd.DataFrame,
    date_end,
    period_days: int,
    duration: int,
) -> pd.DataFrame:
    """Bucket daily sales into equal periods ending on date_end.

    The returned frame includes one extra period before the
    visible window. That baseline is only there so the first
    visible bar has a previous period to grow against. A
    calendar period with no reported days stays blank instead
    of becoming zero and pulling the comparison onto an
    older period.
    """

    date_end = pd.Timestamp(date_end).normalize()
    period_days = int(period_days)
    duration = int(duration)

    working = frame.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"]
    ).dt.normalize()

    # duration visible periods, plus the one before them.
    history_start = date_end - pd.Timedelta(
        days=period_days * (duration + 1) - 1
    )

    calendar_rows = []

    for period_index in range(duration + 1):
        period_end = date_end - pd.Timedelta(
            days=period_days * period_index
        )
        period_start = period_end - pd.Timedelta(
            days=period_days - 1
        )
        calendar_rows.append(
            {
                "period_index": period_index,
                "period_start": period_start,
                "period_end": period_end,
            }
        )

    calendar = pd.DataFrame(calendar_rows)

    scoped = working.loc[
        working["business_date"].between(
            history_start,
            date_end,
            inclusive="both",
        )
    ].copy()

    if not scoped.empty:
        scoped["period_index"] = (
            (date_end - scoped["business_date"]).dt.days
            // period_days
        ).astype(int)
        scoped = scoped[
            scoped["period_index"].between(0, duration)
        ]

    if scoped.empty:
        grouped = pd.DataFrame(
            {"period_index": pd.Series(dtype="int64")}
        )
    else:
        grouped = (
            scoped
            .groupby("period_index", as_index=False)
            .agg(
                **{
                    column_name: (column_name, "sum")
                    for column_name in SALES_NUMERIC_COLUMNS
                },
                days_counted=("business_date", "nunique"),
            )
        )

    calendar = calendar.merge(
        grouped,
        on="period_index",
        how="left",
    )

    for column_name in SALES_NUMERIC_COLUMNS:
        if column_name not in calendar.columns:
            calendar[column_name] = NOT_AVAILABLE

    if "days_counted" not in calendar.columns:
        calendar["days_counted"] = 0

    calendar["days_counted"] = (
        calendar["days_counted"].fillna(0).astype(int)
    )

    missing_period = calendar["days_counted"].eq(0)
    calendar.loc[
        missing_period,
        SALES_NUMERIC_COLUMNS,
    ] = NOT_AVAILABLE

    calendar["is_complete"] = (
        calendar["days_counted"] >= period_days
    )

    previous = calendar[
        ["period_index", "net_sales"]
    ].rename(
        columns={
            "period_index": "previous_index",
            "net_sales": "previous_sales",
        }
    )
    calendar["previous_index"] = calendar["period_index"] + 1
    calendar = calendar.merge(
        previous,
        on="previous_index",
        how="left",
    )
    calendar = calendar.drop(columns=["previous_index"])

    calendar["sales_change"] = (
        calendar["net_sales"] - calendar["previous_sales"]
    )
    calendar["sales_growth"] = (
        (
            calendar["sales_change"]
            / calendar["previous_sales"].abs()
        )
        * 100
    ).replace(
        [float("inf"), float("-inf")],
        NOT_AVAILABLE,
    )

    include_year = bool(
        calendar["period_start"].min().year
        != calendar["period_end"].max().year
    )

    calendar["period_label"] = [
        format_period_label(
            start,
            end,
            period_days,
            include_year,
        )
        for start, end in zip(
            calendar["period_start"],
            calendar["period_end"],
        )
    ]
    calendar["period_range"] = [
        format_period_range(start, end, period_days)
        for start, end in zip(
            calendar["period_start"],
            calendar["period_end"],
        )
    ]

    return calendar.sort_values("period_index").reset_index(drop=True)


def channel_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per channel per period, including the baseline.

    channel_growth compares a channel with the same channel in
    the previous period. The baseline period stays in the
    result so the first visible period has a prior value.
    """

    value_columns = {
        **CHANNEL_COLUMNS,
        "other_sales": "Other",
    }

    long_df = frame.melt(
        id_vars=[
            "period_index",
            "period_label",
            "period_range",
            "period_start",
            "period_end",
            "net_sales",
            "previous_sales",
            "sales_change",
            "days_counted",
            "is_complete",
        ],
        value_vars=list(value_columns),
        var_name="channel_key",
        value_name="channel_sales",
    )

    long_df["channel"] = long_df["channel_key"].map(value_columns)

    previous = long_df[
        ["period_index", "channel_key", "channel_sales"]
    ].rename(
        columns={
            "period_index": "previous_index",
            "channel_sales": "previous_channel_sales",
        }
    )
    long_df["previous_index"] = long_df["period_index"] + 1
    long_df = long_df.merge(
        previous,
        on=["previous_index", "channel_key"],
        how="left",
    )
    long_df = long_df.drop(columns=["previous_index"])

    long_df["channel_growth"] = (
        (
            long_df["channel_sales"]
            - long_df["previous_channel_sales"]
        )
        / long_df["previous_channel_sales"].abs()
        * 100
    ).replace(
        [float("inf"), float("-inf")],
        NOT_AVAILABLE,
    )
    long_df["share_percent"] = (
        long_df["channel_sales"]
        / long_df["net_sales"].replace(0.0, NOT_AVAILABLE)
        * 100
    )
    # Vega's percent format multiplies by 100, so the tooltip
    # uses the 0–1 ratio and still reads as a percent.
    long_df["share_ratio"] = long_df["share_percent"] / 100

    return long_df


def stack_channel_sales(
    frame: pd.DataFrame,
    duration: int,
) -> pd.DataFrame:
    """Visible channel rows for the stacked bars.

    stack_rank 1 is that period's largest channel. The chart
    draws ascending rank from the baseline, which puts the
    largest channel on the bottom and the smallest on top.
    """

    long_df = channel_rows(frame)
    long_df = long_df.loc[
        long_df["period_index"] < duration
    ].copy()
    long_df = long_df[
        long_df["channel_sales"].notna()
        & long_df["channel_sales"].abs().gt(0.005)
    ].copy()

    if long_df.empty:
        return long_df

    # Rank inside the period, not by the axis label, so two
    # buckets cannot share a stack order.
    long_df["stack_rank"] = (
        long_df
        .groupby("period_end", sort=False)["channel_sales"]
        .rank(ascending=False, method="first")
    )

    return long_df.sort_values(
        ["period_end", "stack_rank"],
        ascending=[True, True],
    )


def growth_axis_domain(frame: pd.DataFrame) -> tuple[float, float]:
    """Percent axis that includes 0% and a little padding."""

    growth_values = frame["sales_growth"].dropna()

    if growth_values.empty:
        return -1.0, 1.0

    raw_min = min(0.0, float(growth_values.min()))
    raw_max = max(0.0, float(growth_values.max()))
    padding = max((raw_max - raw_min) * 0.10, 1.0)

    return raw_min - padding, raw_max + padding


def build_trend_chart(
    period_df: pd.DataFrame,
    stacked_df: pd.DataFrame,
    label_order: list[str],
    x_title: str,
    growth_title: str,
    channel_order: list[str],
    sales_axis_max: float,
    growth_axis_min: float,
    growth_axis_max: float,
    label_angle: int,
) -> alt.Chart:
    """Stacked channel bars with a growth line on a second axis."""

    layers = []
    x_encoding = alt.X(
        "period_label:N",
        title=x_title,
        sort=label_order,
        axis=alt.Axis(
            labelAngle=label_angle,
            labelOverlap="greedy",
        ),
    )

    if not stacked_df.empty:
        layers.append(
            alt.Chart(stacked_df)
            .mark_bar(stroke="white", strokeWidth=0.4)
            .encode(
                x=x_encoding,
                y=alt.Y(
                    "channel_sales:Q",
                    title="Net Sales",
                    stack="zero",
                    axis=alt.Axis(
                        format="$,.0f",
                        titleColor="#1f2937",
                    ),
                    scale=alt.Scale(domain=[0, sales_axis_max]),
                ),
                color=alt.Color(
                    "channel:N",
                    title="Channel",
                    sort=channel_order,
                    scale=alt.Scale(
                        domain=channel_order,
                        range=[
                            CHANNEL_COLORS.get(name, "#7f7f7f")
                            for name in channel_order
                        ],
                    ),
                ),
                order=alt.Order("stack_rank:Q", sort="ascending"),
                opacity=alt.condition(
                    alt.datum.is_complete,
                    alt.value(0.92),
                    alt.value(0.4),
                ),
                tooltip=[
                    alt.Tooltip(
                        "period_range:N",
                        title="Period",
                    ),
                    alt.Tooltip("channel:N", title="Channel"),
                    alt.Tooltip(
                        "channel_sales:Q",
                        title="Channel sales",
                        format="$,.0f",
                    ),
                    alt.Tooltip(
                        "share_ratio:Q",
                        title="Share %",
                        format=".1%",
                    ),
                    alt.Tooltip(
                        "channel_growth:Q",
                        title="Channel growth (%)",
                        format="+.2f",
                    ),
                    alt.Tooltip(
                        "days_counted:Q",
                        title="Days counted",
                        format=".0f",
                    ),
                ],
            )
        )

    growth_scale = alt.Scale(
        domain=[growth_axis_min, growth_axis_max],
        zero=False,
    )

    layers.append(
        alt.Chart(period_df)
        .mark_line(
            color=GROWTH_COLOR,
            strokeWidth=3,
            point=alt.OverlayMarkDef(
                color=GROWTH_COLOR,
                size=55,
            ),
        )
        .encode(
            x=x_encoding,
            y=alt.Y(
                "sales_growth:Q",
                title=growth_title,
                axis=alt.Axis(
                    orient="right",
                    format=".1f",
                    labelExpr="datum.value + '%'",
                    titleColor=GROWTH_COLOR,
                    labelColor=GROWTH_COLOR,
                    grid=False,
                ),
                scale=growth_scale,
            ),
            tooltip=[
                alt.Tooltip("period_range:N", title="Period"),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "previous_sales:Q",
                    title="Previous period",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "sales_growth:Q",
                    title="Growth (%)",
                    format="+.2f",
                ),
            ],
        )
    )

    layers.append(
        alt.Chart(pd.DataFrame({"zero_growth": [0.0]}))
        .mark_rule(
            color="#111827",
            strokeWidth=3,
            strokeDash=[8, 4],
        )
        .encode(
            y=alt.Y(
                "zero_growth:Q",
                axis=None,
                scale=growth_scale,
            ),
        )
    )

    return (
        alt.layer(*layers)
        .resolve_scale(y="independent")
        .properties(height=440)
        .configure_legend(orient="bottom", columns=4)
    )


st.title("Sales Growth")
st.caption(
    "Net sales by channel for a date range you choose, "
    "compared with the previous day, week, or 30-day month."
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

selected_interval = st.sidebar.selectbox(
    "Interval",
    options=list(INTERVALS),
    index=list(INTERVALS).index("Month over month"),
    help=(
        "Periods are counted backward from Date end. "
        "A week is 7 days and a month is 30 days."
    ),
)

interval = INTERVALS[selected_interval]
period_days = interval["days"]

min_date = sales_df["business_date"].min().date()
max_date = sales_df["business_date"].max().date()
date_key = "sales_growth_date_end"

if date_key not in st.session_state:
    st.session_state[date_key] = max_date
else:
    chosen_date = st.session_state[date_key]
    if chosen_date < min_date or chosen_date > max_date:
        st.session_state[date_key] = min(
            max(chosen_date, min_date),
            max_date,
        )

date_end = st.sidebar.date_input(
    "Date end",
    min_value=min_date,
    max_value=max_date,
    key=date_key,
    help="Last business date included in the chart.",
)

duration = st.sidebar.number_input(
    interval["duration_label"],
    min_value=1,
    max_value=interval["max_duration"],
    value=interval["default_duration"],
    step=1,
    key=f"sales_growth_duration_{selected_interval}",
    help=(
        f"How many {interval['unit']} to include, ending "
        f"on Date end. "
        + (
            "Each month is 30 days. "
            if period_days == 30
            else ""
        )
        + f"Maximum is {interval['max_duration']}."
    ),
)
duration = int(duration)

if period_days == 1:
    st.sidebar.caption("Each bar is one business day.")
elif period_days == 7:
    st.sidebar.caption(
        "Each bar is 7 days, ending on Date end."
    )
else:
    st.sidebar.caption(
        "Each bar is 30 days, ending on Date end."
    )

if date_end is None:
    st.warning("Pick a date end.")
    st.stop()

if location_df.empty:
    st.warning("No sales data matches the selected filters.")
    st.stop()

location_max = location_df["business_date"].max()

if pd.Timestamp(date_end) > location_max:
    st.caption(
        f"The latest sales date for {selected_location} is "
        f"{location_max:%b %d, %Y}."
    )


# ---------------------------------------------------------
# Period aggregation
# ---------------------------------------------------------

period_df = build_period_frame(
    location_df,
    date_end,
    period_days,
    duration,
)

# period_index 0 is the bucket that ends on Date end.
# The highest index is the hidden comparison baseline.
period_df["channel_total"] = period_df[list(CHANNEL_COLUMNS)].sum(
    axis=1,
    min_count=1,
)
period_df["other_sales"] = (
    period_df["net_sales"] - period_df["channel_total"]
)
period_df.loc[
    period_df["other_sales"] <= 0.50,
    "other_sales",
] = NOT_AVAILABLE

visible = period_df[
    period_df["period_index"] < duration
].copy()

window_start = pd.Timestamp(
    visible.loc[visible["period_index"].idxmax(), "period_start"]
)
window_end = pd.Timestamp(date_end)

if window_start == window_end:
    window_text = f"{window_end:%b %d, %Y}"
else:
    window_text = (
        f"{window_start:%b %d, %Y} – {window_end:%b %d, %Y}"
    )

st.subheader(f"Net Sales: {selected_interval}")
st.caption(
    f"{window_text} · {duration} {interval['unit']} · "
    f"{selected_location}"
)

if visible["net_sales"].notna().sum() == 0:
    st.info(
        f"No sales fall between {window_start:%b %d, %Y} and "
        f"{window_end:%b %d, %Y}."
    )
    st.stop()

current = visible.loc[visible["period_index"] == 0].iloc[0]
previous_rows = period_df.loc[period_df["period_index"] == 1]
previous = (
    previous_rows.iloc[0] if not previous_rows.empty else None
)

current_growth = percent_change(
    current["net_sales"],
    current["previous_sales"],
)


# ---------------------------------------------------------
# Period summary
# ---------------------------------------------------------

summary_columns = st.columns(4)

summary_columns[0].metric(
    "Selected period",
    format_dollars(current["net_sales"]),
    help=(
        f"{current['period_range']} "
        f"({int(current['days_counted'])} of "
        f"{period_days} days)"
    ),
)

previous_help = None

if previous is not None:
    previous_help = (
        f"{previous['period_range']} "
        f"({int(previous['days_counted'])} of "
        f"{period_days} days)"
    )

summary_columns[1].metric(
    "Previous period",
    format_dollars(current["previous_sales"]),
    help=previous_help,
)

summary_columns[2].metric(
    "Change",
    format_dollars(current["sales_change"]),
)

growth_has_value = not pd.isna(current_growth)

summary_columns[3].metric(
    interval["growth_title"],
    (
        f"{current_growth:+.2f}%"
        if growth_has_value
        else "—"
    ),
    delta=(
        f"{current_growth:+.2f}%"
        if growth_has_value
        else None
    ),
)

if not bool(current["is_complete"]):
    st.caption(
        f"The period ending {current['period_end']:%b %d, %Y} "
        f"has {int(current['days_counted'])} of "
        f"{period_days} day(s) of data, so its total is short."
    )


# ---------------------------------------------------------
# Trend and growth
# ---------------------------------------------------------

st.subheader("Trend and Growth")

stacked_df = stack_channel_sales(period_df, duration)

if stacked_df.empty:
    channel_order = []
    sales_peak = 0.0
else:
    channel_order = (
        stacked_df
        .groupby("channel")["channel_sales"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    sales_peak = float(
        stacked_df
        .groupby("period_label")["channel_sales"]
        .sum()
        .max()
    )

if pd.isna(sales_peak) or sales_peak <= 0:
    sales_axis_max = 1.0
else:
    sales_axis_max = sales_peak * 1.05

growth_axis_min, growth_axis_max = growth_axis_domain(visible)

label_order = (
    visible
    .sort_values("period_end")["period_label"]
    .tolist()
)

trend_chart = build_trend_chart(
    visible.sort_values("period_end"),
    stacked_df,
    label_order,
    interval["x_title"],
    interval["growth_title"],
    channel_order,
    sales_axis_max,
    growth_axis_min,
    growth_axis_max,
    label_angle=-40 if duration > 6 else 0,
)

st.altair_chart(trend_chart, use_container_width=True)

st.caption(
    "Each bar stacks channel net sales, with that period's "
    "largest channel at the bottom. Hover a segment for that "
    "channel's share and its growth versus the previous "
    "period. The orange line is net sales growth, and the "
    "dashed line is 0%."
)

channels_exceed_net = (
    visible["channel_total"].fillna(0)
    > visible["net_sales"].fillna(0) + 0.5
).any()

if channels_exceed_net:
    st.caption(
        "On some periods the reported channels add up to "
        "more than net sales, so those bars are taller than "
        "net sales. Other fills the gap when the channels "
        "add up to less."
    )
elif visible["other_sales"].notna().any():
    st.caption(
        "Other is the gap between reported channel sales "
        "and net sales. Revel reports channels independently, "
        "so shares do not always total exactly 100%."
    )

incomplete_labels = (
    visible.loc[~visible["is_complete"]]
    .sort_values("period_end")["period_label"]
    .tolist()
)

if incomplete_labels:
    st.caption(
        "Faded bars have fewer than "
        f"{period_days} day(s) of data: "
        f"{', '.join(incomplete_labels)}. Their totals "
        "and growth are understated."
    )


# ---------------------------------------------------------
# Period breakdown
# ---------------------------------------------------------

with st.expander("View period breakdown"):
    # One row per channel per period, instead of one column
    # per channel. Other is omitted when there is no gap.
    breakdown = channel_rows(period_df)
    breakdown = breakdown.loc[
        (breakdown["period_index"] < duration)
        & ~(
            breakdown["channel"].eq("Other")
            & breakdown["channel_sales"].isna()
        )
        & (
            breakdown["net_sales"].notna()
            | breakdown["channel_sales"].notna()
        )
    ].copy()
    breakdown = breakdown.sort_values(
        ["period_end", "channel_sales"],
        ascending=[False, False],
        na_position="last",
    )
    breakdown["period_start"] = breakdown["period_start"].dt.date
    breakdown["period_end"] = breakdown["period_end"].dt.date

    st.dataframe(
        breakdown[
            [
                "period_start",
                "period_end",
                "days_counted",
                "channel",
                "channel_sales",
                "share_percent",
                "channel_growth",
                "net_sales",
                "previous_sales",
                "sales_change",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "period_start": st.column_config.DateColumn(
                "Period start",
            ),
            "period_end": st.column_config.DateColumn(
                "Period end",
            ),
            "days_counted": st.column_config.NumberColumn(
                "Days counted",
                format="%d",
            ),
            "channel": st.column_config.TextColumn(
                "Channel",
            ),
            "channel_sales": st.column_config.NumberColumn(
                "Sales",
                format="$%,.0f",
            ),
            "share_percent": st.column_config.NumberColumn(
                "Share %",
                format="%.1f%%",
            ),
            "channel_growth": st.column_config.NumberColumn(
                "Channel growth %",
                format="%.2f%%",
            ),
            "net_sales": st.column_config.NumberColumn(
                "Net sales",
                format="$%,.0f",
            ),
            "previous_sales": st.column_config.NumberColumn(
                "Previous period",
                format="$%,.0f",
            ),
            "sales_change": st.column_config.NumberColumn(
                "Change",
                format="$%,.0f",
            ),
        },
    )


st.caption(
    f"Displaying {duration:,} {interval['unit']} of net sales "
    f"for {selected_location.lower()}, ending "
    f"{pd.Timestamp(date_end):%b %d, %Y}."
)
