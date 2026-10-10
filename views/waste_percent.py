import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    MissingSecretError,
    escape_dollar_signs,
    load_sales_data,
    load_waste_log,
    location_filter_controls,
)


CHICKEN_CUTOFF = 0.25
FRIES_CUTOFF = 0.15

OVER_STYLE = (
    "background-color: rgba(180, 35, 24, 0.16); color: #b42318"
)
WITHIN_STYLE = (
    "background-color: rgba(34, 139, 84, 0.18); color: #157347"
)
SERIES_COLORS = {
    "Chicken": "#1f77b4",
    "Fries": "#ff7f0e",
}


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


def format_percent(value) -> str:
    """Waste percent of net sales, to three decimals."""

    if value is None or pd.isna(value):
        return "—"
    return f"{value:.3f}%"


def sales_by_day(
    sales_frame: pd.DataFrame,
    location_keys,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Net sales for each location and day in the calendar range."""

    empty = pd.DataFrame(
        columns=["location", "location_key", "business_date", "net_sales"]
    )
    if (
        sales_frame.empty
        or "net_sales" not in sales_frame.columns
        or "location_key" not in sales_frame.columns
        or "business_date" not in sales_frame.columns
    ):
        return empty

    sales = sales_frame.copy()
    sales["business_date"] = pd.to_datetime(
        sales["business_date"],
        errors="coerce",
    ).dt.normalize()
    sales = sales[
        sales["location_key"].isin(location_keys)
        & sales["business_date"].between(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
        )
    ]
    if sales.empty:
        return empty

    return (
        sales.groupby(
            ["location_key", "business_date"],
            as_index=False,
        )
        .agg(
            location=("location", "first"),
            net_sales=("net_sales", "sum"),
        )
    )


def waste_by_day(waste_frame: pd.DataFrame) -> pd.DataFrame:
    """Chicken and fries waste dollars for each location and day."""

    empty = pd.DataFrame(
        columns=[
            "location",
            "location_key",
            "business_date",
            "chicken_waste",
            "fries_waste",
        ]
    )
    if waste_frame.empty:
        return empty

    grouped = (
        waste_frame.groupby(
            ["location", "location_key", "business_date", "category"],
            as_index=False,
        )["total"]
        .sum()
    )
    wide = grouped.pivot_table(
        index=["location", "location_key", "business_date"],
        columns="category",
        values="total",
        aggfunc="sum",
    ).reset_index()
    wide.columns.name = None
    for category in ("chicken", "fries"):
        if category not in wide.columns:
            wide[category] = 0.0
    return wide.rename(
        columns={
            "chicken": "chicken_waste",
            "fries": "fries_waste",
        }
    )


def daily_waste(
    waste_frame: pd.DataFrame,
    sales_frame: pd.DataFrame,
    location_keys,
    start_date,
    end_date,
) -> pd.DataFrame:
    """One row per location and day, with waste percent of net sales."""

    waste_days = waste_by_day(waste_frame)
    sales_days = sales_by_day(
        sales_frame,
        location_keys,
        start_date,
        end_date,
    )
    waste_days["business_date"] = pd.to_datetime(
        waste_days["business_date"],
        errors="coerce",
    )
    sales_days["business_date"] = pd.to_datetime(
        sales_days["business_date"],
        errors="coerce",
    )
    daily = waste_days.merge(
        sales_days,
        on=["location_key", "business_date"],
        how="outer",
        suffixes=("", "_sales"),
    )
    if "location_sales" in daily.columns:
        daily["location"] = daily["location"].fillna(
            daily["location_sales"]
        )
        daily = daily.drop(columns=["location_sales"])

    if "net_sales" not in daily.columns:
        daily["net_sales"] = float("nan")
    for column in ("chicken_waste", "fries_waste"):
        if column not in daily.columns:
            daily[column] = 0.0
        daily[column] = daily[column].fillna(0)

    usable = daily["net_sales"].notna() & (daily["net_sales"] != 0)
    daily["chicken_percent"] = float("nan")
    daily["fries_percent"] = float("nan")
    daily.loc[usable, "chicken_percent"] = (
        daily.loc[usable, "chicken_waste"]
        / daily.loc[usable, "net_sales"]
        * 100
    )
    daily.loc[usable, "fries_percent"] = (
        daily.loc[usable, "fries_waste"]
        / daily.loc[usable, "net_sales"]
        * 100
    )
    return daily.sort_values(
        ["business_date", "location"]
    ).reset_index(drop=True)


def range_percent(daily: pd.DataFrame, waste_column: str):
    """Waste dollars divided by net sales, for days that have sales."""

    scored = daily[daily["net_sales"].notna() & (daily["net_sales"] != 0)]
    net_sales = float(scored["net_sales"].sum())
    waste_dollars = float(scored[waste_column].sum())
    if net_sales == 0:
        return waste_dollars, net_sales, float("nan")
    return waste_dollars, net_sales, waste_dollars / net_sales * 100


def percent_cell_style(cutoff: float):
    """Green at or under the cutoff, red above it."""

    def style_cell(value) -> str:
        if pd.isna(value):
            return ""
        if value > cutoff:
            return OVER_STYLE
        return WITHIN_STYLE

    return style_cell


def style_detail(
    frame: pd.DataFrame,
    chicken_cutoff: float,
    fries_cutoff: float,
):
    """Color each waste percent against its cutoff."""

    styler = frame.style
    styler = styler.map(
        percent_cell_style(chicken_cutoff),
        subset=["chicken_percent"],
    )
    return styler.map(
        percent_cell_style(fries_cutoff),
        subset=["fries_percent"],
    )


def detail_column_config() -> dict:
    """Readable headers for the daily table."""

    return {
        "business_date": st.column_config.DateColumn("Date"),
        "location": "Location",
        "net_sales": st.column_config.NumberColumn(
            "Net sales",
            format="$%.2f",
        ),
        "chicken_waste": st.column_config.NumberColumn(
            "Chicken waste",
            format="$%.2f",
        ),
        "chicken_percent": st.column_config.NumberColumn(
            "Chicken %",
            format="%.3f%%",
        ),
        "fries_waste": st.column_config.NumberColumn(
            "Fries waste",
            format="$%.2f",
        ),
        "fries_percent": st.column_config.NumberColumn(
            "Fries %",
            format="%.3f%%",
        ),
    }


def daily_chart(
    daily: pd.DataFrame,
    chicken_cutoff: float,
    fries_cutoff: float,
) -> alt.Chart:
    """Daily waste percent, with a line at each cutoff."""

    scored = daily[
        daily["net_sales"].notna() & (daily["net_sales"] != 0)
    ].copy()
    by_day = scored.groupby("business_date", as_index=False).agg(
        chicken_waste=("chicken_waste", "sum"),
        fries_waste=("fries_waste", "sum"),
        net_sales=("net_sales", "sum"),
    )
    by_day["chicken_percent"] = (
        by_day["chicken_waste"] / by_day["net_sales"] * 100
    )
    by_day["fries_percent"] = (
        by_day["fries_waste"] / by_day["net_sales"] * 100
    )
    long = by_day.melt(
        id_vars=["business_date", "net_sales"],
        value_vars=["chicken_percent", "fries_percent"],
        var_name="series",
        value_name="percent",
    )
    long["series"] = long["series"].map(
        {
            "chicken_percent": "Chicken",
            "fries_percent": "Fries",
        }
    )
    waste_dollars = by_day.melt(
        id_vars=["business_date"],
        value_vars=["chicken_waste", "fries_waste"],
        var_name="series",
        value_name="waste",
    )
    waste_dollars["series"] = waste_dollars["series"].map(
        {
            "chicken_waste": "Chicken",
            "fries_waste": "Fries",
        }
    )
    long = long.merge(
        waste_dollars,
        on=["business_date", "series"],
        how="left",
    )
    long["day"] = long["business_date"].dt.strftime("%b %d")
    day_order = (
        long.sort_values("business_date")["day"]
        .drop_duplicates()
        .tolist()
    )

    y_max = max(
        float(long["percent"].max()),
        chicken_cutoff,
        fries_cutoff,
        0.01,
    ) * 1.25
    color = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(
            domain=list(SERIES_COLORS),
            range=list(SERIES_COLORS.values()),
        ),
    )
    lines = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "day:N",
                title=None,
                sort=day_order,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "percent:Q",
                title="Waste % of net sales",
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(format=".3f"),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("day:N", title="Date"),
                alt.Tooltip("series:N", title="Item"),
                alt.Tooltip("percent:Q", title="Waste %", format=".3f"),
                alt.Tooltip("waste:Q", title="Waste $", format="$,.2f"),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.2f",
                ),
            ],
        )
    )
    cutoffs = pd.DataFrame(
        {
            "series": ["Chicken", "Fries"],
            "cutoff": [chicken_cutoff, fries_cutoff],
        }
    )
    rules = (
        alt.Chart(cutoffs)
        .mark_rule(strokeDash=[6, 4])
        .encode(y="cutoff:Q", color=color)
    )
    return (lines + rules).properties(height=320)


st.title("Waste %")
st.caption(
    "Chicken and fries waste dollars as a percent of net sales. "
    "A day is over the cutoff when that percent is higher."
)

try:
    waste_df = load_waste_log()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "Unable to load waste_log_history from Supabase: "
        f"{error}"
    )
    st.stop()

if waste_df.empty:
    st.warning(
        "No chicken or fries waste was returned from "
        "waste_log_history."
    )
    st.stop()

if "location" not in waste_df.columns:
    st.error("waste_log_history has no location column.")
    st.stop()

if "business_date" not in waste_df.columns:
    st.error("waste_log_history has no waste_date column.")
    st.stop()


st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(waste_df)

if location_df.empty:
    st.warning("No waste rows match that location.")
    st.stop()

available_dates = location_df["business_date"].dropna()
if available_dates.empty:
    st.warning("No waste rows have a date.")
    st.stop()

min_date = available_dates.min().date()
max_date = available_dates.max().date()
default_start = max(min_date, max_date - dt.timedelta(days=6))

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    key=f"waste_percent_range_{selected_location}",
)
chicken_cutoff = st.sidebar.number_input(
    "Chicken cutoff (%)",
    min_value=0.0,
    value=CHICKEN_CUTOFF,
    step=0.01,
    format="%.2f",
    help=(
        "Chicken waste is over the cutoff when it exceeds "
        "this percent of net sales."
    ),
)
fries_cutoff = st.sidebar.number_input(
    "Fries cutoff (%)",
    min_value=0.0,
    value=FRIES_CUTOFF,
    step=0.01,
    format="%.2f",
    help=(
        "Fries waste is over the cutoff when it exceeds "
        "this percent of net sales."
    ),
)

bounds = selected_bounds(selected_dates)
if bounds is None:
    st.info("Select a start and end date.")
    st.stop()

start_date, end_date = bounds
range_df = location_df[
    location_df["business_date"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
].copy()
range_label = format_range(start_date, end_date)

try:
    sales_df = load_sales_data()
except Exception:
    sales_df = pd.DataFrame()

location_keys = (
    location_df["location_key"].dropna().unique().tolist()
    if "location_key" in location_df.columns
    else []
)
daily = daily_waste(
    range_df,
    sales_df,
    location_keys,
    start_date,
    end_date,
)
scored = daily[daily["chicken_percent"].notna()].copy()

chicken_dollars, net_sales, chicken_percent = range_percent(
    daily,
    "chicken_waste",
)
fries_dollars, _, fries_percent = range_percent(daily, "fries_waste")

st.subheader(range_label)

if net_sales == 0 or pd.isna(chicken_percent):
    sales_help = "Net sales were not available for this date range."
else:
    sales_help = escape_dollar_signs(
        f"${chicken_dollars:,.2f} chicken waste and "
        f"${fries_dollars:,.2f} fries waste of "
        f"${net_sales:,.2f} net sales in {range_label}. "
        "The figure under each percent is how far it sits "
        "from that item's cutoff."
    )

measured = len(scored)
chicken_over = int((scored["chicken_percent"] > chicken_cutoff).sum())
fries_over = int((scored["fries_percent"] > fries_cutoff).sum())

summary = st.columns(4)
summary[0].metric(
    "Chicken waste %",
    format_percent(chicken_percent),
    delta=None if pd.isna(chicken_percent) else round(
        chicken_percent - chicken_cutoff,
        3,
    ),
    delta_color="inverse",
    help=sales_help,
)
summary[1].metric(
    "Fries waste %",
    format_percent(fries_percent),
    delta=None if pd.isna(fries_percent) else round(
        fries_percent - fries_cutoff,
        3,
    ),
    delta_color="inverse",
    help=sales_help,
)
summary[2].metric(
    "Days over chicken cutoff",
    "—" if measured == 0 else f"{chicken_over} of {measured}",
    help=f"Days above {chicken_cutoff:.2f}% of net sales.",
)
summary[3].metric(
    "Days over fries cutoff",
    "—" if measured == 0 else f"{fries_over} of {measured}",
    help=f"Days above {fries_cutoff:.2f}% of net sales.",
)

if scored.empty:
    st.info(
        "Net sales were not available for these locations and dates, "
        "so waste % cannot be calculated."
    )
    st.stop()

st.altair_chart(
    daily_chart(daily, chicken_cutoff, fries_cutoff),
    use_container_width=True,
)
st.caption(
    "Dashed lines are the chicken and fries cutoffs. "
    "Each point is that day's waste dollars divided by net sales."
)
unmatched = daily[
    daily["chicken_percent"].isna()
    & (
        (daily["chicken_waste"] > 0)
        | (daily["fries_waste"] > 0)
    )
]
if not unmatched.empty:
    st.caption(
        f"{len(unmatched):,} days have waste logged but no net sales, "
        "so they are not included in the percent."
    )

outside = scored[
    (scored["chicken_percent"] > chicken_cutoff)
    | (scored["fries_percent"] > fries_cutoff)
].copy()
outside["business_date"] = outside["business_date"].dt.date

st.subheader("Days over a cutoff")
if outside.empty:
    st.info("Every measured day is within both cutoffs.")
else:
    detail = outside[
        [
            "business_date",
            "location",
            "net_sales",
            "chicken_waste",
            "chicken_percent",
            "fries_waste",
            "fries_percent",
        ]
    ].sort_values(
        ["business_date", "location"],
        ascending=[False, True],
    )
    st.dataframe(
        style_detail(detail, chicken_cutoff, fries_cutoff),
        column_config=detail_column_config(),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        escape_dollar_signs(
            f"{len(detail):,} of {measured:,} days with net sales "
            f"in {range_label} are over a cutoff. "
            "Green is at or under the cutoff, red is over it."
        )
    )
