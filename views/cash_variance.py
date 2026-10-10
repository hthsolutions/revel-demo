import datetime as dt

import pandas as pd
import streamlit as st

from revel_data import (
    CASH_SUMMARY_NUMERIC_COLUMNS,
    MissingSecretError,
    escape_dollar_signs,
    load_cash_summary,
    location_filter_controls,
)


# Columns created for filtering. They are not on the
# Supabase table, so the detail table leaves them out.
HELPER_COLUMNS = [
    "location_key",
]

MONEY_COLUMNS = [
    column_name
    for column_name in CASH_SUMMARY_NUMERIC_COLUMNS
    if column_name != "number_of_no_sales"
]


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


def format_signed_currency(value) -> str:
    """Currency with the sign in front of the dollar mark."""

    if value is None or pd.isna(value):
        return "—"

    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def extreme_row(
    frame: pd.DataFrame,
    pick_largest: bool,
) -> pd.Series | None:
    """Row with the highest or lowest variance."""

    values = frame["variance"].dropna()
    if values.empty:
        return None

    target = values.max() if pick_largest else values.min()
    matches = frame[frame["variance"] == target]
    return matches.sort_values("business_date").iloc[-1]


def detail_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Source columns only, variance descending."""

    hidden = [
        column_name
        for column_name in HELPER_COLUMNS
        if column_name in frame.columns
    ]
    detail = frame.drop(columns=hidden).copy()
    detail = detail.sort_values("variance", ascending=False)

    if "business_date" in detail.columns:
        detail["business_date"] = detail["business_date"].dt.date

    if "extracted_at" in detail.columns:
        detail["extracted_at"] = pd.to_datetime(
            detail["extracted_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)

    return detail


def detail_column_config(
    columns: list[str],
) -> dict:
    """Readable headers and currency formats for the source table."""

    config = {}
    for column_name in columns:
        label = column_name.replace("_", " ").title()
        if column_name == "id":
            label = "ID"

        if column_name in MONEY_COLUMNS:
            config[column_name] = st.column_config.NumberColumn(
                label,
                format="$%.2f",
            )
        elif column_name == "number_of_no_sales":
            config[column_name] = st.column_config.NumberColumn(
                label,
                format="%d",
            )
        elif column_name == "business_date":
            config[column_name] = st.column_config.DateColumn(
                label,
            )
        elif column_name == "extracted_at":
            config[column_name] = st.column_config.DatetimeColumn(
                label,
            )
        elif column_name not in config:
            config[column_name] = label

    return config


st.title("Cash Variance")
st.caption(
    "Declared cash versus expected cash. A day is within "
    "the window when variance is between −X and +X."
)

try:
    cash_df = load_cash_summary()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "Unable to load cash summary from Supabase: "
        f"{error}"
    )
    st.stop()

if cash_df.empty:
    st.warning(
        "No rows were returned from "
        "daily-revel-cash-summary."
    )
    st.stop()

if "variance" not in cash_df.columns:
    st.error(
        "daily-revel-cash-summary has no variance column."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar: location, dates, and the tolerance amount
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(
    cash_df,
)

if location_df.empty:
    st.warning("No cash summary rows match that location.")
    st.stop()

if "business_date" not in location_df.columns:
    st.error(
        "daily-revel-cash-summary has no business_date column."
    )
    st.stop()

available_dates = location_df["business_date"].dropna()
if available_dates.empty:
    st.warning("No cash summary rows have a business date.")
    st.stop()

min_date = available_dates.min().date()
max_date = available_dates.max().date()
default_start = max(min_date, max_date - dt.timedelta(days=6))

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    key=f"cash_variance_range_{selected_location}",
)
tolerance = st.sidebar.number_input(
    "Tolerance X ($)",
    min_value=0.0,
    value=5.0,
    step=1.0,
    format="%.2f",
    help=(
        "Days within ±X are excluded from the table. "
        "Days outside it are listed below."
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

if range_df.empty:
    st.info("No cash summary rows fall in that date range.")
    st.stop()

range_label = format_range(start_date, end_date)
measured = range_df.dropna(subset=["variance"])
within = measured[measured["variance"].abs() <= tolerance]
outside = measured[measured["variance"].abs() > tolerance]
surplus = extreme_row(measured, pick_largest=True)
shortage = extreme_row(measured, pick_largest=False)
tolerance_label = f"${tolerance:,.2f}"


# ---------------------------------------------------------
# KPIs for the selected dates
# ---------------------------------------------------------

st.subheader(range_label)

summary_columns = st.columns(4)

summary_columns[0].metric(
    f"Days within ±{tolerance_label}",
    f"{len(within):,}",
    help=f"{selected_location} · {range_label}",
)

summary_columns[1].metric(
    f"Days outside ±{tolerance_label}",
    f"{len(outside):,}",
)

summary_columns[2].metric(
    "Max Surplus Variance",
    (
        format_signed_currency(surplus["variance"])
        if surplus is not None
        else "—"
    ),
    help=(
        None
        if surplus is None
        else (
            f"{surplus['business_date']:%b %d, %Y}"
            + (
                f" · {surplus['location']}"
                if "location" in surplus.index
                else ""
            )
        )
    ),
)

summary_columns[3].metric(
    "Max Shortage Variance",
    (
        format_signed_currency(shortage["variance"])
        if shortage is not None
        else "—"
    ),
    help=(
        None
        if shortage is None
        else (
            f"{shortage['business_date']:%b %d, %Y}"
            + (
                f" · {shortage['location']}"
                if "location" in shortage.index
                else ""
            )
        )
    ),
)


# ---------------------------------------------------------
# Days outside the tolerance, full source rows
# ---------------------------------------------------------

st.subheader(f"Days outside ±{tolerance_label}")
st.caption(
    escape_dollar_signs(
        f"{len(outside):,} days in {range_label}, sorted by "
        "variance descending. Source: "
        "public.daily-revel-cash-summary."
    )
)

if outside.empty:
    st.info(
        escape_dollar_signs(
            f"Every business day in {range_label} is within "
            f"±{tolerance_label}."
        )
    )
else:
    table = detail_frame(outside)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config=detail_column_config(list(table.columns)),
    )
