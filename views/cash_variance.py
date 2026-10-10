import pandas as pd
import streamlit as st

from revel_data import (
    CASH_SUMMARY_NUMERIC_COLUMNS,
    WEEK_START_OPTIONS,
    MissingSecretError,
    add_week_columns,
    escape_dollar_signs,
    load_cash_summary,
    location_filter_controls,
)


# Columns created for filtering. They are not on the
# Supabase table, so the detail table leaves them out.
HELPER_COLUMNS = [
    "location_key",
    "day_index",
    "week_start",
]

MONEY_COLUMNS = [
    column_name
    for column_name in CASH_SUMMARY_NUMERIC_COLUMNS
    if column_name != "number_of_no_sales"
]


def format_week(week_start) -> str:
    """Label a week by its first and last calendar day."""

    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    return f"{start:%b %d} – {end:%b %d, %Y}"


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
# Sidebar: location, week, and the tolerance amount
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(
    cash_df,
)

if location_df.empty:
    st.warning("No cash summary rows match that location.")
    st.stop()

selected_week_start_name = st.sidebar.selectbox(
    "Week starts on",
    options=list(WEEK_START_OPTIONS.keys()),
)
week_start_weekday = WEEK_START_OPTIONS[selected_week_start_name]

location_df = add_week_columns(location_df, week_start_weekday)
week_starts = sorted(
    location_df["week_start"].dropna().unique(),
    reverse=True,
)
selected_week = st.sidebar.selectbox(
    "Week",
    options=week_starts,
    format_func=format_week,
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

week_df = location_df[
    location_df["week_start"] == selected_week
].copy()

if week_df.empty:
    st.info("No cash summary rows fall in that week.")
    st.stop()

week_label = format_week(selected_week)
measured = week_df.dropna(subset=["variance"])
within = measured[measured["variance"].abs() <= tolerance]
outside = measured[measured["variance"].abs() > tolerance]
surplus = extreme_row(measured, pick_largest=True)
shortage = extreme_row(measured, pick_largest=False)
tolerance_label = f"${tolerance:,.2f}"


# ---------------------------------------------------------
# KPIs for the selected week
# ---------------------------------------------------------

st.subheader(week_label)

summary_columns = st.columns(4)

summary_columns[0].metric(
    f"Days within ±{tolerance_label}",
    f"{len(within):,}",
    help=f"{selected_location} · {week_label}",
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
        f"{len(outside):,} days in {week_label}, sorted by "
        "variance descending. Source: "
        "public.daily-revel-cash-summary."
    )
)

if outside.empty:
    st.info(
        escape_dollar_signs(
            f"Every business day in {week_label} is within "
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
