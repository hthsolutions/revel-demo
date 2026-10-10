import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    MissingSecretError,
    escape_dollar_signs,
    load_discount_reasons,
    load_sales_data,
    location_filter_controls,
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


def net_sales_for(
    sales_frame: pd.DataFrame,
    location_keys,
    start_date,
    end_date,
) -> float:
    """Net sales for the selected locations and calendar range."""

    if (
        sales_frame.empty
        or "net_sales" not in sales_frame.columns
        or "location_key" not in sales_frame.columns
        or "business_date" not in sales_frame.columns
    ):
        return float("nan")

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
        return float("nan")
    return float(sales["net_sales"].sum())


def format_sales_share(discount_dollars: float, net_sales: float) -> str:
    """Discount dollars as a percent of net sales."""

    if pd.isna(net_sales) or net_sales == 0:
        return "—"
    return f"{discount_dollars / net_sales:.1%}"


def reason_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Each reason's share of discount dollars in the frame."""

    grouped = (
        frame.groupby("reason", as_index=False)
        .agg(amount=("amount", "sum"), qty=("qty", "sum"))
    )
    grouped = grouped[grouped["amount"] > 0].copy()
    if grouped.empty:
        return grouped

    total = float(grouped["amount"].sum())
    grouped["share"] = grouped["amount"] / total if total else 0.0
    return grouped.sort_values(
        ["share", "reason"],
        ascending=[False, True],
    )


def share_chart(shares: pd.DataFrame) -> alt.Chart:
    """Horizontal bars, largest share at the top."""

    share_max = float(shares["share"].max())
    if pd.isna(share_max) or share_max <= 0:
        share_max = 0.01
    order = shares["reason"].tolist()
    base = alt.Chart(shares)
    bars = (
        base
        .mark_bar(color="#1f77b4")
        .encode(
            y=alt.Y(
                "reason:N",
                title=None,
                sort=order,
                axis=alt.Axis(labelLimit=0),
            ),
            x=alt.X(
                "share:Q",
                title="Share of discount dollars",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, share_max * 1.22]),
            ),
            tooltip=[
                alt.Tooltip("reason:N", title="Reason"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
                alt.Tooltip(
                    "amount:Q",
                    title="Discount dollars",
                    format="$,.2f",
                ),
                alt.Tooltip("qty:Q", title="Quantity", format=",.0f"),
            ],
        )
    )
    labels = (
        base
        .mark_text(align="left", dx=4, color="#31333F")
        .encode(
            y=alt.Y(
                "reason:N",
                sort=order,
                axis=alt.Axis(labelLimit=0),
            ),
            x=alt.X("share:Q"),
            text=alt.Text("share:Q", format=".1%"),
        )
    )
    return (
        alt.layer(bars, labels)
        .properties(height=max(240, 26 * len(shares)))
    )


st.title("Discount Reason")
st.caption(
    "Share of discount dollars by reason. Loyalty, Standard, "
    "and TOTAL repeat those same dollars, so they are not "
    "separate bars."
)

try:
    reason_df = load_discount_reasons()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "Unable to load discount reasons from Supabase: "
        f"{error}"
    )
    st.stop()

if reason_df.empty:
    st.warning(
        "No discount reasons were returned from "
        "daily-revel-discount-reason."
    )
    st.stop()

if "location" not in reason_df.columns:
    st.error(
        "daily-revel-discount-reason has no location column."
    )
    st.stop()

if "business_date" not in reason_df.columns:
    st.error(
        "daily-revel-discount-reason has no business_date column."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar: location and calendar range
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(
    reason_df,
)

if location_df.empty:
    st.warning("No discount reasons match that location.")
    st.stop()

available_dates = location_df["business_date"].dropna()
if available_dates.empty:
    st.warning("No discount reasons have a business date.")
    st.stop()

min_date = available_dates.min().date()
max_date = available_dates.max().date()
default_start = max(min_date, max_date - dt.timedelta(days=6))

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    key=f"discount_reason_range_{selected_location}",
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
    st.info("No discount reasons fall in that date range.")
    st.stop()

range_label = format_range(start_date, end_date)
shares = reason_shares(range_df)

st.subheader(range_label)

if shares.empty:
    st.info(
        "No discount dollars were recorded in that date range."
    )
    st.stop()

discount_dollars = float(shares["amount"].sum())
top_reason = shares.iloc[0]

try:
    sales_df = load_sales_data()
except Exception:
    sales_df = pd.DataFrame()

location_keys = (
    location_df["location_key"].unique()
    if "location_key" in location_df.columns
    else []
)
net_sales = net_sales_for(
    sales_df,
    location_keys,
    start_date,
    end_date,
)
if pd.isna(net_sales) or net_sales == 0:
    sales_help = "Net sales were not available for this date range."
else:
    sales_help = escape_dollar_signs(
        f"${discount_dollars:,.2f} of ${net_sales:,.2f} net sales "
        f"in {range_label}."
    )

summary = st.columns(3)
summary[0].metric(
    "Discount dollars",
    f"${discount_dollars:,.2f}",
    help=f"{selected_location} · {range_label}",
)
summary[1].metric(
    "Discount % of net sales",
    format_sales_share(discount_dollars, net_sales),
    help=sales_help,
)
summary[2].metric(
    f"{top_reason['reason']} share",
    f"{top_reason['share']:.1%}",
    help=escape_dollar_signs(
        f"${top_reason['amount']:,.2f} in {range_label}."
    ),
)

st.altair_chart(
    share_chart(shares),
    use_container_width=True,
)
st.caption(
    escape_dollar_signs(
        f"{len(shares):,} reasons in {range_label}. "
        "Share is each reason's discount dollars divided by "
        f"${discount_dollars:,.2f}."
    )
)
