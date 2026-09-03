from datetime import date

import altair as alt
import pandas as pd
import streamlit as st
from supabase import Client, create_client


st.set_page_config(
    page_title="Revel Sales Dashboard",
    page_icon="📈",
    layout="wide",
)

TABLE_NAME = "daily-sales-summary-all-revenue-operations"


@st.cache_resource
def get_supabase_client() -> Client:
    """Create and reuse one Supabase client."""

    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


@st.cache_data(ttl=300)
def load_sales_data() -> pd.DataFrame:
    """Retrieve daily net-sales records from Supabase."""

    supabase = get_supabase_client()

    response = (
        supabase
        .table(TABLE_NAME)
        .select(
            "id, location, business_date, net_sales"
        )
        .order("business_date")
        .execute()
    )

    dataframe = pd.DataFrame(response.data)

    if dataframe.empty:
        return dataframe

    dataframe["business_date"] = pd.to_datetime(
        dataframe["business_date"],
        errors="coerce",
    )

    dataframe["net_sales"] = pd.to_numeric(
        dataframe["net_sales"],
        errors="coerce",
    )

    dataframe["location"] = (
        dataframe["location"]
        .fillna("Unknown")
        .astype(str)
    )

    dataframe = dataframe.dropna(
        subset=["business_date", "net_sales"],
    )

    return dataframe.sort_values("business_date")


st.title("Revel Sales Dashboard")
st.caption(
    "Daily net sales extracted from Revel and stored in Supabase."
)

try:
    sales_df = load_sales_data()
except KeyError as error:
    st.error(
        f"Missing Streamlit secret: {error}. "
        "Add SUPABASE_URL and SUPABASE_KEY in your "
        "Streamlit app settings."
    )
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

locations = sorted(
    sales_df["location"].dropna().unique().tolist()
)

selected_location = st.sidebar.selectbox(
    "Location",
    options=["All Locations", *locations],
)

minimum_date = sales_df["business_date"].min().date()
maximum_date = sales_df["business_date"].max().date()

selected_dates = st.sidebar.date_input(
    "Business date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)

if len(selected_dates) == 2:
    selected_start_date, selected_end_date = selected_dates
else:
    selected_start_date = selected_dates[0]
    selected_end_date = selected_dates[0]

if selected_start_date > selected_end_date:
    st.sidebar.error(
        "The start date must be before the end date."
    )
    st.stop()


# Filter by location first while retaining the complete date history.
# Rolling calculations use this full location-level dataset.
location_df = sales_df.copy()

if selected_location != "All Locations":
    location_df = location_df[
        location_df["location"] == selected_location
    ].copy()

# The selected date range controls only the displayed records.
filtered_df = location_df[
    (
        location_df["business_date"].dt.date
        >= selected_start_date
    )
    & (
        location_df["business_date"].dt.date
        <= selected_end_date
    )
].copy()


if filtered_df.empty:
    st.warning(
        "No sales data matches the selected filters."
    )
    st.stop()


# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------

total_net_sales = filtered_df["net_sales"].sum()
average_net_sales = filtered_df["net_sales"].mean()
highest_net_sales = filtered_df["net_sales"].max()

latest_row = filtered_df.sort_values(
    "business_date"
).iloc[-1]

latest_net_sales = latest_row["net_sales"]
latest_business_date = latest_row[
    "business_date"
].strftime("%B %d, %Y")

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Total Net Sales",
    f"${total_net_sales:,.2f}",
)

metric_columns[1].metric(
    "Average Daily Net Sales",
    f"${average_net_sales:,.2f}",
)

metric_columns[2].metric(
    "Highest Daily Net Sales",
    f"${highest_net_sales:,.2f}",
)

metric_columns[3].metric(
    "Latest Net Sales",
    f"${latest_net_sales:,.2f}",
    help=latest_business_date,
)

# ---------------------------------------------------------
# Rolling 30-day net-sales growth
# ---------------------------------------------------------

# Aggregate the complete location history for rolling calculations.
daily_history_df = (
    location_df
    .groupby("business_date", as_index=False)["net_sales"]
    .sum()
    .sort_values("business_date")
)

# Aggregate only the selected viewing period for the chart.
daily_sales_df = (
    filtered_df
    .groupby("business_date", as_index=False)["net_sales"]
    .sum()
    .sort_values("business_date")
)

# Create a continuous daily date range so the calculation
# represents calendar days, even if a date is missing.
complete_date_range = pd.date_range(
    start=daily_history_df["business_date"].min(),
    end=daily_history_df["business_date"].max(),
    freq="D",
)

rolling_df = (
    daily_history_df
    .set_index("business_date")
    .reindex(complete_date_range)
    .rename_axis("business_date")
    .reset_index()
)

# Missing dates remain null so incomplete periods are not
# incorrectly treated as days with zero sales.
rolling_df["rolling_30_day_sales"] = (
    rolling_df["net_sales"]
    .rolling(
        window=30,
        min_periods=30,
    )
    .sum()
)

rolling_df["previous_30_day_sales"] = (
    rolling_df["rolling_30_day_sales"].shift(30)
)

rolling_df["rolling_30_day_growth"] = (
    (
        rolling_df["rolling_30_day_sales"]
        - rolling_df["previous_30_day_sales"]
    )
    / rolling_df["previous_30_day_sales"]
    * 100
)

valid_growth_df = rolling_df.dropna(
    subset=[
        "rolling_30_day_sales",
        "previous_30_day_sales",
        "rolling_30_day_growth",
    ]
).copy()

# Restrict displayed growth results to the selected viewing period.
# The underlying values still use all available historical data.
visible_growth_df = valid_growth_df[
    (
        valid_growth_df["business_date"].dt.date
        >= selected_start_date
    )
    & (
        valid_growth_df["business_date"].dt.date
        <= selected_end_date
    )
].copy()

st.subheader("Rolling 30-Day Performance")

if visible_growth_df.empty:
    st.info(
        "Rolling 30-day growth is unavailable for the selected "
        "dates. The database needs at least 60 days of history "
        "ending within this viewing period."
    )
else:
    latest_growth_row = visible_growth_df.iloc[-1]

    current_30_day_sales = latest_growth_row[
        "rolling_30_day_sales"
    ]

    previous_30_day_sales = latest_growth_row[
        "previous_30_day_sales"
    ]

    rolling_30_day_growth = latest_growth_row[
        "rolling_30_day_growth"
    ]

    growth_period_end = latest_growth_row[
        "business_date"
    ].strftime("%B %d, %Y")

    growth_columns = st.columns(3)

    growth_columns[0].metric(
        "Latest 30-Day Net Sales",
        f"${current_30_day_sales:,.2f}",
        help=f"30-day period ending {growth_period_end}",
    )

    growth_columns[1].metric(
        "Previous 30-Day Net Sales",
        f"${previous_30_day_sales:,.2f}",
    )

    growth_columns[2].metric(
        "Rolling 30-Day Growth",
        f"{rolling_30_day_growth:+.2f}%",
        delta=f"{rolling_30_day_growth:+.2f}%",
        help=(
            "Latest 30 days compared with the immediately "
            "preceding 30 days."
        ),
    )

# ---------------------------------------------------------
# Daily net sales with rolling growth overlay
# ---------------------------------------------------------

st.subheader("Daily Net Sales and Rolling 30-Day Growth")

# Join rolling growth to the aggregated daily-sales data.
chart_df = daily_sales_df.merge(
    rolling_df[
        [
            "business_date",
            "rolling_30_day_growth",
        ]
    ],
    on="business_date",
    how="left",
)

selected_period_average = daily_sales_df["net_sales"].mean()

# Give each reference line the same scale as its related series.
sales_axis_max = max(
    float(chart_df["net_sales"].max()),
    float(selected_period_average),
) * 1.05

visible_growth_values = chart_df[
    "rolling_30_day_growth"
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

# Daily net sales use bars and the left dollar axis.
sales_bars = (
    alt.Chart(chart_df)
    .mark_bar(
        color="#1f77b4",
        opacity=0.75,
    )
    .encode(
        x=alt.X(
            "business_date:T",
            title="Business Date",
            axis=alt.Axis(format="%b %d"),
        ),
        y=alt.Y(
            "net_sales:Q",
            title="Daily Net Sales",
            axis=alt.Axis(
                format="$,.0f",
                titleColor="#1f77b4",
            ),
            scale=alt.Scale(
                domain=[0, sales_axis_max],
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "business_date:T",
                title="Business Date",
                format="%B %d, %Y",
            ),
            alt.Tooltip(
                "net_sales:Q",
                title="Daily Net Sales",
                format="$,.2f",
            ),
        ],
    )
)

# Average daily net sales for the selected viewing period.
average_sales_line = (
    alt.Chart(
        pd.DataFrame({
            "average_net_sales": [selected_period_average],
        })
    )
    .mark_rule(
        color="#0f3d66",
        strokeWidth=2,
        strokeDash=[8, 5],
    )
    .encode(
        y=alt.Y(
            "average_net_sales:Q",
            axis=None,
            scale=alt.Scale(
                domain=[0, sales_axis_max],
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "average_net_sales:Q",
                title="Average Daily Net Sales",
                format="$,.2f",
            ),
        ],
    )
)

# Rolling growth uses a line and the right percentage axis.
growth_line = (
    alt.Chart(chart_df)
    .mark_line(
        color="#ff7f0e",
        strokeWidth=3,
        point=alt.OverlayMarkDef(
            color="#ff7f0e",
            size=45,
        ),
    )
    .encode(
        x=alt.X(
            "business_date:T",
            title="Business Date",
        ),
        y=alt.Y(
            "rolling_30_day_growth:Q",
            title="Rolling 30-Day Growth",
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
                "business_date:T",
                title="Period End",
                format="%B %d, %Y",
            ),
            alt.Tooltip(
                "rolling_30_day_growth:Q",
                title="30-Day Growth (%)",
                format="+.2f",
            ),
        ],
    )
)

# A 0% reference separates positive from negative growth.
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
        tooltip=[
            alt.Tooltip(
                "zero_growth:Q",
                title="Growth Reference",
                format=".0f",
            ),
        ],
    )
)

combined_chart = (
    alt.layer(
        sales_bars,
        average_sales_line,
        growth_line,
        zero_growth_line,
    )
    .resolve_scale(y="independent")
    .properties(height=475)
    .interactive()
)

st.altair_chart(
    combined_chart,
    use_container_width=True,
)

st.caption(
    "Blue bars show daily net sales. The orange line shows "
    "the latest 30 days compared with the preceding 30 days. "
    "The dashed navy line is average daily net sales for the "
    "selected period, and the dashed gray line marks 0% growth. "
    "The growth line begins once 60 days of data are available."
)


# ---------------------------------------------------------
# Source data
# ---------------------------------------------------------

with st.expander("View sales records"):
    display_df = filtered_df[
        [
            "business_date",
            "location",
            "net_sales",
        ]
    ].copy()

    display_df["business_date"] = (
        display_df["business_date"].dt.date
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
        },
    )


st.caption(
    f"Displaying {len(filtered_df):,} records from "
    f"{selected_start_date:%B %d, %Y} through "
    f"{selected_end_date:%B %d, %Y}."
)
