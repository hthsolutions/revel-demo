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


filtered_df = sales_df[
    (
        sales_df["business_date"].dt.date
        >= selected_start_date
    )
    & (
        sales_df["business_date"].dt.date
        <= selected_end_date
    )
].copy()

if selected_location != "All Locations":
    filtered_df = filtered_df[
        filtered_df["location"] == selected_location
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
# Net-sales chart
# ---------------------------------------------------------

st.subheader("Daily Net Sales")

chart = (
    alt.Chart(filtered_df)
    .mark_line(
        point=True,
        strokeWidth=3,
    )
    .encode(
        x=alt.X(
            "business_date:T",
            title="Business Date",
            axis=alt.Axis(format="%b %d"),
        ),
        y=alt.Y(
            "net_sales:Q",
            title="Net Sales",
            axis=alt.Axis(format="$,.0f"),
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color(
            "location:N",
            title="Location",
        ),
        tooltip=[
            alt.Tooltip(
                "business_date:T",
                title="Business Date",
                format="%B %d, %Y",
            ),
            alt.Tooltip(
                "location:N",
                title="Location",
            ),
            alt.Tooltip(
                "net_sales:Q",
                title="Net Sales",
                format="$,.2f",
            ),
        ],
    )
    .properties(height=475)
    .interactive()
)

st.altair_chart(
    chart,
    use_container_width=True,
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
