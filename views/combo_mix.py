import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    WEEK_START_OPTIONS,
    WEEKDAY_NAMES,
    MissingSecretError,
    add_week_columns,
    escape_dollar_signs,
    format_metric_value,
    load_combo_mix,
    load_sales_data,
    location_filter_controls,
    percent_change,
    reindex_weekly,
)


# Meal-build rows are the combo itself. Other rows are the
# drink, side, and sauce sold with it.
MEAL_SUBCATEGORIES = {
    "Meals",
    "Family Pack Chicken Choice",
}


def format_week(week_start) -> str:
    """Label a week by its first and last calendar day."""

    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    return f"{start:%b %d} – {end:%b %d, %Y}"


def format_day(business_date) -> str:
    """Label a business date with its weekday."""

    stamp = pd.Timestamp(business_date)
    return f"{WEEKDAY_NAMES[stamp.weekday()][:3]} {stamp:%m/%d}"


def format_percent(value) -> str:
    """Render a 0–1 share as a percent, or an em dash."""

    if value is None or pd.isna(value):
        return "—"
    return f"{value:.1%}"


def store_net_sales_for(
    sales_frame: pd.DataFrame,
    location_keys,
    business_dates,
) -> float:
    """Total store net sales for the same locations and dates."""

    if sales_frame.empty or "net_sales" not in sales_frame.columns:
        return float("nan")

    store_sales = sales_frame.copy()
    store_sales["business_date"] = pd.to_datetime(
        store_sales["business_date"],
        errors="coerce",
    ).dt.normalize()
    period_dates = set(
        pd.to_datetime(pd.Series(business_dates))
        .dt.normalize()
        .dt.date
    )
    store_sales = store_sales[
        store_sales["location_key"].isin(location_keys)
        & store_sales["business_date"].dt.date.isin(period_dates)
    ]
    if store_sales.empty:
        return float("nan")
    return float(store_sales["net_sales"].sum())


def clean_product_name(name) -> str:
    """Drop the asterisks Revel wraps around combo builds."""

    return " ".join(str(name).replace("*", "").split())


def summarize_combos(frame: pd.DataFrame) -> pd.DataFrame:
    """Net sales, items, and mix for each combo."""

    grouped = (
        frame.groupby("product_class", as_index=False)
        .agg(
            net_sales=("net_sales", "sum"),
            items=("n_items", "sum"),
            discounts=("discounts", "sum"),
            voids=("n_voids", "sum"),
        )
        .sort_values("net_sales", ascending=False)
    )
    sales_total = float(grouped["net_sales"].sum())
    grouped["mix"] = (
        grouped["net_sales"] / sales_total if sales_total else 0.0
    )
    grouped["net_sales_per_item"] = (
        grouped["net_sales"]
        / grouped["items"].replace(0.0, pd.NA)
    )
    return grouped


def share_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each row's share of net sales."""

    chart_df = frame.copy()
    sales_total = float(chart_df["net_sales"].sum())
    chart_df["mix"] = (
        chart_df["net_sales"] / sales_total if sales_total else 0.0
    )
    return chart_df


st.title("Combo Mix")
st.caption(
    "Net sales from combos. Each row's Total column is combo "
    "net sales, and the figures below add those rows up."
)

try:
    combo_df = load_combo_mix()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Unable to load combo mix from Supabase: {error}")
    st.stop()

if combo_df.empty:
    st.warning(
        "No combo rows were returned. Check that "
        "daily-product-mix-combomix has data for this account."
    )
    st.stop()

try:
    sales_df = load_sales_data()
except Exception:
    sales_df = pd.DataFrame()


# ---------------------------------------------------------
# Sidebar: location, week, day, and an optional combo
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(combo_df)

if location_df.empty:
    st.warning("No combo sales match that location.")
    st.stop()

selected_week_start_name = st.sidebar.selectbox(
    "Week starts on",
    options=list(WEEK_START_OPTIONS.keys()),
)
week_start_weekday = WEEK_START_OPTIONS[selected_week_start_name]
weeks_to_display = st.sidebar.slider(
    "Weeks to display",
    min_value=2,
    max_value=26,
    value=8,
)
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

week_df = location_df[
    location_df["week_start"] == selected_week
].copy()

if week_df.empty:
    st.info("No combo sales fall in that week.")
    st.stop()

day_choices = (
    week_df[["business_date"]]
    .drop_duplicates()
    .sort_values("business_date")
)
day_labels = {
    pd.Timestamp(value): format_day(value)
    for value in day_choices["business_date"]
}
selected_day_label = st.sidebar.selectbox(
    "Day",
    options=["All days", *day_labels.values()],
)

combo_rank = summarize_combos(week_df)
combo_names = combo_rank["product_class"].tolist()
selected_combo = st.sidebar.selectbox(
    "Combo",
    options=["All combos", *combo_names],
    help=(
        "Limits the totals, meal builds, and item table. "
        "The mix chart still includes every combo."
    ),
)

period_df = week_df
comparison_label = "the previous week"
if selected_day_label != "All days":
    selected_dates = [
        business_date
        for business_date, label in day_labels.items()
        if label == selected_day_label
    ]
    period_df = week_df[
        week_df["business_date"].isin(selected_dates)
    ].copy()
    comparison_label = "the same weekday in the previous week"

previous_start = pd.Timestamp(selected_week) - pd.Timedelta(days=7)
previous_df = location_df[
    location_df["week_start"] == previous_start
].copy()
if selected_day_label != "All days":
    previous_dates = [
        pd.Timestamp(business_date) - pd.Timedelta(days=7)
        for business_date in selected_dates
    ]
    previous_df = previous_df[
        previous_df["business_date"].isin(previous_dates)
    ].copy()

detail_df = period_df
previous_detail = previous_df
if selected_combo != "All combos":
    detail_df = period_df[
        period_df["product_class"] == selected_combo
    ].copy()
    previous_detail = previous_df[
        previous_df["product_class"] == selected_combo
    ].copy()

if period_df.empty:
    st.info("No combo sales fall on that day.")
    st.stop()


def _sum(frame: pd.DataFrame, column_name: str) -> float:
    if frame.empty or column_name not in frame.columns:
        return 0.0
    return float(frame[column_name].sum())


net_sales = _sum(detail_df, "net_sales")
previous_net_sales = _sum(previous_detail, "net_sales")
sales_growth = percent_change(net_sales, previous_net_sales)
store_net_sales = store_net_sales_for(
    sales_df,
    location_df["location_key"].unique(),
    period_df["business_date"],
)
combo_share_of_sales = (
    net_sales / store_net_sales
    if pd.notna(store_net_sales) and store_net_sales
    else float("nan")
)
combo_rank_period = summarize_combos(period_df)
top_combo = combo_rank_period.iloc[0]
top_combo_name = top_combo["product_class"]
top_combo_of_combos = float(top_combo["mix"])
top_combo_of_sales = (
    float(top_combo["net_sales"]) / store_net_sales
    if pd.notna(store_net_sales) and store_net_sales
    else float("nan")
)

period_title = (
    format_week(selected_week)
    if selected_day_label == "All days"
    else f"{selected_day_label}, {format_week(selected_week)}"
)
if selected_combo != "All combos":
    period_title = f"{selected_combo} · {period_title}"

st.subheader(period_title)

kpi_columns = st.columns(4)
kpi_columns[0].metric(
    "Combo net sales",
    format_metric_value(net_sales, "currency"),
    delta=(
        None
        if pd.isna(sales_growth)
        else f"{sales_growth:+.1f}%"
    ),
    help=(
        f"Sum of the Total column. Compared with "
        f"{comparison_label}: "
        f"{format_metric_value(previous_net_sales, 'currency')}."
    ),
)
kpi_columns[1].metric(
    "Combo share of net sales",
    format_percent(combo_share_of_sales),
    help=(
        "The combo net sales above, divided by total store "
        "net sales for this location, week, and day."
    ),
)
kpi_columns[2].metric(
    "Highest selling combo",
    top_combo_name,
    help=escape_dollar_signs(
        f"{format_metric_value(top_combo['net_sales'], 'currency')} "
        "in combo net sales."
    ),
)
kpi_columns[3].metric(
    f"{top_combo_name} share",
    (
        f"{format_percent(top_combo_of_sales)} of total · "
        f"{format_percent(top_combo_of_combos)} of combos"
    ),
    help=(
        "That combo's sales as a percent of total store net "
        "sales, then as a percent of combo net sales."
    ),
)
st.caption(f"The change on combo net sales is versus {comparison_label}.")


# ---------------------------------------------------------
# Recent weeks, then the mix for the selected week
# ---------------------------------------------------------

weekly_all = (
    location_df.groupby("week_start", as_index=False)
    .agg(net_sales=("net_sales", "sum"))
    .sort_values("week_start")
)
weekly_all = reindex_weekly(weekly_all, location_df["week_start"])
weekly_all["previous_net_sales"] = weekly_all["net_sales"].shift(1)
weekly_all["growth"] = [
    percent_change(current, previous)
    for current, previous in zip(
        weekly_all["net_sales"],
        weekly_all["previous_net_sales"],
    )
]
selected_positions = weekly_all.index[
    weekly_all["week_start"] == pd.Timestamp(selected_week)
]
if len(selected_positions):
    window_end = int(selected_positions[0]) + 1
    window_start = max(0, window_end - weeks_to_display)
    weekly_df = weekly_all.iloc[window_start:window_end].copy()
else:
    weekly_df = weekly_all.tail(weeks_to_display).copy()
weekly_df["week_label"] = weekly_df["week_start"].dt.strftime("%b %d")
weekly_df["is_selected"] = (
    weekly_df["week_start"] == pd.Timestamp(selected_week)
)
week_label_order = weekly_df["week_label"].tolist()

weekly_max = float(weekly_df["net_sales"].max(skipna=True))
if pd.isna(weekly_max) or weekly_max <= 0:
    weekly_axis_max = 1.0
else:
    weekly_axis_max = weekly_max * 1.05

growth_values = weekly_df["growth"].dropna()
if growth_values.empty:
    growth_axis_min = -1.0
    growth_axis_max = 1.0
else:
    raw_growth_min = min(0.0, float(growth_values.min()))
    raw_growth_max = max(0.0, float(growth_values.max()))
    growth_padding = max(
        (raw_growth_max - raw_growth_min) * 0.10,
        1.0,
    )
    growth_axis_min = raw_growth_min - growth_padding
    growth_axis_max = raw_growth_max + growth_padding

weekly_bars = (
    alt.Chart(weekly_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "week_label:N",
            title=None,
            sort=week_label_order,
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "net_sales:Q",
            title="Combo net sales",
            axis=alt.Axis(
                format="$,.0f",
                titleColor="#1f77b4",
            ),
            scale=alt.Scale(domain=[0, weekly_axis_max]),
        ),
        color=alt.condition(
            alt.datum.is_selected,
            alt.value("#1f77b4"),
            alt.value("#9ecae1"),
        ),
        tooltip=[
            alt.Tooltip("week_label:N", title="Week of"),
            alt.Tooltip(
                "net_sales:Q",
                title="Combo net sales",
                format="$,.2f",
            ),
            alt.Tooltip(
                "growth:Q",
                title="Week-over-week growth",
                format="+.1f",
            ),
        ],
    )
)
growth_line = (
    alt.Chart(weekly_df)
    .mark_line(
        color="#ff7f0e",
        strokeWidth=3,
        point=alt.OverlayMarkDef(color="#ff7f0e", size=55),
    )
    .encode(
        x=alt.X(
            "week_label:N",
            title=None,
            sort=week_label_order,
        ),
        y=alt.Y(
            "growth:Q",
            title="Week-over-week growth",
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
            alt.Tooltip("week_label:N", title="Week of"),
            alt.Tooltip(
                "growth:Q",
                title="Week-over-week growth",
                format="+.1f",
            ),
        ],
    )
)
zero_growth_line = (
    alt.Chart(pd.DataFrame({"zero_growth": [0.0]}))
    .mark_rule(color="#6b7280", strokeWidth=2, strokeDash=[5, 5])
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
trend = (
    alt.layer(weekly_bars, growth_line, zero_growth_line)
    .resolve_scale(y="independent")
    .properties(height=320)
)

st.subheader("Combo net sales by week")
st.altair_chart(trend, use_container_width=True)
st.caption(
    "Blue bars are weekly combo net sales. The darker bar is "
    "the week selected above. The orange line is the change "
    "versus the previous week, and the dashed line is 0%."
)

latest_week = location_df["week_start"].max()
days_in_latest_week = location_df.loc[
    location_df["week_start"] == latest_week,
    "business_date",
].nunique()
if days_in_latest_week < 7:
    st.caption(
        "The latest week includes only the business dates "
        f"loaded so far ({days_in_latest_week} of 7)."
    )

mix_df = summarize_combos(period_df)
if pd.notna(store_net_sales) and store_net_sales:
    mix_df["share_of_total"] = mix_df["net_sales"] / store_net_sales
else:
    mix_df["share_of_total"] = float("nan")

share_series = [
    "Share of combo net sales",
    "Share of total net sales",
]
combo_share = mix_df.assign(
    series=share_series[0],
    share=mix_df["mix"],
)
total_share = mix_df.assign(
    series=share_series[1],
    share=mix_df["share_of_total"],
)
mix_long = pd.concat([combo_share, total_share], ignore_index=True)
combo_order = mix_df["product_class"].tolist()
mix_chart = (
    alt.Chart(mix_long)
    .mark_bar()
    .encode(
        y=alt.Y(
            "product_class:N",
            title=None,
            sort=combo_order,
        ),
        yOffset=alt.YOffset(
            "series:N",
            sort=share_series,
        ),
        x=alt.X(
            "share:Q",
            title="Share",
            axis=alt.Axis(format=".0%"),
        ),
        color=alt.Color(
            "series:N",
            title=None,
            sort=share_series,
            scale=alt.Scale(
                domain=share_series,
                range=["#1f77b4", "#ff7f0e"],
            ),
        ),
        tooltip=[
            alt.Tooltip("product_class:N", title="Combo"),
            alt.Tooltip("series:N", title="Share"),
            alt.Tooltip(
                "net_sales:Q",
                title="Combo net sales",
                format="$,.2f",
            ),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    .properties(height=max(52 * len(mix_df), 180))
)
st.subheader("Combo mix")
st.altair_chart(mix_chart, use_container_width=True)
if pd.isna(store_net_sales) or store_net_sales == 0:
    st.caption(
        "Blue is each combo's share of combo net sales. "
        "Total store net sales were not available for an "
        "orange comparison."
    )
else:
    st.caption(
        escape_dollar_signs(
            "Blue is each combo's share of combo net sales. "
            "Orange is that combo's share of total store net sales "
            f"({format_metric_value(store_net_sales, 'currency')})."
        )
    )

combo_table = mix_df.copy()
previous_mix = summarize_combos(previous_df)
combo_table = combo_table.merge(
    previous_mix[["product_class", "net_sales"]].rename(
        columns={"net_sales": "previous_net_sales"}
    ),
    on="product_class",
    how="left",
)
combo_table["vs_prior_week"] = [
    percent_change(current, previous)
    for current, previous in zip(
        combo_table["net_sales"],
        combo_table["previous_net_sales"],
    )
]

st.dataframe(
    combo_table[
        [
            "product_class",
            "net_sales",
            "mix",
            "share_of_total",
            "items",
            "net_sales_per_item",
            "discounts",
            "voids",
            "vs_prior_week",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "product_class": "Combo",
        "net_sales": st.column_config.NumberColumn(
            "Net sales",
            format="$%.2f",
        ),
        "mix": st.column_config.NumberColumn(
            "Share of combo sales",
            format="percent",
        ),
        "share_of_total": st.column_config.NumberColumn(
            "Share of total net sales",
            format="percent",
        ),
        "items": st.column_config.NumberColumn(
            "Items",
            format="%.0f",
        ),
        "net_sales_per_item": st.column_config.NumberColumn(
            "Net sales / item",
            format="$%.2f",
        ),
        "discounts": st.column_config.NumberColumn(
            "Discounts",
            format="$%.2f",
        ),
        "voids": st.column_config.NumberColumn(
            "Voids",
            format="%.0f",
        ),
        "vs_prior_week": st.column_config.NumberColumn(
            "Vs prior week",
            format="%+.1f%%",
        ),
    },
)


# ---------------------------------------------------------
# Inside the selected combo: builds, pieces, and items
# ---------------------------------------------------------

day_count = period_df["business_date"].nunique()
if day_count > 1:
    daily = (
        period_df.groupby(
            ["business_date", "product_class"],
            as_index=False,
        )
        .agg(net_sales=("net_sales", "sum"))
    )
    daily["day_label"] = daily["business_date"].map(format_day)
    day_order = [
        format_day(value)
        for value in sorted(period_df["business_date"].unique())
    ]
    daily_chart = (
        alt.Chart(daily)
        .mark_bar()
        .encode(
            x=alt.X(
                "day_label:N",
                title=None,
                sort=day_order,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "net_sales:Q",
                title="Combo net sales",
                stack=True,
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "product_class:N",
                title="Combo",
                sort=combo_names,
                scale=alt.Scale(scheme="category20"),
            ),
            tooltip=[
                alt.Tooltip("day_label:N", title="Day"),
                alt.Tooltip("product_class:N", title="Combo"),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.2f",
                ),
            ],
        )
        .properties(height=320)
    )
    st.subheader("Combo net sales by day")
    st.altair_chart(daily_chart, use_container_width=True)

if detail_df.empty:
    st.info(f"No {selected_combo} sales in this selection.")
    st.stop()

builds = detail_df[
    detail_df["product_subcategory"].isin(MEAL_SUBCATEGORIES)
].copy()
if not builds.empty:
    builds["build"] = builds["product_name"].map(clean_product_name)
    build_summary = share_frame(
        builds.groupby(["product_class", "build"], as_index=False)
        .agg(
            net_sales=("net_sales", "sum"),
            items=("n_items", "sum"),
        )
    )
    build_order = (
        build_summary.groupby("build")["net_sales"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    build_chart = (
        alt.Chart(build_summary)
        .mark_bar()
        .encode(
            y=alt.Y(
                "build:N",
                title=None,
                sort=build_order,
            ),
            yOffset=alt.YOffset(
                "product_class:N",
                sort=combo_names,
            ),
            x=alt.X(
                "mix:Q",
                title="Share of combo net sales",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "product_class:N",
                title="Combo",
                sort=combo_names,
                scale=alt.Scale(scheme="category20"),
            ),
            tooltip=[
                alt.Tooltip("build:N", title="Build"),
                alt.Tooltip("product_class:N", title="Combo"),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.2f",
                ),
                alt.Tooltip("mix:Q", title="Mix", format=".1%"),
                alt.Tooltip("items:Q", title="Items", format=",.0f"),
            ],
        )
        .properties(height=max(40 * len(build_order), 160))
    )
    st.subheader("Meal build mix")
    st.altair_chart(build_chart, use_container_width=True)
    st.caption(
        "Regular, spicy, and other builds. Share uses net sales "
        "on the meal row, not the drink and side rows."
    )

items_df = detail_df.copy()
items_df["item"] = items_df["product_name"].map(clean_product_name)
item_summary = (
    items_df.groupby(
        ["product_class", "product_subcategory", "item"],
        as_index=False,
    )
    .agg(
        net_sales=("net_sales", "sum"),
        items=("n_items", "sum"),
        discounts=("discounts", "sum"),
        voids=("n_voids", "sum"),
    )
    .sort_values("net_sales", ascending=False)
)
item_total = float(item_summary["net_sales"].sum())
item_summary["mix"] = (
    item_summary["net_sales"] / item_total if item_total else 0.0
)

st.subheader("Items")
st.dataframe(
    item_summary[
        [
            "product_class",
            "product_subcategory",
            "item",
            "items",
            "net_sales",
            "mix",
            "discounts",
            "voids",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "product_class": "Combo",
        "product_subcategory": "Piece",
        "item": "Item",
        "items": st.column_config.NumberColumn(
            "Items",
            format="%.0f",
        ),
        "net_sales": st.column_config.NumberColumn(
            "Net sales",
            format="$%.2f",
        ),
        "mix": st.column_config.NumberColumn(
            "Mix",
            format="percent",
        ),
        "discounts": st.column_config.NumberColumn(
            "Discounts",
            format="$%.2f",
        ),
        "voids": st.column_config.NumberColumn(
            "Voids",
            format="%.0f",
        ),
    },
)
