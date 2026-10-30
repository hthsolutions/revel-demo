import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    WEEKDAY_NAMES,
    MissingSecretError,
    escape_dollar_signs,
    load_combo_mix,
    load_product_mix,
    load_sales_data,
    location_filter_controls,
    percent_change,
)


# Fixed-length buckets counted backward from Date end.
# A month is 30 days so every month is the same length,
# matching Sales Growth and Labor %.
INTERVALS = {
    "Daily": {
        "days": 1,
        "max_duration": 60,
        "default_duration": 30,
        "unit": "days",
        "x_title": "Business date",
        "duration_label": "Duration (days)",
        "growth_title": "Day-over-day growth",
        "row_name": "day",
        "series_name": "daily",
        "comparison": "the previous day",
        "prior_column": "Vs prior day",
    },
    "Weekly": {
        "days": 7,
        "max_duration": 52,
        "default_duration": 8,
        "unit": "weeks",
        "x_title": "Week",
        "duration_label": "Duration (weeks)",
        "growth_title": "Week-over-week growth",
        "row_name": "week",
        "series_name": "weekly",
        "comparison": "the previous week",
        "prior_column": "Vs prior week",
    },
    "Monthly": {
        "days": 30,
        "max_duration": 12,
        "default_duration": 6,
        "unit": "months",
        "x_title": "30-day period",
        "duration_label": "Duration (months)",
        "growth_title": "Month-over-month growth",
        "row_name": "month",
        "series_name": "monthly",
        "comparison": "the previous month",
        "prior_column": "Vs prior month",
    },
}


# Meal-build rows are the combo itself. Other rows are the
# drink, side, and sauce sold with it.
MEAL_SUBCATEGORIES = {
    "Meals",
    "Family Pack Chicken Choice",
}


def format_period_label(
    start,
    end,
    period_days: int,
    include_year: bool,
) -> str:
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
    """Full date range shown in the heading and tooltips."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if period_days == 1:
        return f"{end:%b %d, %Y}"

    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def assign_periods(
    frame: pd.DataFrame,
    date_end,
    period_days: int,
    duration: int,
) -> pd.DataFrame:
    """Tag each row with its period, including one baseline period.

    period_index 0 ends on Date end. The highest index is the
    period before the chart, used only so the first bar has a
    previous period to grow against.
    """

    date_end = pd.Timestamp(date_end).normalize()
    period_days = int(period_days)
    duration = int(duration)
    history_start = date_end - pd.Timedelta(
        days=period_days * (duration + 1) - 1
    )

    working = frame.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"],
        errors="coerce",
    ).dt.normalize()
    scoped = working.loc[
        working["business_date"].between(
            history_start,
            date_end,
            inclusive="both",
        )
    ].copy()
    if scoped.empty:
        scoped["period_index"] = pd.Series(dtype="int64")
        return scoped

    scoped["period_index"] = (
        (date_end - scoped["business_date"]).dt.days // period_days
    ).astype(int)
    return scoped.loc[
        scoped["period_index"].between(0, duration)
    ].copy()


def build_period_calendar(
    date_end,
    period_days: int,
    count: int,
) -> pd.DataFrame:
    """One row per period, newest last, including the baseline."""

    date_end = pd.Timestamp(date_end).normalize()
    period_days = int(period_days)
    count = int(count)
    rows = []
    for period_index in range(count):
        period_end = date_end - pd.Timedelta(
            days=period_days * period_index
        )
        period_start = period_end - pd.Timedelta(days=period_days - 1)
        rows.append(
            {
                "period_index": period_index,
                "period_start": period_start,
                "period_end": period_end,
            }
        )

    calendar = pd.DataFrame(rows).sort_values("period_end")
    include_year = (
        calendar["period_start"].min().year
        != calendar["period_end"].max().year
    )
    calendar["period_label"] = [
        format_period_label(start, end, period_days, include_year)
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
    return calendar.reset_index(drop=True)


def format_day(business_date) -> str:
    """Label a business date with its weekday."""

    stamp = pd.Timestamp(business_date)
    return f"{WEEKDAY_NAMES[stamp.weekday()][:3]} {stamp:%m/%d}"


def format_dollars(value) -> str:
    """Nearest dollar, with a thousands separator."""

    if value is None or pd.isna(value):
        return "—"

    rounded = int(round(float(value)))
    sign = "-" if rounded < 0 else ""
    return f"{sign}${abs(rounded):,}"


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


def _text(frame: pd.DataFrame, column_name: str) -> pd.Series:
    """A text column, or blanks when the column is absent."""

    if column_name not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column_name].fillna("").astype(str)


def cookie_sales_mask(frame: pd.DataFrame) -> pd.Series:
    """Food cookies, plus every Chocolate Chunk Cookie."""

    if frame.empty:
        return pd.Series(dtype="bool")
    name = _text(frame, "product_name").str.upper()
    subcategory = _text(frame, "product_subcategory").str.upper()
    return (
        (
            name.str.contains("COOKIE", regex=False)
            & subcategory.eq("FOOD")
        )
        | name.str.contains("CHOCOLATE CHUNK COOKIE", regex=False)
    )


def shake_sales_mask(frame: pd.DataFrame) -> pd.Series:
    """Products whose name contains shake."""

    if frame.empty:
        return pd.Series(dtype="bool")
    name = _text(frame, "product_name").str.lower()
    return name.str.contains("shake", regex=False)


def large_drink_upgrade_mask(frame: pd.DataFrame) -> pd.Series:
    """Large drinks on a combo other than the 5 Finger Meal."""

    if frame.empty:
        return pd.Series(dtype="bool")
    name = _text(frame, "product_name").str.upper()
    subcategory = _text(frame, "product_subcategory").str.upper()
    combo = _text(frame, "product_class")
    return (
        subcategory.eq("DRINKS")
        & combo.str.casefold().ne("5 finger meal")
        & name.str.contains("LARGE", regex=False)
    )


def rows_for_period(
    frame: pd.DataFrame,
    location_keys,
    period_start,
    period_end,
) -> pd.DataFrame:
    """Lines for these stores between the period bounds."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=getattr(frame, "columns", []))
    working = frame.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"],
        errors="coerce",
    ).dt.normalize()
    return working.loc[
        working["location_key"].isin(location_keys)
        & working["business_date"].between(
            pd.Timestamp(period_start),
            pd.Timestamp(period_end),
            inclusive="both",
        )
    ].copy()


def masked_sum(frame: pd.DataFrame, mask: pd.Series, column_name: str) -> float:
    """Sum one column on the matching rows."""

    if frame.empty or mask.empty or column_name not in frame.columns:
        return 0.0
    return float(frame.loc[mask, column_name].sum())


def format_count(value) -> str:
    """Whole number with a thousands separator."""

    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}"


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
    "Net sales from combos. A day is one business date, a "
    "week is 7 days, and a month is 30 days, counted "
    "backward from Date end. Each row's Total column is "
    "combo net sales, and the figures below add those rows up."
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

try:
    product_df = load_product_mix()
    product_error = None
except Exception as error:
    product_df = pd.DataFrame()
    product_error = str(error)


# ---------------------------------------------------------
# Sidebar: location, period, and an optional combo
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_location, location_df = location_filter_controls(combo_df)

if location_df.empty:
    st.warning("No combo sales match that location.")
    st.stop()

location_df["business_date"] = pd.to_datetime(
    location_df["business_date"],
    errors="coerce",
).dt.normalize()
location_df = location_df.dropna(subset=["business_date"])

if location_df.empty:
    st.warning("Combo rows for this location have no business date.")
    st.stop()

selected_interval = st.sidebar.selectbox(
    "Interval",
    options=list(INTERVALS),
    index=list(INTERVALS).index("Weekly"),
    help=(
        "Periods are counted backward from Date end. "
        "A week is 7 days and a month is 30 days. "
        "This sets the summary, the trend, and the mix."
    ),
)
interval = INTERVALS[selected_interval]
period_days = interval["days"]
period_name = interval["row_name"]
comparison_label = interval["comparison"]

min_date = location_df["business_date"].min().date()
max_date = location_df["business_date"].max().date()
date_key = "combo_mix_date_end"

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
    help=(
        "Last business date included in the summary, "
        "the trend, and the mix."
    ),
)

duration = st.sidebar.number_input(
    interval["duration_label"],
    min_value=1,
    max_value=interval["max_duration"],
    value=interval["default_duration"],
    step=1,
    key=f"combo_mix_duration_{selected_interval}",
    help=(
        f"How many {interval['unit']} to include, ending "
        "on Date end. "
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
    st.sidebar.caption("Each period is one business day.")
elif period_days == 7:
    st.sidebar.caption("Each period is 7 days, ending on Date end.")
else:
    st.sidebar.caption("Each period is 30 days, ending on Date end.")

if date_end is None:
    st.warning("Pick a date end.")
    st.stop()

scoped = assign_periods(location_df, date_end, period_days, duration)
calendar = build_period_calendar(
    date_end,
    period_days,
    duration + 1,
)
period_df = scoped.loc[scoped["period_index"].eq(0)].copy()
previous_df = scoped.loc[scoped["period_index"].eq(1)].copy()

combo_rank = summarize_combos(period_df) if not period_df.empty else pd.DataFrame(
    columns=["product_class"]
)
combo_names = combo_rank["product_class"].tolist()
selected_combo = st.sidebar.selectbox(
    "Combo",
    options=["All combos", *combo_names],
    help=(
        "Limits the totals, meal builds, and item table "
        "to the period ending on Date end. "
        "The mix chart and trend still include every combo."
    ),
)

detail_df = period_df
previous_detail = previous_df
if selected_combo != "All combos":
    detail_df = period_df[
        period_df["product_class"] == selected_combo
    ].copy()
    previous_detail = previous_df[
        previous_df["product_class"] == selected_combo
    ].copy()

visible_rows = scoped.loc[scoped["period_index"].lt(duration)]
if visible_rows.empty:
    window_end = pd.Timestamp(date_end)
    window_start = window_end - pd.Timedelta(
        days=period_days * duration - 1
    )
    st.info(
        "No combo sales fall between "
        f"{window_start:%b %d, %Y} and {window_end:%b %d, %Y}."
    )
    st.stop()

selected_period = calendar.loc[calendar["period_index"].eq(0)].iloc[0]
period_title = selected_period["period_range"]
if selected_combo != "All combos":
    period_title = f"{selected_combo} · {period_title}"


def _sum(frame: pd.DataFrame, column_name: str) -> float:
    if frame.empty or column_name not in frame.columns:
        return 0.0
    return float(frame[column_name].sum())


if period_df.empty:
    st.subheader(period_title)
    st.info(
        "No combo sales fall in "
        f"{selected_period['period_range']}."
    )
else:
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
    top_builds = period_df[
        (period_df["product_class"] == top_combo_name)
        & period_df["product_subcategory"].isin(MEAL_SUBCATEGORIES)
    ]
    if top_builds.empty:
        top_build_name = "—"
    else:
        top_build_name = (
            top_builds.assign(
                build=top_builds["product_name"].map(clean_product_name)
            )
            .groupby("build")["net_sales"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )

    st.subheader(period_title)

    sales_columns = st.columns(3)
    sales_columns[0].metric(
        "Combo net sales",
        format_dollars(net_sales),
        delta=(
            None
            if pd.isna(sales_growth)
            else f"{sales_growth:+.1f}%"
        ),
        help=(
            f"Sum of the Total column. Compared with "
            f"{comparison_label}: "
            f"{format_dollars(previous_net_sales)}."
        ),
    )
    sales_columns[1].metric(
        "Combo share of net sales",
        format_percent(combo_share_of_sales),
        help=(
            "The combo net sales above, divided by total store "
            "net sales for this location and the selected period."
        ),
    )
    sales_columns[2].metric(
        "Highest selling combo",
        top_combo_name,
        help=escape_dollar_signs(
            f"{format_dollars(top_combo['net_sales'])} "
            "in combo net sales."
        ),
    )
    top_columns = st.columns(3)
    top_columns[0].metric(
        f"{top_combo_name} share of total net sales",
        format_percent(top_combo_of_sales),
    )
    top_columns[1].metric(
        f"{top_combo_name} share of combo net sales",
        format_percent(top_combo_of_combos),
    )
    top_columns[2].metric(
        f"Highest selling {top_combo_name} type",
        top_build_name,
    )

    location_keys = location_df["location_key"].unique()
    current_products = rows_for_period(
        product_df,
        location_keys,
        selected_period["period_start"],
        selected_period["period_end"],
    )
    previous_period = calendar.loc[calendar["period_index"].eq(1)]
    if previous_period.empty:
        previous_products = product_df.iloc[0:0].copy()
    else:
        previous_products = rows_for_period(
            product_df,
            location_keys,
            previous_period.iloc[0]["period_start"],
            previous_period.iloc[0]["period_end"],
        )

    if product_error is None:
        cookie_sales = masked_sum(
            current_products,
            cookie_sales_mask(current_products),
            "net_sales",
        )
        previous_cookie_sales = masked_sum(
            previous_products,
            cookie_sales_mask(previous_products),
            "net_sales",
        )
        shake_sales = masked_sum(
            current_products,
            shake_sales_mask(current_products),
            "net_sales",
        )
        previous_shake_sales = masked_sum(
            previous_products,
            shake_sales_mask(previous_products),
            "net_sales",
        )
    else:
        cookie_sales = None
        previous_cookie_sales = None
        shake_sales = None
        previous_shake_sales = None

    large_drinks = masked_sum(
        detail_df,
        large_drink_upgrade_mask(detail_df),
        "n_items",
    )
    previous_large_drinks = masked_sum(
        previous_detail,
        large_drink_upgrade_mask(previous_detail),
        "n_items",
    )
    cookie_growth = percent_change(cookie_sales, previous_cookie_sales)
    shake_growth = percent_change(shake_sales, previous_shake_sales)
    large_drink_growth = percent_change(
        large_drinks,
        previous_large_drinks,
    )

    def growth_delta(change) -> str | None:
        if pd.isna(change):
            return None
        return f"{change:+.1f}%"

    extra_columns = st.columns(3)
    extra_columns[0].metric(
        "Cookie sales",
        format_dollars(cookie_sales),
        delta=growth_delta(cookie_growth),
        help=escape_dollar_signs(
            "Food items whose name contains Cookie, plus "
            "Chocolate Chunk Cookie. This is store sales for "
            f"the period, compared with {comparison_label}: "
            f"{format_dollars(previous_cookie_sales)}."
        ),
    )
    extra_columns[1].metric(
        "Shake sales",
        format_dollars(shake_sales),
        delta=growth_delta(shake_growth),
        help=escape_dollar_signs(
            "Items whose name contains shake. This is store "
            f"sales for the period, compared with {comparison_label}: "
            f"{format_dollars(previous_shake_sales)}."
        ),
    )
    extra_columns[2].metric(
        "Large drink upgrades",
        format_count(large_drinks),
        delta=growth_delta(large_drink_growth),
        help=(
            "Large drinks sold with a combo, excluding the "
            "5 Finger Meal. The count is the number of items. "
            f"Compared with {comparison_label}: "
            f"{format_count(previous_large_drinks)}."
        ),
    )
    if product_error:
        st.caption(
            "Cookie and shake sales are unavailable: "
            f"{product_error}"
        )

    st.caption(
        f"The change on combo net sales is versus {comparison_label}."
    )


# ---------------------------------------------------------
# Trend across the selected periods, then the mix
# ---------------------------------------------------------

grouped_periods = (
    scoped.groupby("period_index", as_index=False)
    .agg(net_sales=("net_sales", "sum"))
)
trend_df = calendar.merge(
    grouped_periods,
    on="period_index",
    how="left",
)
trend_df = trend_df.sort_values("period_end")
trend_df["previous_net_sales"] = trend_df["net_sales"].shift(1)
trend_df["growth"] = [
    percent_change(current, previous)
    for current, previous in zip(
        trend_df["net_sales"],
        trend_df["previous_net_sales"],
    )
]
trend_df = trend_df.loc[trend_df["period_index"].lt(duration)].copy()
trend_df["is_selected"] = trend_df["period_index"].eq(0)
period_label_order = trend_df["period_label"].tolist()
label_angle = 0 if duration <= 12 else -40

trend_max = float(trend_df["net_sales"].max(skipna=True))
if pd.isna(trend_max) or trend_max <= 0:
    trend_axis_max = 1.0
else:
    trend_axis_max = trend_max * 1.05

growth_values = trend_df["growth"].dropna()
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

period_bars = (
    alt.Chart(trend_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "period_label:N",
            title=None,
            sort=period_label_order,
            axis=alt.Axis(labelAngle=label_angle),
        ),
        y=alt.Y(
            "net_sales:Q",
            title="Combo net sales",
            axis=alt.Axis(
                format="$,.0f",
                titleColor="#1f77b4",
            ),
            scale=alt.Scale(domain=[0, trend_axis_max]),
        ),
        color=alt.condition(
            alt.datum.is_selected,
            alt.value("#1f77b4"),
            alt.value("#9ecae1"),
        ),
        tooltip=[
            alt.Tooltip("period_range:N", title=interval["x_title"]),
            alt.Tooltip(
                "net_sales:Q",
                title="Combo net sales",
                format="$,.0f",
            ),
            alt.Tooltip(
                "growth:Q",
                title=interval["growth_title"],
                format="+.1f",
            ),
        ],
    )
)
growth_line = (
    alt.Chart(trend_df)
    .mark_line(
        color="#ff7f0e",
        strokeWidth=3,
        point=alt.OverlayMarkDef(color="#ff7f0e", size=55),
    )
    .encode(
        x=alt.X(
            "period_label:N",
            title=None,
            sort=period_label_order,
        ),
        y=alt.Y(
            "growth:Q",
            title=interval["growth_title"],
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
            alt.Tooltip("period_range:N", title=interval["x_title"]),
            alt.Tooltip(
                "growth:Q",
                title=interval["growth_title"],
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
    alt.layer(period_bars, growth_line, zero_growth_line)
    .resolve_scale(y="independent")
    .properties(height=320)
)

st.subheader(f"Combo net sales by {period_name}")
st.altair_chart(trend, use_container_width=True)
st.caption(
    f"Blue bars are {interval['series_name']} combo net sales. "
    f"The darker bar is the {period_name} ending on Date end. "
    f"The orange line is the change versus the previous "
    f"{period_name}, and the dashed line is 0%."
)

days_in_selected = int(period_df["business_date"].nunique())
if 0 < days_in_selected < period_days:
    st.caption(
        f"The {period_name} ending on Date end includes only "
        "the business dates loaded so far "
        f"({days_in_selected} of {period_days})."
    )

if period_df.empty:
    st.stop()

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
mix_long["bar_label"] = mix_long["share"].map(
    lambda value: "" if pd.isna(value) else f"{value:.1%}"
)
combo_order = mix_df["product_class"].tolist()
share_max = float(mix_long["share"].max(skipna=True))
if pd.isna(share_max) or share_max <= 0:
    share_max = 0.01
mix_base = alt.Chart(mix_long)
mix_bars = (
    mix_base
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
            scale=alt.Scale(domain=[0, share_max * 1.28]),
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
                format="$,.0f",
            ),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
)
mix_labels = (
    mix_base
    .transform_filter("isValid(datum.share)")
    .mark_text(align="left", dx=4, fontSize=11)
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
            scale=alt.Scale(domain=[0, share_max * 1.28]),
        ),
        text="bar_label:N",
        color=alt.value("#1a1a1a"),
    )
)
mix_chart = (
    alt.layer(mix_bars, mix_labels)
    .properties(height=max(56 * len(mix_df), 200))
    .configure_view(strokeWidth=0, clip=False)
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
            f"({format_dollars(store_net_sales)})."
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
combo_table["vs_prior_period"] = [
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
            "vs_prior_period",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "product_class": "Combo",
        "net_sales": st.column_config.NumberColumn(
            "Net sales",
            format="$%,.0f",
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
            format="$%,.0f",
        ),
        "discounts": st.column_config.NumberColumn(
            "Discounts",
            format="$%,.0f",
        ),
        "voids": st.column_config.NumberColumn(
            "Voids",
            format="%.0f",
        ),
        "vs_prior_period": st.column_config.NumberColumn(
            interval["prior_column"],
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
    daily["stack_rank"] = (
        daily.groupby("business_date", sort=False)["net_sales"]
        .rank(ascending=False, method="first")
    )
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
                axis=alt.Axis(
                    labelAngle=0 if day_count <= 10 else -40,
                ),
            ),
            y=alt.Y(
                "net_sales:Q",
                title="Combo net sales",
                stack="zero",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "product_class:N",
                title="Combo",
                sort=combo_names,
                scale=alt.Scale(scheme="category20"),
            ),
            order=alt.Order("stack_rank:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("day_label:N", title="Day"),
                alt.Tooltip("product_class:N", title="Combo"),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.0f",
                ),
            ],
        )
        .properties(height=320)
    )
    st.subheader("Combo net sales by day")
    st.altair_chart(daily_chart, use_container_width=True)
    st.caption(
        "Each bar stacks combo net sales, with that day's "
        "largest combo at the bottom and the smallest at the top."
    )

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
                    format="$,.0f",
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
            format="$%,.0f",
        ),
        "mix": st.column_config.NumberColumn(
            "Mix",
            format="percent",
        ),
        "discounts": st.column_config.NumberColumn(
            "Discounts",
            format="$%,.0f",
        ),
        "voids": st.column_config.NumberColumn(
            "Voids",
            format="%.0f",
        ),
    },
)
