import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    MissingSecretError,
    escape_dollar_signs,
    format_metric_value,
    load_cogs_summary,
    load_salary_data,
    load_sales_data,
    load_shift_data,
)


# Store profit margin, in percent points. A month that
# lands on both cutoffs keeps this margin.
STORE_MARGIN_PERCENT = {
    "LEANDER": 17.0,
    "MARBLE FALLS": 7.0,
    "LAMPASAS": 7.0,
    "TEMPLE": 22.0,
}

# Shown first when several stores share a chart.
STORE_ORDER = ["Leander", "Marble Falls", "Lampasas", "Temple"]

HOURLY_LABOR_CUTOFF = 18.0
COGS_CUTOFF = 31.5

# Roles paid from the salary feed. Their shift rows are
# not hourly labor. GM and DM stay out even when the
# salary extract is empty.
SALARIED_ROLES = {"GM", "DM"}

STORE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

OVER_STYLE = (
    "background-color: rgba(180, 35, 24, 0.16); color: #b42318"
)
WITHIN_STYLE = (
    "background-color: rgba(34, 139, 84, 0.18); color: #157347"
)

PERIOD_KEYS = ["location_key", "start_date", "end_date"]


def salaried_roles(salary_frame: pd.DataFrame) -> set[str]:
    """Role names paid as salary, so they stay out of hourly labor."""

    keys = set(SALARIED_ROLES)
    if salary_frame is not None and not salary_frame.empty:
        if "role" in salary_frame.columns:
            keys.update(
                salary_frame["role"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )
    keys.discard("")
    return keys


def ordered_locations(names) -> list[str]:
    """Leander, Marble Falls, Lampasas, then Temple, then any other store."""

    present = {str(name) for name in names if pd.notna(name)}
    preferred = [name for name in STORE_ORDER if name in present]
    rest = sorted(present - set(preferred), key=str.lower)
    return preferred + rest


def month_periods(cogs: pd.DataFrame) -> pd.DataFrame:
    """One row per store and month, with that month's COGS dollars.

    A month whose cost lines are all blank stays blank
    instead of becoming zero.
    """

    empty = pd.DataFrame(
        columns=[
            "location_key",
            "start_date",
            "end_date",
            "location",
            "cogs_amount",
        ]
    )
    if cogs.empty:
        return empty

    grouped = (
        cogs
        .groupby(PERIOD_KEYS, as_index=False)
        .agg(
            location=("location", "first"),
            cogs_amount=("amount", "sum"),
            reported_lines=("amount", "count"),
        )
    )
    missing_amount = grouped["reported_lines"].eq(0)
    if bool(missing_amount.any()):
        grouped.loc[missing_amount, "cogs_amount"] = float("nan")
    return grouped.drop(columns=["reported_lines"])


def sales_by_day(sales: pd.DataFrame) -> pd.DataFrame:
    """Net sales for each store and business date."""

    empty = pd.DataFrame(
        columns=["business_date", "location_key", "net_sales"]
    )
    if (
        sales is None
        or sales.empty
        or "net_sales" not in sales.columns
        or "location_key" not in sales.columns
        or "business_date" not in sales.columns
    ):
        return empty

    working = sales.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"],
        errors="coerce",
    ).dt.normalize()
    working = working.dropna(subset=["business_date", "location_key"])
    if working.empty:
        return empty

    return (
        working
        .groupby(["business_date", "location_key"], as_index=False)
        ["net_sales"]
        .sum()
    )


def hourly_wages_by_day(
    shift_frame: pd.DataFrame,
    salaried: set[str],
) -> pd.DataFrame:
    """Hourly shift wages for each store and business date."""

    empty = pd.DataFrame(
        columns=["business_date", "location_key", "hourly_wages"]
    )
    if (
        shift_frame is None
        or shift_frame.empty
        or "shift_wages" not in shift_frame.columns
        or "location_key" not in shift_frame.columns
        or "business_date" not in shift_frame.columns
    ):
        return empty

    working = shift_frame.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"],
        errors="coerce",
    ).dt.normalize()

    if "role" in working.columns and salaried:
        role = (
            working["role"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        working = working.loc[~role.isin(salaried)].copy()

    if working.empty:
        return empty

    working["hourly_wages"] = pd.to_numeric(
        working["shift_wages"],
        errors="coerce",
    ).fillna(0.0)
    working = working.dropna(subset=["business_date", "location_key"])
    if working.empty:
        return empty

    return (
        working
        .groupby(["business_date", "location_key"], as_index=False)
        ["hourly_wages"]
        .sum()
    )


def sum_in_periods(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Sum a daily measure into each store month.

    A business date counts when it falls on start_date,
    on end_date, or on a day between them.
    """

    empty = pd.DataFrame(
        columns=[*PERIOD_KEYS, value_column, "days"]
    )
    if (
        daily.empty
        or periods.empty
        or value_column not in daily.columns
    ):
        return empty

    scoped = daily.merge(periods[PERIOD_KEYS], on="location_key", how="inner")
    if scoped.empty:
        return empty

    in_month = scoped["business_date"].between(
        scoped["start_date"],
        scoped["end_date"],
        inclusive="both",
    )
    scoped = scoped.loc[in_month & scoped["business_date"].notna()]
    if scoped.empty:
        return empty

    return (
        scoped
        .groupby(PERIOD_KEYS, as_index=False)
        .agg(
            **{
                value_column: (value_column, "sum"),
                "days": ("business_date", "nunique"),
            }
        )
    )


def assign_percent(numerator, denominator) -> pd.Series:
    """Percent of the denominator, blank when sales are missing."""

    percent = pd.Series(float("nan"), index=numerator.index, dtype="float64")
    sales = pd.to_numeric(denominator, errors="coerce")
    usable = sales.notna() & sales.ne(0) & numerator.notna()
    if not bool(usable.any()):
        return percent

    percent.loc[usable] = (
        numerator.loc[usable].astype(float)
        / sales.loc[usable].astype(float)
        * 100
    )
    return percent


def build_store_months(
    cogs: pd.DataFrame,
    sales: pd.DataFrame,
    shifts: pd.DataFrame,
    salary: pd.DataFrame,
) -> pd.DataFrame:
    """COGS %, hourly labor %, and estimated net profit by store month.

    Actual margin starts at the store margin. Each point of
    hourly labor above 18% reduces it by a point, and each
    point of COGS above 31.5% reduces it by a point. A month
    under a cutoff gains those points back. Estimated net
    profit is that month's net sales times the actual margin.
    """

    periods = month_periods(cogs)
    if periods.empty:
        return periods

    sales_totals = sum_in_periods(
        sales_by_day(sales),
        periods,
        "net_sales",
    ).rename(columns={"days": "sales_days"})
    wage_totals = sum_in_periods(
        hourly_wages_by_day(shifts, salaried_roles(salary)),
        periods,
        "hourly_wages",
    ).rename(columns={"days": "labor_days"})

    frame = periods.merge(sales_totals, on=PERIOD_KEYS, how="left")
    frame = frame.merge(wage_totals, on=PERIOD_KEYS, how="left")
    frame["days_in_month"] = (
        (frame["end_date"] - frame["start_date"]).dt.days + 1
    ).astype(int)
    frame["cogs_percent"] = assign_percent(
        frame["cogs_amount"],
        frame["net_sales"],
    )
    frame["hourly_percent"] = assign_percent(
        frame["hourly_wages"],
        frame["net_sales"],
    )
    frame["base_margin_percent"] = frame["location_key"].map(
        STORE_MARGIN_PERCENT
    )
    frame["actual_margin_percent"] = (
        frame["base_margin_percent"]
        + (HOURLY_LABOR_CUTOFF - frame["hourly_percent"])
        + (COGS_CUTOFF - frame["cogs_percent"])
    )
    frame["estimated_net_profit"] = (
        frame["net_sales"] * frame["actual_margin_percent"] / 100
    )
    frame["sales_coverage"] = [
        "—"
        if pd.isna(counted)
        else f"{int(counted)} of {int(total)}"
        for counted, total in zip(
            frame["sales_days"],
            frame["days_in_month"],
        )
    ]
    return frame.sort_values(
        ["start_date", "location"],
        ascending=[False, True],
    ).reset_index(drop=True)


def format_whole_dollars(value) -> str:
    """Currency with no cents, or a dash when the value is blank."""

    if value is None or pd.isna(value):
        return "—"
    return f"${value:,.0f}"


def percent_cell_style(cutoff: float):
    """Green at or under the cutoff, red above it."""

    def style_cell(value) -> str:
        if pd.isna(value):
            return ""
        if value > cutoff:
            return OVER_STYLE
        return WITHIN_STYLE

    return style_cell


def style_table(frame: pd.DataFrame):
    """Color COGS % and hourly labor % against their cutoffs."""

    styler = frame.style
    styler = styler.map(
        percent_cell_style(COGS_CUTOFF),
        subset=["cogs_percent"],
    )
    return styler.map(
        percent_cell_style(HOURLY_LABOR_CUTOFF),
        subset=["hourly_percent"],
    )


def table_column_config() -> dict:
    """Readable headers for the store-month table."""

    return {
        "location": "Store",
        "start_date": st.column_config.DateColumn(
            "Month",
            help="First day of the COGS month.",
            format="MMM YYYY",
        ),
        "net_sales": st.column_config.NumberColumn(
            "Net sales",
            format="$%,.0f",
            help=(
                "Net sales whose business date falls in "
                "this month."
            ),
        ),
        "cogs_amount": st.column_config.NumberColumn(
            "COGS $",
            format="$%,.0f",
            help="Sum of cogs_summary amount for this store and month.",
        ),
        "cogs_percent": st.column_config.NumberColumn(
            "COGS %",
            format="%.2f%%",
            help=(
                "COGS dollars divided by net sales. "
                f"The cutoff is {COGS_CUTOFF:.1f}%."
            ),
        ),
        "hourly_percent": st.column_config.NumberColumn(
            "Hourly labor %",
            format="%.2f%%",
            help=(
                "Hourly shift wages divided by net sales. "
                "Salaried roles are left out. "
                f"The cutoff is {HOURLY_LABOR_CUTOFF:.1f}%."
            ),
        ),
        "base_margin_percent": st.column_config.NumberColumn(
            "Store margin",
            format="%.1f%%",
        ),
        "actual_margin_percent": st.column_config.NumberColumn(
            "Actual margin",
            format="%.2f%%",
            help=(
                "Store margin + (hourly cutoff − hourly labor %) "
                "+ (COGS cutoff − COGS %)."
            ),
        ),
        "estimated_net_profit": st.column_config.NumberColumn(
            "Estimated net profit",
            format="$%,.0f",
            help="Net sales times the actual margin.",
        ),
        "sales_coverage": st.column_config.TextColumn(
            "Sales days",
            help=(
                "Business dates with net sales, out of the "
                "days in the month. COGS is the full month, "
                "so a short sales feed makes COGS % high."
            ),
        ),
    }


def store_color_scale(locations: list[str]) -> alt.Scale:
    """One color per store, stable across both charts."""

    return alt.Scale(
        domain=locations,
        range=STORE_COLORS[: len(locations)],
    )


def with_month_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one axis label per calendar month, such as Aug 2026."""

    labeled = frame.copy()
    labeled["month_label"] = [
        f"{pd.Timestamp(month):%b %Y}"
        for month in labeled["start_date"]
    ]
    return labeled


def month_axis(frame: pd.DataFrame) -> alt.X:
    """One slot per month, in calendar order.

    A continuous date axis turns two months into a row of
    daily ticks that all read Aug 2026.
    """

    order = (
        frame.sort_values("start_date")["month_label"]
        .drop_duplicates()
        .tolist()
    )
    angle = -40 if len(order) > 6 else 0
    return alt.X(
        "month_label:N",
        title=None,
        sort=order,
        scale=alt.Scale(domain=order),
        axis=alt.Axis(labelAngle=angle, labelOverlap=False),
    )


def cogs_chart(frame: pd.DataFrame, locations: list[str]) -> alt.Chart:
    """COGS % by month, with a line at the cutoff."""

    scored = with_month_label(frame[frame["cogs_percent"].notna()])
    color = alt.Color(
        "location:N",
        title="Store",
        sort=locations,
        scale=store_color_scale(locations),
    )
    y_max = max(
        float(scored["cogs_percent"].max()),
        COGS_CUTOFF,
        1.0,
    ) * 1.15
    lines = (
        alt.Chart(scored)
        .mark_line(point=alt.OverlayMarkDef(size=80))
        .encode(
            x=month_axis(scored),
            y=alt.Y(
                "cogs_percent:Q",
                title="COGS % of net sales",
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(format=".1f"),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("location:N", title="Store"),
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip(
                    "cogs_percent:Q",
                    title="COGS %",
                    format=".2f",
                ),
                alt.Tooltip(
                    "cogs_amount:Q",
                    title="COGS $",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.0f",
                ),
            ],
        )
    )
    cutoff = (
        alt.Chart(pd.DataFrame({"cutoff": [COGS_CUTOFF]}))
        .mark_rule(strokeDash=[6, 4], color="#b42318")
        .encode(y="cutoff:Q")
    )
    return (lines + cutoff).properties(height=340)


def profit_chart(frame: pd.DataFrame, locations: list[str]) -> alt.Chart:
    """Estimated net profit by store and month."""

    scored = with_month_label(
        frame[frame["estimated_net_profit"].notna()]
    )
    return (
        alt.Chart(scored)
        .mark_bar()
        .encode(
            x=month_axis(scored),
            xOffset=alt.XOffset("location:N", sort=locations),
            y=alt.Y(
                "estimated_net_profit:Q",
                title="Estimated net profit",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "location:N",
                title="Store",
                sort=locations,
                scale=store_color_scale(locations),
            ),
            tooltip=[
                alt.Tooltip("location:N", title="Store"),
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip(
                    "estimated_net_profit:Q",
                    title="Estimated net profit",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "actual_margin_percent:Q",
                    title="Actual margin %",
                    format=".2f",
                ),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net sales",
                    format="$,.0f",
                ),
            ],
        )
        .properties(height=340)
    )


def month_choices(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Months present in the table, newest first."""

    if frame.empty:
        return []
    months = pd.to_datetime(frame["start_date"]).drop_duplicates()
    return [pd.Timestamp(month) for month in sorted(months, reverse=True)]


def month_label(month) -> str:
    """September 2026 style label for a month start."""

    return f"{pd.Timestamp(month):%B %Y}"


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.title("COGS")
st.caption(
    "COGS % is the month's cost of goods divided by net sales. "
    "Net sales are the daily sales whose business date falls "
    "in that month. Estimated net profit starts from each "
    "store's margin, then moves by the same number of points "
    "that hourly labor and COGS miss their cutoffs."
)

try:
    cogs_df = load_cogs_summary()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Unable to load cogs_summary from Supabase: {error}")
    st.stop()

if cogs_df.empty:
    st.warning("No COGS rows were returned from cogs_summary.")
    st.stop()

sales_error = None
shift_error = None
salary_error = None

try:
    sales_df = load_sales_data()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    sales_df = pd.DataFrame()
    sales_error = str(error)

try:
    shift_df = load_shift_data()
except Exception as error:
    shift_df = pd.DataFrame()
    shift_error = str(error)

try:
    salary_df = load_salary_data()
except Exception as error:
    salary_df = pd.DataFrame()
    salary_error = str(error)

if sales_error:
    st.warning(f"Sales are unavailable: {sales_error}")
if shift_error:
    st.warning(f"Hourly labor is unavailable: {shift_error}")
if salary_error:
    st.warning(
        "Salary roles are unavailable, so only GM and DM "
        f"are left out of hourly labor: {salary_error}"
    )

months = build_store_months(cogs_df, sales_df, shift_df, salary_df)

st.sidebar.header("Filters")

location_names = ordered_locations(months["location"])
selected_locations = st.sidebar.multiselect(
    "Store",
    options=location_names,
    default=location_names,
)
month_starts = month_choices(months)
selected_month_labels = st.sidebar.multiselect(
    "Month",
    options=[month_label(month) for month in month_starts],
    default=[month_label(month) for month in month_starts],
)

st.sidebar.header("Margins")
st.sidebar.caption(
    "Leander 17%. Marble Falls 7%. Lampasas 7%. Temple 22%."
)
st.sidebar.caption(
    f"Hourly labor cutoff {HOURLY_LABOR_CUTOFF:.1f}%. "
    f"COGS cutoff {COGS_CUTOFF:.1f}%."
)
st.sidebar.caption(
    "Actual margin = store margin "
    "+ (hourly cutoff − hourly labor %) "
    "+ (COGS cutoff − COGS %). "
    "Estimated net profit = net sales × actual margin."
)

if not selected_locations or not selected_month_labels:
    st.info("Select at least one store and one month.")
    st.stop()

selected_months = {
    month_label(month)
    for month in month_starts
    if month_label(month) in selected_month_labels
}
view = months[
    months["location"].isin(selected_locations)
    & months["start_date"].map(month_label).isin(selected_months)
].copy()

if view.empty:
    st.info("No COGS rows match the selected stores and months.")
    st.stop()

latest_month = view["start_date"].max()
latest = view.loc[view["start_date"].eq(latest_month)].copy()
latest["location"] = pd.Categorical(
    latest["location"],
    categories=ordered_locations(latest["location"]),
    ordered=True,
)
latest = latest.sort_values("location")

st.subheader(month_label(latest_month))
store_count = len(latest)
metric_columns = st.columns(store_count)
profit_columns = st.columns(store_count)

for index, row in enumerate(latest.itertuples(index=False)):
    cogs_percent = row.cogs_percent
    if pd.isna(cogs_percent):
        delta = None
        help_text = (
            f"{row.location} has COGS for this month, and the "
            "sales table has no net sales in the same dates."
        )
    else:
        delta = f"{cogs_percent - COGS_CUTOFF:+.2f} pts"
        help_text = escape_dollar_signs(
            f"{format_whole_dollars(row.cogs_amount)} COGS of "
            f"{format_whole_dollars(row.net_sales)} net sales. "
            f"Sales days: {row.sales_coverage}."
        )
    metric_columns[index].metric(
        f"{row.location} COGS %",
        format_metric_value(cogs_percent, "percent"),
        delta=delta,
        delta_color="inverse",
        help=help_text,
    )
    profit_columns[index].metric(
        f"{row.location} profit",
        format_whole_dollars(row.estimated_net_profit),
        help=escape_dollar_signs(
            "Actual margin "
            f"{format_metric_value(row.actual_margin_percent, 'percent')}. "
            "Hourly labor "
            f"{format_metric_value(row.hourly_percent, 'percent')} "
            f"against a {HOURLY_LABOR_CUTOFF:.1f}% cutoff."
        ),
    )

chart_locations = ordered_locations(view["location"])
scored = view[view["cogs_percent"].notna()]

st.subheader("COGS % by month")
if scored.empty:
    st.info(
        "COGS % needs net sales in the same month. The sales "
        "table has no business dates inside the selected months "
        "for these stores."
    )
else:
    st.altair_chart(
        cogs_chart(view, chart_locations),
        use_container_width=True,
    )
    st.caption(
        f"The dashed line is the {COGS_CUTOFF:.1f}% COGS cutoff. "
        "Each point is that store's COGS dollars divided by net "
        "sales for the month."
    )

profit_rows = view[view["estimated_net_profit"].notna()]
st.subheader("Estimated net profit")
if profit_rows.empty:
    st.info(
        "Estimated net profit needs net sales and hourly labor "
        "for the same month, and a margin for the store."
    )
else:
    st.altair_chart(
        profit_chart(view, chart_locations),
        use_container_width=True,
    )
    st.caption(
        "Net sales times the actual margin. The actual margin "
        f"is the store margin, plus {HOURLY_LABOR_CUTOFF:.1f}% "
        "minus hourly labor %, plus "
        f"{COGS_CUTOFF:.1f}% minus COGS %."
    )

display_columns = [
    "location",
    "start_date",
    "net_sales",
    "cogs_amount",
    "cogs_percent",
    "hourly_percent",
    "base_margin_percent",
    "actual_margin_percent",
    "estimated_net_profit",
    "sales_coverage",
]
table = view[display_columns].sort_values(
    ["start_date", "location"],
    ascending=[False, True],
)

st.subheader("Store months")
st.dataframe(
    style_table(table),
    column_config=table_column_config(),
    hide_index=True,
    use_container_width=True,
)

short_months = view[
    view["sales_days"].notna()
    & view["sales_days"].lt(view["days_in_month"])
]
if not short_months.empty:
    st.caption(
        f"{len(short_months):,} store month(s) have net sales on "
        "fewer days than the month contains. COGS still covers "
        "the full month, so those COGS % figures are high "
        "relative to a complete sales feed."
    )

missing_sales = view[view["net_sales"].isna()]
if not missing_sales.empty:
    missing_names = ", ".join(ordered_locations(missing_sales["location"]))
    st.caption(
        f"{missing_names} have COGS for a selected month and no "
        "matching net sales, so COGS % and estimated net profit "
        "are blank for those rows."
    )

unknown_margin = view[view["base_margin_percent"].isna()]
if not unknown_margin.empty:
    unknown_names = ", ".join(
        ordered_locations(unknown_margin["location"])
    )
    st.caption(
        f"{unknown_names} have no hardcoded margin, so estimated "
        "net profit is blank for those stores."
    )
