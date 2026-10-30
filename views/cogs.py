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


# Shown first in the store picker.
STORE_ORDER = ["Leander", "Marble Falls", "Lampasas", "Temple"]

HOURLY_LABOR_CUTOFF = 18.0
DEFAULT_COGS_PERCENT = 31.5
DEFAULT_STORE_MARGIN_PERCENT = 10.0

# Chart series. Net sales is drawn first.
BASIS_ORDER = ["Net sales", "Gross sales"]

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
    """Net and gross sales for each store and business date."""

    empty = pd.DataFrame(
        columns=[
            "business_date",
            "location_key",
            "net_sales",
            "gross_sales",
        ]
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
    if "gross_sales" not in working.columns:
        working["gross_sales"] = float("nan")
    working["gross_sales"] = pd.to_numeric(
        working["gross_sales"],
        errors="coerce",
    )
    working = working.dropna(subset=["business_date", "location_key"])
    if working.empty:
        return empty

    grouped = (
        working
        .groupby(["business_date", "location_key"], as_index=False)
        .agg(
            net_sales=("net_sales", "sum"),
            gross_sales=(
                "gross_sales",
                lambda values: values.sum(min_count=1),
            ),
        )
    )
    return grouped


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


def sum_present(values: pd.Series) -> float:
    """Sum a column, and stay blank when every value is blank."""

    return values.sum(min_count=1)


def sum_in_periods(
    daily: pd.DataFrame,
    periods: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Sum a daily measure into each store month.

    A business date counts when it falls on start_date,
    on end_date, or on a day between them. A month whose
    values are all blank stays blank instead of becoming zero.
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
                value_column: (value_column, sum_present),
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
    """COGS % and hourly labor % of net and gross sales by store month.

    Margin and estimated profit are applied later from the
    targets the user enters.
    """

    periods = month_periods(cogs)
    if periods.empty:
        return periods

    daily_sales = sales_by_day(sales)
    sales_totals = sum_in_periods(
        daily_sales,
        periods,
        "net_sales",
    ).rename(columns={"days": "sales_days"})
    gross_totals = sum_in_periods(
        daily_sales,
        periods,
        "gross_sales",
    ).drop(columns=["days"])
    wage_totals = sum_in_periods(
        hourly_wages_by_day(shifts, salaried_roles(salary)),
        periods,
        "hourly_wages",
    ).rename(columns={"days": "labor_days"})

    frame = periods.merge(sales_totals, on=PERIOD_KEYS, how="left")
    frame = frame.merge(gross_totals, on=PERIOD_KEYS, how="left")
    frame = frame.merge(wage_totals, on=PERIOD_KEYS, how="left")
    frame["days_in_month"] = (
        (frame["end_date"] - frame["start_date"]).dt.days + 1
    ).astype(int)
    frame["cogs_percent_net"] = assign_percent(
        frame["cogs_amount"],
        frame["net_sales"],
    )
    frame["cogs_percent_gross"] = assign_percent(
        frame["cogs_amount"],
        frame["gross_sales"],
    )
    frame["hourly_percent_net"] = assign_percent(
        frame["hourly_wages"],
        frame["net_sales"],
    )
    frame["hourly_percent_gross"] = assign_percent(
        frame["hourly_wages"],
        frame["gross_sales"],
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


def apply_margin_inputs(
    frame: pd.DataFrame,
    store_margin_percent: float,
    cogs_net_percent: float,
    cogs_gross_percent: float,
) -> pd.DataFrame:
    """Actual margin and estimated profit from the entered targets.

    Actual margin starts at the store margin. Each point of
    hourly labor above 18% reduces it by a point, and each
    point of COGS above its target reduces it by a point.
    A month under a target gains those points back. Net sales
    and gross sales each use their own hourly labor %, COGS %,
    and COGS target. Estimated profit is that basis's sales
    times its actual margin.
    """

    scored = frame.copy()
    scored["store_margin_percent"] = float(store_margin_percent)
    scored["actual_margin_net"] = (
        scored["store_margin_percent"]
        + (HOURLY_LABOR_CUTOFF - scored["hourly_percent_net"])
        + (float(cogs_net_percent) - scored["cogs_percent_net"])
    )
    scored["actual_margin_gross"] = (
        scored["store_margin_percent"]
        + (HOURLY_LABOR_CUTOFF - scored["hourly_percent_gross"])
        + (float(cogs_gross_percent) - scored["cogs_percent_gross"])
    )
    scored["estimated_profit_net"] = (
        scored["net_sales"] * scored["actual_margin_net"] / 100
    )
    scored["estimated_profit_gross"] = (
        scored["gross_sales"] * scored["actual_margin_gross"] / 100
    )
    return scored


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


def style_table(
    frame: pd.DataFrame,
    cogs_net_percent: float,
    cogs_gross_percent: float,
):
    """Color COGS % and hourly labor % against their targets."""

    styler = frame.style
    styler = styler.map(
        percent_cell_style(cogs_net_percent),
        subset=["cogs_percent_net"],
    )
    styler = styler.map(
        percent_cell_style(cogs_gross_percent),
        subset=["cogs_percent_gross"],
    )
    return styler.map(
        percent_cell_style(HOURLY_LABOR_CUTOFF),
        subset=["hourly_percent_net", "hourly_percent_gross"],
    )


def table_column_config(
    cogs_net_percent: float,
    cogs_gross_percent: float,
) -> dict:
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
        "gross_sales": st.column_config.NumberColumn(
            "Gross sales",
            format="$%,.0f",
            help=(
                "Gross sales whose business date falls in "
                "this month."
            ),
        ),
        "cogs_amount": st.column_config.NumberColumn(
            "COGS $",
            format="$%,.0f",
            help="Sum of cogs_summary amount for this store and month.",
        ),
        "cogs_percent_net": st.column_config.NumberColumn(
            "COGS % of net",
            format="%.2f%%",
            help=(
                "COGS dollars divided by net sales. "
                f"The target is {cogs_net_percent:.1f}%."
            ),
        ),
        "cogs_percent_gross": st.column_config.NumberColumn(
            "COGS % of gross",
            format="%.2f%%",
            help=(
                "COGS dollars divided by gross sales. "
                f"The target is {cogs_gross_percent:.1f}%."
            ),
        ),
        "hourly_percent_net": st.column_config.NumberColumn(
            "Hourly labor % of net",
            format="%.2f%%",
            help=(
                "Hourly shift wages divided by net sales. "
                "Salaried roles are left out. "
                f"The cutoff is {HOURLY_LABOR_CUTOFF:.1f}%."
            ),
        ),
        "hourly_percent_gross": st.column_config.NumberColumn(
            "Hourly labor % of gross",
            format="%.2f%%",
            help=(
                "Hourly shift wages divided by gross sales. "
                "Salaried roles are left out. "
                f"The cutoff is {HOURLY_LABOR_CUTOFF:.1f}%."
            ),
        ),
        "store_margin_percent": st.column_config.NumberColumn(
            "Store margin",
            format="%.1f%%",
            help="Store margin entered for this view.",
        ),
        "actual_margin_net": st.column_config.NumberColumn(
            "Actual margin, net",
            format="%.2f%%",
            help=(
                "Store margin + (hourly cutoff − hourly labor % of net) "
                "+ (COGS % of net sales − COGS % of net)."
            ),
        ),
        "actual_margin_gross": st.column_config.NumberColumn(
            "Actual margin, gross",
            format="%.2f%%",
            help=(
                "Store margin + (hourly cutoff − hourly labor % of gross) "
                "+ (COGS % of gross sales − COGS % of gross)."
            ),
        ),
        "estimated_profit_net": st.column_config.NumberColumn(
            "Profit on net",
            format="$%,.0f",
            help="Net sales times the actual margin for net sales.",
        ),
        "estimated_profit_gross": st.column_config.NumberColumn(
            "Profit on gross",
            format="$%,.0f",
            help="Gross sales times the actual margin for gross sales.",
        ),
        "sales_coverage": st.column_config.TextColumn(
            "Sales days",
            help=(
                "Business dates with sales, out of the "
                "days in the month. COGS is the full month, "
                "so a short sales feed makes COGS % high."
            ),
        ),
    }


def basis_color(show_legend: bool = True) -> alt.Color:
    """Net sales in blue, gross sales in orange."""

    return alt.Color(
        "basis:N",
        title=None,
        sort=BASIS_ORDER,
        scale=alt.Scale(
            domain=BASIS_ORDER,
            range=STORE_COLORS[: len(BASIS_ORDER)],
        ),
        legend=alt.Legend() if show_legend else None,
    )


def to_basis_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per month for net sales, and one for gross sales."""

    specs = (
        (
            "Net sales",
            "cogs_percent_net",
            "hourly_percent_net",
            "actual_margin_net",
            "estimated_profit_net",
            "net_sales",
        ),
        (
            "Gross sales",
            "cogs_percent_gross",
            "hourly_percent_gross",
            "actual_margin_gross",
            "estimated_profit_gross",
            "gross_sales",
        ),
    )
    pieces = []
    for (
        basis,
        cogs_column,
        hourly_column,
        margin_column,
        profit_column,
        sales_column,
    ) in specs:
        piece = pd.DataFrame(
            {
                "start_date": frame["start_date"].to_numpy(),
                "location": frame["location"].to_numpy(),
                "cogs_amount": frame["cogs_amount"].to_numpy(),
                "sales_coverage": frame["sales_coverage"].to_numpy(),
                "basis": basis,
                "sales": frame[sales_column].to_numpy(),
                "cogs_percent": frame[cogs_column].to_numpy(),
                "hourly_percent": frame[hourly_column].to_numpy(),
                "actual_margin_percent": frame[margin_column].to_numpy(),
                "estimated_profit": frame[profit_column].to_numpy(),
            }
        )
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


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


def cutoff_rules(
    cogs_net_percent: float,
    cogs_gross_percent: float,
) -> alt.Chart:
    """Dashed target for each sales basis.

    One shared target is a single red line. Two different
    targets use the same colors as the COGS lines.
    """

    if cogs_net_percent == cogs_gross_percent:
        return (
            alt.Chart(pd.DataFrame({"cutoff": [cogs_net_percent]}))
            .mark_rule(strokeDash=[6, 4], color="#b42318")
            .encode(y="cutoff:Q")
        )

    targets = pd.DataFrame(
        {
            "basis": BASIS_ORDER,
            "cutoff": [cogs_net_percent, cogs_gross_percent],
        }
    )
    return (
        alt.Chart(targets)
        .mark_rule(strokeDash=[6, 4])
        .encode(y="cutoff:Q", color=basis_color(show_legend=False))
    )


def cogs_chart(
    frame: pd.DataFrame,
    cogs_net_percent: float,
    cogs_gross_percent: float,
) -> alt.Chart:
    """COGS % of net sales and of gross sales for one month.

    Bars stand in for a line because the page shows one month.
    Dashed rules are the COGS targets.
    """

    scored = with_month_label(
        to_basis_rows(frame)
    )
    scored = scored[scored["cogs_percent"].notna()]
    y_max = max(
        float(scored["cogs_percent"].max()),
        float(cogs_net_percent),
        float(cogs_gross_percent),
        1.0,
    ) * 1.15
    bars = (
        alt.Chart(scored)
        .mark_bar(size=72)
        .encode(
            x=month_axis(scored),
            xOffset=alt.XOffset("basis:N", sort=BASIS_ORDER),
            y=alt.Y(
                "cogs_percent:Q",
                title="COGS %",
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(format=".1f"),
            ),
            color=basis_color(),
            tooltip=[
                alt.Tooltip("basis:N", title="Sales"),
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
                    "sales:Q",
                    title="Sales",
                    format="$,.0f",
                ),
            ],
        )
    )
    return (bars + cutoff_rules(
        cogs_net_percent,
        cogs_gross_percent,
    )).properties(height=340)


def profit_chart(frame: pd.DataFrame) -> alt.Chart:
    """Estimated profit on net sales and on gross sales."""

    scored = with_month_label(to_basis_rows(frame))
    scored = scored[scored["estimated_profit"].notna()]
    return (
        alt.Chart(scored)
        .mark_bar()
        .encode(
            x=month_axis(scored),
            xOffset=alt.XOffset("basis:N", sort=BASIS_ORDER),
            y=alt.Y(
                "estimated_profit:Q",
                title="Estimated profit",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=basis_color(),
            tooltip=[
                alt.Tooltip("basis:N", title="Sales"),
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip(
                    "estimated_profit:Q",
                    title="Estimated profit",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "actual_margin_percent:Q",
                    title="Actual margin %",
                    format=".2f",
                ),
                alt.Tooltip(
                    "sales:Q",
                    title="Sales",
                    format="$,.0f",
                ),
            ],
        )
        .properties(height=340)
    )


def render_latest_metrics(
    row: pd.Series,
    cogs_net_percent: float,
    cogs_gross_percent: float,
) -> None:
    """One month: COGS %, labor cutoff, actual margins, and profit."""

    cogs_columns = st.columns(3)
    cogs_specs = (
        (
            "COGS % of net sales",
            "cogs_percent_net",
            cogs_net_percent,
            "net_sales",
            "net sales",
        ),
        (
            "COGS % of gross sales",
            "cogs_percent_gross",
            cogs_gross_percent,
            "gross_sales",
            "gross sales",
        ),
    )
    for index, (
        label,
        percent_column,
        target,
        sales_column,
        sales_name,
    ) in enumerate(cogs_specs):
        percent = row[percent_column]
        sales = row[sales_column]
        if pd.isna(percent):
            delta = None
            help_text = (
                f"{row['location']} has COGS for this month, and "
                f"the sales table has no {sales_name} in the "
                "same dates."
            )
        else:
            delta = f"{percent - target:+.2f} pts"
            help_text = escape_dollar_signs(
                f"{format_whole_dollars(row['cogs_amount'])} COGS of "
                f"{format_whole_dollars(sales)} {sales_name}. "
                f"Target {target:.1f}%. "
                f"Sales days: {row['sales_coverage']}."
            )
        cogs_columns[index].metric(
            label,
            format_metric_value(percent, "percent"),
            delta=delta,
            delta_color="inverse",
            help=help_text,
        )

    cogs_columns[2].metric(
        "Hourly labor cutoff",
        format_metric_value(HOURLY_LABOR_CUTOFF, "percent"),
        help=(
            "Hourly labor is "
            f"{format_metric_value(row['hourly_percent_net'], 'percent')} "
            "of net sales and "
            f"{format_metric_value(row['hourly_percent_gross'], 'percent')} "
            "of gross sales. Each point above this cutoff lowers "
            "actual margin by a point."
        ),
    )

    result_columns = st.columns(4)
    margin_specs = (
        (
            "Actual margin, net",
            "actual_margin_net",
            "hourly_percent_net",
            "cogs_percent_net",
            cogs_net_percent,
            "net sales",
        ),
        (
            "Actual margin, gross",
            "actual_margin_gross",
            "hourly_percent_gross",
            "cogs_percent_gross",
            cogs_gross_percent,
            "gross sales",
        ),
    )
    for index, (
        label,
        margin_column,
        hourly_column,
        cogs_column,
        cogs_target,
        sales_name,
    ) in enumerate(margin_specs):
        result_columns[index].metric(
            label,
            format_metric_value(row[margin_column], "percent"),
            help=(
                f"Store margin plus ({HOURLY_LABOR_CUTOFF:.1f}% − hourly "
                f"labor % of {sales_name}) plus ({cogs_target:.1f}% − "
                f"COGS % of {sales_name}). Hourly labor "
                f"{format_metric_value(row[hourly_column], 'percent')}. "
                "COGS "
                f"{format_metric_value(row[cogs_column], 'percent')}."
            ),
        )

    profit_specs = (
        (
            "Profit on net sales",
            "estimated_profit_net",
            "actual_margin_net",
            "net sales",
        ),
        (
            "Profit on gross sales",
            "estimated_profit_gross",
            "actual_margin_gross",
            "gross sales",
        ),
    )
    for index, (
        label,
        profit_column,
        margin_column,
        sales_name,
    ) in enumerate(profit_specs):
        result_columns[index + 2].metric(
            label,
            format_whole_dollars(row[profit_column]),
            help=escape_dollar_signs(
                f"{sales_name.capitalize()} times the actual margin "
                f"of {format_metric_value(row[margin_column], 'percent')}."
            ),
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
    "Pick one store, then set the store margin and the COGS "
    "targets for net sales and gross sales. COGS % is that "
    "month's cost of goods divided by sales whose business "
    "date falls in the month. Estimated profit starts from "
    "the store margin, then moves by the same number of points "
    "that hourly labor and COGS miss their targets."
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
if not location_names:
    st.info("No stores were returned with COGS.")
    st.stop()

default_store_index = 0
for index, name in enumerate(location_names):
    if str(name).casefold() == "leander":
        default_store_index = index
        break

selected_location = st.sidebar.selectbox(
    "Store",
    options=location_names,
    index=default_store_index,
)

store_months = months.loc[
    months["location"] == selected_location
].copy()
month_starts = month_choices(store_months)
month_options = [month_label(month) for month in month_starts]
if month_options:
    selected_month_label = st.sidebar.selectbox(
        "Month",
        options=month_options,
        index=0,
        key=f"cogs_month_{selected_location}",
    )
else:
    selected_month_label = ""

st.sidebar.header("Margins")
cogs_net_percent = st.sidebar.number_input(
    "COGS % of net sales",
    min_value=0.0,
    max_value=100.0,
    value=DEFAULT_COGS_PERCENT,
    step=0.1,
    format="%.1f",
    help=(
        "Target COGS as a percent of net sales. Each point "
        "above this lowers the actual margin on net sales "
        "by a point."
    ),
)
cogs_gross_percent = st.sidebar.number_input(
    "COGS % of gross sales",
    min_value=0.0,
    max_value=100.0,
    value=DEFAULT_COGS_PERCENT,
    step=0.1,
    format="%.1f",
    help=(
        "Target COGS as a percent of gross sales. Each point "
        "above this lowers the actual margin on gross sales "
        "by a point."
    ),
)
store_margin_percent = st.sidebar.number_input(
    "Store margin",
    min_value=0.0,
    max_value=100.0,
    value=DEFAULT_STORE_MARGIN_PERCENT,
    step=0.1,
    format="%.1f",
    help=(
        "Starting margin, in percent. Hourly labor and COGS "
        "move the actual margin from here."
    ),
)
st.sidebar.caption(
    f"Hourly labor cutoff {HOURLY_LABOR_CUTOFF:.1f}%."
)
st.sidebar.caption(
    "Actual margin = store margin "
    "+ (hourly cutoff − hourly labor %) "
    "+ (COGS target − COGS %). "
    "Estimated profit = sales × actual margin. "
    "Net sales and gross sales are calculated separately."
)

if not month_options:
    st.info(f"No COGS months for {selected_location}.")
    st.stop()

view = store_months.loc[
    store_months["start_date"].map(month_label).eq(selected_month_label)
].copy()

if view.empty:
    st.info("No COGS rows match the selected store and month.")
    st.stop()

view = apply_margin_inputs(
    view,
    store_margin_percent,
    cogs_net_percent,
    cogs_gross_percent,
)

latest_month = view["start_date"].max()
latest = view.loc[view["start_date"].eq(latest_month)].iloc[0]

st.subheader(f"{latest['location']} · {month_label(latest_month)}")
render_latest_metrics(
    latest,
    cogs_net_percent,
    cogs_gross_percent,
)

has_cogs_percent = (
    view["cogs_percent_net"].notna()
    | view["cogs_percent_gross"].notna()
)
st.subheader("COGS % by month")
if not bool(has_cogs_percent.any()):
    st.info(
        "COGS % needs net sales or gross sales in the same "
        "month. The sales table has no business dates inside "
        "the selected month for this store."
    )
else:
    st.altair_chart(
        cogs_chart(view, cogs_net_percent, cogs_gross_percent),
        use_container_width=True,
    )
    if cogs_net_percent == cogs_gross_percent:
        target_text = (
            f"The dashed line is the {cogs_net_percent:.1f}% "
            "COGS target for both net sales and gross sales."
        )
    else:
        target_text = (
            "Dashed lines mark the targets: "
            f"{cogs_net_percent:.1f}% of net sales and "
            f"{cogs_gross_percent:.1f}% of gross sales."
        )
    st.caption(
        target_text
        + " Each bar is that month's COGS dollars divided "
        "by sales."
    )

has_profit = (
    view["estimated_profit_net"].notna()
    | view["estimated_profit_gross"].notna()
)
st.subheader("Estimated profit")
if not bool(has_profit.any()):
    st.info(
        "Estimated profit needs sales and hourly labor for "
        "the same month."
    )
else:
    st.altair_chart(
        profit_chart(view),
        use_container_width=True,
    )
    st.caption(
        "Sales times the actual margin. The actual margin "
        f"is the store margin of {store_margin_percent:.1f}%, "
        f"plus {HOURLY_LABOR_CUTOFF:.1f}% minus hourly labor %, "
        "plus the COGS target minus COGS %. Net sales and "
        "gross sales each use their own hourly labor %, "
        "COGS %, and COGS target."
    )

display_columns = [
    "location",
    "start_date",
    "net_sales",
    "gross_sales",
    "cogs_amount",
    "cogs_percent_net",
    "cogs_percent_gross",
    "hourly_percent_net",
    "hourly_percent_gross",
    "store_margin_percent",
    "actual_margin_net",
    "actual_margin_gross",
    "estimated_profit_net",
    "estimated_profit_gross",
    "sales_coverage",
]
table = view[display_columns].sort_values(
    ["start_date", "location"],
    ascending=[False, True],
)

st.subheader("Store months")
st.dataframe(
    style_table(table, cogs_net_percent, cogs_gross_percent),
    column_config=table_column_config(
        cogs_net_percent,
        cogs_gross_percent,
    ),
    hide_index=True,
    use_container_width=True,
)

short_months = view[
    view["sales_days"].notna()
    & view["sales_days"].lt(view["days_in_month"])
]
if not short_months.empty:
    st.caption(
        "This month has sales on fewer days than it contains. "
        "COGS still covers the full month, so the COGS % "
        "figures are high relative to a complete sales feed."
    )

if bool(view["net_sales"].isna().any()):
    st.caption(
        "This month has COGS and no matching net sales, so "
        "COGS % of net sales, the net actual margin, and "
        "profit on net sales are blank."
    )

if bool(view["gross_sales"].isna().any()):
    st.caption(
        "This month has COGS and no matching gross sales, so "
        "COGS % of gross sales, the gross actual margin, and "
        "profit on gross sales are blank."
    )
