import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    NOT_AVAILABLE,
    MissingSecretError,
    escape_dollar_signs,
    format_metric_value,
    load_salary_data,
    load_sales_data,
    load_shift_data,
    percent_change,
)


# Fixed-length buckets counted backward from the date the
# user picks. A month is 30 days so every month is the
# same length, matching Sales Growth.
INTERVALS = {
    "Daily": {
        "days": 1,
        "max_duration": 60,
        "default_duration": 30,
        "unit": "days",
        "x_title": "Business date",
        "duration_label": "Duration (days)",
        "summary_title": "Selected Day vs Previous Day",
    },
    "Weekly": {
        "days": 7,
        "max_duration": 52,
        "default_duration": 8,
        "unit": "weeks",
        "x_title": "Week",
        "duration_label": "Duration (weeks)",
        "summary_title": "Selected Week vs Previous Week",
    },
    "Monthly": {
        "days": 30,
        "max_duration": 12,
        "default_duration": 6,
        "unit": "months",
        "x_title": "30-day period",
        "duration_label": "Duration (months)",
        "summary_title": "Selected Month vs Previous Month",
    },
}

# Columns summed when rolling daily labor up to a period.
PERIOD_SUM_COLUMNS = [
    "net_sales",
    "hourly_labor",
    "gm_labor",
    "dm_labor",
    "other_salary_labor",
    "salary_labor",
    "total_labor",
    "labor_hours",
    "ot_hours",
    "ot_labor",
]

# daily_salary role names, matched after trimming and
# upper-casing. Anything else still counts as salary.
SALARY_ROLE_COLUMNS = {
    "GM": "gm_wages",
    "DM": "dm_wages",
}

BAR_TYPE_LABELS = {
    "hourly_labor": "Hourly",
    "gm_labor": "GM",
    "dm_labor": "DM",
}

SEGMENT_COLORS = {
    "Hourly": "#1f77b4",
    "GM": "#9467bd",
    "DM": "#2ca02c",
    "Total": "#ff7f0e",
    "Other salary": "#7f7f7f",
}


def color_scale(labels: list[str]) -> alt.Scale:
    """Colors shared by the bars and the matching rate lines."""

    return alt.Scale(
        domain=labels,
        range=[SEGMENT_COLORS[label] for label in labels],
    )


def stack_labor_types(
    frame: pd.DataFrame,
    id_columns: list[str],
    type_labels: dict[str, str],
) -> pd.DataFrame:
    """Reshape to one row per labor type for stacked bars."""

    long_df = frame.melt(
        id_vars=id_columns,
        value_vars=list(type_labels.keys()),
        var_name="labor_type",
        value_name="labor_dollars",
    )

    long_df["labor_type"] = long_df["labor_type"].map(
        type_labels
    )

    return long_df


def rank_stack(
    frame: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> tuple[pd.DataFrame, list]:
    """Largest category at the bottom, then smaller ones above."""

    ranked = frame.copy()

    if ranked.empty:
        ranked["stack_rank"] = pd.Series(dtype="int64")
        return ranked, []

    totals = (
        ranked.groupby(category_column, dropna=False)[value_column]
        .sum()
        .reset_index()
        .sort_values(
            [value_column, category_column],
            ascending=[False, True],
        )
    )
    category_order = totals[category_column].tolist()
    rank_by_category = {
        category: index
        for index, category in enumerate(category_order)
    }
    ranked["stack_rank"] = ranked[category_column].map(
        rank_by_category
    )

    return ranked, category_order


def format_whole_dollars(value) -> str:
    """Currency with no cents."""

    if value is None or pd.isna(value):
        return "—"

    return f"${value:,.0f}"


def melt_labor_rates(
    frame: pd.DataFrame,
    id_columns: list[str],
    rate_series: dict[str, str],
) -> pd.DataFrame:
    """One row per hourly, GM, DM, and total labor rate."""

    long_df = frame.melt(
        id_vars=id_columns,
        value_vars=list(rate_series.keys()),
        var_name="rate_series",
        value_name="rate_percent",
    )

    long_df["rate_series"] = long_df["rate_series"].map(
        rate_series
    )

    return long_df


def percent_axis_limits(
    frame: pd.DataFrame,
    rate_columns: list[str],
    extra_values: list[float],
) -> tuple[float, float]:
    """Y domain that fits every rate and every target."""

    values = (
        frame[rate_columns]
        .to_numpy(dtype=float)
        .ravel()
    )
    values = values[~pd.isna(values)]
    candidates = [*values.tolist(), *extra_values]

    if not candidates or max(candidates) <= 0:
        return 0.0, 1.0

    return 0.0, max(candidates) * 1.08


def percent_of_sales(labor_dollars, net_sales) -> float:
    """Labor dollars as a percent of sales, or NaN."""

    if pd.isna(labor_dollars) or pd.isna(net_sales):
        return NOT_AVAILABLE

    if net_sales == 0:
        return NOT_AVAILABLE

    return labor_dollars / net_sales * 100


def salaried_role_keys(salary_frame: pd.DataFrame) -> set[str]:
    """Role names paid from the salary feed, not hourly shifts."""

    keys = {role.strip().upper() for role in SALARY_ROLE_COLUMNS}
    if not salary_frame.empty and "role" in salary_frame.columns:
        keys.update(
            salary_frame["role"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    keys.discard("")
    return keys


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two series, leaving zero denominators blank."""

    return numerator.astype(float) / denominator.astype(
        float
    ).replace(0.0, NOT_AVAILABLE)


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
    """Full date range shown in tooltips."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if period_days == 1:
        return f"{end:%b %d, %Y}"

    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def build_labor_period_frame(
    frame: pd.DataFrame,
    date_end,
    period_days: int,
    duration: int,
) -> pd.DataFrame:
    """Bucket daily labor into equal periods ending on date_end.

    A period with no reported days stays blank instead of
    becoming zero.
    """

    date_end = pd.Timestamp(date_end).normalize()
    period_days = int(period_days)
    duration = int(duration)
    sum_columns = [
        column_name
        for column_name in PERIOD_SUM_COLUMNS
        if column_name in frame.columns
    ]

    working = frame.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"]
    ).dt.normalize()

    history_start = date_end - pd.Timedelta(
        days=period_days * duration - 1
    )

    calendar_rows = []

    for period_index in range(duration):
        period_end = date_end - pd.Timedelta(
            days=period_days * period_index
        )
        period_start = period_end - pd.Timedelta(
            days=period_days - 1
        )
        calendar_rows.append(
            {
                "period_index": period_index,
                "period_start": period_start,
                "period_end": period_end,
            }
        )

    calendar = pd.DataFrame(calendar_rows)

    scoped = working.loc[
        working["business_date"].between(
            history_start,
            date_end,
            inclusive="both",
        )
    ].copy()

    if not scoped.empty:
        scoped["period_index"] = (
            (date_end - scoped["business_date"]).dt.days
            // period_days
        ).astype(int)
        scoped = scoped[
            scoped["period_index"].between(0, duration - 1)
        ]

    if scoped.empty:
        grouped = pd.DataFrame(
            {"period_index": pd.Series(dtype="int64")}
        )
    else:
        grouped = (
            scoped
            .groupby("period_index", as_index=False)
            .agg(
                **{
                    column_name: (column_name, "sum")
                    for column_name in sum_columns
                },
                days_counted=("business_date", "nunique"),
            )
        )

    calendar = calendar.merge(
        grouped,
        on="period_index",
        how="left",
    )

    for column_name in sum_columns:
        if column_name not in calendar.columns:
            calendar[column_name] = NOT_AVAILABLE

    if "days_counted" not in calendar.columns:
        calendar["days_counted"] = 0

    calendar["days_counted"] = (
        calendar["days_counted"].fillna(0).astype(int)
    )

    missing_period = calendar["days_counted"].eq(0)
    calendar.loc[missing_period, sum_columns] = NOT_AVAILABLE

    if "labor_hours" in calendar.columns:
        calendar["sales_per_labor_hour"] = safe_ratio(
            calendar["net_sales"],
            calendar["labor_hours"],
        )

    calendar["is_complete"] = (
        calendar["days_counted"] >= period_days
    )

    # A period rate is total dollars over total sales, not
    # the average of the daily rates.
    rate_sources = [
        ("labor_percent", "total_labor", "net_sales"),
        ("hourly_percent", "hourly_labor", "net_sales"),
        ("gm_percent", "gm_labor", "net_sales"),
        ("dm_percent", "dm_labor", "net_sales"),
        ("other_percent", "other_salary_labor", "net_sales"),
        ("ot_labor_percent", "ot_labor", "total_labor"),
    ]

    for rate_name, numerator, denominator in rate_sources:
        if (
            numerator not in calendar.columns
            or denominator not in calendar.columns
        ):
            continue

        calendar[rate_name] = (
            safe_ratio(
                calendar[numerator],
                calendar[denominator],
            )
            * 100
        )

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

    calendar = (
        calendar
        .sort_values("period_end")
        .reset_index(drop=True)
    )

    if "labor_percent" in calendar.columns:
        calendar["previous_labor_percent"] = calendar[
            "labor_percent"
        ].shift(1)
        calendar["labor_percent_change"] = (
            calendar["labor_percent"]
            - calendar["previous_labor_percent"]
        )

    return calendar


def rows_in_periods(
    frame: pd.DataFrame,
    date_end,
    period_days: int,
    duration: int,
) -> pd.DataFrame:
    """Keep rows inside the selected periods and tag each one.

    period_index 0 is the bucket that ends on date_end, matching
    the labor rate chart.
    """

    tagged = frame.copy()

    if tagged.empty:
        tagged["period_index"] = pd.Series(dtype="int64")
        return tagged

    date_end = pd.Timestamp(date_end).normalize()
    period_days = int(period_days)
    duration = int(duration)
    history_start = date_end - pd.Timedelta(
        days=period_days * duration - 1
    )

    tagged["business_date"] = pd.to_datetime(
        tagged["business_date"]
    ).dt.normalize()
    tagged = tagged.loc[
        tagged["business_date"].between(
            history_start,
            date_end,
            inclusive="both",
        )
    ].copy()

    if tagged.empty:
        tagged["period_index"] = pd.Series(dtype="int64")
        return tagged

    tagged["period_index"] = (
        (date_end - tagged["business_date"]).dt.days
        // period_days
    ).astype(int)

    return tagged.loc[
        tagged["period_index"].between(0, duration - 1)
    ].copy()


def count_employees_by_period(
    shift_frame: pd.DataFrame,
    labor_frame: pd.DataFrame,
    date_end,
    period_days: int,
    duration: int,
) -> pd.DataFrame:
    """Unique employees on days that count toward each period."""

    empty = pd.DataFrame(
        {
            "period_index": pd.Series(dtype="int64"),
            "employee_count": pd.Series(dtype="int64"),
        }
    )

    if (
        shift_frame.empty
        or labor_frame.empty
        or "employee" not in shift_frame.columns
    ):
        return empty

    tagged = rows_in_periods(
        shift_frame,
        date_end,
        period_days,
        duration,
    )

    if tagged.empty:
        return empty

    labor_days = (
        labor_frame[["business_date", "location_key"]]
        .drop_duplicates()
        .copy()
    )
    labor_days["business_date"] = pd.to_datetime(
        labor_days["business_date"]
    ).dt.normalize()

    tagged = tagged.merge(
        labor_days,
        on=["business_date", "location_key"],
        how="inner",
    )

    if tagged.empty:
        return empty

    counted = (
        tagged
        .groupby("period_index", as_index=False)
        .agg(employee_count=("employee", "nunique"))
    )
    counted["employee_count"] = (
        counted["employee_count"].astype(int)
    )

    return counted


st.title("Labor % of Net Sales")
st.caption(
    "Hourly shift wages plus prorated GM and DM salary, "
    "measured against net sales."
)

try:
    sales_df = load_sales_data()
    shift_df = load_shift_data()
    salary_df = load_salary_data()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        f"Unable to load data from Supabase: {error} "
        "Confirm the table names in revel_data.py "
        "(SALES_TABLE, SHIFT_TABLE, SALARY_TABLE)."
    )
    st.stop()

if sales_df.empty:
    st.warning("No sales records were returned from Supabase.")
    st.stop()

if shift_df.empty and salary_df.empty:
    st.warning(
        "No labor records were returned from Supabase. "
        "Check the shift and salary table names, "
        "credentials, and Row Level Security settings."
    )
    st.stop()


# ---------------------------------------------------------
# Daily labor assembled from three tables
# ---------------------------------------------------------

daily_sales_df = (
    sales_df
    .groupby(["business_date", "location_key"], as_index=False)
    .agg(
        location=("location", "first"),
        net_sales=("net_sales", "sum"),
    )
)

def empty_daily_frame(value_columns: list[str]) -> pd.DataFrame:
    """Build a typed empty frame so merges keep their dtypes."""

    columns = {
        "business_date": pd.Series(dtype="datetime64[ns]"),
        "location_key": pd.Series(dtype="object"),
    }

    for column_name in value_columns:
        columns[column_name] = pd.Series(dtype="float64")

    return pd.DataFrame(columns)


if shift_df.empty:
    daily_hourly_df = empty_daily_frame(
        [
            "hourly_wages",
            "regular_hours",
            "ot_hours",
            "ot_wages",
            "shift_count",
            "employee_count",
            "hourly_labor_hours",
        ]
    )
else:
    daily_hourly_df = (
        shift_df
        .groupby(
            ["business_date", "location_key"],
            as_index=False,
        )
        .agg(
            # shift_wages is the per-shift total. ot_wages
            # is the overtime portion of that same shift.
            hourly_wages=("shift_wages", "sum"),
            regular_hours=("regular_hours", "sum"),
            ot_hours=("ot_hours", "sum"),
            ot_wages=("ot_wages", "sum"),
            shift_count=("shift_wages", "size"),
            employee_count=("employee", "nunique"),
        )
    )
    role_key = (
        shift_df["role"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    hourly_only = shift_df.loc[
        ~role_key.isin(salaried_role_keys(salary_df))
    ].copy()
    if hourly_only.empty:
        daily_hourly_df["hourly_labor_hours"] = 0.0
    else:
        hourly_only["hourly_labor_hours"] = (
            hourly_only["regular_hours"].fillna(0)
            + hourly_only["ot_hours"].fillna(0)
        )
        hourly_hours = (
            hourly_only
            .groupby(
                ["business_date", "location_key"],
                as_index=False,
            )["hourly_labor_hours"]
            .sum()
        )
        daily_hourly_df = daily_hourly_df.merge(
            hourly_hours,
            on=["business_date", "location_key"],
            how="left",
        )
        daily_hourly_df["hourly_labor_hours"] = (
            daily_hourly_df["hourly_labor_hours"].fillna(0.0)
        )

salary_wage_columns = [
    "gm_wages",
    "dm_wages",
    "other_salary_wages",
    "salary_wages",
]

if salary_df.empty:
    daily_salary_df = empty_daily_frame(salary_wage_columns)
else:
    salary_by_role = salary_df.copy()
    salary_by_role["role_key"] = (
        salary_by_role["role"].str.strip().str.upper()
    )

    daily_salary_df = (
        salary_by_role
        .pivot_table(
            index=["business_date", "location_key"],
            columns="role_key",
            values="daily_salary",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    daily_salary_df.columns.name = None

    for role_key, column_name in SALARY_ROLE_COLUMNS.items():
        if role_key in daily_salary_df.columns:
            daily_salary_df = daily_salary_df.rename(
                columns={role_key: column_name}
            )
        else:
            daily_salary_df[column_name] = 0.0

    named_wage_columns = set(SALARY_ROLE_COLUMNS.values())
    extra_role_columns = [
        column_name
        for column_name in daily_salary_df.columns
        if column_name not in {
            "business_date",
            "location_key",
            *named_wage_columns,
        }
    ]

    if extra_role_columns:
        daily_salary_df["other_salary_wages"] = (
            daily_salary_df[extra_role_columns].sum(axis=1)
        )
        daily_salary_df = daily_salary_df.drop(
            columns=extra_role_columns
        )
    else:
        daily_salary_df["other_salary_wages"] = 0.0

    daily_salary_df["salary_wages"] = (
        daily_salary_df["gm_wages"]
        + daily_salary_df["dm_wages"]
        + daily_salary_df["other_salary_wages"]
    )

# Labor percent is only meaningful on days that report both
# sales and hourly labor, so this join is deliberately an
# inner join on the hourly data.
labor_df = daily_sales_df.merge(
    daily_hourly_df,
    on=["business_date", "location_key"],
    how="inner",
).merge(
    daily_salary_df,
    on=["business_date", "location_key"],
    how="left",
)

labor_df[salary_wage_columns] = (
    labor_df[salary_wage_columns].fillna(0.0)
)


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    labor_df["location"].dropna().unique().tolist()
)

selected_location = st.sidebar.selectbox(
    "Location",
    options=["All Locations", *available_locations],
)

if selected_location != "All Locations":
    labor_df = labor_df[
        labor_df["location"] == selected_location
    ].copy()
    location_sales_df = daily_sales_df[
        daily_sales_df["location"] == selected_location
    ].copy()
else:
    labor_df = labor_df.copy()
    location_sales_df = daily_sales_df.copy()

if labor_df.empty:
    st.warning(
        "No days have both sales and labor records for the "
        "selected location. The sales table covers "
        f"{len(location_sales_df):,} day(s), but none of "
        "them line up with a shift record."
    )
    st.stop()

selected_interval = st.sidebar.selectbox(
    "Interval",
    options=list(INTERVALS),
    index=list(INTERVALS).index("Weekly"),
    help=(
        "Periods are counted backward from Date end. "
        "A week is 7 days and a month is 30 days. "
        "This sets the summary metrics, the labor "
        "charts, and the labor breakdown table."
    ),
)

interval = INTERVALS[selected_interval]
period_days = interval["days"]

min_date = labor_df["business_date"].min().date()
max_date = labor_df["business_date"].max().date()
date_key = "labor_rate_date_end"

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
        "the labor charts, and the breakdown table."
    ),
)

duration = st.sidebar.number_input(
    interval["duration_label"],
    min_value=1,
    max_value=interval["max_duration"],
    value=interval["default_duration"],
    step=1,
    key=f"labor_rate_duration_{selected_interval}",
    help=(
        f"How many {interval['unit']} to include, ending "
        f"on Date end. "
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
    st.sidebar.caption(
        "Each period is 7 days, ending on Date end."
    )
else:
    st.sidebar.caption(
        "Each period is 30 days, ending on Date end."
    )

hourly_target_percent = st.sidebar.number_input(
    "Hourly labor %",
    min_value=0.0,
    max_value=100.0,
    value=18.0,
    step=0.5,
    help="Target for hourly shift wages as a percent of net sales.",
)

gm_target_percent = st.sidebar.number_input(
    "GM labor %",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.5,
    help="Target for GM salary as a percent of net sales.",
)

dm_target_percent = st.sidebar.number_input(
    "DM labor %",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.5,
    help="Target for DM salary as a percent of net sales.",
)

target_labor_percent = (
    hourly_target_percent
    + gm_target_percent
    + dm_target_percent
)

st.sidebar.metric(
    "Target labor %",
    f"{target_labor_percent:.1f}%",
    help="Hourly labor % plus GM labor % plus DM labor %.",
)


# ---------------------------------------------------------
# Labor dollars, hours, and rate
# ---------------------------------------------------------

labor_df["hourly_labor"] = labor_df["hourly_wages"]
labor_df["gm_labor"] = labor_df["gm_wages"]
labor_df["dm_labor"] = labor_df["dm_wages"]
labor_df["other_salary_labor"] = labor_df["other_salary_wages"]
labor_df["salary_labor"] = (
    labor_df["gm_labor"]
    + labor_df["dm_labor"]
    + labor_df["other_salary_labor"]
)
labor_df["ot_labor"] = labor_df["ot_wages"]
labor_df["total_labor"] = (
    labor_df["hourly_labor"] + labor_df["salary_labor"]
)
labor_df["labor_hours"] = labor_df["hourly_labor_hours"].fillna(
    0.0
)

labor_df["labor_percent"] = (
    safe_ratio(
        labor_df["total_labor"],
        labor_df["net_sales"],
    )
    * 100
)

labor_df["hourly_percent"] = (
    safe_ratio(
        labor_df["hourly_labor"],
        labor_df["net_sales"],
    )
    * 100
)

labor_df["gm_percent"] = (
    safe_ratio(
        labor_df["gm_labor"],
        labor_df["net_sales"],
    )
    * 100
)

labor_df["dm_percent"] = (
    safe_ratio(
        labor_df["dm_labor"],
        labor_df["net_sales"],
    )
    * 100
)

labor_df["other_percent"] = (
    safe_ratio(
        labor_df["other_salary_labor"],
        labor_df["net_sales"],
    )
    * 100
)

bar_type_labels = dict(BAR_TYPE_LABELS)

if float(labor_df["other_salary_labor"].sum()) > 0:
    bar_type_labels["other_salary_labor"] = "Other salary"

labor_df["sales_per_labor_hour"] = safe_ratio(
    labor_df["net_sales"],
    labor_df["labor_hours"],
)


# ---------------------------------------------------------
# Coverage: labor data usually lags the sales feed
# ---------------------------------------------------------

labor_day_count = len(labor_df)
sales_day_count = len(location_sales_df)
days_without_labor = sales_day_count - labor_day_count

earliest_labor_date = labor_df["business_date"].min()
latest_labor_date = labor_df["business_date"].max()

coverage_message = (
    f"Labor data covers {labor_day_count:,} day(s) from "
    f"{earliest_labor_date:%b %d, %Y} through "
    f"{latest_labor_date:%b %d, %Y}."
)

if days_without_labor > 0:
    st.info(
        f"{coverage_message} The sales table has "
        f"{days_without_labor:,} additional day(s) with no "
        "matching shift records; those days are excluded so "
        "they do not show as 0% labor."
    )
else:
    st.caption(coverage_message)

days_without_salary = int(
    (labor_df["salary_wages"] == 0).sum()
)

if days_without_salary:
    st.caption(
        f"{days_without_salary:,} day(s) have no salaried "
        "pay recorded and count hourly wages only."
    )


# ---------------------------------------------------------
# Selected period vs the period before it
# ---------------------------------------------------------

st.subheader(interval["summary_title"])


def period_coverage(row) -> str:
    """Date range and how many days of that bucket reported."""

    return (
        f"{row['period_range']} "
        f"({int(row['days_counted'])} of {period_days} days)"
    )


def period_delta(current_value, previous_row, column_name):
    """Signed percent change against the previous period."""

    if previous_row is None:
        return None

    change = percent_change(
        current_value,
        previous_row[column_name],
    )

    if pd.isna(change):
        return None

    return f"{change:+.2f}%"


if date_end is None:
    st.warning("Pick a date end.")
    summary_current = None
    summary_previous = None
else:
    summary_df = build_labor_period_frame(
        labor_df,
        date_end,
        period_days,
        2,
    )
    summary_current = summary_df.loc[
        summary_df["period_index"] == 0
    ].iloc[0]
    previous_rows = summary_df.loc[
        summary_df["period_index"] == 1
    ]
    summary_previous = (
        previous_rows.iloc[0]
        if not previous_rows.empty
        else None
    )

if summary_current is not None:
    current_coverage = period_coverage(summary_current)
    labor_percent_change = NOT_AVAILABLE

    if (
        summary_previous is not None
        and not pd.isna(summary_current["labor_percent"])
        and not pd.isna(summary_previous["labor_percent"])
    ):
        labor_percent_change = (
            summary_current["labor_percent"]
            - summary_previous["labor_percent"]
        )

    summary_columns = st.columns(4)

    summary_columns[0].metric(
        "Labor % of Net Sales",
        format_metric_value(
            summary_current["labor_percent"],
            "percent",
        ),
        delta=(
            f"{labor_percent_change:+.2f} pts"
            if not pd.isna(labor_percent_change)
            else None
        ),
        # A falling labor rate is a good outcome.
        delta_color="inverse",
        help=current_coverage,
    )

    summary_columns[1].metric(
        "Total Labor",
        format_whole_dollars(summary_current["total_labor"]),
        delta=period_delta(
            summary_current["total_labor"],
            summary_previous,
            "total_labor",
        ),
        delta_color="inverse",
        help=current_coverage,
    )

    summary_columns[2].metric(
        "Net Sales",
        format_whole_dollars(summary_current["net_sales"]),
        delta=period_delta(
            summary_current["net_sales"],
            summary_previous,
            "net_sales",
        ),
        help=current_coverage,
    )

    summary_columns[3].metric(
        "SpLH",
        format_whole_dollars(
            summary_current["sales_per_labor_hour"]
        ),
        delta=period_delta(
            summary_current["sales_per_labor_hour"],
            summary_previous,
            "sales_per_labor_hour",
        ),
        help=(
            f"{current_coverage}. Net sales divided by "
            "hourly shift hours. GM, DM, and other "
            "salaried roles are excluded."
        ),
    )

    previous_labor_percent = (
        summary_previous["labor_percent"]
        if summary_previous is not None
        else NOT_AVAILABLE
    )
    previous_label = format_metric_value(
        previous_labor_percent,
        "percent",
    )
    hourly_label = format_whole_dollars(
        summary_current["hourly_labor"]
    )
    gm_label = format_whole_dollars(summary_current["gm_labor"])
    dm_label = format_whole_dollars(summary_current["dm_labor"])
    current_hourly_percent = format_metric_value(
        percent_of_sales(
            summary_current["hourly_labor"],
            summary_current["net_sales"],
        ),
        "percent",
    )
    current_gm_percent = format_metric_value(
        percent_of_sales(
            summary_current["gm_labor"],
            summary_current["net_sales"],
        ),
        "percent",
    )
    current_dm_percent = format_metric_value(
        percent_of_sales(
            summary_current["dm_labor"],
            summary_current["net_sales"],
        ),
        "percent",
    )

    st.caption(
        escape_dollar_signs(
            f"Previous period was {previous_label}. Target is "
            f"{target_labor_percent:.1f}% "
            f"({hourly_target_percent:.1f}% hourly + "
            f"{gm_target_percent:.1f}% GM + "
            f"{dm_target_percent:.1f}% DM). Hourly pay was "
            f"{hourly_label} ({current_hourly_percent} of net "
            f"sales), GM salary was {gm_label} "
            f"({current_gm_percent}), and DM salary was "
            f"{dm_label} ({current_dm_percent})."
        )
    )

    if int(summary_current["days_counted"]) == 0:
        st.info(
            f"No labor falls in {summary_current['period_range']}."
        )
    elif not bool(summary_current["is_complete"]):
        st.caption(
            f"The period ending "
            f"{summary_current['period_end']:%b %d, %Y} has "
            f"{int(summary_current['days_counted'])} of "
            f"{period_days} day(s) of data, so its total is short."
        )

    if (
        summary_previous is None
        or int(summary_previous["days_counted"]) == 0
    ):
        st.info(
            "The previous period has no labor, so there is "
            "nothing to compare against yet."
        )


# ---------------------------------------------------------
# Labor dollars and labor rate
# ---------------------------------------------------------

st.subheader("Labor Dollars and Labor Rate")

if date_end is None:
    st.warning("Pick a date end.")
    rate_df = pd.DataFrame()
else:
    rate_df = build_labor_period_frame(
        labor_df,
        date_end,
        period_days,
        duration,
    )

if date_end is not None and not rate_df.empty:
    window_start = pd.Timestamp(rate_df["period_start"].min())
    window_end = pd.Timestamp(date_end)

    if window_start.normalize() == window_end.normalize():
        window_text = f"{window_end:%b %d, %Y}"
    else:
        window_text = (
            f"{window_start:%b %d, %Y} – {window_end:%b %d, %Y}"
        )

    st.caption(
        f"{window_text} · {duration} {interval['unit']} · "
        f"{selected_location}"
    )

    location_max = pd.Timestamp(
        labor_df["business_date"].max()
    )

    if pd.Timestamp(date_end) > location_max:
        st.caption(
            f"The latest labor date for {selected_location} is "
            f"{location_max:%b %d, %Y}."
        )

rate_has_values = (
    date_end is not None
    and not rate_df.empty
    and (
        rate_df["net_sales"].notna().any()
        or rate_df["total_labor"].notna().any()
    )
)

if not rate_has_values:
    if date_end is not None:
        st.info(
            "No labor falls in the selected date range."
        )
else:
    period_label_order = rate_df["period_label"].tolist()
    label_angle = -40 if duration > 6 else 0
    period_axis = alt.X(
        "period_label:N",
        title=interval["x_title"],
        sort=period_label_order,
        axis=alt.Axis(
            labelAngle=label_angle,
            labelOverlap="greedy",
        ),
    )

    weekly_labor_max = float(
        rate_df["total_labor"].max()
    )

    if pd.isna(weekly_labor_max) or weekly_labor_max <= 0:
        labor_axis_max = 1.0
    else:
        labor_axis_max = weekly_labor_max * 1.05

    percent_axis_min, percent_axis_max = percent_axis_limits(
        rate_df,
        ["labor_percent"],
        [float(target_labor_percent)],
    )

    target_df = pd.DataFrame(
        {
            "rate_series": ["Total"],
            "target": [float(target_labor_percent)],
        }
    )

    weekly_stacked_df, weekly_type_order = rank_stack(
        stack_labor_types(
            rate_df,
            [
                "period_label",
                "period_range",
                "is_complete",
                "total_labor",
                "net_sales",
                "days_counted",
            ],
            bar_type_labels,
        ),
        "labor_type",
        "labor_dollars",
    )

    labor_dollar_bars = (
        alt.Chart(weekly_stacked_df)
        .mark_bar()
        .encode(
            x=period_axis,
            y=alt.Y(
                "labor_dollars:Q",
                title="Total Labor",
                stack="zero",
                axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(domain=[0, labor_axis_max]),
            ),
            order=alt.Order("stack_rank:Q", sort="ascending"),
            color=alt.Color(
                "labor_type:N",
                title="Labor Type",
                scale=color_scale(weekly_type_order),
                sort=weekly_type_order,
            ),
            opacity=alt.condition(
                alt.datum.is_complete,
                alt.value(0.85),
                alt.value(0.4),
            ),
            tooltip=[
                alt.Tooltip(
                    "period_range:N",
                    title="Period",
                ),
                alt.Tooltip("labor_type:N", title="Labor Type"),
                alt.Tooltip(
                    "labor_dollars:Q",
                    title="Segment",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "total_labor:Q",
                    title="Total Labor",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net Sales",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "days_counted:Q",
                    title="Days Counted",
                    format=".0f",
                ),
            ],
        )
    )

    weekly_rates_df = melt_labor_rates(
        rate_df,
        ["period_label", "period_range"],
        {"labor_percent": "Total"},
    )

    total_rate_scale = color_scale(["Total"])

    percent_scale = alt.Scale(
        domain=[percent_axis_min, percent_axis_max],
        zero=False,
    )

    labor_rate_lines = (
        alt.Chart(weekly_rates_df)
        .mark_line(
            strokeWidth=2.5,
            point=alt.OverlayMarkDef(size=55),
        )
        .encode(
            x=period_axis,
            y=alt.Y(
                "rate_percent:Q",
                title="Labor % of Net Sales",
                axis=alt.Axis(
                    orient="right",
                    format=".1f",
                    labelExpr="datum.value + '%'",
                ),
                scale=percent_scale,
            ),
            color=alt.Color(
                "rate_series:N",
                title="Labor %",
                scale=total_rate_scale,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "period_range:N",
                    title="Period",
                ),
                alt.Tooltip(
                    "rate_series:N",
                    title="Rate",
                ),
                alt.Tooltip(
                    "rate_percent:Q",
                    title="Labor %",
                    format=".2f",
                ),
            ],
        )
    )

    net_sales_values = rate_df["net_sales"].dropna()

    if net_sales_values.empty:
        net_sales_axis_min = 0.0
        net_sales_axis_max = 1.0
    else:
        lowest_net_sales = float(net_sales_values.min())
        highest_net_sales = float(net_sales_values.max())
        net_sales_axis_min = lowest_net_sales * 0.75
        net_sales_axis_max = highest_net_sales * 1.25
        if net_sales_axis_max <= net_sales_axis_min:
            net_sales_axis_max = net_sales_axis_min + 1.0

    net_sales_scale = alt.Scale(
        domain=[net_sales_axis_min, net_sales_axis_max],
        zero=False,
        nice=False,
    )

    net_sales_line = (
        alt.Chart(rate_df)
        .mark_line(
            color="#17becf",
            strokeWidth=2.5,
            point=alt.OverlayMarkDef(
                size=55,
                color="#17becf",
            ),
        )
        .encode(
            x=period_axis,
            y=alt.Y(
                "net_sales:Q",
                title="Net Sales",
                axis=alt.Axis(
                    orient="right",
                    offset=90,
                    format="$,.0f",
                    titleColor="#17becf",
                    labelColor="#17becf",
                    grid=False,
                ),
                scale=net_sales_scale,
            ),
            tooltip=[
                alt.Tooltip(
                    "period_range:N",
                    title="Period",
                ),
                alt.Tooltip(
                    "net_sales:Q",
                    title="Net Sales",
                    format="$,.0f",
                ),
            ],
        )
    )

    target_rules = (
        alt.Chart(target_df)
        .mark_rule(
            strokeWidth=2,
            strokeDash=[8, 5],
        )
        .encode(
            y=alt.Y(
                "target:Q",
                axis=None,
                scale=percent_scale,
            ),
            color=alt.Color(
                "rate_series:N",
                scale=total_rate_scale,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "rate_series:N",
                    title="Target",
                ),
                alt.Tooltip(
                    "target:Q",
                    title="Target %",
                    format=".1f",
                ),
            ],
        )
    )

    weekly_labor_chart = (
        alt.layer(
            labor_dollar_bars,
            net_sales_line,
            labor_rate_lines,
            target_rules,
        )
        .resolve_scale(
            y="independent",
            color="independent",
        )
        .properties(height=440)
    )

    st.altair_chart(
        weekly_labor_chart,
        use_container_width=True,
    )

    st.caption(
        "Bars show hourly, GM, and DM labor dollars, with the "
        "largest segment at the bottom. The line is total labor "
        "as a percent of net sales, and the dashed line is the "
        "total labor target. The outer right axis, in teal, is "
        "net sales. That axis runs from 75% of the lowest period "
        "to 125% of the highest period shown."
    )


# ---------------------------------------------------------
# Hourly labor dollars by role
# ---------------------------------------------------------

st.subheader("Hourly Labor Dollars by role")

role_frames = []
role_window_ready = date_end is not None and not rate_df.empty

if not role_window_ready:
    st.info(
        "Pick a date end to see hourly labor dollars by role."
    )
else:
    if not shift_df.empty:
        hourly_roles_df = shift_df.copy()

        if selected_location != "All Locations":
            hourly_roles_df = hourly_roles_df[
                hourly_roles_df["location"] == selected_location
            ]

        hourly_roles_df = rows_in_periods(
            hourly_roles_df,
            date_end,
            period_days,
            duration,
        )

        if not hourly_roles_df.empty:
            # Salaried roles are paid from the salary feed.
            # Shift rows for those roles are left out here.
            salaried_roles = salaried_role_keys(salary_df)
            shift_role_key = (
                hourly_roles_df["role"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )
            hourly_roles_df = hourly_roles_df.loc[
                ~shift_role_key.isin(salaried_roles)
            ].copy()

        if not hourly_roles_df.empty:
            hourly_roles_df["role"] = (
                hourly_roles_df["role"]
                .fillna("Unassigned")
                .astype(str)
                .str.strip()
            )
            role_frames.append(
                hourly_roles_df
                .groupby(["period_index", "role"], as_index=False)
                .agg(labor_dollars=("shift_wages", "sum"))
            )

    if not role_frames:
        st.info(
            "No hourly labor records by role fall inside "
            "the selected date range."
        )
    else:
        role_df = (
            pd.concat(role_frames, ignore_index=True)
            .groupby(["period_index", "role"], as_index=False)
            .agg(labor_dollars=("labor_dollars", "sum"))
        )
        role_df = role_df[role_df["labor_dollars"] > 0].copy()

        if role_df.empty:
            st.info(
                "No hourly labor records by role fall inside "
                "the selected date range."
            )
        else:
            period_labels = rate_df[
                [
                    "period_index",
                    "period_label",
                    "period_range",
                    "is_complete",
                ]
            ]
            role_df = role_df.merge(
                period_labels,
                on="period_index",
                how="inner",
            )
            role_df, role_order = rank_stack(
                role_df,
                "role",
                "labor_dollars",
            )
            # Stack each period by that period's labor dollars.
            # A role's total across the selection can rank it
            # differently from one period.
            role_df = role_df.sort_values(
                ["period_index", "labor_dollars", "role"],
                ascending=[True, False, True],
            )
            role_df["labor_top"] = (
                role_df
                .groupby("period_index", sort=False)["labor_dollars"]
                .cumsum()
            )
            role_df["labor_base"] = (
                role_df["labor_top"] - role_df["labor_dollars"]
            )
            period_label_order = rate_df["period_label"].tolist()
            label_angle = -40 if duration > 6 else 0

            role_chart = (
                alt.Chart(role_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "period_label:N",
                        title=interval["x_title"],
                        sort=period_label_order,
                        scale=alt.Scale(domain=period_label_order),
                        axis=alt.Axis(
                            labelAngle=label_angle,
                            labelOverlap="greedy",
                        ),
                    ),
                    y=alt.Y(
                        "labor_base:Q",
                        title="Hourly Labor Dollars",
                        stack=None,
                        axis=alt.Axis(format="$,.0f"),
                    ),
                    y2=alt.Y2("labor_top:Q"),
                    color=alt.Color(
                        "role:N",
                        title="Role",
                        scale=alt.Scale(scheme="tableau10"),
                        sort=role_order,
                    ),
                    opacity=alt.condition(
                        alt.datum.is_complete,
                        alt.value(0.85),
                        alt.value(0.4),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "period_range:N",
                            title="Period",
                        ),
                        alt.Tooltip("role:N", title="Role"),
                        alt.Tooltip(
                            "labor_dollars:Q",
                            title="Hourly Labor Dollars",
                            format="$,.0f",
                        ),
                    ],
                )
                .properties(height=400)
            )

            st.altair_chart(
                role_chart,
                use_container_width=True,
            )

            st.caption(
                "Each bar uses the same periods as the chart "
                "above. Bars are hourly shift wages by role, "
                "and salaried roles are left out. Totals cover "
                "every day with an hourly labor record in that "
                "period, including days the sales feed has not "
                "reported, so they can exceed the hourly labor "
                "totals above. Within each bar, roles are "
                "ordered by that period's labor dollars, largest "
                "at the bottom."
            )


# ---------------------------------------------------------
# Total overtime labor dollars and hours
# ---------------------------------------------------------

st.subheader("Total OT Labor $ per Period")

if date_end is None or rate_df.empty:
    st.info("Pick a date end to see overtime by period.")
else:
    labor_period_days = rows_in_periods(
        labor_df,
        date_end,
        period_days,
        duration,
    )
    counted_ot_days = (
        labor_period_days[
            ["business_date", "location_key", "period_index"]
        ]
        .drop_duplicates()
    )

    ot_role_df = pd.DataFrame(
        columns=[
            "period_index",
            "role",
            "ot_labor",
            "ot_hours",
        ]
    )

    if not shift_df.empty and not counted_ot_days.empty:
        ot_role_source = shift_df.copy()

        if selected_location != "All Locations":
            ot_role_source = ot_role_source[
                ot_role_source["location"] == selected_location
            ]

        ot_role_source = rows_in_periods(
            ot_role_source,
            date_end,
            period_days,
            duration,
        )
        ot_role_source = ot_role_source.merge(
            counted_ot_days,
            on=["business_date", "location_key", "period_index"],
            how="inner",
        )

        if not ot_role_source.empty:
            # Salaried roles are paid from the salary feed
            # and are left out of overtime dollars and hours.
            salaried_roles = salaried_role_keys(salary_df)
            ot_role_key = (
                ot_role_source["role"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )
            ot_role_source = ot_role_source.loc[
                ~ot_role_key.isin(salaried_roles)
            ].copy()

        if not ot_role_source.empty:
            ot_role_df = (
                ot_role_source
                .groupby(
                    ["period_index", "role"],
                    as_index=False,
                )
                .agg(
                    ot_labor=("ot_wages", "sum"),
                    ot_hours=("ot_hours", "sum"),
                )
            )
            ot_role_df = ot_role_df[
                (ot_role_df["ot_labor"] > 0)
                | (ot_role_df["ot_hours"] > 0)
            ].copy()

    period_labels = rate_df[
        [
            "period_index",
            "period_label",
            "period_range",
            "is_complete",
            "total_labor",
        ]
    ].copy()

    if not ot_role_df.empty:
        ot_role_df = ot_role_df.merge(
            period_labels[
                [
                    "period_index",
                    "period_label",
                    "period_range",
                    "is_complete",
                ]
            ],
            on="period_index",
            how="inner",
        )
        ot_role_df, ot_role_order = rank_stack(
            ot_role_df,
            "role",
            "ot_labor",
        )
        # Stack each period by that period's overtime pay.
        # A role's total across the selection can rank it
        # differently from one week, such as 9/14–9/20.
        ot_role_df = ot_role_df.sort_values(
            ["period_index", "ot_labor", "role"],
            ascending=[True, False, True],
        )
        ot_role_df["ot_labor_top"] = (
            ot_role_df
            .groupby("period_index", sort=False)["ot_labor"]
            .cumsum()
        )
        ot_role_df["ot_labor_base"] = (
            ot_role_df["ot_labor_top"] - ot_role_df["ot_labor"]
        )
    else:
        ot_role_order = []

    # The line uses the same hourly roles as the bars, so
    # salaried overtime is left out of both dollars and hours.
    if not ot_role_df.empty:
        ot_by_period = (
            ot_role_df
            .groupby("period_index", as_index=False)
            .agg(
                ot_labor=("ot_labor", "sum"),
                ot_hours=("ot_hours", "sum"),
            )
        )
    else:
        ot_by_period = pd.DataFrame(
            columns=["period_index", "ot_labor", "ot_hours"]
        )

    ot_percent_df = period_labels[
        [
            "period_index",
            "period_label",
            "period_range",
            "total_labor",
        ]
    ].merge(
        ot_by_period,
        on="period_index",
        how="left",
    )
    reported_period = ot_percent_df["total_labor"].notna()
    ot_percent_df.loc[reported_period, "ot_labor"] = (
        ot_percent_df.loc[reported_period, "ot_labor"].fillna(0.0)
    )
    ot_percent_df.loc[reported_period, "ot_hours"] = (
        ot_percent_df.loc[reported_period, "ot_hours"].fillna(0.0)
    )
    ot_percent_df["ot_labor_percent"] = (
        safe_ratio(
            ot_percent_df["ot_labor"],
            ot_percent_df["total_labor"],
        )
        * 100
    )

    stacked_ot_peak = (
        ot_role_df.groupby("period_label")["ot_labor"].sum().max()
        if not ot_role_df.empty
        else 0.0
    )
    period_ot_peak = ot_percent_df["ot_labor"].max()
    ot_labor_candidates = [
        float(value)
        for value in [stacked_ot_peak, period_ot_peak]
        if pd.notna(value)
    ]
    ot_labor_peak = max(ot_labor_candidates, default=0.0)

    if ot_labor_peak <= 0:
        ot_labor_axis_max = 1.0
    else:
        ot_labor_axis_max = ot_labor_peak * 1.08

    ot_labor_scale = alt.Scale(domain=[0, ot_labor_axis_max])
    period_label_order = rate_df["period_label"].tolist()
    label_angle = -40 if duration > 6 else 0
    ot_period_axis = alt.X(
        "period_label:N",
        title=interval["x_title"],
        sort=period_label_order,
        scale=alt.Scale(domain=period_label_order),
        axis=alt.Axis(
            labelAngle=label_angle,
            labelOverlap="greedy",
        ),
    )

    ot_chart_layers = []

    if not ot_role_df.empty:
        ot_labor_bars = (
            alt.Chart(ot_role_df)
            .mark_bar()
            .encode(
                x=ot_period_axis,
                y=alt.Y(
                    "ot_labor_base:Q",
                    title="Total OT Labor $",
                    stack=None,
                    axis=alt.Axis(format="$,.0f"),
                    scale=ot_labor_scale,
                ),
                y2=alt.Y2("ot_labor_top:Q"),
                color=alt.Color(
                    "role:N",
                    title="Role",
                    scale=alt.Scale(scheme="tableau10"),
                    sort=ot_role_order,
                ),
                opacity=alt.condition(
                    alt.datum.is_complete,
                    alt.value(0.85),
                    alt.value(0.4),
                ),
                tooltip=[
                    alt.Tooltip(
                        "period_range:N",
                        title="Period",
                    ),
                    alt.Tooltip("role:N", title="Role"),
                    alt.Tooltip(
                        "ot_hours:Q",
                        title="OT Hours",
                        format=",.1f",
                    ),
                    alt.Tooltip(
                        "ot_labor:Q",
                        title="OT Labor $",
                        format="$,.0f",
                    ),
                ],
            )
        )
        ot_chart_layers.append(ot_labor_bars)

    ot_percent_peak = ot_percent_df["ot_labor_percent"].max()

    if pd.isna(ot_percent_peak) or ot_percent_peak <= 0:
        ot_percent_axis_max = 1.0
    else:
        ot_percent_axis_max = float(ot_percent_peak) * 1.08

    ot_percent_scale = alt.Scale(
        domain=[0, ot_percent_axis_max],
    )

    ot_percent_line = (
        alt.Chart(ot_percent_df)
        .mark_line(
            color="#6a3d9a",
            strokeWidth=2.5,
            point=alt.OverlayMarkDef(
                size=55,
                color="#6a3d9a",
            ),
        )
        .encode(
            x=ot_period_axis,
            y=alt.Y(
                "ot_labor_percent:Q",
                title="$ OT % of Total Labor Dollars",
                axis=alt.Axis(
                    orient="right",
                    format=".1f",
                    labelExpr="datum.value + '%'",
                    titleColor="#6a3d9a",
                    labelColor="#6a3d9a",
                    grid=False,
                ),
                scale=ot_percent_scale,
            ),
            tooltip=[
                alt.Tooltip(
                    "period_range:N",
                    title="Period",
                ),
                alt.Tooltip(
                    "ot_labor_percent:Q",
                    title="$ OT % of Total Labor Dollars",
                    format=".1f",
                ),
                alt.Tooltip(
                    "ot_hours:Q",
                    title="OT Hours",
                    format=",.1f",
                ),
                alt.Tooltip(
                    "ot_labor:Q",
                    title="OT Labor $",
                    format="$,.0f",
                ),
                alt.Tooltip(
                    "total_labor:Q",
                    title="Total Labor $",
                    format="$,.0f",
                ),
            ],
        )
    )

    ot_chart_layers.append(ot_percent_line)

    ot_labor_chart = (
        alt.layer(*ot_chart_layers)
        .resolve_scale(y="independent")
        .properties(height=400)
    )

    st.altair_chart(ot_labor_chart, use_container_width=True)

    st.caption(
        "Each bar uses the same periods as the charts above. "
        "Bars are overtime pay on the left axis, split by "
        "role. Within each bar, that period's largest "
        "overtime amount sits at the bottom. Hover a segment "
        "for that role's overtime hours and overtime pay. "
        "The right axis is overtime pay "
        "as a percent of total labor. Overtime dollars and "
        "hours leave out salaried roles."
    )


# ---------------------------------------------------------
# Labor breakdown for the selected interval
# ---------------------------------------------------------

with st.expander(
    f"View {selected_interval.lower()} labor breakdown"
):
    if date_end is None:
        st.caption("Pick a date end to see the breakdown.")
    else:
        # One extra period supplies Prior Labor % for the
        # oldest row, then drops out of the table.
        breakdown_df = build_labor_period_frame(
            labor_df,
            date_end,
            period_days,
            duration + 1,
        )
        employee_counts = count_employees_by_period(
            shift_df,
            labor_df,
            date_end,
            period_days,
            duration,
        )
        breakdown_df = breakdown_df.merge(
            employee_counts,
            on="period_index",
            how="left",
        )
        breakdown_df = breakdown_df.loc[
            breakdown_df["period_index"] < duration
        ].copy()

        missing_period = breakdown_df["days_counted"].eq(0)
        reported_period = ~missing_period
        breakdown_df.loc[missing_period, "employee_count"] = (
            pd.NA
        )
        breakdown_df.loc[reported_period, "employee_count"] = (
            breakdown_df.loc[
                reported_period,
                "employee_count",
            ].fillna(0)
        )
        breakdown_df["employee_count"] = (
            breakdown_df["employee_count"].astype("Int64")
        )
        breakdown_df["labor_vs_target"] = (
            breakdown_df["labor_percent"]
            - float(target_labor_percent)
        )

        show_other_salary = (
            "other_salary_labor" in breakdown_df.columns
            and breakdown_df["other_salary_labor"]
            .fillna(0)
            .sum()
            > 0
        )

        display_columns = [
            "period_start",
            "period_end",
            "days_counted",
            "net_sales",
            "hourly_labor",
            "gm_labor",
            "dm_labor",
        ]

        if show_other_salary:
            display_columns.append("other_salary_labor")

        display_columns.extend(
            [
                "total_labor",
                "ot_labor",
                "labor_percent",
                "previous_labor_percent",
                "labor_percent_change",
                "labor_vs_target",
                "hourly_percent",
                "gm_percent",
                "dm_percent",
            ]
        )

        if show_other_salary:
            display_columns.append("other_percent")

        display_columns.extend(
            [
                "ot_labor_percent",
                "labor_hours",
                "ot_hours",
                "sales_per_labor_hour",
                "employee_count",
            ]
        )
        display_columns = [
            column_name
            for column_name in display_columns
            if column_name in breakdown_df.columns
        ]

        table_df = (
            breakdown_df[display_columns]
            .sort_values("period_end", ascending=False)
            .copy()
        )
        table_df["period_start"] = table_df[
            "period_start"
        ].dt.date
        table_df["period_end"] = table_df["period_end"].dt.date

        st.caption(
            f"{duration} {interval['unit']} ending "
            f"{pd.Timestamp(date_end):%b %d, %Y}, "
            "newest first. Dollars are summed, and each "
            "rate uses those totals. Employees is the "
            "number of distinct people with a shift on a "
            "day that has both sales and labor."
        )

        column_config = {
            "period_start": st.column_config.DateColumn(
                "Period start",
            ),
            "period_end": st.column_config.DateColumn(
                "Period end",
            ),
            "days_counted": st.column_config.NumberColumn(
                "Days",
                format="%d",
                help=(
                    f"Days in this period with both sales "
                    f"and labor. A full period is "
                    f"{period_days}."
                ),
            ),
            "net_sales": st.column_config.NumberColumn(
                "Net Sales",
                format="$%,.0f",
            ),
            "hourly_labor": st.column_config.NumberColumn(
                "Hourly",
                format="$%,.0f",
            ),
            "gm_labor": st.column_config.NumberColumn(
                "GM",
                format="$%,.0f",
            ),
            "dm_labor": st.column_config.NumberColumn(
                "DM",
                format="$%,.0f",
            ),
            "other_salary_labor": st.column_config.NumberColumn(
                "Other salary",
                format="$%,.0f",
            ),
            "total_labor": st.column_config.NumberColumn(
                "Total Labor",
                format="$%,.0f",
            ),
            "ot_labor": st.column_config.NumberColumn(
                "OT Labor",
                format="$%,.0f",
                help="Overtime pay, including the GM role.",
            ),
            "labor_percent": st.column_config.NumberColumn(
                "Labor %",
                format="%.2f%%",
            ),
            "previous_labor_percent": (
                st.column_config.NumberColumn(
                    "Prior Labor %",
                    format="%.2f%%",
                )
            ),
            "labor_percent_change": (
                st.column_config.NumberColumn(
                    "Change (pts)",
                    format="%.2f",
                    help=(
                        "Labor % minus the previous "
                        "period's labor %, in points."
                    ),
                )
            ),
            "labor_vs_target": st.column_config.NumberColumn(
                "vs Target (pts)",
                format="%.2f",
                help=(
                    "Labor % minus the target in the "
                    "sidebar. Positive is above target."
                ),
            ),
            "hourly_percent": st.column_config.NumberColumn(
                "Hourly %",
                format="%.2f%%",
            ),
            "gm_percent": st.column_config.NumberColumn(
                "GM %",
                format="%.2f%%",
            ),
            "dm_percent": st.column_config.NumberColumn(
                "DM %",
                format="%.2f%%",
            ),
            "other_percent": st.column_config.NumberColumn(
                "Other %",
                format="%.2f%%",
            ),
            "ot_labor_percent": st.column_config.NumberColumn(
                "OT % of Labor",
                format="%.2f%%",
                help="Overtime pay as a percent of total labor.",
            ),
            "labor_hours": st.column_config.NumberColumn(
                "Hourly Hours",
                format="%.1f",
                help=(
                    "Hourly shift hours. GM, DM, and other "
                    "salaried roles are excluded."
                ),
            ),
            "ot_hours": st.column_config.NumberColumn(
                "OT Hours",
                format="%.1f",
                help=(
                    "Overtime hours from the shift feed, "
                    "including the GM role."
                ),
            ),
            "sales_per_labor_hour": (
                st.column_config.NumberColumn(
                    "Sales / Hour",
                    format="$%,.0f",
                    help=(
                        "Net sales divided by hourly shift "
                        "hours."
                    ),
                )
            ),
            "employee_count": st.column_config.NumberColumn(
                "Employees",
                format="%d",
            ),
        }

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                column_name: column_config[column_name]
                for column_name in display_columns
            },
        )


if date_end is None:
    st.caption(
        f"Labor for {selected_location.lower()}."
    )
else:
    st.caption(
        f"Displaying {duration:,} {interval['unit']} of labor "
        f"for {selected_location.lower()}, ending "
        f"{pd.Timestamp(date_end):%b %d, %Y}."
    )
