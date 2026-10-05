"""Shared Supabase access and week-bucketing helpers.

Every page imports from here so that table names, data
cleaning, and week definitions stay consistent.
"""

import pandas as pd
import streamlit as st
from supabase import Client, create_client


# ---------------------------------------------------------
# Supabase tables. Update these if a table is renamed.
# ---------------------------------------------------------

SALES_TABLE = "daily-sales-summary-all-revenue-operations"
SHIFT_TABLE = "daily_employee_shift_timeworked_summary"
SALARY_TABLE = "daily_salary"

# PostgREST returns at most 1,000 rows per request.
PAGE_SIZE = 1000

# Numeric columns retrieved from the sales table.
SALES_NUMERIC_COLUMNS = [
    "net_sales",
    "gross_sales",
    "total_orders",
    "total_discounts",
    "to_go",
    "eat_in",
    "drive_through",
    "pickup",
    "dd_marketplace",
    "uber_eats",
]

SHIFT_NUMERIC_COLUMNS = [
    "hours",
    "regular_hours",
    "ot_hours",
    "regular_wages",
    "ot_wages",
    "shift_wages",
]

# pandas uses Monday=0 through Sunday=6 for weekday numbers.
WEEK_START_OPTIONS = {
    "Monday": 0,
    "Sunday": 6,
}

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

NOT_AVAILABLE = float("nan")


class MissingSecretError(RuntimeError):
    """Raised when Supabase credentials are not configured."""


@st.cache_resource
def get_supabase_client() -> Client:
    """Create and reuse one Supabase client."""

    # Streamlit raises different errors depending on whether
    # the secrets file is missing or just incomplete, and the
    # fix is the same either way.
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
    except Exception as error:
        raise MissingSecretError(
            "Add SUPABASE_URL and SUPABASE_KEY in your "
            "Streamlit app settings."
        ) from error

    return create_client(supabase_url, supabase_key)


def fetch_table(
    table_name: str,
    columns: list[str],
    order_column: str,
) -> pd.DataFrame:
    """Read an entire table, one page of rows at a time."""

    supabase = get_supabase_client()
    selected_columns = ", ".join(columns)

    records: list[dict] = []
    page_start = 0

    while True:
        response = (
            supabase
            .table(table_name)
            .select(selected_columns)
            .order(order_column)
            .range(page_start, page_start + PAGE_SIZE - 1)
            .execute()
        )

        page_records = response.data or []
        records.extend(page_records)

        if len(page_records) < PAGE_SIZE:
            break

        page_start += PAGE_SIZE

    return pd.DataFrame(records)


def _coerce_numeric(
    dataframe: pd.DataFrame,
    column_names: list[str],
) -> pd.DataFrame:
    """Force columns to numbers, adding any that are absent."""

    for column_name in column_names:
        if column_name not in dataframe.columns:
            dataframe[column_name] = 0.0

        dataframe[column_name] = pd.to_numeric(
            dataframe[column_name],
            errors="coerce",
        )

    return dataframe


def _add_location_key(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a case-insensitive key for joining locations.

    The sales and shift tables report "Leander" while the
    salary table reports "LEANDER", so joins need a key that
    ignores casing and stray whitespace.
    """

    dataframe["location"] = (
        dataframe["location"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    dataframe["location_key"] = (
        dataframe["location"].str.upper()
    )

    return dataframe


@st.cache_data(ttl=300)
def load_sales_data() -> pd.DataFrame:
    """Retrieve daily sales records from Supabase."""

    dataframe = fetch_table(
        SALES_TABLE,
        ["id", "location", "business_date", *SALES_NUMERIC_COLUMNS],
        "business_date",
    )

    if dataframe.empty:
        return dataframe

    dataframe["business_date"] = pd.to_datetime(
        dataframe["business_date"],
        errors="coerce",
    )

    dataframe = _coerce_numeric(
        dataframe,
        SALES_NUMERIC_COLUMNS,
    )

    dataframe = _add_location_key(dataframe)

    dataframe = dataframe.dropna(
        subset=["business_date", "net_sales"],
    )

    # Channel and order columns are optional in the source
    # feed, so treat gaps as zero rather than dropping days.
    supporting_columns = [
        column_name
        for column_name in SALES_NUMERIC_COLUMNS
        if column_name != "net_sales"
    ]

    dataframe[supporting_columns] = (
        dataframe[supporting_columns].fillna(0.0)
    )

    return dataframe.sort_values("business_date")


@st.cache_data(ttl=300)
def load_shift_data() -> pd.DataFrame:
    """Retrieve hourly shift records from Supabase.

    One row is one worked shift. shift_wages is the
    per-shift total. regular_wages and ot_wages split
    that total and are safe to sum on their own.
    """

    dataframe = fetch_table(
        SHIFT_TABLE,
        [
            "record_key",
            "date",
            "location",
            "employee",
            "role",
            *SHIFT_NUMERIC_COLUMNS,
        ],
        "date",
    )

    if dataframe.empty:
        return dataframe

    dataframe["business_date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    ).dt.normalize()

    dataframe = _coerce_numeric(
        dataframe,
        SHIFT_NUMERIC_COLUMNS,
    )

    dataframe = _add_location_key(dataframe)

    dataframe["role"] = (
        dataframe["role"].fillna("Unassigned").astype(str)
    )

    dataframe = dataframe.dropna(
        subset=["business_date", "shift_wages"],
    )

    dataframe[SHIFT_NUMERIC_COLUMNS] = (
        dataframe[SHIFT_NUMERIC_COLUMNS].fillna(0.0)
    )

    return dataframe.sort_values("business_date")


# Clock columns seen on Revel time-worked extracts. The
# first pair that exists on the shift table is used.
CLOCK_COLUMN_PAIRS = [
    ("clock_in", "clock_out"),
    ("start_time", "end_time"),
    ("shift_start", "shift_end"),
    ("time_in", "time_out"),
    ("in_time", "out_time"),
]


@st.cache_data(ttl=300)
def get_shift_columns() -> list[str]:
    """Column names on the shift table, from a single row."""

    supabase = get_supabase_client()
    response = (
        supabase
        .table(SHIFT_TABLE)
        .select("*")
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return []
    return list(rows[0].keys())


def resolve_clock_columns(
    columns: list[str],
) -> tuple[str, str] | None:
    """Pick the clock-in and clock-out columns, if present."""

    lookup = {name.lower(): name for name in columns}
    for clock_in, clock_out in CLOCK_COLUMN_PAIRS:
        if clock_in in lookup and clock_out in lookup:
            return lookup[clock_in], lookup[clock_out]
    return None


@st.cache_data(ttl=300)
def load_shift_clocks() -> pd.DataFrame:
    """Clock-in and clock-out for every shift.

    Columns are record_key, clock_in, and clock_out.
    The frame is empty when the table has no clock columns.
    """

    empty = pd.DataFrame(
        columns=["record_key", "clock_in", "clock_out"]
    )
    columns = get_shift_columns()
    pair = resolve_clock_columns(columns)
    if pair is None:
        return empty

    clock_in, clock_out = pair
    clocks = fetch_table(
        SHIFT_TABLE,
        ["record_key", clock_in, clock_out],
        "date",
    )
    if clocks.empty:
        return empty

    return clocks.rename(
        columns={
            clock_in: "clock_in",
            clock_out: "clock_out",
        }
    )


@st.cache_data(ttl=300)
def load_salary_data() -> pd.DataFrame:
    """Retrieve daily salaried-labor records from Supabase.

    One row is one salaried role at one location on one day,
    already prorated into a daily amount.
    """

    dataframe = fetch_table(
        SALARY_TABLE,
        [
            "date",
            "location",
            "role",
            "daily_salary",
        ],
        "date",
    )

    if dataframe.empty:
        return dataframe

    # These timestamps arrive as midnight UTC, so read them
    # as UTC and drop the zone to avoid shifting the day.
    dataframe["business_date"] = (
        pd.to_datetime(
            dataframe["date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    dataframe = _coerce_numeric(dataframe, ["daily_salary"])

    dataframe = _add_location_key(dataframe)

    dataframe["role"] = (
        dataframe["role"].fillna("Salaried").astype(str)
    )

    dataframe = dataframe.dropna(
        subset=["business_date", "daily_salary"],
    )

    return dataframe.sort_values("business_date")


def add_week_columns(
    dataframe: pd.DataFrame,
    week_start_weekday: int,
    date_column: str = "business_date",
) -> pd.DataFrame:
    """Add day_index and week_start for weekly bucketing.

    day_index is the position of a date inside its own week,
    where 0 is the configured week start day.
    """

    dataframe = dataframe.copy()

    dataframe["day_index"] = (
        dataframe[date_column].dt.weekday - week_start_weekday
    ) % 7

    dataframe["week_start"] = (
        dataframe[date_column]
        - pd.to_timedelta(dataframe["day_index"], unit="D")
    )

    return dataframe


def reindex_weekly(
    dataframe: pd.DataFrame,
    week_starts: pd.Series | None = None,
) -> pd.DataFrame:
    """Place weekly rows on a continuous weekly calendar.

    A week with no records then stays blank instead of
    shifting a comparison onto an older week. Pass the
    untrimmed week_start values so that a week emptied by
    week-to-date trimming still appears as a gap.
    """

    if week_starts is None or week_starts.empty:
        week_starts = dataframe["week_start"]

    complete_week_range = pd.date_range(
        start=min(week_starts.min(), dataframe["week_start"].min()),
        end=max(week_starts.max(), dataframe["week_start"].max()),
        freq="7D",
    )

    return (
        dataframe
        .set_index("week_start")
        .reindex(complete_week_range)
        .rename_axis("week_start")
        .reset_index()
    )


def format_metric_value(value, value_format: str) -> str:
    """Render a metric value using its display format."""

    if value is None or pd.isna(value):
        return "—"

    if value_format == "currency":
        return f"${value:,.2f}"

    if value_format == "count":
        return f"{value:,.0f}"

    if value_format == "percent":
        return f"{value:,.2f}%"

    if value_format == "hours":
        return f"{value:,.1f}"

    return f"{value:,.2f}"


def escape_dollar_signs(text: str) -> str:
    """Keep Streamlit markdown from reading $...$ as LaTeX.

    Two dollar amounts in one caption would otherwise be
    rendered as a math expression and lose both signs.
    """

    return text.replace("$", r"\$")


def percent_change(current_value, previous_value):
    """Percent change, or NaN when it is undefined."""

    if current_value is None or previous_value is None:
        return NOT_AVAILABLE

    if pd.isna(current_value) or pd.isna(previous_value):
        return NOT_AVAILABLE

    if previous_value == 0:
        return NOT_AVAILABLE

    return (
        (current_value - previous_value) / abs(previous_value)
    ) * 100


def location_filter_controls(
    dataframe: pd.DataFrame,
    label: str = "Location",
) -> tuple[str, pd.DataFrame]:
    """Render a location picker and apply it to the data."""

    locations = sorted(
        dataframe["location"].dropna().unique().tolist()
    )

    selected_location = st.sidebar.selectbox(
        label,
        options=["All Locations", *locations],
    )

    if selected_location == "All Locations":
        return selected_location, dataframe.copy()

    return selected_location, dataframe[
        dataframe["location"] == selected_location
    ].copy()
