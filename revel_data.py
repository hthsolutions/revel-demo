"""Shared Supabase access and week-bucketing helpers.

Every page imports from here so that table names, data
cleaning, and week definitions stay consistent.
"""

import json

import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from supabase import Client, create_client


# ---------------------------------------------------------
# Supabase tables. Update these if a table is renamed.
# ---------------------------------------------------------

SALES_TABLE = "daily-sales-summary-all-revenue-operations"
SHIFT_TABLE = "daily_employee_shift_timeworked_summary"
SALARY_TABLE = "daily_salary"
HOURLY_SALES_TABLE = "revel_hourly_sales"
SHIFT_SUMMARY_TABLE = SHIFT_TABLE
COMBO_MIX_TABLE = "daily-product-mix-combomix"
HME_TABLE = "daily-hme-rcd-summary"
CASH_SUMMARY_TABLE = "daily-revel-cash-summary"
DISCOUNT_REASON_TABLE = "daily-revel-discount-reason"
WASTE_LOG_TABLE = "waste_log_history"

# HME departure timestamps are store-local. A visit before
# 6:00 AM belongs to the previous business date, matching
# the hourly sales extract.
STORE_TIMEZONE = ZoneInfo("America/Chicago")
BUSINESS_DAY_START_HOUR = 6

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
def get_table_columns(table_name: str) -> list[str]:
    """Column names on a table, from a single row."""

    supabase = get_supabase_client()
    response = (
        supabase
        .table(table_name)
        .select("*")
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return []
    return list(rows[0].keys())


@st.cache_data(ttl=300)
def get_shift_columns() -> list[str]:
    """Column names on the shift table, from a single row."""

    return get_table_columns(SHIFT_TABLE)


def _pick_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Return the first candidate present, ignoring case and spaces."""

    lookup = {
        name.strip().lower().replace(" ", "_"): name
        for name in columns
    }
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


def _resolve_clock_pair(columns: list[str]) -> tuple[str, str] | None:
    """Pick clock-in and clock-out columns, if the table has them."""

    lookup = {
        name.strip().lower().replace(" ", "_"): name
        for name in columns
    }
    for clock_in, clock_out in CLOCK_COLUMN_PAIRS:
        if clock_in in lookup and clock_out in lookup:
            return lookup[clock_in], lookup[clock_out]
    return None


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


HOURLY_SALES_COLUMNS = [
    "location",
    "business_date",
    "interval_label",
    "time",
    "transactions",
    "items",
    "sales",
    "extracted_at",
]

HOURLY_SALES_NUMERIC_COLUMNS = [
    "transactions",
    "items",
    "sales",
]


@st.cache_data(ttl=300)
def load_hourly_sales() -> pd.DataFrame:
    """Retrieve 15-minute sales intervals from Supabase.

    One row is one location and one interval. The extract
    labels a business day from 6:00 AM through 5:59 AM.
    """

    available = get_table_columns(HOURLY_SALES_TABLE)
    selected = [
        column_name
        for column_name in (
            _pick_column(available, [candidate])
            for candidate in HOURLY_SALES_COLUMNS
        )
        if column_name is not None
    ]
    order_column = _pick_column(
        available,
        ["business_date", "extracted_at", "id"],
    )

    if not selected or order_column is None:
        return pd.DataFrame(columns=HOURLY_SALES_COLUMNS)

    dataframe = fetch_table(
        HOURLY_SALES_TABLE,
        selected,
        order_column,
    )

    if dataframe.empty:
        return dataframe

    rename = {}
    for candidate in HOURLY_SALES_COLUMNS:
        actual = _pick_column(list(dataframe.columns), [candidate])
        if actual is not None and actual != candidate:
            rename[actual] = candidate
    if rename:
        dataframe = dataframe.rename(columns=rename)

    missing = [
        column_name
        for column_name in ("business_date", "location", "sales")
        if column_name not in dataframe.columns
    ]
    if missing:
        found = ", ".join(available) or "none"
        raise ValueError(
            "revel_hourly_sales is missing "
            + ", ".join(missing)
            + f". Columns on the table: {found}."
        )

    dataframe["business_date"] = (
        pd.to_datetime(
            dataframe["business_date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if "time" not in dataframe.columns:
        dataframe["time"] = dataframe.get("interval_label")

    dataframe = _coerce_numeric(
        dataframe,
        HOURLY_SALES_NUMERIC_COLUMNS,
    )
    dataframe[HOURLY_SALES_NUMERIC_COLUMNS] = (
        dataframe[HOURLY_SALES_NUMERIC_COLUMNS].fillna(0.0)
    )
    dataframe = _add_location_key(dataframe)
    dataframe = dataframe.dropna(subset=["business_date"])

    if (
        "extracted_at" in dataframe.columns
        and dataframe["time"].notna().any()
    ):
        dataframe["extracted_at"] = pd.to_datetime(
            dataframe["extracted_at"],
            errors="coerce",
            utc=True,
        )
        dataframe = dataframe.sort_values("extracted_at")
        dataframe = dataframe.drop_duplicates(
            subset=["location_key", "business_date", "time"],
            keep="last",
        )

    return dataframe.sort_values(
        ["business_date", "location", "time"]
    )


# Canonical names for the HME car-departure feed. The source
# table uses the export's mixed-case headers.
HME_FIELDS = {
    "location": ["StoreLocation", "location", "store"],
    "departure_time": ["Departure_Time", "departure_time"],
    "event_name": ["Event_Name", "event_name", "event"],
    "total_cars": ["Total_Cars", "total_cars"],
    "menu_board": ["Menu_Board", "menu_board"],
    "greet": ["Greet", "greet"],
    "service": ["Service", "service"],
    "lane_queue": ["Lane_Queue", "lane_queue"],
    "lane_total": ["Lane_Total", "lane_total"],
}

DRIVE_THRU_TIME_COLUMNS = [
    "menu_board",
    "greet",
    "service",
    "lane_queue",
    "lane_total",
]


def _pick_hme_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Match an HME column, ignoring case and underscores."""

    found = _pick_column(columns, candidates)
    if found is not None:
        return found

    def squash(name: str) -> str:
        return (
            name.strip().lower().replace(" ", "").replace("_", "")
        )

    lookup = {squash(name): name for name in columns}
    for candidate in candidates:
        match = lookup.get(squash(candidate))
        if match is not None:
            return match
    return None


def _parse_store_timestamps(series: pd.Series) -> pd.Series:
    """Parse departure timestamps into naive Central Time.

    Values without a timezone are already store-local clock
    times. Converting those as UTC would shift the hour.
    """

    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = (
            parsed.dt.tz_convert(STORE_TIMEZONE).dt.tz_localize(None)
        )
    return parsed


def assign_business_clock(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the business date and clock hour for each departure.

    Hours before 6:00 AM belong to the previous business
    date, matching hourly sales and labor placement.
    """

    dataframe = dataframe.copy()
    stamps = dataframe["departure_time"]
    hours = stamps.dt.hour
    dates = stamps.dt.normalize()
    early = hours < BUSINESS_DAY_START_HOUR
    # Nullable so a blank timestamp does not crash the cast.
    dataframe["hour"] = hours.astype("Int64")
    dataframe["business_date"] = dates.where(
        ~early.fillna(False),
        dates - pd.Timedelta(days=1),
    )
    return dataframe


def rollup_drive_thru_times(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Car-weighted average of each lane time, in seconds.

    Each departure counts once. Averaging daily rates first
    would let a slow, quiet day count the same as a busy one.
    """

    value_columns = [
        "cars",
        *DRIVE_THRU_TIME_COLUMNS,
        "avg_cars_in_lane",
    ]
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, *value_columns])

    working = frame.copy()
    working["_car"] = 1
    aggregations = {
        "cars": ("_car", "sum"),
        "cars_in_lane_sum": ("total_cars", "sum"),
        "cars_in_lane_n": ("total_cars", "count"),
    }
    for column_name in DRIVE_THRU_TIME_COLUMNS:
        aggregations[f"{column_name}_sum"] = (column_name, "sum")
        aggregations[f"{column_name}_n"] = (column_name, "count")

    grouped = working.groupby(group_columns, as_index=False).agg(
        **aggregations
    )
    for column_name in DRIVE_THRU_TIME_COLUMNS:
        counted = grouped[f"{column_name}_n"].replace(0, pd.NA)
        grouped[column_name] = (
            grouped[f"{column_name}_sum"] / counted
        )
        grouped = grouped.drop(
            columns=[f"{column_name}_sum", f"{column_name}_n"]
        )

    grouped["avg_cars_in_lane"] = (
        grouped["cars_in_lane_sum"]
        / grouped["cars_in_lane_n"].replace(0, pd.NA)
    )
    grouped = grouped.drop(
        columns=["cars_in_lane_sum", "cars_in_lane_n"]
    )
    return grouped


@st.cache_data(ttl=300)
def load_hme_departures() -> pd.DataFrame:
    """Retrieve HME drive-thru car departures from Supabase.

    One row is one car leaving the lane. Time columns are
    seconds. total_cars is how many cars were still stacked
    behind that departure, not a daily car count.
    """

    available = get_table_columns(HME_TABLE)
    selected: list[str] = []
    rename: dict[str, str] = {}
    for canonical, candidates in HME_FIELDS.items():
        actual = _pick_hme_column(available, candidates)
        if actual is None or actual in selected:
            continue
        selected.append(actual)
        rename[actual] = canonical

    required = {"location", "departure_time", "lane_total"}
    missing = sorted(required - set(rename.values()))
    if missing or not selected:
        found = ", ".join(available) or "none"
        raise ValueError(
            f"{HME_TABLE} is missing "
            + ", ".join(missing or ["departure_time"])
            + f". Columns on the table: {found}."
        )

    order_column = _pick_hme_column(
        available,
        HME_FIELDS["departure_time"],
    )
    dataframe = fetch_table(HME_TABLE, selected, order_column)
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.rename(columns=rename)
    for column_name in ("event_name", "total_cars", *DRIVE_THRU_TIME_COLUMNS):
        if column_name not in dataframe.columns:
            dataframe[column_name] = pd.NA

    dataframe["departure_time"] = _parse_store_timestamps(
        dataframe["departure_time"]
    )
    dataframe = _coerce_numeric(
        dataframe,
        ["total_cars", *DRIVE_THRU_TIME_COLUMNS],
    )
    dataframe = dataframe.dropna(
        subset=["departure_time", "lane_total"]
    )

    if "event_name" in dataframe.columns:
        event_name = (
            dataframe["event_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        departures = dataframe[event_name == "car_departure"]
        if not departures.empty:
            dataframe = departures.copy()

    dataframe = _add_location_key(dataframe)
    dataframe = assign_business_clock(dataframe)
    dataframe = dataframe.drop_duplicates(
        subset=["location_key", "departure_time"],
        keep="last",
    )
    return dataframe.sort_values(
        ["departure_time", "location"]
    )


@st.cache_data(ttl=300)
def load_shift_summary() -> pd.DataFrame:
    """Retrieve shift clocks used to place labor on an hour.

    The frame always has location, role, employee, shift_date,
    clock_in, clock_out, and hours. Clock columns are empty
    when the source table does not store them.
    """

    empty = pd.DataFrame(
        columns=[
            "location",
            "location_key",
            "role",
            "employee",
            "shift_date",
            "clock_in",
            "clock_out",
            "hours",
        ]
    )
    available = get_table_columns(SHIFT_SUMMARY_TABLE)
    if not available:
        return empty

    date_column = _pick_column(
        available,
        ["business_date", "date", "shift_date"],
    )
    location_column = _pick_column(
        available,
        ["location", "establishment", "store"],
    )
    role_column = _pick_column(
        available,
        ["role", "job_title", "position"],
    )
    employee_column = _pick_column(
        available,
        ["employee", "employee_name", "name"],
    )
    hours_column = _pick_column(
        available,
        ["hours", "total_hours", "labor_hours", "worked_hours"],
    )
    regular_hours_column = None
    ot_hours_column = None
    if hours_column is None:
        regular_hours_column = _pick_column(
            available,
            ["regular_hours"],
        )
        ot_hours_column = _pick_column(
            available,
            ["ot_hours"],
        )
    clock_pair = _resolve_clock_pair(available)
    order_column = date_column or _pick_column(
        available,
        ["id", "record_key"],
    ) or available[0]

    selected = []
    for column_name in (
        date_column,
        location_column,
        role_column,
        employee_column,
        hours_column,
        regular_hours_column,
        ot_hours_column,
        *(clock_pair or ()),
    ):
        if column_name and column_name not in selected:
            selected.append(column_name)

    dataframe = fetch_table(
        SHIFT_SUMMARY_TABLE,
        selected,
        order_column,
    )
    if dataframe.empty:
        return empty

    rename = {}
    if date_column:
        rename[date_column] = "shift_date"
    if location_column:
        rename[location_column] = "location"
    if role_column:
        rename[role_column] = "role"
    if employee_column:
        rename[employee_column] = "employee"
    if hours_column:
        rename[hours_column] = "hours"
    if regular_hours_column:
        rename[regular_hours_column] = "regular_hours"
    if ot_hours_column:
        rename[ot_hours_column] = "ot_hours"
    if clock_pair:
        rename[clock_pair[0]] = "clock_in"
        rename[clock_pair[1]] = "clock_out"
    dataframe = dataframe.rename(columns=rename)

    for column_name in ("role", "employee", "clock_in", "clock_out"):
        if column_name not in dataframe.columns:
            dataframe[column_name] = pd.NA

    if "hours" not in dataframe.columns:
        if (
            "regular_hours" in dataframe.columns
            or "ot_hours" in dataframe.columns
        ):
            regular_hours = (
                dataframe["regular_hours"]
                if "regular_hours" in dataframe.columns
                else 0.0
            )
            ot_hours = (
                dataframe["ot_hours"]
                if "ot_hours" in dataframe.columns
                else 0.0
            )
            dataframe["hours"] = (
                pd.to_numeric(regular_hours, errors="coerce").fillna(0.0)
                + pd.to_numeric(ot_hours, errors="coerce").fillna(0.0)
            )
        else:
            dataframe["hours"] = 0.0

    if "location" not in dataframe.columns:
        dataframe["location"] = "Unknown"

    if "shift_date" not in dataframe.columns:
        dataframe["shift_date"] = pd.Series(
            pd.NaT,
            index=dataframe.index,
            dtype="datetime64[ns]",
        )
    else:
        dataframe["shift_date"] = pd.to_datetime(
            dataframe["shift_date"],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)
    dataframe["shift_date"] = dataframe["shift_date"].dt.normalize()

    dataframe = _coerce_numeric(dataframe, ["hours"])
    dataframe["hours"] = dataframe["hours"].fillna(0.0)
    dataframe["role"] = (
        dataframe["role"].fillna("Unassigned").astype(str)
    )
    dataframe["employee"] = (
        dataframe["employee"].fillna("Unknown").astype(str)
    )
    dataframe = _add_location_key(dataframe)

    return dataframe


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


COMBO_MIX_COLUMNS = [
    "business_date",
    "location",
    "product_class",
    "product_category",
    "product_subcategory",
    "product_name",
    "n_items",
    "n_voids",
    "total",
    "discount",
    "order_discount",
]

COMBO_MIX_NUMERIC_COLUMNS = [
    "n_items",
    "n_voids",
    "total",
    "discount",
    "order_discount",
]

COMBO_MIX_TEXT_COLUMNS = [
    "product_class",
    "product_category",
    "product_subcategory",
    "product_name",
]


@st.cache_data(ttl=300)
def load_combo_mix() -> pd.DataFrame:
    """Retrieve combo component rows from Supabase.

    Each row is one item sold inside a combo. The source
    `total` column is combo net sales, so summing it is the
    combo net sales for that selection.
    """

    available = get_table_columns(COMBO_MIX_TABLE)
    selected = [
        column_name
        for column_name in (
            _pick_column(available, [candidate])
            for candidate in COMBO_MIX_COLUMNS
        )
        if column_name is not None
    ]
    order_column = _pick_column(
        available,
        ["business_date", "id"],
    )

    if not selected or order_column is None:
        return pd.DataFrame(
            columns=[
                *COMBO_MIX_COLUMNS,
                "net_sales",
                "discounts",
                "location_key",
            ]
        )

    dataframe = fetch_table(
        COMBO_MIX_TABLE,
        selected,
        order_column,
    )

    if dataframe.empty:
        return dataframe

    rename = {}
    for candidate in COMBO_MIX_COLUMNS:
        actual = _pick_column(list(dataframe.columns), [candidate])
        if actual is not None and actual != candidate:
            rename[actual] = candidate
    if rename:
        dataframe = dataframe.rename(columns=rename)

    missing = [
        column_name
        for column_name in (
            "business_date",
            "location",
            "product_class",
            "product_name",
            "total",
        )
        if column_name not in dataframe.columns
    ]
    if missing:
        found = ", ".join(available) or "none"
        raise ValueError(
            f"{COMBO_MIX_TABLE} is missing "
            + ", ".join(missing)
            + f". Columns on the table: {found}."
        )

    dataframe["business_date"] = (
        pd.to_datetime(
            dataframe["business_date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    dataframe = _coerce_numeric(
        dataframe,
        COMBO_MIX_NUMERIC_COLUMNS,
    )
    dataframe[COMBO_MIX_NUMERIC_COLUMNS] = (
        dataframe[COMBO_MIX_NUMERIC_COLUMNS].fillna(0.0)
    )

    for column_name in COMBO_MIX_TEXT_COLUMNS:
        if column_name not in dataframe.columns:
            dataframe[column_name] = "Unassigned"
        dataframe[column_name] = (
            dataframe[column_name]
            .fillna("Unassigned")
            .astype(str)
            .str.strip()
            .replace("", "Unassigned")
        )

    dataframe = _add_location_key(dataframe)
    dataframe = dataframe.dropna(subset=["business_date"])
    dataframe["net_sales"] = dataframe["total"]
    dataframe["discounts"] = (
        dataframe["discount"] + dataframe["order_discount"]
    )

    grain = [
        "location_key",
        "business_date",
        "product_class",
        "product_category",
        "product_subcategory",
        "product_name",
    ]
    dataframe = dataframe.drop_duplicates(subset=grain, keep="last")

    return dataframe.sort_values(
        ["business_date", "location", "product_class", "product_name"]
    )


CASH_SUMMARY_NUMERIC_COLUMNS = [
    "starting_cash",
    "cash_payments",
    "pay_ins",
    "pay_outs",
    "safe_drops",
    "total_expected_cash",
    "expected_cash_from_tills",
    "other",
    "declared_cash",
    "variance",
    "expected_total_cash_to_business",
    "actual_total_cash_to_business",
    "number_of_no_sales",
]


@st.cache_data(ttl=300)
def load_cash_summary() -> pd.DataFrame:
    """Retrieve every column of the daily cash summary."""

    columns = get_table_columns(CASH_SUMMARY_TABLE)
    if not columns:
        return pd.DataFrame()

    order_column = _pick_column(
        columns,
        ["business_date", "id"],
    )
    if order_column is None:
        order_column = columns[0]

    dataframe = fetch_table(
        CASH_SUMMARY_TABLE,
        columns,
        order_column,
    )

    if dataframe.empty:
        return dataframe

    if "business_date" in dataframe.columns:
        dataframe["business_date"] = (
            pd.to_datetime(
                dataframe["business_date"],
                errors="coerce",
                utc=True,
            )
            .dt.tz_localize(None)
            .dt.normalize()
        )

    numeric_columns = [
        column_name
        for column_name in CASH_SUMMARY_NUMERIC_COLUMNS
        if column_name in dataframe.columns
    ]
    dataframe = _coerce_numeric(dataframe, numeric_columns)

    if "location" in dataframe.columns:
        dataframe = _add_location_key(dataframe)

    if "business_date" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["business_date"])
        return dataframe.sort_values("business_date")

    return dataframe


# Loyalty and Standard are a second split of the same
# dollars as the named reasons. TOTAL repeats that amount.
_DISCOUNT_ROLLUP_REASONS = {
    "total",
    "loyalty",
    "standard",
}


def _parse_discount_payload(value) -> dict:
    """Turn a jsonb value into a reason-to-amounts mapping."""

    if value is None:
        return {}
    if isinstance(value, float) and pd.isna(value):
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
    if isinstance(value, dict):
        return value
    return {}


def _discount_payload_column(dataframe: pd.DataFrame) -> str | None:
    """Column that holds the reason object."""

    for name in dataframe.columns:
        key = name.strip().lower().replace(" ", "_")
        if key == "discount_reasons":
            return name

    for name in dataframe.columns:
        series = dataframe[name].dropna()
        if series.empty:
            continue
        sample = series.iloc[0]
        if isinstance(sample, dict):
            return name
        if isinstance(sample, str) and sample.strip().startswith("{"):
            return name
    return None


def _is_discount_rollup(reason: str) -> bool:
    return reason.strip().lower() in _DISCOUNT_ROLLUP_REASONS


def _reason_records(
    dataframe: pd.DataFrame,
    payload_column: str,
    keep_rollups: bool,
) -> list[dict]:
    records = []
    for row in dataframe.to_dict("records"):
        payload = _parse_discount_payload(row.get(payload_column))
        for reason, stats in payload.items():
            reason_name = str(reason).strip()
            if not reason_name:
                continue
            if reason_name.lower() == "total":
                continue
            if _is_discount_rollup(reason_name) and not keep_rollups:
                continue
            if not isinstance(stats, dict):
                continue
            records.append(
                {
                    "location": row.get("location"),
                    "business_date": row.get("business_date"),
                    "reason": reason_name,
                    "qty": stats.get("qty"),
                    "amount": stats.get("total"),
                }
            )
    return records


def expand_discount_reasons(dataframe: pd.DataFrame) -> pd.DataFrame:
    """One row per location, day, and named discount reason."""

    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "location",
                "business_date",
                "reason",
                "qty",
                "amount",
            ]
        )

    payload_column = _discount_payload_column(dataframe)
    if payload_column is None:
        return pd.DataFrame()

    records = _reason_records(
        dataframe,
        payload_column,
        keep_rollups=False,
    )
    if not records:
        records = _reason_records(
            dataframe,
            payload_column,
            keep_rollups=True,
        )
    if not records:
        return pd.DataFrame()

    expanded = pd.DataFrame(records)
    expanded["business_date"] = (
        pd.to_datetime(
            expanded["business_date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )
    expanded = _coerce_numeric(expanded, ["qty", "amount"])
    expanded = expanded.dropna(subset=["business_date", "reason"])

    if "location" in expanded.columns:
        expanded = _add_location_key(expanded)

    return expanded.sort_values(
        ["business_date", "location", "reason"]
    )


@st.cache_data(ttl=300)
def load_discount_reasons() -> pd.DataFrame:
    """Named discount reasons from the daily reason extract."""

    columns = get_table_columns(DISCOUNT_REASON_TABLE)
    if not columns:
        return pd.DataFrame()

    order_column = _pick_column(
        columns,
        ["business_date", "id"],
    )
    if order_column is None:
        order_column = columns[0]

    dataframe = fetch_table(
        DISCOUNT_REASON_TABLE,
        columns,
        order_column,
    )
    return expand_discount_reasons(dataframe)


WASTE_LOG_COLUMNS = [
    "waste_date",
    "location",
    "item",
    "uom",
    "qty",
    "each_amount",
    "total",
]

WASTE_LOG_NUMERIC_COLUMNS = [
    "qty",
    "each_amount",
    "total",
]


def waste_category(item) -> str | None:
    """Chicken and fries are the only waste lines this view scores."""

    name = str(item).strip().upper()
    if "CHICKEN" in name:
        return "chicken"
    if "FRIES" in name or name.startswith("FRY"):
        return "fries"
    return None


@st.cache_data(ttl=300)
def load_waste_log() -> pd.DataFrame:
    """Chicken and fries waste dollars from the waste log."""

    columns = get_table_columns(WASTE_LOG_TABLE)
    if not columns:
        return pd.DataFrame()

    selected = [
        column
        for column in WASTE_LOG_COLUMNS
        if column in columns
    ]
    if not {"waste_date", "location", "item", "total"} <= set(selected):
        return pd.DataFrame()

    order_column = (
        "waste_date" if "waste_date" in columns else selected[0]
    )
    dataframe = fetch_table(
        WASTE_LOG_TABLE,
        selected,
        order_column,
    )
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.rename(columns={"waste_date": "business_date"})
    dataframe["business_date"] = pd.to_datetime(
        dataframe["business_date"],
        errors="coerce",
    ).dt.normalize()
    for column in WASTE_LOG_NUMERIC_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).fillna(0)

    dataframe["category"] = dataframe["item"].map(waste_category)
    dataframe = dataframe[dataframe["category"].notna()].copy()
    dataframe = _add_location_key(dataframe)
    return dataframe.sort_values(
        ["business_date", "location", "category"]
    ).reset_index(drop=True)
