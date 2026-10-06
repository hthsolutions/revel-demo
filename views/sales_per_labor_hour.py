import importlib
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

import revel_data

# A deploy can leave the already-imported data module in memory
# while this page is read fresh from disk.
if not hasattr(revel_data, "SHIFT_SUMMARY_TABLE"):
    revel_data = importlib.reload(revel_data)

from revel_data import (
    SHIFT_SUMMARY_TABLE,
    WEEK_START_OPTIONS,
    WEEKDAY_NAMES,
    MissingSecretError,
    add_week_columns,
    get_table_columns,
    load_hourly_sales,
    load_shift_summary,
)


# The hourly sales extract starts each business date at
# 6:00 AM and continues through 5:59 AM the next morning.
BUSINESS_DAY_START_HOUR = 6
STORE_TIMEZONE = ZoneInfo("America/Chicago")


def format_week(week_start) -> str:
    """Label a week by its first and last calendar day."""

    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    return f"{start:%b %d} – {end:%b %d, %Y}"


def format_day(business_date) -> str:
    """Label a business date with its weekday."""

    stamp = pd.Timestamp(business_date)
    return f"{WEEKDAY_NAMES[stamp.weekday()][:3]} {stamp:%m/%d}"


def hour_label(hour: int) -> str:
    """Render 0–23 as a clock hour, such as 6 AM or 12 PM."""

    hour = int(hour) % 24
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour} {suffix}"


def hour_from_interval(label) -> int | None:
    """Read the starting clock hour from a 15-minute label."""

    if label is None or pd.isna(label):
        return None

    text = (
        str(label)
        .replace("–", "-")
        .replace("—", "-")
        .split("-")[0]
        .strip()
    )
    if text == "" or text.lower() in {"none", "nat", "null"}:
        return None

    parsed = pd.to_datetime(text, format="%I:%M %p", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.hour)


def parse_store_clock(value):
    """Parse a clock timestamp into naive Central Time."""

    if value is None or pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        parsed = value
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert(STORE_TIMEZONE).tz_localize(None)
        return pd.Timestamp(parsed)

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nat", "null", "<na>"}:
        return pd.NaT

    has_zone = (
        text.endswith("Z")
        or text.endswith("z")
        or "+" in text[10:]
        or text[10:].count("-") > 0
    )
    parsed = pd.to_datetime(text, errors="coerce", utc=has_zone)
    if pd.isna(parsed):
        return pd.NaT

    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(STORE_TIMEZONE).tz_localize(None)

    return pd.Timestamp(parsed)


def interpret_clock(shift_date, value):
    """Combine a shift date with a time-only clock when needed."""

    if value is None or pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return parse_store_clock(value)

    text = str(value).strip()
    looks_like_time_only = (
        "T" not in text
        and "-" not in text
        and "/" not in text
    )
    if not looks_like_time_only:
        return parse_store_clock(text)

    if shift_date is None or pd.isna(shift_date):
        return pd.NaT

    parsed_time = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed_time):
        return pd.NaT

    base = pd.Timestamp(shift_date).normalize()
    return base + pd.Timedelta(
        hours=int(parsed_time.hour),
        minutes=int(parsed_time.minute),
        seconds=int(parsed_time.second),
    )


def resolve_shift_span(shift_date, clock_in_raw, clock_out_raw, hours):
    """Return the clock-in and clock-out used to place a shift."""

    clock_in = interpret_clock(shift_date, clock_in_raw)
    clock_out = interpret_clock(shift_date, clock_out_raw)

    if (
        pd.notna(clock_in)
        and pd.notna(clock_out)
        and clock_out <= clock_in
    ):
        clock_out = clock_out + pd.Timedelta(days=1)

    if pd.isna(clock_out) and pd.notna(clock_in):
        duration = float(hours or 0)
        if duration > 0:
            clock_out = clock_in + pd.Timedelta(hours=duration)

    return clock_in, clock_out


def allocate_shift_hours(clock_in, clock_out):
    """Split a shift into business-date and clock-hour pieces.

    Hours before 6:00 AM belong to the previous business
    date, matching the hourly sales extract.
    """

    if clock_in is None or pd.isna(clock_in):
        return []
    if clock_out is None or pd.isna(clock_out) or clock_out <= clock_in:
        return []

    pieces = []
    cursor = pd.Timestamp(clock_in)
    stop_at = pd.Timestamp(clock_out)

    while cursor < stop_at:
        hour_floor = cursor.replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        hour_end = hour_floor + pd.Timedelta(hours=1)
        piece_end = min(stop_at, hour_end)
        worked = (piece_end - cursor).total_seconds() / 3600
        clock_hour = int(cursor.hour)
        if clock_hour < BUSINESS_DAY_START_HOUR:
            business_date = (
                hour_floor.normalize() - pd.Timedelta(days=1)
            )
        else:
            business_date = hour_floor.normalize()
        if worked > 0:
            pieces.append((business_date, clock_hour, worked))
        cursor = piece_end

    return pieces


def preferred_location(names: pd.Series) -> str:
    """Pick one display name when sources spell a store differently."""

    unique = [str(name) for name in names.dropna().unique()]
    if not unique:
        return "Unknown"
    mixed_case = [name for name in unique if name != name.upper()]
    return sorted(mixed_case or unique)[0]


def with_hour_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the business-day hour order and its display label."""

    frame = frame.copy()
    frame["hour_index"] = (
        frame["hour"] - BUSINESS_DAY_START_HOUR
    ) % 24
    frame["hour_label"] = frame["hour"].map(hour_label)
    return frame


def productivity(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Sum sales and labor, then divide once per group."""

    grouped = (
        frame
        .groupby(group_columns, as_index=False)
        .agg(
            sales=("sales", "sum"),
            labor_hours=("labor_hours", "sum"),
            transactions=("transactions", "sum"),
        )
    )
    grouped["sales_per_labor_hour"] = (
        grouped["sales"]
        / grouped["labor_hours"].where(grouped["labor_hours"] > 0)
    )
    return with_hour_fields(grouped)


@st.cache_data(ttl=300)
def labor_hours_by_hour(shift_df: pd.DataFrame) -> pd.DataFrame:
    """Place each shift onto the business hours it overlaps."""

    empty = pd.DataFrame(
        columns=[
            "location_key",
            "location",
            "business_date",
            "hour",
            "labor_hours",
        ]
    )
    if shift_df.empty:
        return empty

    records = []
    for shift in shift_df.itertuples(index=False):
        clock_in, clock_out = resolve_shift_span(
            shift.shift_date,
            shift.clock_in,
            shift.clock_out,
            shift.hours,
        )
        for business_date, hour, worked in allocate_shift_hours(
            clock_in,
            clock_out,
        ):
            records.append(
                {
                    "location_key": shift.location_key,
                    "location": shift.location,
                    "business_date": business_date,
                    "hour": hour,
                    "labor_hours": worked,
                }
            )

    if not records:
        return empty

    placed = pd.DataFrame(records)
    placed["business_date"] = pd.to_datetime(
        placed["business_date"]
    ).dt.normalize()
    return (
        placed
        .groupby(
            ["location_key", "location", "business_date", "hour"],
            as_index=False,
        )["labor_hours"]
        .sum()
    )


st.title("Sales per Labor Hour per Hour")
st.caption(
    "Sales in each clock hour divided by the labor hours "
    "clocked during that same hour. The business day runs "
    "from 6:00 AM through 5:59 AM the next morning. "
    "GM shifts are excluded."
)

try:
    sales_df = load_hourly_sales()
    shift_df = load_shift_summary()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "Unable to load hourly sales or shift summary "
        f"from Supabase: {error}"
    )
    st.stop()

if sales_df.empty:
    st.warning("No hourly sales records were returned.")
    st.stop()

if shift_df.empty:
    st.warning("No shift summary records were returned.")
    st.stop()

shift_df = shift_df[
    shift_df["role"].astype(str).str.strip().str.upper() != "GM"
].copy()

if shift_df.empty:
    st.warning("No shifts remain after excluding GM.")
    st.stop()

sales_df = sales_df.copy()
sales_df["hour"] = sales_df["time"].map(hour_from_interval)
if (
    "interval_label" in sales_df.columns
    and sales_df["hour"].isna().any()
):
    sales_df["hour"] = sales_df["hour"].fillna(
        sales_df["interval_label"].map(hour_from_interval)
    )
unreadable_intervals = int(sales_df["hour"].isna().sum())
if unreadable_intervals:
    st.warning(
        f"{unreadable_intervals:,} sales intervals had no "
        "readable start time and were skipped."
    )
sales_df = sales_df.dropna(subset=["hour"])
if sales_df.empty:
    st.warning("Hourly sales intervals have no readable times.")
    st.stop()
sales_df["hour"] = sales_df["hour"].astype(int)

labor_df = labor_hours_by_hour(shift_df)
if labor_df.empty:
    try:
        column_names = get_table_columns(SHIFT_SUMMARY_TABLE)
    except Exception:
        column_names = []
    detail = (
        "Columns on revel_shift_summary: " + ", ".join(column_names)
        if column_names
        else "revel_shift_summary returned no clock columns."
    )
    st.error(
        "Labor hours can't be placed on the clock without "
        "a clock-in and clock-out. " + detail
    )
    st.stop()

sales_hours = (
    sales_df
    .groupby(
        ["location_key", "business_date", "hour"],
        as_index=False,
    )
    .agg(
        location=("location", "first"),
        sales=("sales", "sum"),
        transactions=("transactions", "sum"),
    )
)
combined = sales_hours.merge(
    labor_df.rename(columns={"location": "labor_location"}),
    on=["location_key", "business_date", "hour"],
    how="outer",
)
combined["location"] = combined["location"].fillna(
    combined["labor_location"]
)
combined = combined.drop(columns=["labor_location"])
combined["location"] = combined["location"].fillna("Unknown")
combined["sales"] = combined["sales"].fillna(0.0)
combined["transactions"] = combined["transactions"].fillna(0.0)
combined["labor_hours"] = combined["labor_hours"].fillna(0.0)
combined["hour"] = combined["hour"].astype(int)
combined["business_date"] = pd.to_datetime(
    combined["business_date"]
).dt.normalize()
location_names = (
    combined
    .groupby("location_key")["location"]
    .agg(preferred_location)
)
combined["location"] = combined["location_key"].map(location_names)


# ---------------------------------------------------------
# Sidebar: one location, one week, and an optional day
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    combined["location"].dropna().unique().tolist()
)
selected_location = st.sidebar.selectbox(
    "Location",
    options=["All Locations", *available_locations],
)
selected_week_start_name = st.sidebar.selectbox(
    "Week starts on",
    options=list(WEEK_START_OPTIONS.keys()),
)
week_start_weekday = WEEK_START_OPTIONS[selected_week_start_name]

if selected_location != "All Locations":
    combined = combined[
        combined["location"] == selected_location
    ].copy()

if combined.empty:
    st.warning("No sales or labor hours match that location.")
    st.stop()

combined = add_week_columns(combined, week_start_weekday)
week_starts = sorted(
    combined["week_start"].dropna().unique(),
    reverse=True,
)
selected_week = st.sidebar.selectbox(
    "Week",
    options=week_starts,
    format_func=format_week,
)

week_df = combined[
    combined["week_start"] == selected_week
].copy()

if week_df.empty:
    st.info("No sales or labor hours fall in that week.")
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

if selected_day_label != "All days":
    selected_dates = [
        business_date
        for business_date, label in day_labels.items()
        if label == selected_day_label
    ]
    week_df = week_df[
        week_df["business_date"].isin(selected_dates)
    ].copy()

total_sales = float(week_df["sales"].sum())
total_labor_hours = float(week_df["labor_hours"].sum())
overall_rate = (
    total_sales / total_labor_hours
    if total_labor_hours
    else float("nan")
)

st.subheader(
    format_week(selected_week)
    if selected_day_label == "All days"
    else f"{selected_day_label}, {format_week(selected_week)}"
)

kpi_columns = st.columns(3)
kpi_columns[0].metric("Sales", f"${total_sales:,.2f}")
kpi_columns[1].metric(
    "Labor Hours",
    f"{total_labor_hours:,.2f} hrs",
)
kpi_columns[2].metric(
    "Sales per Labor Hour",
    "—" if pd.isna(overall_rate) else f"${overall_rate:,.2f}",
)

hour_totals = productivity(week_df, ["hour"])
hour_totals = hour_totals.sort_values("hour_index")
active_hours = hour_totals[
    (hour_totals["sales"] > 0) | (hour_totals["labor_hours"] > 0)
].copy()

if active_hours.empty:
    st.info("That selection has no sales or labor hours.")
    st.stop()

day_totals = productivity(
    week_df,
    ["business_date", "hour"],
)
day_totals["day_label"] = day_totals["business_date"].map(format_day)
day_order = [
    format_day(value)
    for value in sorted(day_totals["business_date"].unique())
]
active_hour_keys = active_hours[
    ["hour", "hour_index", "hour_label"]
].drop_duplicates()
day_dates = day_totals[["business_date", "day_label"]].drop_duplicates()
day_grid = day_dates.merge(active_hour_keys, how="cross")
day_plot = day_grid.merge(
    day_totals.drop(columns=["hour_index", "hour_label"]),
    on=["business_date", "day_label", "hour"],
    how="left",
)
day_plot["sales_per_labor_hour"] = (
    day_plot["sales"]
    / day_plot["labor_hours"].where(day_plot["labor_hours"] > 0)
)

def hour_axis() -> alt.X:
    """Clock-hour axis in business-day order, 6 AM through 5 AM."""

    return alt.X(
        "hour_label:N",
        title="Hour",
        sort=alt.SortField(field="hour_index"),
    )
rate_tooltip = [
    alt.Tooltip("hour_label:N", title="Hour"),
    alt.Tooltip(
        "sales_per_labor_hour:Q",
        title="Sales per Labor Hour",
        format="$,.2f",
    ),
    alt.Tooltip("sales:Q", title="Sales", format="$,.2f"),
    alt.Tooltip(
        "labor_hours:Q",
        title="Labor Hours",
        format=",.2f",
    ),
    alt.Tooltip(
        "transactions:Q",
        title="Transactions",
        format=",.0f",
    ),
]

day_lines = (
    alt.Chart(day_plot)
    .mark_line(
        strokeWidth=2,
        point=alt.OverlayMarkDef(size=45),
    )
    .encode(
        x=hour_axis(),
        y=alt.Y(
            "sales_per_labor_hour:Q",
            title="Sales per Labor Hour",
            axis=alt.Axis(format="$,.0f"),
        ),
        color=alt.Color(
            "day_label:N",
            title="Day",
            sort=day_order,
        ),
        tooltip=[
            alt.Tooltip("day_label:N", title="Day"),
            *rate_tooltip,
        ],
    )
)

chart_layers = [day_lines]
if selected_day_label == "All days" and len(day_order) > 1:
    week_line = (
        alt.Chart(active_hours)
        .mark_line(
            color="#222222",
            strokeWidth=3.5,
            point=alt.OverlayMarkDef(size=70, color="#222222"),
        )
        .encode(
            x=hour_axis(),
            y=alt.Y(
                "sales_per_labor_hour:Q",
                title="Sales per Labor Hour",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=rate_tooltip,
        )
    )
    chart_layers.append(week_line)

rate_chart = (
    alt.layer(*chart_layers)
    .resolve_scale(color="independent", y="shared")
    .properties(height=420)
)
st.altair_chart(rate_chart, use_container_width=True)
if selected_day_label == "All days" and len(day_order) > 1:
    st.caption(
        "Each colored line is one business day. The dark "
        "line pools the week: sales in that hour divided "
        "by labor hours in that hour."
    )
else:
    st.caption(
        "Each point is sales during that hour divided by "
        "labor hours clocked during that hour."
    )

labor_bars = (
    alt.Chart(active_hours)
    .mark_bar(color="#1f77b4", opacity=0.75)
    .encode(
        x=hour_axis(),
        y=alt.Y(
            "labor_hours:Q",
            title="Labor Hours",
            axis=alt.Axis(format=",.1f"),
        ),
        tooltip=rate_tooltip,
    )
)
sales_line = (
    alt.Chart(active_hours)
    .mark_line(
        color="#ff7f0e",
        strokeWidth=2.5,
        point=alt.OverlayMarkDef(size=55, color="#ff7f0e"),
    )
    .encode(
        x=hour_axis(),
        y=alt.Y(
            "sales:Q",
            title="Sales",
            axis=alt.Axis(
                orient="right",
                format="$,.0f",
                titleColor="#ff7f0e",
                labelColor="#ff7f0e",
            ),
        ),
        tooltip=rate_tooltip,
    )
)
mix_chart = (
    alt.layer(labor_bars, sales_line)
    .resolve_scale(y="independent")
    .properties(height=360)
)

st.subheader("Sales and Labor Hours")
st.altair_chart(mix_chart, use_container_width=True)
st.caption(
    "Bars are labor hours on the left axis. The line is "
    "sales on the right axis, for the same hours."
)

table_df = active_hours.sort_values("hour_index")[
    [
        "hour_label",
        "sales",
        "labor_hours",
        "sales_per_labor_hour",
        "transactions",
    ]
].copy()
st.subheader("Hourly breakdown")
st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "hour_label": "Hour",
        "sales": st.column_config.NumberColumn(
            "Sales",
            format="$%.2f",
        ),
        "labor_hours": st.column_config.NumberColumn(
            "Labor Hours",
            format="%.2f",
        ),
        "sales_per_labor_hour": st.column_config.NumberColumn(
            "Sales per Labor Hour",
            format="$%.2f",
        ),
        "transactions": st.column_config.NumberColumn(
            "Transactions",
            format="%d",
        ),
    },
)

st.caption(
    f"Displaying {format_week(selected_week)} for "
    f"{selected_location.lower()}, weeks starting "
    f"{selected_week_start_name}. Labor hours are the "
    "time between clock-in and clock-out."
)
