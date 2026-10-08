import importlib
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

import revel_data

if not hasattr(revel_data, "load_hme_departures"):
    revel_data = importlib.reload(revel_data)

from revel_data import (
    BUSINESS_DAY_START_HOUR,
    DRIVE_THRU_TIME_COLUMNS,
    WEEK_START_OPTIONS,
    WEEKDAY_NAMES,
    MissingSecretError,
    add_week_columns,
    load_hme_departures,
    load_hourly_sales,
    load_shift_summary,
    percent_change,
    rollup_drive_thru_times,
)


STORE_TIMEZONE = ZoneInfo("America/Chicago")

# Lane time at or under this stays green. Longer times are red.
GOAL_MINUTES = 4.0
HEATMAP_RED = ((103, 0, 13), (239, 59, 44))
HEATMAP_GREEN = ((102, 189, 99), (0, 68, 27))


def format_week(week_start) -> str:
    """Label a week by its first and last calendar day."""

    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    return f"{start:%b %d} – {end:%b %d, %Y}"


def format_day(business_date) -> str:
    """Label a business date with its weekday."""

    stamp = pd.Timestamp(business_date)
    return f"{WEEKDAY_NAMES[stamp.weekday()][:3]} {stamp:%m/%d}"


def format_duration(seconds) -> str:
    """Render seconds as minutes and seconds, such as 3:12."""

    if seconds is None or pd.isna(seconds):
        return "—"
    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    minutes, remainder = divmod(total, 60)
    return f"{sign}{minutes}:{remainder:02d}"


def _srgb_channel(value: float) -> float:
    value = value / 255
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _mix_color(start, end, amount: float):
    blend = min(max(amount, 0.0), 1.0)
    return tuple(
        round(start[index] + (end[index] - start[index]) * blend)
        for index in range(3)
    )


def heatmap_top(minutes_max: float) -> float:
    """Keep the 4-minute break on the scale."""

    if pd.isna(minutes_max) or minutes_max <= 0:
        return GOAL_MINUTES
    return max(float(minutes_max), GOAL_MINUTES)


def heatmap_fill(minutes, top: float):
    """Green at 4 minutes or under, red when the lane runs longer.

    Darker green is a faster lane. Darker red is a slower one.
    """

    if minutes is None or pd.isna(minutes):
        return None
    amount = float(minutes)
    if amount <= GOAL_MINUTES:
        return _mix_color(
            HEATMAP_GREEN[1],
            HEATMAP_GREEN[0],
            amount / GOAL_MINUTES,
        )
    span = max(float(top) - GOAL_MINUTES, 0.01)
    return _mix_color(
        HEATMAP_RED[1],
        HEATMAP_RED[0],
        (amount - GOAL_MINUTES) / span,
    )


def heatmap_color_hex(minutes, top: float):
    fill = heatmap_fill(minutes, top)
    if fill is None:
        return None
    return "#{:02x}{:02x}{:02x}".format(*fill)


def heatmap_label_color(minutes, top: float) -> str:
    """Use white type on dark red and dark green cells."""

    fill = heatmap_fill(minutes, top)
    if fill is None:
        return "#1a1a1a"
    red, green, blue = (_srgb_channel(part) for part in fill)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    # Tiles near 4:00 are light green, so they need dark type.
    return "#ffffff" if luminance < 0.30 else "#1a1a1a"



def format_seconds_delta(current, previous):
    """Signed second change, or None when either side is missing."""

    if current is None or previous is None:
        return None
    if pd.isna(current) or pd.isna(previous):
        return None
    return f"{float(current) - float(previous):+.0f} sec"


def preferred_location(names: pd.Series) -> str:
    """Pick one display name when sources spell a store differently."""

    unique = [str(name) for name in names.dropna().unique()]
    if not unique:
        return "Unknown"
    mixed_case = [name for name in unique if name != name.upper()]
    return sorted(mixed_case or unique)[0]


def hour_label(hour: int) -> str:
    """Render 0–23 as a clock hour, such as 6 AM or 12 PM."""

    hour = int(hour) % 24
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour} {suffix}"


def with_hour_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add business-day hour order and a display label."""

    frame = frame.copy()
    frame["hour"] = frame["hour"].astype(int)
    frame["hour_index"] = (
        frame["hour"] - BUSINESS_DAY_START_HOUR
    ) % 24
    frame["hour_label"] = frame["hour"].map(hour_label)
    return frame


def interval_start_parts(label) -> tuple[int | None, int | None]:
    """Read the starting hour and minute from a 15-minute label."""

    if label is None or pd.isna(label):
        return None, None

    text = (
        str(label)
        .replace("–", "-")
        .replace("—", "-")
        .split("-")[0]
        .strip()
    )
    if text == "" or text.lower() in {"none", "nat", "null"}:
        return None, None

    parsed = pd.to_datetime(text, format="%I:%M %p", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None, None
    return int(parsed.hour), int(parsed.minute)


def hour_from_interval(label) -> int | None:
    """Read the starting clock hour from a 15-minute label."""

    hour, _minute = interval_start_parts(label)
    return hour


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


def labor_hours_by_hour(shift_df: pd.DataFrame) -> pd.DataFrame:
    """Place each shift onto the business hours it overlaps."""

    empty = pd.DataFrame(
        columns=[
            "location_key",
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
            ["location_key", "business_date", "hour"],
            as_index=False,
        )["labor_hours"]
        .sum()
    )


def sales_by_hour(sales_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sum 15-minute sales into clock hours."""

    empty = pd.DataFrame(
        columns=[
            "location_key",
            "business_date",
            "hour",
            "sales",
            "transactions",
        ]
    )
    if sales_df.empty or "time" not in sales_df.columns:
        return empty, 0

    working = sales_df.copy()
    working["hour"] = working["time"].map(hour_from_interval)
    if (
        "interval_label" in working.columns
        and working["hour"].isna().any()
    ):
        working["hour"] = working["hour"].fillna(
            working["interval_label"].map(hour_from_interval)
        )
    skipped = int(working["hour"].isna().sum())
    working = working.dropna(subset=["hour"])
    if working.empty:
        return empty, skipped

    working["hour"] = working["hour"].astype(int)
    working["business_date"] = pd.to_datetime(
        working["business_date"]
    ).dt.normalize()
    if "transactions" not in working.columns:
        working["transactions"] = 0.0

    grouped = (
        working
        .groupby(
            ["location_key", "business_date", "hour"],
            as_index=False,
        )
        .agg(
            sales=("sales", "sum"),
            transactions=("transactions", "sum"),
        )
    )
    return grouped, skipped



def period_rates(frame: pd.DataFrame) -> dict:
    """One car-weighted average for a set of departures."""

    if frame.empty:
        return {}
    rolled = rollup_drive_thru_times(
        frame.assign(_group=1),
        ["_group"],
    )
    if rolled.empty:
        return {}
    return rolled.iloc[0].to_dict()


def period_splh(frame: pd.DataFrame) -> float:
    """Sales per labor hour for a set of store hours."""

    if frame.empty or "labor_hours" not in frame.columns:
        return float("nan")
    labor_hours = float(frame["labor_hours"].fillna(0).sum())
    if labor_hours <= 0:
        return float("nan")
    return float(frame["sales"].fillna(0).sum()) / labor_hours


def add_duration_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add minute:second labels for every lane-time column present."""

    frame = frame.copy()
    for column_name in DRIVE_THRU_TIME_COLUMNS:
        if column_name not in frame.columns:
            continue
        frame[f"{column_name}_label"] = frame[column_name].map(
            format_duration
        )
    return frame


st.title("Drive-Thru Times")
st.caption(
    "Average HME lane time for each car that leaves the "
    "drive-thru, and that time next to sales per labor hour."
)

try:
    hme_df = load_hme_departures()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        "Unable to load drive-thru times from Supabase: "
        f"{error}"
    )
    st.stop()

sales_error = None
shift_error = None
try:
    sales_df = load_hourly_sales()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    sales_df = pd.DataFrame()
    sales_error = str(error)

try:
    shift_df = load_shift_summary()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    shift_df = pd.DataFrame()
    shift_error = str(error)

if hme_df.empty:
    st.warning("No drive-thru departures were returned.")
    st.stop()

if not sales_df.empty and "location_key" in sales_df.columns:
    name_frames = [
        hme_df[["location_key", "location"]],
        sales_df[["location_key", "location"]],
    ]
    if not shift_df.empty and "location_key" in shift_df.columns:
        name_frames.append(shift_df[["location_key", "location"]])
    location_names = (
        pd.concat(name_frames, ignore_index=True)
        .groupby("location_key")["location"]
        .agg(preferred_location)
    )
    hme_df = hme_df.copy()
    hme_df["location"] = (
        hme_df["location_key"].map(location_names).fillna(hme_df["location"])
    )


# ---------------------------------------------------------
# Sidebar: location, week, day, and how much history
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    hme_df["location"].dropna().unique().tolist()
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
    hme_df = hme_df[hme_df["location"] == selected_location].copy()

if hme_df.empty:
    st.warning("No drive-thru departures match that location.")
    st.stop()

store_keys = set(hme_df["location_key"].unique())
if not sales_df.empty and "location_key" in sales_df.columns:
    sales_df = sales_df[
        sales_df["location_key"].isin(store_keys)
    ].copy()
if not shift_df.empty and "location_key" in shift_df.columns:
    shift_df = shift_df[
        shift_df["location_key"].isin(store_keys)
    ].copy()
    shift_df = shift_df[
        shift_df["role"].astype(str).str.strip().str.upper() != "GM"
    ].copy()

hme_df = add_week_columns(hme_df, week_start_weekday)
week_starts = sorted(
    hme_df["week_start"].dropna().unique(),
    reverse=True,
)
selected_week = st.sidebar.selectbox(
    "Week",
    options=week_starts,
    format_func=format_week,
)
weeks_to_display = st.sidebar.slider(
    "Weeks of history",
    min_value=2,
    max_value=26,
    value=8,
    help=(
        "Days on the drive-thru time heatmap, ending with "
        "the selected week. The summary and the hourly "
        "breakdown use the selected week only."
    ),
)

week_cars = hme_df[hme_df["week_start"] == selected_week]
day_choices = (
    week_cars[["business_date"]]
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
    help=(
        "Applies to the summary and the hourly breakdown. "
        "The history chart still shows every day."
    ),
)

history_start = pd.Timestamp(selected_week) - pd.Timedelta(
    days=7 * (weeks_to_display - 1)
)
history_df = hme_df[
    (hme_df["week_start"] >= history_start)
    & (hme_df["week_start"] <= pd.Timestamp(selected_week))
].copy()

if selected_day_label == "All days":
    slice_df = week_cars.copy()
    slice_dates = set(
        pd.to_datetime(week_cars["business_date"]).dt.normalize()
    )
else:
    slice_dates = {
        business_date
        for business_date, label in day_labels.items()
        if label == selected_day_label
    }
    slice_df = week_cars[
        pd.to_datetime(week_cars["business_date"]).dt.normalize().isin(
            slice_dates
        )
    ].copy()

previous_week = pd.Timestamp(selected_week) - pd.Timedelta(days=7)
if selected_day_label == "All days":
    previous_cars = hme_df[hme_df["week_start"] == previous_week]
else:
    previous_dates = {
        business_date - pd.Timedelta(days=7)
        for business_date in slice_dates
    }
    previous_cars = hme_df[
        pd.to_datetime(hme_df["business_date"]).dt.normalize().isin(
            previous_dates
        )
    ]

current_times = period_rates(slice_df)
previous_times = period_rates(previous_cars)

sales_skipped = 0
if sales_df.empty:
    hourly_store = pd.DataFrame(
        columns=[
            "location_key",
            "business_date",
            "hour",
            "sales",
            "transactions",
            "labor_hours",
        ]
    )
else:
    hourly_sales, sales_skipped = sales_by_hour(sales_df)
    hourly_labor = labor_hours_by_hour(shift_df)
    if hourly_sales.empty and hourly_labor.empty:
        hourly_store = pd.DataFrame(
            columns=[
                "location_key",
                "business_date",
                "hour",
                "sales",
                "transactions",
                "labor_hours",
            ]
        )
    else:
        hourly_store = hourly_sales.merge(
            hourly_labor,
            on=["location_key", "business_date", "hour"],
            how="outer",
        )
        hourly_store["sales"] = hourly_store["sales"].fillna(0.0)
        hourly_store["transactions"] = (
            hourly_store["transactions"].fillna(0.0)
        )
        hourly_store["labor_hours"] = (
            hourly_store["labor_hours"].fillna(0.0)
        )
        hourly_store["business_date"] = pd.to_datetime(
            hourly_store["business_date"]
        ).dt.normalize()
        hourly_store = hourly_store.dropna(
            subset=["business_date", "hour"]
        )
        hourly_store["hour"] = hourly_store["hour"].astype(int)

if not hourly_store.empty:
    hourly_store = add_week_columns(hourly_store, week_start_weekday)
    hourly_slice = hourly_store[
        hourly_store["week_start"] == pd.Timestamp(selected_week)
    ].copy()
    if selected_day_label != "All days":
        hourly_slice = hourly_slice[
            hourly_slice["business_date"].isin(slice_dates)
        ].copy()
    previous_hours = hourly_store[
        hourly_store["week_start"] == previous_week
    ].copy()
    if selected_day_label != "All days":
        previous_hours = previous_hours[
            previous_hours["business_date"].isin(previous_dates)
        ].copy()
else:
    hourly_slice = hourly_store
    previous_hours = hourly_store

current_splh = period_splh(hourly_slice)
previous_splh = period_splh(previous_hours)
splh_change = percent_change(current_splh, previous_splh)

if sales_skipped:
    st.warning(
        f"{sales_skipped:,} sales intervals had no readable "
        "start time and were skipped."
    )
if sales_error:
    st.warning(f"Hourly sales are unavailable: {sales_error}")
if shift_error:
    st.warning(f"Shift clocks are unavailable: {shift_error}")


# ---------------------------------------------------------
# Selected week or day
# ---------------------------------------------------------

slice_title = (
    format_week(selected_week)
    if selected_day_label == "All days"
    else f"{selected_day_label}, {format_week(selected_week)}"
)
st.subheader(slice_title)

lane_delta = format_seconds_delta(
    current_times.get("lane_total"),
    previous_times.get("lane_total"),
)
kpi_columns = st.columns(4)
kpi_columns[0].metric(
    "Cars",
    f"{int(current_times.get('cars', 0)):,}",
)
kpi_columns[1].metric(
    "Lane total",
    format_duration(current_times.get("lane_total")),
    delta=lane_delta,
    delta_color="inverse",
    help="Average time from entering the lane to leaving it.",
)
kpi_columns[2].metric(
    "Greet",
    format_duration(current_times.get("greet")),
    help=(
        "Average time until the first greeting. On this "
        "feed that time sits inside the menu-board time."
    ),
)
kpi_columns[3].metric(
    "SpLH",
    "—" if pd.isna(current_splh) else f"${current_splh:,.2f}",
    delta=(
        None
        if pd.isna(splh_change)
        else f"{splh_change:+.1f}%"
    ),
    help=(
        "Sales during this selection divided by labor hours "
        "clocked in the same hours. GM shifts are excluded."
    ),
)

menu_label = format_duration(current_times.get("menu_board"))
service_label = format_duration(current_times.get("service"))
queue_label = format_duration(current_times.get("lane_queue"))
st.caption(
    f"Menu board {menu_label}. Service {service_label}. "
    f"Lane queue {queue_label}. "
    "The change on lane total and SpLH is versus the "
    "previous week"
    + (
        "."
        if selected_day_label == "All days"
        else ", same weekday."
    )
)

if (
    selected_day_label == "All days"
    and not previous_cars.empty
    and week_cars["business_date"].nunique()
    < previous_cars["business_date"].nunique()
):
    st.caption(
        "The selected week has fewer days with drive-thru "
        "cars than the week before it, so that change "
        "compares a shorter period with a longer one."
    )


# ---------------------------------------------------------
# Drive-thru time by day and hour
# ---------------------------------------------------------

st.subheader("Drive-thru time by day")

if history_df.empty:
    st.info("No drive-thru departures fall in that history.")
else:
    day_cells = rollup_drive_thru_times(
        history_df,
        ["business_date", "hour"],
    )
    day_cells = with_hour_fields(day_cells)
    day_cells = add_duration_labels(day_cells)
    day_cells["minutes"] = day_cells["lane_total"] / 60.0
    history_dates = sorted(day_cells["business_date"].unique())
    years_span = (
        pd.Series(pd.to_datetime(history_dates)).dt.year.nunique() > 1
    )

    def history_day_label(business_date) -> str:
        stamp = pd.Timestamp(business_date)
        if years_span:
            return (
                f"{WEEKDAY_NAMES[stamp.weekday()][:3]} "
                f"{stamp:%m/%d/%y}"
            )
        return format_day(stamp)

    day_cells["day_label"] = day_cells["business_date"].map(
        history_day_label
    )
    day_order = [history_day_label(value) for value in history_dates]
    row_order = list(day_order)
    if len(day_order) > 1:
        pooled = rollup_drive_thru_times(history_df, ["hour"])
        pooled = with_hour_fields(pooled)
        pooled = add_duration_labels(pooled)
        pooled["minutes"] = pooled["lane_total"] / 60.0
        pooled["day_label"] = "All"
        day_cells = pd.concat([day_cells, pooled], ignore_index=True)
        row_order = [*day_order, "All"]

    scale_top = heatmap_top(day_cells["minutes"].max(skipna=True))
    day_cells["fill_color"] = day_cells["minutes"].map(
        lambda value: heatmap_color_hex(value, scale_top)
    )
    day_cells["label_color"] = day_cells["minutes"].map(
        lambda value: heatmap_label_color(value, scale_top)
    )
    legend_steps = 80
    legend_df = pd.DataFrame(
        {
            "minutes": [
                scale_top * step / legend_steps
                for step in range(legend_steps + 1)
            ]
        }
    )
    legend_df["minutes_end"] = legend_df["minutes"].shift(-1)
    legend_df = legend_df.dropna()
    legend_df["fill_color"] = legend_df["minutes"].map(
        lambda value: heatmap_color_hex(value, scale_top)
    )
    legend_ticks = [0, GOAL_MINUTES]
    if scale_top > GOAL_MINUTES + 0.05:
        legend_ticks.append(float(scale_top))

    heatmap_tooltip = [
        alt.Tooltip("day_label:N", title="Day"),
        alt.Tooltip("hour_label:N", title="Hour"),
        alt.Tooltip("lane_total_label:N", title="Lane total"),
        alt.Tooltip("menu_board_label:N", title="Menu board"),
        alt.Tooltip("greet_label:N", title="Greet"),
        alt.Tooltip("service_label:N", title="Service"),
        alt.Tooltip("lane_queue_label:N", title="Lane queue"),
        alt.Tooltip("cars:Q", title="Cars", format=",.0f"),
        alt.Tooltip(
            "avg_cars_in_lane:Q",
            title="Avg cars in lane",
            format=".1f",
        ),
    ]
    hour_sort = alt.SortField(field="hour_index")
    heatmap_base = alt.Chart(day_cells)
    heatmap_rects = (
        heatmap_base
        .transform_filter("isValid(datum.minutes)")
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X(
                "hour_label:N",
                title="Hour",
                sort=hour_sort,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "day_label:N",
                title=None,
                sort=row_order,
            ),
            color=alt.Color(
                "fill_color:N",
                scale=None,
                legend=None,
            ),
            tooltip=heatmap_tooltip,
        )
    )
    heatmap_labels = (
        heatmap_base
        .transform_filter("isValid(datum.minutes)")
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X(
                "hour_label:N",
                title="Hour",
                sort=hour_sort,
            ),
            y=alt.Y(
                "day_label:N",
                title=None,
                sort=row_order,
            ),
            text=alt.Text("lane_total_label:N"),
            color=alt.Color(
                "label_color:N",
                scale=None,
                legend=None,
            ),
        )
    )
    heatmap_legend = (
        alt.Chart(legend_df)
        .mark_rect()
        .encode(
            y=alt.Y(
                "minutes:Q",
                title="Minutes",
                scale=alt.Scale(domain=[0, scale_top], nice=False),
                axis=alt.Axis(format=".1f", values=legend_ticks),
            ),
            y2="minutes_end:Q",
            color=alt.Color(
                "fill_color:N",
                scale=None,
                legend=None,
            ),
        )
        .properties(width=18, height=max(len(row_order), 1) * 46)
    )
    heatmap = (
        alt.hconcat(
            alt.layer(heatmap_rects, heatmap_labels).properties(
                width=alt.Step(72),
                height=alt.Step(46),
            ),
            heatmap_legend,
        )
        .configure_view(strokeWidth=0, clip=False)
    )
    st.altair_chart(heatmap, use_container_width=False)

    history_from = history_df["business_date"].min()
    history_to = history_df["business_date"].max()
    history_caption = (
        "Each cell is the average lane time for cars that "
        "left during that hour. 4:00 and under is green, "
        "and darker green is a faster lane. Above 4:00 is "
        "red, and darker red is a slower lane. "
    )
    if len(day_order) > 1:
        history_caption += (
            "The All row pools every day in this history. "
        )
    st.caption(
        history_caption
        + f"History runs {history_from:%b %d, %Y} through "
        + f"{history_to:%b %d, %Y}."
    )


# ---------------------------------------------------------
# Hourly breakdown
# ---------------------------------------------------------

time_by_hour = rollup_drive_thru_times(slice_df, ["hour"])
if not hourly_slice.empty:
    splh_by_hour = (
        hourly_slice
        .groupby("hour", as_index=False)
        .agg(
            sales=("sales", "sum"),
            transactions=("transactions", "sum"),
            labor_hours=("labor_hours", "sum"),
        )
    )
    splh_by_hour["splh"] = (
        splh_by_hour["sales"]
        / splh_by_hour["labor_hours"].where(
            splh_by_hour["labor_hours"] > 0
        )
    )
else:
    splh_by_hour = pd.DataFrame(
        columns=["hour", "sales", "transactions", "labor_hours", "splh"]
    )

if time_by_hour.empty and splh_by_hour.empty:
    st.info("That selection has no drive-thru times or sales.")
else:
    by_hour = time_by_hour.merge(
        splh_by_hour,
        on="hour",
        how="outer",
    )
    by_hour = with_hour_fields(by_hour)

    table = by_hour.sort_values("hour_index").copy()
    table = table[
        (table["cars"].fillna(0) > 0)
        | (table["sales"].fillna(0) > 0)
        | (table["labor_hours"].fillna(0) > 0)
    ]
    if not table.empty:
        table_display = pd.DataFrame(
            {
                "Hour": table["hour_label"],
                "Cars": table["cars"].fillna(0).round().astype(int),
                "Lane total": table["lane_total"].map(format_duration),
                "Menu board": table["menu_board"].map(format_duration),
                "Greet": table["greet"].map(format_duration),
                "Service": table["service"].map(format_duration),
                "Lane queue": table["lane_queue"].map(format_duration),
                "Avg cars in lane": table["avg_cars_in_lane"].round(1),
                "Sales": table["sales"],
                "Transactions": table["transactions"],
                "Labor hours": table["labor_hours"],
                "SpLH": table["splh"],
            }
        )
        st.subheader("Hourly breakdown")
        st.dataframe(
            table_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cars": st.column_config.NumberColumn(format="%d"),
                "Avg cars in lane": st.column_config.NumberColumn(
                    format="%.1f",
                    help=(
                        "Average number of cars still stacked "
                        "when a car left the lane."
                    ),
                ),
                "Sales": st.column_config.NumberColumn(format="$%.2f"),
                "Transactions": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Labor hours": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "SpLH": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        st.caption(
            "Avg cars in lane is the stack still behind a car "
            "at departure, not the count of cars served."
        )


st.caption(
    f"Drive-thru times for {selected_location.lower()}, "
    f"weeks starting {selected_week_start_name}. "
    "Each average weights every car equally. "
    "Sales per labor hour uses the same hourly sales and "
    "shift clocks as the SpLH page."
)
