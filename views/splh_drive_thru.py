import importlib

import altair as alt
import pandas as pd
import streamlit as st

import revel_data

if not hasattr(revel_data, "load_hme_departures"):
    revel_data = importlib.reload(revel_data)

from revel_data import (
    BUSINESS_DAY_START_HOUR,
    DRIVE_THRU_TIME_COLUMNS,
    NOT_AVAILABLE,
    SHIFT_SUMMARY_TABLE,
    STORE_TIMEZONE,
    WEEKDAY_NAMES,
    MissingSecretError,
    get_table_columns,
    load_hme_departures,
    load_hourly_sales,
    load_shift_summary,
    percent_change,
    rollup_drive_thru_times,
)


# Weeks only. Each week is 7 days, counted backward from Date end.
INTERVALS = {
    "Weekly": {
        "days": 7,
        "max_duration": 52,
        "default_duration": 8,
        "unit": "weeks",
        "x_title": "Week",
        "duration_label": "Duration (weeks)",
        "summary_title": "Selected Week vs Previous Week",
    },
}

GREEN_SPLH = 70.0
GOAL_MINUTES = 4.0
# Hourly charts and the hourly table use this clock only.
DISPLAY_START_HOUR = 10
DISPLAY_END_HOUR = 23
SPLH_RED = ((103, 0, 13), (239, 59, 44))
SPLH_GREEN = ((102, 189, 99), (0, 68, 27))
LANE_RED = ((103, 0, 13), (239, 59, 44))
LANE_GREEN = ((102, 189, 99), (0, 68, 27))


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
    """Full date range shown in tooltips and metric help."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if period_days == 1:
        return f"{end:%b %d, %Y}"

    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def format_duration(seconds) -> str:
    """Render seconds as minutes and seconds, such as 3:12."""

    if seconds is None or pd.isna(seconds):
        return "—"

    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    minutes, remainder = divmod(total, 60)
    return f"{sign}{minutes}:{remainder:02d}"


def format_money(value) -> str:
    """Dollars and cents, or a dash when the value is missing."""

    if value is None or pd.isna(value):
        return "—"

    return f"${float(value):,.2f}"


def format_hours(value) -> str:
    """Labor hours with two decimals."""

    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):,.2f} hrs"


def signed_percent(current, previous):
    """Percent-change label, or None when it cannot be computed."""

    change = percent_change(current, previous)
    if pd.isna(change):
        return None
    return f"{change:+.1f}%"


def format_seconds_delta(current, previous):
    """Signed second change, or None when either side is missing."""

    if current is None or previous is None:
        return None
    if pd.isna(current) or pd.isna(previous):
        return None
    return f"{float(current) - float(previous):+.0f} sec"


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


def _label_color(fill, luminance_cutoff: float) -> str:
    if fill is None:
        return "#1a1a1a"
    red, green, blue = (_srgb_channel(part) for part in fill)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance < luminance_cutoff:
        return "#ffffff"
    return "#1a1a1a"


def splh_scale_top(rate_max: float) -> float:
    """Keep the green end of the scale above the entered break."""

    break_at = max(float(GREEN_SPLH), 0.0)
    if pd.isna(rate_max) or rate_max <= 0:
        return break_at + 0.01
    return max(float(rate_max), break_at + 0.01)


def splh_fill(value, top: float):
    """Red below the entered SpLH, and green from that amount up."""

    if value is None or pd.isna(value):
        return None
    amount = float(value)
    break_at = max(float(GREEN_SPLH), 0.0)
    if amount >= break_at:
        span = max(float(top) - break_at, 0.01)
        return _mix_color(
            SPLH_GREEN[0],
            SPLH_GREEN[1],
            (amount - break_at) / span,
        )
    if break_at <= 0:
        return _mix_color(SPLH_RED[0], SPLH_RED[1], 1.0)
    return _mix_color(SPLH_RED[0], SPLH_RED[1], amount / break_at)


def splh_color_hex(value, top: float):
    fill = splh_fill(value, top)
    if fill is None:
        return None
    return "#{:02x}{:02x}{:02x}".format(*fill)


def splh_label_color(value, top: float) -> str:
    return _label_color(splh_fill(value, top), 0.42)


def lane_scale_top(minutes_max: float) -> float:
    """Keep the entered lane goal on the scale."""

    break_at = max(float(GOAL_MINUTES), 0.0)
    if pd.isna(minutes_max) or minutes_max <= 0:
        return break_at
    return max(float(minutes_max), break_at)


def lane_fill(minutes, top: float):
    """Green at the entered goal or under, red when the lane is slower."""

    if minutes is None or pd.isna(minutes):
        return None
    amount = float(minutes)
    break_at = max(float(GOAL_MINUTES), 0.0)
    if break_at <= 0:
        if amount <= 0:
            return LANE_GREEN[0]
        span = max(float(top), 0.01)
        return _mix_color(LANE_RED[1], LANE_RED[0], amount / span)
    if amount <= break_at:
        return _mix_color(
            LANE_GREEN[1],
            LANE_GREEN[0],
            amount / break_at,
        )
    span = max(float(top) - break_at, 0.01)
    return _mix_color(
        LANE_RED[1],
        LANE_RED[0],
        (amount - break_at) / span,
    )


def lane_color_hex(minutes, top: float):
    fill = lane_fill(minutes, top)
    if fill is None:
        return None
    return "#{:02x}{:02x}{:02x}".format(*fill)


def lane_label_color(minutes, top: float) -> str:
    return _label_color(lane_fill(minutes, top), 0.30)


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
    hour, _minute = interval_start_parts(label)
    return hour


def rush_time_choices() -> list[tuple[str, int]]:
    """Clock times every 15 minutes, in business-day order."""

    choices = []
    for step in range((24 * 4) + 1):
        minutes_from_open = step * 15
        clock_minutes = (6 * 60 + minutes_from_open) % (24 * 60)
        hour = clock_minutes // 60
        minute = clock_minutes % 60
        suffix = "AM" if hour < 12 else "PM"
        display_hour = hour % 12 or 12
        label = f"{display_hour}:{minute:02d} {suffix}"
        if step == 24 * 4:
            label = "6:00 AM next day"
        choices.append((label, minutes_from_open))
    return choices


RUSH_TIME_CHOICES = rush_time_choices()
RUSH_TIME_LABELS = [label for label, _minutes in RUSH_TIME_CHOICES]
RUSH_TIME_MINUTES = dict(RUSH_TIME_CHOICES)


def rush_splh(
    quarter_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    start_minutes: int,
    end_minutes: int,
) -> float:
    """Sales per labor hour for one window on the business clock."""

    if end_minutes <= start_minutes:
        return NOT_AVAILABLE
    if quarter_df.empty and hour_df.empty:
        return NOT_AVAILABLE

    sales = 0.0
    if not quarter_df.empty:
        quarters = quarter_df.copy()
        quarters["quarter_start"] = (
            (quarters["hour"] - BUSINESS_DAY_START_HOUR) % 24
        ) * 60 + quarters["minute"]
        in_window = quarters[
            (quarters["quarter_start"] >= start_minutes)
            & (quarters["quarter_start"] < end_minutes)
        ]
        if not in_window.empty:
            sales = float(in_window["sales"].sum())

    labor_hours = 0.0
    if not hour_df.empty:
        hours = hour_df.copy()
        hour_start = (
            (hours["hour"] - BUSINESS_DAY_START_HOUR) % 24
        ) * 60
        overlap_minutes = (
            hour_start.add(60).clip(upper=end_minutes)
            - hour_start.clip(lower=start_minutes)
        ).clip(lower=0)
        labor_hours = float(
            (hours["labor_hours"] * overlap_minutes / 60).sum()
        )

    if not labor_hours:
        return NOT_AVAILABLE
    return sales / labor_hours


def preferred_location(names: pd.Series) -> str:
    """Pick one display name when sources spell a store differently."""

    unique = [str(name) for name in names.dropna().unique()]
    if not unique:
        return "Unknown"
    mixed_case = [name for name in unique if name != name.upper()]
    return sorted(mixed_case or unique)[0]


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
    """Split a shift into business-date and clock-hour pieces."""

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


@st.cache_data(ttl=300)
def labor_hours_by_hour(shift_df: pd.DataFrame) -> pd.DataFrame:
    """Place each shift onto the business hours it overlaps."""

    empty = pd.DataFrame(
        columns=[
            "location_key",
            "location",
            "business_date",
            "hour",
            "employee",
            "role",
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
        employee = str(getattr(shift, "employee", "Unknown") or "Unknown")
        role = str(getattr(shift, "role", "Unassigned") or "Unassigned")
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
                    "employee": employee,
                    "role": role,
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
            [
                "location_key",
                "location",
                "business_date",
                "hour",
                "employee",
                "role",
            ],
            as_index=False,
        )["labor_hours"]
        .sum()
    )


def labor_hour_totals(detail: pd.DataFrame) -> pd.DataFrame:
    """Labor hours per store, business date, and clock hour."""

    empty = pd.DataFrame(
        columns=[
            "location_key",
            "location",
            "business_date",
            "hour",
            "labor_hours",
        ]
    )
    if detail.empty:
        return empty
    return (
        detail
        .groupby(
            ["location_key", "location", "business_date", "hour"],
            as_index=False,
        )["labor_hours"]
        .sum()
    )


def replace_labor_hours(
    combined: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Swap in labor hours for the selected employees and roles."""

    frame = combined.drop(columns=["labor_hours"], errors="ignore")
    totals = labor_hour_totals(detail)
    if totals.empty:
        frame["labor_hours"] = 0.0
        return frame
    totals = totals.drop(columns=["location"], errors="ignore")
    frame = frame.merge(
        totals,
        on=["location_key", "business_date", "hour"],
        how="left",
    )
    frame["labor_hours"] = frame["labor_hours"].fillna(0.0)
    return frame


def employee_name_lists(
    detail: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Comma-separated employee names for each heatmap cell."""

    if detail.empty:
        return pd.DataFrame(columns=[*group_columns, "employees"])
    listed = (
        detail
        .groupby(group_columns, as_index=False)["employee"]
        .agg(
            lambda names: ", ".join(
                sorted(
                    {
                        str(name).strip()
                        for name in names
                        if str(name).strip()
                    }
                )
            )
        )
    )
    return listed.rename(columns={"employee": "employees"})


def attach_employee_names(
    grid: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Add the people clocked in for each week and hour."""

    labeled = grid.copy()
    labeled["employees"] = "—"
    if detail.empty or labeled.empty:
        return labeled

    by_period = employee_name_lists(
        detail,
        ["period_index", "hour"],
    )
    if "period_index" in labeled.columns and not by_period.empty:
        labeled = labeled.drop(columns=["employees"]).merge(
            by_period,
            on=["period_index", "hour"],
            how="left",
        )
    by_hour = employee_name_lists(detail, ["hour"])
    if "row_label" in labeled.columns and not by_hour.empty:
        hour_names = dict(
            zip(by_hour["hour"], by_hour["employees"])
        )
        all_rows = labeled["row_label"].eq("All")
        labeled.loc[all_rows, "employees"] = (
            labeled.loc[all_rows, "hour"].map(hour_names)
        )
    labeled["employees"] = labeled["employees"].fillna("—")
    labeled.loc[labeled["employees"].eq(""), "employees"] = "—"
    return labeled


def keep_multiselect(key: str, options: list[str]) -> list[str]:
    """Keep a multiselect on the options that still exist."""

    if key not in st.session_state:
        st.session_state[key] = options
        return options
    previous = list(st.session_state[key])
    kept = [item for item in previous if item in options]
    if previous and not kept:
        kept = list(options)
    st.session_state[key] = kept
    return kept


def empty_hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "location_key",
            "location",
            "business_date",
            "hour",
            "sales",
            "transactions",
            "labor_hours",
        ]
    )


def empty_quarter_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "location",
            "location_key",
            "business_date",
            "hour",
            "minute",
            "sales",
            "transactions",
        ]
    )


def prepare_hourly_sales(
    sales_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split 15-minute sales into clock hours and quarter rows."""

    if sales_df.empty or "time" not in sales_df.columns:
        return empty_hourly_frame(), empty_quarter_frame(), 0

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
        return empty_hourly_frame(), empty_quarter_frame(), skipped

    working["hour"] = working["hour"].astype(int)
    working["business_date"] = pd.to_datetime(
        working["business_date"]
    ).dt.normalize()
    if "transactions" not in working.columns:
        working["transactions"] = 0.0
    if "location" not in working.columns:
        working["location"] = "Unknown"

    clock_parts = working["time"].map(interval_start_parts)
    working["minute"] = pd.Series(
        [parts[1] for parts in clock_parts],
        index=working.index,
    )
    if (
        "interval_label" in working.columns
        and working["minute"].isna().any()
    ):
        fallback_parts = working["interval_label"].map(
            interval_start_parts
        )
        working["minute"] = working["minute"].fillna(
            pd.Series(
                [parts[1] for parts in fallback_parts],
                index=working.index,
            )
        )
    working["minute"] = working["minute"].fillna(0).astype(int)

    quarters = working[
        [
            "location",
            "location_key",
            "business_date",
            "hour",
            "minute",
            "sales",
            "transactions",
        ]
    ].copy()
    hourly = (
        working
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
    return hourly, quarters, skipped


def combine_sales_and_labor(
    sales_hours: pd.DataFrame,
    labor_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per store, business date, and clock hour."""

    if sales_hours.empty and labor_df.empty:
        return empty_hourly_frame()

    if sales_hours.empty:
        combined = labor_df.copy()
        combined["sales"] = 0.0
        combined["transactions"] = 0.0
    elif labor_df.empty:
        combined = sales_hours.copy()
        combined["labor_hours"] = 0.0
    else:
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
    return combined


def productivity(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
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
    if "hour" in grouped.columns:
        grouped = with_hour_fields(grouped)
    return grouped


def build_period_calendar(date_end, period_days: int, count: int) -> pd.DataFrame:
    """Equal periods ending on date_end. Index 0 is the latest."""

    date_end = pd.Timestamp(date_end).normalize()
    period_days = int(period_days)
    rows = []
    for period_index in range(int(count)):
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

    calendar = pd.DataFrame(rows)
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
    return (
        calendar
        .sort_values("period_end")
        .reset_index(drop=True)
    )


def rows_in_periods(
    frame: pd.DataFrame,
    date_end,
    period_days: int,
    duration: int,
) -> pd.DataFrame:
    """Keep rows inside the selected periods and tag each one."""

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


def compliance_rate(frame: pd.DataFrame):
    """Share of departures at or under the entered lane goal."""

    if frame.empty or "lane_total" not in frame.columns:
        return NOT_AVAILABLE
    times = frame["lane_total"].dropna()
    if times.empty:
        return NOT_AVAILABLE
    on_goal = (times / 60.0) <= GOAL_MINUTES
    return float(on_goal.sum()) / float(len(times))


def slice_period(frame: pd.DataFrame, period_index: int) -> pd.DataFrame:
    if frame.empty or "period_index" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["period_index"] == period_index].copy()


def build_period_metrics(
    hourly: pd.DataFrame,
    quarters: pd.DataFrame,
    cars: pd.DataFrame,
    calendar: pd.DataFrame,
    lunch_start: int,
    lunch_end: int,
    dinner_start: int,
    dinner_end: int,
) -> pd.DataFrame:
    """SpLH and drive-thru totals for every period in the calendar."""

    rows = []
    for period in calendar.itertuples(index=False):
        period_index = int(period.period_index)
        hours = slice_period(hourly, period_index)
        period_cars = slice_period(cars, period_index)
        period_quarters = slice_period(quarters, period_index)

        if hours.empty:
            sales = NOT_AVAILABLE
            labor_hours = NOT_AVAILABLE
            transactions = NOT_AVAILABLE
            splh = NOT_AVAILABLE
        else:
            sales = float(hours["sales"].sum())
            labor_hours = float(hours["labor_hours"].sum())
            transactions = float(hours["transactions"].sum())
            if labor_hours:
                splh = sales / labor_hours
            else:
                splh = NOT_AVAILABLE

        date_parts = []
        if not hours.empty:
            date_parts.append(hours["business_date"])
        if not period_cars.empty:
            date_parts.append(period_cars["business_date"])
        if date_parts:
            days_counted = int(pd.concat(date_parts).nunique())
        else:
            days_counted = 0

        times = period_rates(period_cars)
        rows.append(
            {
                "period_index": period_index,
                "days_counted": days_counted,
                "sales": sales,
                "labor_hours": labor_hours,
                "transactions": transactions,
                "splh": splh,
                "lunch_splh": rush_splh(
                    period_quarters,
                    hours,
                    lunch_start,
                    lunch_end,
                ),
                "dinner_splh": rush_splh(
                    period_quarters,
                    hours,
                    dinner_start,
                    dinner_end,
                ),
                "cars": times.get("cars", 0) if times else 0,
                "lane_total": times.get("lane_total", NOT_AVAILABLE),
                "menu_board": times.get("menu_board", NOT_AVAILABLE),
                "greet": times.get("greet", NOT_AVAILABLE),
                "service": times.get("service", NOT_AVAILABLE),
                "lane_queue": times.get("lane_queue", NOT_AVAILABLE),
                "avg_cars_in_lane": times.get(
                    "avg_cars_in_lane",
                    NOT_AVAILABLE,
                ),
                "compliance": compliance_rate(period_cars),
            }
        )

    metrics = pd.DataFrame(rows)
    return calendar.merge(metrics, on="period_index", how="left")


def add_duration_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column_name in DRIVE_THRU_TIME_COLUMNS:
        if column_name not in frame.columns:
            continue
        frame[f"{column_name}_label"] = frame[column_name].map(
            format_duration
        )
    return frame


def paint_heatmap(frame, value_column, fill_hex, label_color, top):
    painted = frame.copy()
    painted["fill_color"] = painted[value_column].map(
        lambda value: fill_hex(value, top)
    )
    painted["label_color"] = painted[value_column].map(
        lambda value: label_color(value, top)
    )
    return painted


def heatmap_legend(top, fill_hex, tick_values, title, number_format, row_count):
    steps = 80
    legend = pd.DataFrame(
        {"value": [top * step / steps for step in range(steps + 1)]}
    )
    legend["value_end"] = legend["value"].shift(-1)
    legend = legend.dropna()
    legend["fill_color"] = legend["value"].map(
        lambda value: fill_hex(value, top)
    )
    return (
        alt.Chart(legend)
        .mark_rect()
        .encode(
            y=alt.Y(
                "value:Q",
                title=title,
                scale=alt.Scale(domain=[0, top], nice=False),
                axis=alt.Axis(
                    format=number_format,
                    values=tick_values,
                ),
            ),
            y2="value_end:Q",
            color=alt.Color("fill_color:N", scale=None, legend=None),
        )
        .properties(width=18, height=max(row_count, 1) * 46)
    )


def render_heatmap(
    frame: pd.DataFrame,
    value_field: str,
    text_field: str,
    text_format: str | None,
    row_order: list[str],
    hour_order: list[str],
    tooltip: list,
    legend_chart,
):
    """Week-by-hour heatmap. Both charts share these axis domains."""

    hour_axis_enc = alt.X(
        "hour_label:N",
        title="Hour",
        sort=hour_order,
        scale=alt.Scale(domain=hour_order),
        axis=alt.Axis(labelAngle=0, labelOverlap=False),
    )
    row_axis_enc = alt.Y(
        "row_label:N",
        title=None,
        sort=row_order,
        scale=alt.Scale(domain=row_order),
    )
    base = alt.Chart(frame).transform_filter(
        f"isValid(datum.{value_field})"
    )
    if text_format:
        text = alt.Text(f"{text_field}:Q", format=text_format)
    else:
        text = alt.Text(f"{text_field}:N")

    rects = (
        base
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=hour_axis_enc,
            y=row_axis_enc,
            color=alt.Color("fill_color:N", scale=None, legend=None),
            tooltip=tooltip,
        )
    )
    labels = (
        base
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(
            x=hour_axis_enc,
            y=row_axis_enc,
            text=text,
            color=alt.Color("label_color:N", scale=None, legend=None),
        )
    )
    return (
        alt.hconcat(
            alt.layer(rects, labels).properties(
                width=alt.Step(72),
                height=alt.Step(46),
            ),
            legend_chart,
        )
        .configure_view(strokeWidth=0, clip=False)
    )


def display_hour_frame() -> pd.DataFrame:
    """10 AM through 11 PM, in business-day order."""

    return with_hour_fields(
        pd.DataFrame(
            {
                "hour": list(
                    range(DISPLAY_START_HOUR, DISPLAY_END_HOUR + 1)
                )
            }
        )
    )


def keep_weekday(frame: pd.DataFrame, weekday_name: str) -> pd.DataFrame:
    """Keep one weekday, or every day when the choice is All days."""

    if (
        weekday_name == "All days"
        or frame.empty
        or "business_date" not in frame.columns
    ):
        return frame

    weekday_index = WEEKDAY_NAMES.index(weekday_name)
    dates = pd.to_datetime(frame["business_date"])
    return frame.loc[dates.dt.weekday.eq(weekday_index)].copy()


def in_display_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose clock hour is from 10 AM through 11 PM."""

    if frame.empty or "hour" not in frame.columns:
        return frame
    hours = pd.to_numeric(frame["hour"], errors="coerce")
    return frame.loc[
        hours.between(DISPLAY_START_HOUR, DISPLAY_END_HOUR)
    ].copy()


def hour_axis(label_order: list[str]) -> alt.X:
    return alt.X(
        "hour_label:N",
        title="Hour",
        sort=label_order,
        scale=alt.Scale(domain=label_order),
        axis=alt.Axis(labelAngle=0, labelOverlap=False),
    )


st.title("SpLH & Drive-Thru")
st.caption(
    "Sales per labor hour and drive-thru lane time for the "
    "same weeks. Hourly charts run from 10:00 AM through "
    "11:00 PM. GM shifts are excluded."
)

sales_error = None
shift_error = None
hme_error = None

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

try:
    hme_df = load_hme_departures()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    hme_df = pd.DataFrame()
    hme_error = str(error)

if not shift_df.empty:
    if "role" not in shift_df.columns:
        shift_df["role"] = "Unassigned"
    shift_df["role"] = (
        shift_df["role"].fillna("Unassigned").astype(str).str.strip()
    )
    shift_df.loc[shift_df["role"] == "", "role"] = "Unassigned"
    if "employee" not in shift_df.columns:
        shift_df["employee"] = "Unknown"
    shift_df["employee"] = (
        shift_df["employee"].fillna("Unknown").astype(str).str.strip()
    )
    shift_df.loc[shift_df["employee"] == "", "employee"] = "Unknown"
    shift_df = shift_df[shift_df["role"].str.upper() != "GM"].copy()

sales_hours, quarter_df, sales_skipped = prepare_hourly_sales(sales_df)
labor_error = None
try:
    labor_detail = labor_hours_by_hour(shift_df)
    labor_df = labor_hour_totals(labor_detail)
except Exception as error:
    labor_detail = pd.DataFrame()
    labor_df = pd.DataFrame()
    labor_error = str(error)

if labor_df.empty and not shift_df.empty:
    try:
        column_names = get_table_columns(SHIFT_SUMMARY_TABLE)
    except Exception:
        column_names = []
    if column_names:
        labor_error = (
            "Labor hours can't be placed on the clock without "
            "a clock-in and clock-out. Columns on "
            f"{SHIFT_SUMMARY_TABLE}: " + ", ".join(column_names)
        )
    elif labor_error is None:
        labor_error = (
            "Labor hours can't be placed on the clock without "
            "a clock-in and clock-out."
        )

combined = combine_sales_and_labor(sales_hours, labor_df)
if hme_df.empty:
    hme_df = pd.DataFrame(
        columns=[
            "location_key",
            "location",
            "business_date",
            "hour",
            "lane_total",
            "total_cars",
        ]
    )

name_frames = []
for source in (combined, quarter_df, hme_df):
    if (
        not source.empty
        and "location_key" in source.columns
        and "location" in source.columns
    ):
        name_frames.append(source[["location_key", "location"]])

if name_frames:
    location_names = (
        pd.concat(name_frames, ignore_index=True)
        .groupby("location_key")["location"]
        .agg(preferred_location)
    )
    for source_name in ("combined", "quarter_df", "hme_df"):
        source = locals()[source_name]
        if source.empty or "location_key" not in source.columns:
            continue
        source = source.copy()
        source["location"] = (
            source["location_key"]
            .map(location_names)
            .fillna(source["location"])
        )
        if source_name == "combined":
            combined = source
        elif source_name == "quarter_df":
            quarter_df = source
        else:
            hme_df = source

if combined.empty and (
    hme_df.empty or hme_df["business_date"].dropna().empty
):
    st.warning(
        "No hourly sales, shift clocks, or drive-thru "
        "departures were returned."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    set(combined["location"].dropna().tolist())
    | set(hme_df["location"].dropna().tolist())
)
selected_location = st.sidebar.selectbox(
    "Location",
    options=["All Locations", *available_locations],
)

if selected_location != "All Locations":
    combined = combined[
        combined["location"] == selected_location
    ].copy()
    if not quarter_df.empty:
        quarter_df = quarter_df[
            quarter_df["location"] == selected_location
        ].copy()
    hme_df = hme_df[hme_df["location"] == selected_location].copy()

if not labor_detail.empty and selected_location != "All Locations":
    location_keys = set()
    if not combined.empty and "location_key" in combined.columns:
        location_keys.update(combined["location_key"].dropna().tolist())
    if not hme_df.empty and "location_key" in hme_df.columns:
        location_keys.update(hme_df["location_key"].dropna().tolist())
    labor_detail = labor_detail[
        labor_detail["location_key"].isin(location_keys)
    ].copy()

if combined.empty and hme_df.empty:
    st.warning("No sales, labor, or drive-thru times match that location.")
    st.stop()

date_parts = []
if not combined.empty:
    date_parts.append(combined["business_date"])
if not hme_df.empty:
    date_parts.append(pd.to_datetime(hme_df["business_date"]))
available_dates = pd.concat(date_parts).dropna()
min_date = available_dates.min().date()
max_date = available_dates.max().date()
date_key = "splh_drive_thru_date_end"

if date_key not in st.session_state:
    st.session_state[date_key] = max_date
else:
    chosen_date = st.session_state[date_key]
    if chosen_date < min_date or chosen_date > max_date:
        st.session_state[date_key] = min(
            max(chosen_date, min_date),
            max_date,
        )

interval = INTERVALS["Weekly"]
period_days = interval["days"]

date_end = st.sidebar.date_input(
    "Date end",
    min_value=min_date,
    max_value=max_date,
    key=date_key,
    help=(
        "Last business date included in the summary, "
        "the charts, and the tables."
    ),
)
duration = st.sidebar.number_input(
    interval["duration_label"],
    min_value=1,
    max_value=interval["max_duration"],
    value=interval["default_duration"],
    step=1,
    key="splh_drive_thru_duration_Weekly",
    help=(
        "How many weeks to include, ending on Date end. "
        "Each week is 7 days. Maximum is "
        f"{interval['max_duration']}."
    ),
)
duration = int(duration)
st.sidebar.caption("Each week is 7 days, ending on Date end.")

selected_weekday = st.sidebar.selectbox(
    "Weekday",
    options=["All days", *WEEKDAY_NAMES],
    help=(
        "Limit each week to that weekday. "
        "All days keeps the full 7 days."
    ),
)

if not labor_detail.empty:
    available_roles = sorted(
        labor_detail["role"].dropna().unique(),
        key=str.lower,
    )
    keep_multiselect("splh_drive_thru_roles", available_roles)
    selected_roles = st.sidebar.multiselect(
        "Roles",
        options=available_roles,
        key="splh_drive_thru_roles",
        placeholder="Select roles",
        help=(
            "Labor hours and the employee names use these "
            "roles. Sales and drive-thru times stay for the "
            "whole selection."
        ),
    )
    if not selected_roles:
        st.info("Select at least one role.")
        st.stop()
    labor_detail = labor_detail[
        labor_detail["role"].isin(selected_roles)
    ].copy()

    available_employees = sorted(
        labor_detail["employee"].dropna().unique(),
        key=str.lower,
    )
    keep_multiselect(
        "splh_drive_thru_employees",
        available_employees,
    )
    selected_employees = st.sidebar.multiselect(
        "Employees",
        options=available_employees,
        key="splh_drive_thru_employees",
        placeholder="Select employees",
        help=(
            "Labor hours and the employee names use these "
            "people. Sales and drive-thru times stay for the "
            "whole selection."
        ),
    )
    if not selected_employees:
        st.info("Select at least one employee.")
        st.stop()
    labor_detail = labor_detail[
        labor_detail["employee"].isin(selected_employees)
    ].copy()
    combined = replace_labor_hours(combined, labor_detail)

st.sidebar.header("Color breaks")
GREEN_SPLH = float(
    st.sidebar.number_input(
        "SpLH green ($)",
        min_value=0.0,
        value=70.0,
        step=1.0,
        key="splh_drive_thru_green_splh",
        help=(
            "Hourly SpLH at or above this amount is green "
            "on the heatmap. Below it is red."
        ),
    )
)
GOAL_MINUTES = float(
    st.sidebar.number_input(
        "Lane goal (minutes)",
        min_value=0.0,
        value=4.0,
        step=0.25,
        format="%.2f",
        key="splh_drive_thru_lane_goal",
        help=(
            "Lane time at or under this is green, and it "
            "is the compliance cutoff. 4.00 is 4:00. "
            "Each step is 15 seconds."
        ),
    )
)
splh_break_label = (
    f"${GREEN_SPLH:,.0f}"
    if float(GREEN_SPLH).is_integer()
    else f"${GREEN_SPLH:,.2f}"
)
lane_goal_label = format_duration(GOAL_MINUTES * 60)
st.sidebar.caption(
    f"Green SpLH starts at {splh_break_label}. "
    f"Green lane time is {lane_goal_label} or under."
)

st.sidebar.header("Rush windows")
lunch_start_label = st.sidebar.selectbox(
    "Lunch rush start",
    options=RUSH_TIME_LABELS,
    index=RUSH_TIME_LABELS.index("11:00 AM"),
)
lunch_end_label = st.sidebar.selectbox(
    "Lunch rush end",
    options=RUSH_TIME_LABELS,
    index=RUSH_TIME_LABELS.index("1:00 PM"),
)
dinner_start_label = st.sidebar.selectbox(
    "Dinner rush start",
    options=RUSH_TIME_LABELS,
    index=RUSH_TIME_LABELS.index("5:00 PM"),
)
dinner_end_label = st.sidebar.selectbox(
    "Dinner rush end",
    options=RUSH_TIME_LABELS,
    index=RUSH_TIME_LABELS.index("8:00 PM"),
)

if sales_skipped:
    st.warning(
        f"{sales_skipped:,} sales intervals had no readable "
        "start time and were skipped."
    )
if sales_error:
    st.warning(f"Hourly sales are unavailable: {sales_error}")
if shift_error:
    st.warning(f"Shift clocks are unavailable: {shift_error}")
if labor_error and combined["labor_hours"].fillna(0).sum() == 0:
    st.warning(labor_error)
if hme_error:
    st.warning(f"Drive-thru times are unavailable: {hme_error}")

if date_end is None:
    st.warning("Pick a date end.")
    st.stop()

# One extra period supplies the comparison for the latest bucket.
calendar = build_period_calendar(date_end, period_days, duration + 1)
combined = rows_in_periods(combined, date_end, period_days, duration + 1)
labor_detail = rows_in_periods(
    labor_detail,
    date_end,
    period_days,
    duration + 1,
)
quarter_df = rows_in_periods(
    quarter_df,
    date_end,
    period_days,
    duration + 1,
)
hme_df = rows_in_periods(hme_df, date_end, period_days, duration + 1)
combined = keep_weekday(combined, selected_weekday)
labor_detail = keep_weekday(labor_detail, selected_weekday)
quarter_df = keep_weekday(quarter_df, selected_weekday)
hme_df = keep_weekday(hme_df, selected_weekday)

metrics = build_period_metrics(
    combined,
    quarter_df,
    hme_df,
    calendar,
    RUSH_TIME_MINUTES[lunch_start_label],
    RUSH_TIME_MINUTES[lunch_end_label],
    RUSH_TIME_MINUTES[dinner_start_label],
    RUSH_TIME_MINUTES[dinner_end_label],
)
visible = metrics.loc[metrics["period_index"] < duration].copy()
current = metrics.loc[metrics["period_index"] == 0].iloc[0]
previous_rows = metrics.loc[metrics["period_index"] == 1]
previous = previous_rows.iloc[0] if not previous_rows.empty else None

window_start = pd.Timestamp(visible["period_start"].min())
window_end = pd.Timestamp(date_end)
if window_start.normalize() == window_end.normalize():
    window_text = f"{window_end:%b %d, %Y}"
else:
    window_text = f"{window_start:%b %d, %Y} – {window_end:%b %d, %Y}"

rush_is_valid = (
    RUSH_TIME_MINUTES[lunch_end_label]
    > RUSH_TIME_MINUTES[lunch_start_label]
    and RUSH_TIME_MINUTES[dinner_end_label]
    > RUSH_TIME_MINUTES[dinner_start_label]
)


# ---------------------------------------------------------
# Selected period vs the period before it
# ---------------------------------------------------------

st.subheader(interval["summary_title"])
weekday_text = (
    ""
    if selected_weekday == "All days"
    else f" · {selected_weekday}s"
)
st.caption(
    f"{window_text} · {duration} {interval['unit']}"
    f"{weekday_text} · {selected_location}"
)

if selected_weekday == "All days":
    current_range = (
        f"{current['period_range']} "
        f"({int(current['days_counted'])} of {period_days} days)"
    )
else:
    current_range = f"{selected_weekday}, {current['period_range']}"
summary_columns = st.columns(3)
summary_columns[0].metric(
    "Sales",
    format_money(current["sales"]),
    delta=signed_percent(
        current["sales"],
        None if previous is None else previous["sales"],
    ),
    help=current_range,
)
summary_columns[1].metric(
    "Labor Hours",
    format_hours(current["labor_hours"]),
    help=current_range + ". GM shifts are excluded.",
)
summary_columns[2].metric(
    "SpLH",
    format_money(current["splh"]),
    delta=signed_percent(
        current["splh"],
        None if previous is None else previous["splh"],
    ),
    help=(
        current_range
        + ". Sales divided by hourly labor hours."
    ),
)

rush_columns = st.columns(2)
rush_columns[0].metric(
    "Lunch SpLH",
    format_money(current["lunch_splh"]),
    delta=signed_percent(
        current["lunch_splh"],
        None if previous is None else previous["lunch_splh"],
    ),
    help=(
        f"{lunch_start_label} – {lunch_end_label}. {current_range}."
    ),
)
rush_columns[1].metric(
    "Dinner SpLH",
    format_money(current["dinner_splh"]),
    delta=signed_percent(
        current["dinner_splh"],
        None if previous is None else previous["dinner_splh"],
    ),
    help=(
        f"{dinner_start_label} – {dinner_end_label}. {current_range}."
    ),
)

lane_delta = None
compliance_delta = None
if previous is not None:
    lane_delta = format_seconds_delta(
        current["lane_total"],
        previous["lane_total"],
    )
    if (
        not pd.isna(current["compliance"])
        and not pd.isna(previous["compliance"])
    ):
        points = (
            float(current["compliance"]) - float(previous["compliance"])
        ) * 100
        compliance_delta = f"{points:+.1f} pts"

lane_columns = st.columns(3)
lane_columns[0].metric(
    "Cars",
    "—"
    if pd.isna(current["cars"])
    else f"{int(current['cars']):,}",
    delta=signed_percent(
        current["cars"],
        None if previous is None else previous["cars"],
    ),
    help=current_range,
)
lane_columns[1].metric(
    "Lane total",
    format_duration(current["lane_total"]),
    delta=lane_delta,
    delta_color="inverse",
    help="Average time from entering the lane to leaving it.",
)
compliance_value = current["compliance"]
lane_columns[2].metric(
    "% Compliance",
    "—"
    if pd.isna(compliance_value)
    else f"{float(compliance_value) * 100:.1f}%",
    delta=compliance_delta,
    help=(
        "Share of cars in this period whose lane total "
        f"was {lane_goal_label} or under."
    ),
)

st.caption(
    f"Menu board {format_duration(current['menu_board'])}. "
    f"Greet {format_duration(current['greet'])}. "
    f"Service {format_duration(current['service'])}. "
    f"Lane queue {format_duration(current['lane_queue'])}. "
    "Changes compare this period with the previous one."
)

if not rush_is_valid:
    st.caption(
        "A rush end time has to be later than its start "
        "time. Overnight windows can end at 6:00 AM next day."
    )
else:
    st.caption(
        f"Lunch SpLH is {lunch_start_label}–{lunch_end_label}. "
        f"Dinner SpLH is {dinner_start_label}–{dinner_end_label}."
    )

if int(current["days_counted"]) == 0:
    if selected_weekday == "All days":
        st.info(f"No activity falls in {current['period_range']}.")
    else:
        st.info(
            f"No {selected_weekday} activity falls in "
            f"{current['period_range']}."
        )
elif (
    selected_weekday == "All days"
    and int(current["days_counted"]) < period_days
):
    st.caption(
        f"The period ending {pd.Timestamp(current['period_end']):%b %d, %Y} "
        f"has {int(current['days_counted'])} of {period_days} day(s) "
        "of data, so its totals are short."
    )

if previous is None or int(previous["days_counted"]) == 0:
    if selected_weekday == "All days":
        st.info(
            "The previous week has no sales, labor, or drive-thru "
            "cars, so there is nothing to compare against yet."
        )
    else:
        st.info(
            f"The previous week has no {selected_weekday} sales, "
            "labor, or drive-thru cars, so there is nothing to "
            "compare against yet."
        )


window_hourly = combined.loc[
    combined["period_index"] < duration
].copy() if "period_index" in combined.columns else combined.iloc[0:0]
window_cars = hme_df.loc[
    hme_df["period_index"] < duration
].copy() if "period_index" in hme_df.columns else hme_df.iloc[0:0]


def period_rows() -> pd.DataFrame:
    """Visible periods, oldest first, for heatmap rows."""

    return (
        visible[["period_index", "period_label", "period_range"]]
        .sort_values("period_index", ascending=False)
        .copy()
    )


# ---------------------------------------------------------
# SpLH and drive-thru heatmaps, on the same weeks and hours
# ---------------------------------------------------------

row_meta = period_rows()
row_order = row_meta["period_label"].tolist()
if len(row_order) > 1:
    row_order = [*row_order, "All"]

shared_hours = display_hour_frame()
hour_order = shared_hours["hour_label"].tolist()
window_hourly = in_display_window(window_hourly)
window_cars = in_display_window(window_cars)
window_labor = labor_detail.loc[
    labor_detail["period_index"] < duration
].copy() if (
    not labor_detail.empty and "period_index" in labor_detail.columns
) else labor_detail.iloc[0:0]
window_labor = in_display_window(window_labor)

st.subheader("SpLH by week")

if not hour_order:
    st.info(
        "No hourly sales, labor hours, or drive-thru departures "
        "fall in the selected weeks."
    )
else:
    if window_hourly.empty:
        splh_cells = pd.DataFrame(
            columns=["period_index", "hour", "sales_per_labor_hour"]
        )
    else:
        splh_cells = productivity(
            window_hourly,
            ["period_index", "hour"],
        )
    splh_grid = row_meta.merge(shared_hours, how="cross")
    if not splh_cells.empty:
        splh_grid = splh_grid.merge(
            splh_cells.drop(
                columns=["hour_index", "hour_label"],
                errors="ignore",
            ),
            on=["period_index", "hour"],
            how="left",
        )
    else:
        splh_grid["sales_per_labor_hour"] = NOT_AVAILABLE
    splh_grid["row_label"] = splh_grid["period_label"]

    if len(row_meta) > 1:
        pooled = (
            productivity(window_hourly, ["hour"])
            if not window_hourly.empty
            else pd.DataFrame(columns=["hour"])
        )
        all_grid = shared_hours.copy()
        all_grid["row_label"] = "All"
        all_grid["period_range"] = window_text
        if "sales_per_labor_hour" in pooled.columns:
            all_grid = all_grid.merge(
                pooled.drop(
                    columns=["hour_index", "hour_label"],
                    errors="ignore",
                ),
                on="hour",
                how="left",
            )
        splh_grid = pd.concat([splh_grid, all_grid], ignore_index=True)

    splh_grid = attach_employee_names(splh_grid, window_labor)

    scale_top = splh_scale_top(
        splh_grid["sales_per_labor_hour"].max(skipna=True)
    )
    splh_grid = paint_heatmap(
        splh_grid,
        "sales_per_labor_hour",
        splh_color_hex,
        splh_label_color,
        scale_top,
    )
    legend_ticks = [0, GREEN_SPLH, scale_top]
    splh_heatmap = render_heatmap(
        splh_grid,
        "sales_per_labor_hour",
        "sales_per_labor_hour",
        "$.0f",
        row_order,
        hour_order,
        [
            alt.Tooltip("row_label:N", title="Week"),
            alt.Tooltip("period_range:N", title="Dates"),
            alt.Tooltip("hour_label:N", title="Hour"),
            alt.Tooltip(
                "sales_per_labor_hour:Q",
                title="SpLH",
                format="$,.2f",
            ),
            alt.Tooltip("sales:Q", title="Sales", format="$,.2f"),
            alt.Tooltip(
                "labor_hours:Q",
                title="Labor Hours",
                format=",.2f",
            ),
            alt.Tooltip("employees:N", title="Employees"),
            alt.Tooltip(
                "transactions:Q",
                title="Transactions",
                format=",.0f",
            ),
        ],
        heatmap_legend(
            scale_top,
            splh_color_hex,
            legend_ticks,
            "SpLH",
            "$,.0f",
            len(row_order),
        ),
    )
    st.altair_chart(splh_heatmap, use_container_width=False)
    splh_caption = (
        "Each cell is sales during that hour divided by labor "
        "hours clocked during that hour. Rates under "
        f"{splh_break_label} are red. Rates of {splh_break_label} "
        "and above are green, darker as the rate rises. "
    )
    if len(row_order) > 1:
        splh_caption += "The All row pools every week in the selection. "
    st.caption(
        splh_caption
        + "Both heatmaps show 10:00 AM through 11:00 PM, "
        + "with the same weeks. Employee names are the "
        + "people clocked in during that hour."
    )

    st.subheader("Drive-thru time by week")
    if (
        window_cars.empty
        or "hour" not in window_cars.columns
        or "lane_total" not in window_cars.columns
    ):
        lane_hours = pd.DataFrame()
        lane_cells = pd.DataFrame()
    else:
        lane_hours = window_cars.dropna(subset=["hour"]).copy()
        if lane_hours.empty:
            lane_cells = pd.DataFrame()
        else:
            lane_hours["hour"] = lane_hours["hour"].astype(int)
            lane_cells = rollup_drive_thru_times(
                lane_hours,
                ["period_index", "hour"],
            )
            lane_cells = with_hour_fields(lane_cells)
            lane_cells = add_duration_labels(lane_cells)
            lane_cells["minutes"] = lane_cells["lane_total"] / 60.0

    lane_grid = row_meta.merge(shared_hours, how="cross")
    if lane_cells.empty:
        lane_grid["minutes"] = NOT_AVAILABLE
        lane_grid["lane_total_label"] = pd.NA
    else:
        lane_grid = lane_grid.merge(
            lane_cells.drop(
                columns=["hour_index", "hour_label"],
                errors="ignore",
            ),
            on=["period_index", "hour"],
            how="left",
        )
    lane_grid["row_label"] = lane_grid["period_label"]

    if len(row_meta) > 1:
        all_grid = shared_hours.copy()
        all_grid["row_label"] = "All"
        all_grid["period_range"] = window_text
        if not lane_hours.empty:
            pooled_lane = rollup_drive_thru_times(lane_hours, ["hour"])
            pooled_lane = with_hour_fields(pooled_lane)
            pooled_lane = add_duration_labels(pooled_lane)
            pooled_lane["minutes"] = pooled_lane["lane_total"] / 60.0
            all_grid = all_grid.merge(
                pooled_lane.drop(
                    columns=["hour_index", "hour_label"],
                    errors="ignore",
                ),
                on="hour",
                how="left",
            )
        else:
            all_grid["minutes"] = NOT_AVAILABLE
            all_grid["lane_total_label"] = pd.NA
        lane_grid = pd.concat([lane_grid, all_grid], ignore_index=True)

    lane_grid = attach_employee_names(lane_grid, window_labor)

    lane_scale = lane_scale_top(lane_grid["minutes"].max(skipna=True))
    lane_grid = paint_heatmap(
        lane_grid,
        "minutes",
        lane_color_hex,
        lane_label_color,
        lane_scale,
    )
    lane_ticks = [0, GOAL_MINUTES]
    if lane_scale > GOAL_MINUTES + 0.05:
        lane_ticks.append(float(lane_scale))
    lane_heatmap = render_heatmap(
        lane_grid,
        "minutes",
        "lane_total_label",
        None,
        row_order,
        hour_order,
        [
            alt.Tooltip("row_label:N", title="Week"),
            alt.Tooltip("period_range:N", title="Dates"),
            alt.Tooltip("hour_label:N", title="Hour"),
            alt.Tooltip("lane_total_label:N", title="Lane total"),
            alt.Tooltip("menu_board_label:N", title="Menu board"),
            alt.Tooltip("greet_label:N", title="Greet"),
            alt.Tooltip("service_label:N", title="Service"),
            alt.Tooltip("lane_queue_label:N", title="Lane queue"),
            alt.Tooltip("cars:Q", title="Cars", format=",.0f"),
            alt.Tooltip(
                "avg_cars_in_lane:Q",
                title="Avg cars in lane per departure event",
                format=".2f",
            ),
            alt.Tooltip("employees:N", title="Employees"),
        ],
        heatmap_legend(
            lane_scale,
            lane_color_hex,
            lane_ticks,
            "Minutes",
            ".1f",
            len(row_order),
        ),
    )
    st.altair_chart(lane_heatmap, use_container_width=False)
    lane_caption = (
        "Each cell is the average lane time for cars that "
        f"left during that hour. {lane_goal_label} and under "
        "is green, and darker green is a faster lane. Above "
        f"{lane_goal_label} is red, and darker red is a "
        "slower lane. "
    )
    if len(row_order) > 1:
        lane_caption += "The All row pools every week in the selection. "
    st.caption(
        lane_caption
        + "Both heatmaps show 10:00 AM through 11:00 PM, "
        + "with the same weeks. Employee names are the "
        + "people clocked in during that hour. "
        + f"Showing {window_text}."
    )

if (
    not window_hourly.empty
    and (
        (window_hourly["sales"] > 0)
        | (window_hourly["labor_hours"] > 0)
    ).any()
):
    active_hours = productivity(window_hourly, ["hour"])
    active_hours = active_hours.loc[
        (active_hours["sales"] > 0) | (active_hours["labor_hours"] > 0)
    ].sort_values("hour_index")
    hour_label_order = hour_order
    rate_tooltip = [
        alt.Tooltip("hour_label:N", title="Hour"),
        alt.Tooltip(
            "sales_per_labor_hour:Q",
            title="SpLH",
            format="$,.2f",
        ),
        alt.Tooltip("sales:Q", title="Sales", format="$,.2f"),
        alt.Tooltip("labor_hours:Q", title="Labor Hours", format=",.2f"),
        alt.Tooltip("transactions:Q", title="Transactions", format=",.0f"),
    ]

    labor_bars = (
        alt.Chart(active_hours)
        .mark_bar(color="#1f77b4", opacity=0.75)
        .encode(
            x=hour_axis(hour_label_order),
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
            x=hour_axis(hour_label_order),
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
    st.subheader("Sales and Labor Hours")
    st.altair_chart(
        alt.layer(labor_bars, sales_line)
        .resolve_scale(y="independent")
        .properties(height=360),
        use_container_width=True,
    )
    st.caption(
        "Bars are labor hours on the left axis. The line is "
        f"sales on the right axis, from 10:00 AM through "
        f"11:00 PM, pooled across {window_text}."
    )

    share_hours = active_hours.copy()
    labor_total = float(share_hours["labor_hours"].fillna(0).sum())
    sales_total = float(share_hours["sales"].fillna(0).sum())
    if labor_total:
        share_hours["labor_hour_percent"] = (
            share_hours["labor_hours"].fillna(0) / labor_total * 100
        )
    else:
        share_hours["labor_hour_percent"] = NOT_AVAILABLE
    if sales_total:
        share_hours["sales_percent"] = (
            share_hours["sales"].fillna(0) / sales_total * 100
        )
    else:
        share_hours["sales_percent"] = NOT_AVAILABLE

    share_series_order = [
        "Labor Hours % of Total",
        "Sales % of Total",
    ]
    share_long = pd.concat(
        [
            share_hours.assign(
                series=share_series_order[0],
                percent=share_hours["labor_hour_percent"],
            ),
            share_hours.assign(
                series=share_series_order[1],
                percent=share_hours["sales_percent"],
            ),
        ],
        ignore_index=True,
    )
    share_long["percent_label"] = share_long["percent"].map(
        lambda value: "" if pd.isna(value) else f"{value:.1f}%"
    )
    percent_max = share_long["percent"].max(skipna=True)
    if pd.isna(percent_max) or percent_max <= 0:
        percent_max = 1.0
    percent_scale = alt.Scale(domain=[0, float(percent_max) * 1.2])
    percent_axis = alt.Axis(format=".1f", labelExpr="datum.value + '%'")
    share_bars = (
        alt.Chart(share_long)
        .mark_bar()
        .encode(
            x=hour_axis(hour_label_order),
            xOffset=alt.XOffset("series:N", sort=share_series_order),
            y=alt.Y(
                "percent:Q",
                title="Percent of Total",
                axis=percent_axis,
                scale=percent_scale,
            ),
            color=alt.Color(
                "series:N",
                title=None,
                sort=share_series_order,
                scale=alt.Scale(
                    domain=share_series_order,
                    range=["#1f77b4", "#ff7f0e"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("hour_label:N", title="Hour"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip(
                    "percent:Q",
                    title="Percent of Total",
                    format=".1f",
                ),
                alt.Tooltip(
                    "labor_hours:Q",
                    title="Labor Hours",
                    format=",.2f",
                ),
                alt.Tooltip("sales:Q", title="Sales", format="$,.2f"),
            ],
        )
    )
    share_labels = (
        alt.Chart(share_long.dropna(subset=["percent"]))
        .mark_text(fontSize=11, dy=-8)
        .encode(
            x=hour_axis(hour_label_order),
            xOffset=alt.XOffset("series:N", sort=share_series_order),
            y=alt.Y("percent:Q", axis=None, scale=percent_scale),
            text=alt.Text("percent_label:N"),
            color=alt.value("#222222"),
        )
    )
    st.subheader("Hourly Sales & Labor Distribution")
    st.altair_chart(
        alt.layer(share_bars, share_labels).properties(height=360),
        use_container_width=True,
    )
    st.caption(
        "Each hour's share of sales and labor hours across the "
        "selected periods. The gap between the bars is where "
        "labor is heavy or light relative to sales."
    )



# ---------------------------------------------------------
# Period breakdown
# ---------------------------------------------------------

st.subheader("Period breakdown")

period_table = visible.sort_values("period_end", ascending=False).copy()
period_display = pd.DataFrame(
    {
        "Period start": period_table["period_start"].dt.date,
        "Period end": period_table["period_end"].dt.date,
        "Days": period_table["days_counted"].astype(int),
        "Sales": period_table["sales"],
        "Labor hours": period_table["labor_hours"],
        "Transactions": period_table["transactions"],
        "SpLH": period_table["splh"],
        "Lunch SpLH": period_table["lunch_splh"],
        "Dinner SpLH": period_table["dinner_splh"],
        "Cars": period_table["cars"],
        "Lane total": period_table["lane_total"].map(format_duration),
        "Menu board": period_table["menu_board"].map(format_duration),
        "Greet": period_table["greet"].map(format_duration),
        "Service": period_table["service"].map(format_duration),
        "Lane queue": period_table["lane_queue"].map(format_duration),
        "Avg cars in lane Per Departure Event": round(period_table["avg_cars_in_lane"], 2),
        "% Compliance": period_table["compliance"] * 100,
    }
)
st.dataframe(
    period_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Days": st.column_config.NumberColumn(
            format="%d",
            help=f"Days with sales, labor, or a drive-thru car. A full period is {period_days}.",
        ),
        "Sales": st.column_config.NumberColumn(format="$%.2f"),
        "Labor hours": st.column_config.NumberColumn(format="%.2f"),
        "Transactions": st.column_config.NumberColumn(format="%.0f"),
        "SpLH": st.column_config.NumberColumn(format="$%.2f"),
        "Lunch SpLH": st.column_config.NumberColumn(format="$%.2f"),
        "Dinner SpLH": st.column_config.NumberColumn(format="$%.2f"),
        "Cars": st.column_config.NumberColumn(format="%d"),
        "Avg cars in lane per departure event": st.column_config.NumberColumn(format="%.2f"),
        "% Compliance": st.column_config.NumberColumn(
            format="%.1f%%",
            help=(
                "Share of cars whose lane total was "
                f"{lane_goal_label} or under."
            ),
        ),
    },
)
st.caption(
    "One row per selected period, newest first. SpLH and "
    "lane time use the period totals. Lane times weight "
    "every car equally."
)


# ---------------------------------------------------------
# Hourly breakdown for the whole window
# ---------------------------------------------------------

st.subheader("Hourly breakdown")

if window_cars.empty or "hour" not in window_cars.columns:
    time_by_hour = pd.DataFrame()
else:
    timed = window_cars.dropna(subset=["hour"]).copy()
    if timed.empty:
        time_by_hour = pd.DataFrame()
    else:
        timed["hour"] = timed["hour"].astype(int)
        time_by_hour = rollup_drive_thru_times(timed, ["hour"])

if window_hourly.empty:
    splh_by_hour = pd.DataFrame(
        columns=["hour", "sales", "transactions", "labor_hours", "splh"]
    )
else:
    splh_by_hour = (
        window_hourly
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

if time_by_hour.empty and splh_by_hour.empty:
    st.info("That selection has no drive-thru times or sales.")
else:
    if time_by_hour.empty:
        by_hour = splh_by_hour.copy()
    elif splh_by_hour.empty:
        by_hour = time_by_hour.copy()
    else:
        by_hour = time_by_hour.merge(splh_by_hour, on="hour", how="outer")
    by_hour = with_hour_fields(by_hour)
    activity = pd.Series(False, index=by_hour.index)
    for column_name in ("cars", "sales", "labor_hours"):
        if column_name in by_hour.columns:
            activity = activity | by_hour[column_name].fillna(0).gt(0)
    by_hour = by_hour.loc[activity].sort_values("hour_index")

    if by_hour.empty:
        st.info("That selection has no drive-thru times or sales.")
    else:
        hourly_display = pd.DataFrame(
            {
                "Hour": by_hour["hour_label"],
                "Cars": by_hour["cars"].fillna(0).round().astype(int)
                if "cars" in by_hour.columns
                else 0,
                "Lane total": by_hour["lane_total"].map(format_duration)
                if "lane_total" in by_hour.columns
                else "—",
                "Menu board": by_hour["menu_board"].map(format_duration)
                if "menu_board" in by_hour.columns
                else "—",
                "Greet": by_hour["greet"].map(format_duration)
                if "greet" in by_hour.columns
                else "—",
                "Service": by_hour["service"].map(format_duration)
                if "service" in by_hour.columns
                else "—",
                "Lane queue": by_hour["lane_queue"].map(format_duration)
                if "lane_queue" in by_hour.columns
                else "—",
                "Avg cars in lane per departure event": round(by_hour["avg_cars_in_lane"], 2)
                if "avg_cars_in_lane" in by_hour.columns
                else NOT_AVAILABLE,
                "Sales": by_hour["sales"] if "sales" in by_hour.columns else NOT_AVAILABLE,
                "Transactions": by_hour["transactions"]
                if "transactions" in by_hour.columns
                else NOT_AVAILABLE,
                "Labor hours": by_hour["labor_hours"]
                if "labor_hours" in by_hour.columns
                else NOT_AVAILABLE,
                "SpLH": by_hour["splh"] if "splh" in by_hour.columns else NOT_AVAILABLE,
            }
        )
        st.dataframe(
            hourly_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cars": st.column_config.NumberColumn(format="%d"),
                "Avg cars in lane per departure event": st.column_config.NumberColumn(
                    format="%.2f",
                    help=(
                        "Average number of cars still stacked "
                        "when a car left the lane."
                    ),
                ),
                "Sales": st.column_config.NumberColumn(format="$%.2f"),
                "Transactions": st.column_config.NumberColumn(format="%.0f"),
                "Labor hours": st.column_config.NumberColumn(format="%.2f"),
                "SpLH": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        st.caption(
            f"Hours from 10:00 AM through 11:00 PM, pooled "
            f"across {window_text}. Avg cars in lane is the "
            "stack still behind a car at departure, not the "
            "count of cars served."
        )

st.caption(
    f"Displaying {duration:,} {interval['unit']} for "
    f"{selected_location.lower()}, ending "
    f"{pd.Timestamp(date_end):%b %d, %Y}"
    + (
        "."
        if selected_weekday == "All days"
        else f", {selected_weekday}s only."
    )
    + " SpLH uses the same hourly sales and shift clocks as "
    "SpLH per Hour. Lane time weights every car equally."
)
