from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    WEEK_START_OPTIONS,
    WEEKDAY_NAMES,
    MissingSecretError,
    add_week_columns,
    get_shift_columns,
    load_shift_clocks,
    load_shift_data,
)


# Each day column is a clock, midnight at the left
# and the next midnight at the right.
DAY_HOUR_SPAN = 24
STORE_TIMEZONE = ZoneInfo("America/Chicago")

SEGMENT_ORDER = ["Regular", "Overtime"]
SEGMENT_COLORS = ["#2ca02c", "#d62728"]


def format_week(week_start) -> str:
    """Label a week by its first and last calendar day."""

    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    return f"{start:%b %d} – {end:%b %d, %Y}"


def format_clock(timestamp) -> str:
    """Render a clock time without a leading zero."""

    if timestamp is None or pd.isna(timestamp):
        return ""
    return pd.Timestamp(timestamp).strftime("%I:%M %p").lstrip("0")


def day_labels(week_start, week_start_weekday: int) -> pd.DataFrame:
    """One row for each day of the selected week, in week order."""

    start = pd.Timestamp(week_start)
    frame = pd.DataFrame({"day_index": range(7)})
    frame["business_date"] = start + pd.to_timedelta(
        frame["day_index"],
        unit="D",
    )
    frame["day_label"] = [
        (
            f"{WEEKDAY_NAMES[(week_start_weekday + index) % 7][:3]}"
            f" {date.day}"
        )
        for index, date in enumerate(frame["business_date"])
    ]
    return frame


def parse_store_clock(value):
    """Parse a clock timestamp into naive Central Time."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    text = str(value).strip()
    if text == "" or text.lower() == "none" or text.lower() == "nat":
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


def split_on_midnights(start, end):
    """Break one interval into pieces that stay inside a calendar day."""

    pieces = []
    cursor = pd.Timestamp(start)
    stop_at = pd.Timestamp(end)

    while cursor < stop_at:
        next_midnight = cursor.normalize() + pd.Timedelta(days=1)
        piece_end = min(stop_at, next_midnight)
        start_hour = (
            cursor - cursor.normalize()
        ).total_seconds() / 3600
        end_hour = (
            piece_end - cursor.normalize()
        ).total_seconds() / 3600
        if piece_end == next_midnight:
            end_hour = DAY_HOUR_SPAN
        pieces.append(
            (cursor.normalize(), start_hour, end_hour)
        )
        cursor = piece_end

    return pieces


def paint_shift(clock_in, clock_out, regular_hours, ot_hours):
    """Color a shift green, then red for its overtime tail.

    The bar runs from clock-in to clock-out. Overtime is the
    last portion of that span.
    """

    if clock_in is None or pd.isna(clock_in):
        return []

    regular = max(float(regular_hours or 0), 0.0)
    overtime = max(float(ot_hours or 0), 0.0)
    if clock_out is None or pd.isna(clock_out):
        clock_out = clock_in + pd.Timedelta(
            hours=regular + overtime
        )
    if clock_out <= clock_in:
        return []

    red_start = clock_out - pd.Timedelta(hours=overtime)
    if red_start < clock_in:
        red_start = clock_in

    painted = []
    if red_start > clock_in:
        painted.append((clock_in, red_start, "Regular"))
    if clock_out > red_start:
        painted.append((red_start, clock_out, "Overtime"))
    return painted


st.title("Hourly Employee Hours")
st.caption(
    "One week at a time. Each day is a clock from midnight "
    "to midnight. Green is the shift on that clock, and it "
    "turns red for the overtime at the end of the shift."
)

try:
    shift_df = load_shift_data()
    clock_df = load_shift_clocks()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(
        f"Unable to load shift data from Supabase: {error} "
        "Confirm SHIFT_TABLE in revel_data.py."
    )
    st.stop()

if shift_df.empty:
    st.warning("No hourly shift records were returned.")
    st.stop()

if clock_df.empty:
    clock_columns = []
    try:
        clock_columns = get_shift_columns()
    except Exception:
        clock_columns = []
    time_like = [
        name
        for name in clock_columns
        if any(
            token in name.lower()
            for token in ("clock", "start", "end", "time")
        )
    ]
    detail = (
        "Time-like columns on the shift table: "
        + ", ".join(time_like)
        if time_like
        else "The shift table has no clock-in or clock-out column."
    )
    st.error(
        "Shifts can't be placed on the clock without a "
        "clock-in and clock-out. " + detail
    )
    st.stop()

shift_df = shift_df.copy()
shift_df["employee"] = (
    shift_df["employee"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)
shift_df.loc[shift_df["employee"] == "", "employee"] = "Unknown"
shift_df = shift_df.merge(clock_df, on="record_key", how="left")
shift_df["clock_in"] = shift_df["clock_in"].map(parse_store_clock)
shift_df["clock_out"] = shift_df["clock_out"].map(parse_store_clock)


# ---------------------------------------------------------
# Sidebar: one location and one week
# ---------------------------------------------------------

st.sidebar.header("Filters")

available_locations = sorted(
    shift_df["location"].dropna().unique().tolist()
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
    shift_df = shift_df[
        shift_df["location"] == selected_location
    ].copy()

if shift_df.empty:
    st.warning("No hourly shifts match that location.")
    st.stop()

shift_df = add_week_columns(shift_df, week_start_weekday)

week_starts = sorted(
    shift_df["week_start"].dropna().unique(),
    reverse=True,
)
selected_week = st.sidebar.selectbox(
    "Week",
    options=week_starts,
    format_func=format_week,
)

week_df = shift_df[
    shift_df["week_start"] == selected_week
].copy()

if week_df.empty:
    st.info("No hourly shifts fall in that week.")
    st.stop()

week_df["shift_hours"] = (
    week_df["regular_hours"].fillna(0)
    + week_df["ot_hours"].fillna(0)
)
week_df = week_df.sort_values(
    ["employee", "clock_in"],
    na_position="last",
)
week_df["cumulative_hours"] = (
    week_df
    .groupby("employee")["shift_hours"]
    .cumsum()
)

days_df = day_labels(selected_week, week_start_weekday)
day_by_date = {
    pd.Timestamp(row.business_date).normalize(): row.day_label
    for row in days_df.itertuples(index=False)
}

segment_rows = []
for shift in week_df.itertuples(index=False):
    painted = paint_shift(
        shift.clock_in,
        shift.clock_out,
        shift.regular_hours,
        shift.ot_hours,
    )
    for start, end, segment in painted:
        for day, start_hour, end_hour in split_on_midnights(
            start,
            end,
        ):
            day_label = day_by_date.get(pd.Timestamp(day).normalize())
            if day_label is None or end_hour <= start_hour:
                continue
            segment_rows.append(
                {
                    "employee": shift.employee,
                    "day_label": day_label,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "segment": segment,
                    "clock_in_label": format_clock(shift.clock_in),
                    "clock_out_label": format_clock(shift.clock_out),
                    "regular_hours": float(shift.regular_hours or 0),
                    "ot_hours": float(shift.ot_hours or 0),
                    "shift_hours": float(shift.shift_hours or 0),
                    "cumulative_hours": float(
                        shift.cumulative_hours or 0
                    ),
                }
            )

if not segment_rows:
    st.info(
        "That week has shifts, but none have a clock-in "
        "and clock-out that can be drawn."
    )
    st.stop()

segments_df = pd.DataFrame(segment_rows)
segments_df["bar_opacity"] = 1.0

# Most hours at the top, fewest at the bottom. The
# number beside each name is that week's paid hours.
week_totals = (
    week_df
    .groupby("employee", as_index=False)["shift_hours"]
    .sum()
)
shown_employees = set(segments_df["employee"])
week_totals = week_totals[
    week_totals["employee"].isin(shown_employees)
].copy()
week_totals["name_key"] = week_totals["employee"].str.lower()
week_totals = week_totals.sort_values(
    ["shift_hours", "name_key"],
    ascending=[False, True],
)
week_totals["employee_label"] = week_totals.apply(
    lambda row: f"{row.employee} ({row.shift_hours:.1f})",
    axis=1,
)
label_by_employee = dict(
    zip(
        week_totals["employee"],
        week_totals["employee_label"],
    )
)
segments_df["employee_label"] = segments_df["employee"].map(
    label_by_employee
)
employee_order = week_totals["employee_label"].tolist()
day_label_order = days_df["day_label"].tolist()
missing_days = [
    label
    for label in day_label_order
    if label not in set(segments_df["day_label"])
]
if missing_days:
    placeholders = pd.DataFrame(
        {
            "employee": week_totals["employee"].iloc[0],
            "employee_label": employee_order[0],
            "day_label": missing_days,
            "start_hour": 0.0,
            "end_hour": 0.0,
            "segment": "Regular",
            "bar_opacity": 0.0,
        }
    )
    segments_df = pd.concat(
        [segments_df, placeholders],
        ignore_index=True,
    )

row_height = max(360, 24 * len(employee_order))
hour_axis = alt.Axis(values=[0, 12, 24], labelFlush=True)
hour_scale = alt.Scale(domain=[0, DAY_HOUR_SPAN])

shifts = (
    alt.Chart(segments_df)
    .mark_bar()
    .encode(
        y=alt.Y(
            "employee_label:N",
            title=None,
            sort=employee_order,
            axis=alt.Axis(labelLimit=280),
        ),
        x=alt.X(
            "start_hour:Q",
            title=None,
            scale=hour_scale,
            axis=hour_axis,
        ),
        x2="end_hour:Q",
        opacity=alt.Opacity(
            "bar_opacity:Q",
            scale=alt.Scale(domain=[0, 1], range=[0, 1]),
            legend=None,
        ),
        color=alt.Color(
            "segment:N",
            title=None,
            sort=SEGMENT_ORDER,
            scale=alt.Scale(
                domain=SEGMENT_ORDER,
                range=SEGMENT_COLORS,
            ),
            legend=alt.Legend(orient="top"),
        ),
        column=alt.Column(
            "day_label:N",
            title=None,
            sort=day_label_order,
            header=alt.Header(
                labelOrient="top",
                labelAlign="center",
            ),
            spacing=8,
        ),
        tooltip=[
            alt.Tooltip("employee:N", title="Employee"),
            alt.Tooltip("day_label:N", title="Day"),
            alt.Tooltip("clock_in_label:N", title="Clock In"),
            alt.Tooltip("clock_out_label:N", title="Clock Out"),
            alt.Tooltip(
                "regular_hours:Q",
                title="Regular Hours",
                format=".1f",
            ),
            alt.Tooltip(
                "ot_hours:Q",
                title="Overtime Hours",
                format=".1f",
            ),
            alt.Tooltip(
                "shift_hours:Q",
                title="Shift Hours",
                format=".1f",
            ),
            alt.Tooltip(
                "cumulative_hours:Q",
                title="Cumulative Hours",
                format=".1f",
            ),
        ],
    )
)

hours_chart = shifts.properties(width=128, height=row_height)

st.subheader(format_week(selected_week))
st.altair_chart(hours_chart, use_container_width=True)
st.caption(
    f"{len(employee_order):,} hourly employee(s). "
    "Names run from the most hours at the top to the "
    "fewest at the bottom. The number beside each name "
    "is that employee's total hours for the week. "
    "Each bar sits on the hours they were clocked in "
    "and turns red for overtime at the end of the shift."
)
