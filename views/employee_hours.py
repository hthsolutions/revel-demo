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
# Focused day: a label every two hours, from midnight to midnight.
FOCUS_HOUR_TICKS = list(range(0, DAY_HOUR_SPAN + 1, 2))
CLOCK_LABEL_EXPR = (
    "datum.value == 0 || datum.value == 24 ? '12a' "
    ": datum.value == 12 ? '12p' "
    ": datum.value < 12 ? datum.value + 'a' "
    ": (datum.value - 12) + 'p'"
)

SEGMENT_ORDER = ["Regular", "Overtime"]
SEGMENT_COLORS = ["#2ca02c", "#d62728"]
# Same facet size and name-column width on both charts, so
# a shift on the overtime chart sits on the same clock below.
FACET_WIDTH = 128
Y_LABEL_EXTENT = 280
ROLE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#9edae5",
    "#dbdb8d",
    "#c7c7c7",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
]


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


def paint_role_span(
    clock_in,
    clock_out,
    regular_hours,
    ot_hours,
    role: str,
):
    """Same clock span as the overtime bar, in one role color."""

    painted = paint_shift(
        clock_in,
        clock_out,
        regular_hours,
        ot_hours,
    )
    if not painted:
        return []
    return [(painted[0][0], painted[-1][1], role)]


def build_segment_rows(week_df: pd.DataFrame, day_by_date, color_by: str):
    """One row per colored piece of a shift on a calendar day."""

    rows = []
    for shift in week_df.itertuples(index=False):
        if color_by == "pay":
            painted = paint_shift(
                shift.clock_in,
                shift.clock_out,
                shift.regular_hours,
                shift.ot_hours,
            )
        else:
            painted = paint_role_span(
                shift.clock_in,
                shift.clock_out,
                shift.regular_hours,
                shift.ot_hours,
                shift.role,
            )
        for start, end, segment in painted:
            for day, start_hour, end_hour in split_on_midnights(
                start,
                end,
            ):
                day_label = day_by_date.get(
                    pd.Timestamp(day).normalize()
                )
                if day_label is None or end_hour <= start_hour:
                    continue
                rows.append(
                    {
                        "employee": shift.employee,
                        "role": shift.role,
                        "day_label": day_label,
                        "start_hour": start_hour,
                        "end_hour": end_hour,
                        "segment": segment,
                        "clock_in_label": format_clock(shift.clock_in),
                        "clock_out_label": format_clock(shift.clock_out),
                        "regular_hours": float(
                            shift.regular_hours or 0
                        ),
                        "ot_hours": float(shift.ot_hours or 0),
                        "shift_hours": float(shift.shift_hours or 0),
                        "cumulative_hours": float(
                            shift.cumulative_hours or 0
                        ),
                    }
                )
    return rows


def add_missing_days(
    frame: pd.DataFrame,
    employee_order: list[str],
    first_employee: str,
    day_label_order: list[str],
    filler_segment: str,
) -> pd.DataFrame:
    """Keep every weekday column, even when that day has no shifts."""

    missing_days = [
        label
        for label in day_label_order
        if label not in set(frame["day_label"])
    ]
    if not missing_days:
        return frame
    placeholders = pd.DataFrame(
        {
            "employee": first_employee,
            "employee_label": employee_order[0],
            "day_label": missing_days,
            "start_hour": 0.0,
            "end_hour": 0.0,
            "segment": filler_segment,
            "bar_opacity": 0.0,
        }
    )
    return pd.concat([frame, placeholders], ignore_index=True)


def labels_on_day(
    frame: pd.DataFrame,
    day_label: str,
    employee_order: list[str],
) -> list[str]:
    """Employees who have a visible shift on one day, in week order."""

    present = set(
        frame.loc[
            (frame["day_label"] == day_label)
            & (frame["bar_opacity"].fillna(0) > 0),
            "employee_label",
        ].dropna()
    )
    return [label for label in employee_order if label in present]


def hours_chart(
    frame: pd.DataFrame,
    color_domain: list[str],
    color_range: list[str],
    tooltip: list,
    employee_order: list[str],
    day_label_order: list[str],
    row_height: int,
    hour_ticks: list[int] | None = None,
    clock_labels: bool = False,
    single_day: bool = False,
):
    """One week of clock bars, or one day when a day is selected.

    Both charts use this layout so the clocks stay aligned.
    """

    name_axis = alt.Axis(
        labelLimit=Y_LABEL_EXTENT,
        minExtent=Y_LABEL_EXTENT,
        maxExtent=Y_LABEL_EXTENT,
    )
    ticks = hour_ticks or [0, 12, 24]
    hour_axis = alt.Axis(
        values=ticks,
        labelFlush=True,
        labelOverlap=False if clock_labels else True,
        labelExpr=CLOCK_LABEL_EXPR if clock_labels else None,
    )
    encodings = dict(
        y=alt.Y(
            "employee_label:N",
            title=None,
            sort=employee_order,
            scale=alt.Scale(domain=employee_order),
            axis=name_axis,
        ),
        x=alt.X(
            "start_hour:Q",
            title=None,
            scale=alt.Scale(domain=[0, DAY_HOUR_SPAN]),
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
            sort=color_domain,
            scale=alt.Scale(
                domain=color_domain,
                range=color_range,
            ),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=tooltip,
    )
    if not single_day:
        encodings["column"] = alt.Column(
            "day_label:N",
            title=None,
            sort=day_label_order,
            header=alt.Header(
                labelOrient="top",
                labelAlign="center",
            ),
            spacing=8,
        )
    chart_width = FACET_WIDTH * 7 if single_day else FACET_WIDTH
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(**encodings)
        .properties(width=chart_width, height=row_height)
    )


st.title("Hourly Employee Hours")
st.caption(
    "One week at a time. Each day is a clock from midnight "
    "to midnight. The top chart turns green for the shift "
    "and red for overtime at the end of it. The chart below "
    "colors those same shifts by role. GM shifts are excluded."
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
shift_df["role"] = (
    shift_df["role"].fillna("Unassigned").astype(str).str.strip()
)
shift_df.loc[shift_df["role"] == "", "role"] = "Unassigned"
shift_df = shift_df[shift_df["role"].str.upper() != "GM"].copy()

if shift_df.empty:
    st.warning("No hourly shifts remain after excluding GM.")
    st.stop()


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

available_roles = sorted(
    week_df["role"].unique(),
    key=str.lower,
)
role_key = "employee_hours_roles"
if role_key not in st.session_state:
    st.session_state[role_key] = available_roles
else:
    previous_roles = list(st.session_state[role_key])
    kept_roles = [
        role for role in previous_roles if role in available_roles
    ]
    if previous_roles and not kept_roles:
        kept_roles = available_roles
    st.session_state[role_key] = kept_roles

selected_roles = st.sidebar.multiselect(
    "Roles",
    options=available_roles,
    key=role_key,
    placeholder="Select roles",
    help=(
        "Both charts show shifts for the selected roles. "
        "All roles in the week are selected until you change this."
    ),
)
if not selected_roles:
    st.info("Select at least one role to see hours.")
    st.stop()

week_df = week_df[week_df["role"].isin(selected_roles)].copy()
if week_df.empty:
    st.info(
        "No hourly shifts for the selected roles fall in that week."
    )
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

total_hours = float(week_df["shift_hours"].sum())
regular_hours_total = float(
    week_df["regular_hours"].fillna(0).sum()
)
ot_hours_total = float(week_df["ot_hours"].fillna(0).sum())
ot_percent = (
    ot_hours_total / total_hours * 100
    if total_hours
    else 0.0
)
employees_with_ot = int(
    (
        week_df.groupby("employee")["ot_hours"].sum().fillna(0)
        > 0
    ).sum()
)

days_df = day_labels(selected_week, week_start_weekday)
day_by_date = {
    pd.Timestamp(row.business_date).normalize(): row.day_label
    for row in days_df.itertuples(index=False)
}

pay_rows = build_segment_rows(week_df, day_by_date, "pay")
role_rows = build_segment_rows(week_df, day_by_date, "role")

st.subheader(format_week(selected_week))

kpi_columns = st.columns(5)
kpi_columns[0].metric(
    "Total Hours",
    f"{total_hours:,.1f} hrs",
)
kpi_columns[1].metric(
    "Non-OT Total Hours",
    f"{regular_hours_total:,.1f} hrs",
)
kpi_columns[2].metric(
    "OT Total Hours",
    f"{ot_hours_total:,.1f} hrs",
)
kpi_columns[3].metric(
    "OT Hours % of Total Hours",
    f"{ot_percent:,.1f}%",
)
kpi_columns[4].metric(
    "Employees with OT",
    f"{employees_with_ot:,.0f}",
)

if not pay_rows or not role_rows:
    st.info(
        "That week has shifts, but none have a clock-in "
        "and clock-out that can be drawn."
    )
    st.stop()

pay_df = pd.DataFrame(pay_rows)
role_df = pd.DataFrame(role_rows)
pay_df["bar_opacity"] = 1.0
role_df["bar_opacity"] = 1.0

# Most hours at the top, fewest at the bottom. The
# number beside each name is that week's paid hours.
week_totals = (
    week_df
    .groupby("employee", as_index=False)["shift_hours"]
    .sum()
)
shown_employees = set(pay_df["employee"])
week_totals = week_totals[
    week_totals["employee"].isin(shown_employees)
].copy()
week_totals["name_key"] = week_totals["employee"].str.lower()
week_totals = week_totals.sort_values(
    ["shift_hours", "name_key"],
    ascending=[False, True],
)
week_totals["employee_label"] = week_totals.apply(
    lambda row: f"{row.employee} ({row.shift_hours:.2f} hrs)",
    axis=1,
)
label_by_employee = dict(
    zip(
        week_totals["employee"],
        week_totals["employee_label"],
    )
)
pay_df["employee_label"] = pay_df["employee"].map(label_by_employee)
role_df["employee_label"] = role_df["employee"].map(label_by_employee)
employee_order = week_totals["employee_label"].tolist()
day_label_order = days_df["day_label"].tolist()
first_employee = week_totals["employee"].iloc[0]
role_order = sorted(role_df["segment"].dropna().unique(), key=str.lower)
role_colors = [
    ROLE_PALETTE[index % len(ROLE_PALETTE)]
    for index in range(len(role_order))
]

pay_df = add_missing_days(
    pay_df,
    employee_order,
    first_employee,
    day_label_order,
    "Regular",
)
role_df = add_missing_days(
    role_df,
    employee_order,
    first_employee,
    day_label_order,
    role_order[0],
)

row_height = max(360, 24 * len(employee_order))
shared_tooltip = [
    alt.Tooltip("employee:N", title="Employee"),
    alt.Tooltip("role:N", title="Role"),
    alt.Tooltip("day_label:N", title="Day"),
    alt.Tooltip("clock_in_label:N", title="Clock In"),
    alt.Tooltip("clock_out_label:N", title="Clock Out"),
    alt.Tooltip(
        "regular_hours:Q",
        title="Regular Hours",
        format=".2f",
    ),
    alt.Tooltip(
        "ot_hours:Q",
        title="Overtime Hours",
        format=".2f",
    ),
    alt.Tooltip(
        "shift_hours:Q",
        title="Shift Hours",
        format=".2f",
    ),
    alt.Tooltip(
        "cumulative_hours:Q",
        title="Cumulative Hours",
        format=".2f",
    ),
]
day_options = ["All days", *day_label_order]
day_key = "employee_hours_day"
if (
    day_key not in st.session_state
    or st.session_state[day_key] not in day_options
):
    st.session_state[day_key] = "All days"
selected_day = st.pills(
    "Day",
    options=day_options,
    key=day_key,
    help=(
        "All days shows the week. Pick one day to open "
        "that clock with a mark every two hours."
    ),
)
focused_day = selected_day not in (None, "All days")
if focused_day:
    day_employees = labels_on_day(
        pay_df,
        selected_day,
        employee_order,
    )
    chart_frame_pay = pay_df[
        pay_df["day_label"] == selected_day
    ].copy()
    chart_frame_role = role_df[
        role_df["day_label"] == selected_day
    ].copy()
    chart_employees = day_employees
    chart_days = [selected_day]
    chart_height = max(360, 24 * max(len(day_employees), 1))
    hour_ticks = FOCUS_HOUR_TICKS
    clock_labels = True
    single_day = True
else:
    chart_frame_pay = pay_df
    chart_frame_role = role_df
    chart_employees = employee_order
    chart_days = day_label_order
    chart_height = row_height
    hour_ticks = [0, 12, 24]
    clock_labels = False
    single_day = False

if focused_day and not chart_employees:
    st.info(f"No shifts were clocked on {selected_day}.")
else:
    chart_args = dict(
        employee_order=chart_employees,
        day_label_order=chart_days,
        row_height=chart_height,
        hour_ticks=hour_ticks,
        clock_labels=clock_labels,
        single_day=single_day,
    )
    pay_title = "Regular and overtime"
    role_title = "Hours by role"
    if focused_day:
        pay_title = f"Regular and overtime — {selected_day}"
        role_title = f"Hours by role — {selected_day}"
    st.subheader(pay_title)
    st.altair_chart(
        hours_chart(
            chart_frame_pay,
            SEGMENT_ORDER,
            SEGMENT_COLORS,
            [
                *shared_tooltip,
                alt.Tooltip("segment:N", title="Hours"),
            ],
            **chart_args,
        ),
        use_container_width=True,
    )
    st.subheader(role_title)
    st.altair_chart(
        hours_chart(
            chart_frame_role,
            role_order,
            role_colors,
            shared_tooltip,
            **chart_args,
        ),
        use_container_width=True,
    )

if focused_day:
    focus_caption = (
        f"{selected_day} only. The clock is marked every "
        "two hours. Choose All days to return to the week."
    )
else:
    focus_caption = (
        "Choose a day above to open that clock with a "
        "mark every two hours."
    )
st.caption(
    f"{len(employee_order):,} hourly employee(s). "
    "Both charts use the same names and the same clock, "
    "so a bar below is the same shift as the bar above it. "
    "Names run from the most hours at the top to the "
    "fewest at the bottom. The number beside each name is "
    "that employee's total hours for the selected roles. "
    "The top bar turns red for overtime at the end of the "
    "shift. The lower bar colors that whole shift by role. "
    + focus_caption
)
