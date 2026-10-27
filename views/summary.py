from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from revel_data import (
    DRIVE_THRU_TIME_COLUMNS,
    STORE_TIMEZONE,
    MissingSecretError,
    load_google_reviews,
    load_hme_departures,
    load_salary_data,
    load_sales_data,
    load_shift_data,
    rollup_drive_thru_times,
)


# A full year of the selected week. Partial weeks are
# still multiplied by 52, and the caption says so.
WEEKS_IN_YEAR = 52

# Roles paid from the salary feed. Their hours stay out
# of SpLH. GM overtime is also left out of OT hours.
SALARIED_ROLES = {"GM", "DM"}

# Slices smaller than this keep their percent in the
# legend only, so 15 labels can still be read.
SLICE_LABEL_MIN_SHARE = 0.035


def format_duration(seconds) -> str:
    """Render seconds as minutes and seconds, such as 3:12."""

    if seconds is None or pd.isna(seconds):
        return "—"

    total = int(round(float(seconds)))
    sign = "-" if total < 0 else ""
    total = abs(total)
    minutes, remainder = divmod(total, 60)
    return f"{sign}{minutes}:{remainder:02d}"


def format_week_range(start, end) -> str:
    """Label a week by its first and last day."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.year == end.year:
        return f"{start:%b %d} – {end:%b %d, %Y}"
    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def preferred_location(names: pd.Series) -> str:
    """Pick one display name when sources spell a store differently."""

    unique = [str(name) for name in names.dropna().unique()]
    if not unique:
        return "Unknown"
    mixed_case = [name for name in unique if name != name.upper()]
    return sorted(mixed_case or unique)[0]


def numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """A numeric series, or zeros when the column is absent."""

    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def role_keys(frame: pd.DataFrame) -> pd.Series:
    """Upper-case role names aligned to the frame index."""

    if frame.empty or "role" not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return (
        frame["role"].fillna("").astype(str).str.strip().str.upper()
    )


def salaried_role_keys(salary_frame: pd.DataFrame) -> set[str]:
    """Role names paid as salary, so their hours stay out of SpLH."""

    keys = set(SALARIED_ROLES)
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


def dated(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that have a store and a business date."""

    if (
        frame.empty
        or "business_date" not in frame.columns
        or "location_key" not in frame.columns
    ):
        return pd.DataFrame(
            columns=["location_key", "location", "business_date"]
        )

    working = frame.copy()
    working["business_date"] = pd.to_datetime(
        working["business_date"],
        errors="coerce",
    ).dt.normalize()
    working["location_key"] = (
        working["location_key"].astype(str).str.strip()
    )
    working = working.dropna(subset=["business_date"])
    return working.loc[working["location_key"] != ""].copy()


def store_catalog(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per store that appears in any loaded table."""

    parts = []
    for frame in frames:
        if (
            frame.empty
            or "location_key" not in frame.columns
            or "location" not in frame.columns
        ):
            continue
        parts.append(frame[["location_key", "location"]])

    empty = pd.DataFrame(columns=["location_key", "location"])
    if not parts:
        return empty

    names = pd.concat(parts, ignore_index=True)
    names = names.dropna(subset=["location_key"])
    names["location_key"] = names["location_key"].astype(str).str.strip()
    names = names.loc[names["location_key"] != ""]
    if names.empty:
        return empty

    listed = (
        names
        .groupby("location_key", as_index=False)["location"]
        .agg(preferred_location)
    )
    return listed.sort_values(
        "location",
        key=lambda series: series.str.lower(),
    ).reset_index(drop=True)


def in_window(
    frame: pd.DataFrame,
    start,
    end,
) -> pd.DataFrame:
    """Rows whose business date falls in the inclusive window."""

    if frame.empty or "business_date" not in frame.columns:
        return frame.iloc[0:0]
    dates = pd.to_datetime(frame["business_date"]).dt.normalize()
    mask = dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return frame.loc[mask]


def sum_by_store(
    frame: pd.DataFrame,
    column: str,
    start,
    end,
) -> pd.DataFrame:
    """Sum one measure per store inside the window."""

    empty = pd.DataFrame(columns=["location_key", column])
    if frame.empty or column not in frame.columns:
        return empty
    window = in_window(frame, start, end)
    if window.empty:
        return empty
    return (
        window
        .groupby("location_key", as_index=False)[column]
        .sum()
    )


def sales_by_day(sales_frame: pd.DataFrame) -> pd.DataFrame:
    """Net sales for each store and business date."""

    sales = dated(sales_frame)
    if sales.empty or "net_sales" not in sales.columns:
        return pd.DataFrame(
            columns=[
                "business_date",
                "location_key",
                "location",
                "net_sales",
            ]
        )
    sales["net_sales"] = pd.to_numeric(
        sales["net_sales"],
        errors="coerce",
    )
    return (
        sales
        .groupby(["business_date", "location_key"], as_index=False)
        .agg(
            location=("location", "first"),
            net_sales=("net_sales", "sum"),
        )
    )


def wages_by_day(shift_frame: pd.DataFrame) -> pd.DataFrame:
    """Shift wages for each store and business date."""

    shifts = dated(shift_frame)
    if shifts.empty:
        return pd.DataFrame(
            columns=["business_date", "location_key", "hourly_wages"]
        )
    shifts["hourly_wages"] = numeric_column(shifts, "shift_wages")
    return (
        shifts
        .groupby(["business_date", "location_key"], as_index=False)
        ["hourly_wages"]
        .sum()
    )


def hours_by_day(
    shift_frame: pd.DataFrame,
    salaried: set[str],
) -> pd.DataFrame:
    """Hourly labor hours, with salaried roles left out."""

    shifts = dated(shift_frame)
    empty = pd.DataFrame(
        columns=["business_date", "location_key", "hourly_labor_hours"]
    )
    if shifts.empty:
        return empty
    hourly = shifts.loc[~role_keys(shifts).isin(salaried)].copy()
    if hourly.empty:
        return empty
    hourly["hourly_labor_hours"] = (
        numeric_column(hourly, "regular_hours")
        + numeric_column(hourly, "ot_hours")
    )
    return (
        hourly
        .groupby(["business_date", "location_key"], as_index=False)
        ["hourly_labor_hours"]
        .sum()
    )


def ot_by_day(shift_frame: pd.DataFrame) -> pd.DataFrame:
    """Overtime hours for each store and day. GM is left out."""

    shifts = dated(shift_frame)
    empty = pd.DataFrame(
        columns=["business_date", "location_key", "ot_hours"]
    )
    if shifts.empty:
        return empty
    hourly = shifts.loc[role_keys(shifts) != "GM"].copy()
    if hourly.empty:
        return empty
    hourly["ot_hours"] = numeric_column(hourly, "ot_hours")
    return (
        hourly
        .groupby(["business_date", "location_key"], as_index=False)
        ["ot_hours"]
        .sum()
    )


def lane_for_window(
    hme_frame: pd.DataFrame,
    start,
    end,
) -> tuple[pd.DataFrame, float]:
    """Car-weighted lane time per store, and for every store together."""

    empty = pd.DataFrame(columns=["location_key", "lane_total", "cars"])
    if (
        hme_frame.empty
        or "lane_total" not in hme_frame.columns
        or "location_key" not in hme_frame.columns
    ):
        return empty, float("nan")

    window = in_window(dated(hme_frame), start, end).copy()
    if window.empty:
        return empty, float("nan")
    for column_name in DRIVE_THRU_TIME_COLUMNS:
        if column_name not in window.columns:
            window[column_name] = pd.NA
    if "total_cars" not in window.columns:
        window["total_cars"] = pd.NA

    by_store = rollup_drive_thru_times(window, ["location_key"])
    overall = rollup_drive_thru_times(
        window.assign(_all="all"),
        ["_all"],
    )
    overall_seconds = (
        float(overall["lane_total"].iloc[0])
        if not overall.empty and overall["lane_total"].notna().any()
        else float("nan")
    )
    keep = [
        column
        for column in ("location_key", "lane_total", "cars")
        if column in by_store.columns
    ]
    return by_store[keep], overall_seconds


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, leaving a blank where the denominator is missing or zero."""

    base = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce") / base.where(base > 0)


def _total(series: pd.Series) -> float:
    """Sum a column. An all-blank column stays blank."""

    if series is None or series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").sum(min_count=1))


def store_window(
    stores: pd.DataFrame,
    sales_daily: pd.DataFrame,
    wage_daily: pd.DataFrame,
    hours_daily: pd.DataFrame,
    ot_daily: pd.DataFrame,
    hme_frame: pd.DataFrame,
    start,
    end,
    annualize: bool,
    shifts_loaded: bool,
) -> pd.DataFrame:
    """Day or week KPIs, plus one pooled row for every store."""

    sales = sum_by_store(sales_daily, "net_sales", start, end)
    wages = sum_by_store(wage_daily, "hourly_wages", start, end)
    hours = sum_by_store(hours_daily, "hourly_labor_hours", start, end)
    overtime = sum_by_store(ot_daily, "ot_hours", start, end)
    lanes, overall_lane = lane_for_window(hme_frame, start, end)

    frame = stores.merge(sales, on="location_key", how="left")
    frame = frame.merge(wages, on="location_key", how="left")
    frame = frame.merge(hours, on="location_key", how="left")
    frame = frame.merge(overtime, on="location_key", how="left")
    if "lane_total" in lanes.columns:
        frame = frame.merge(
            lanes[["location_key", "lane_total"]],
            on="location_key",
            how="left",
        )
    else:
        frame["lane_total"] = float("nan")

    # A store with sales and no punches recorded zero hourly
    # wages. Leave the columns blank when the shift extract
    # itself failed to load.
    if shifts_loaded:
        frame["hourly_wages"] = frame["hourly_wages"].fillna(0.0)
        frame["hourly_labor_hours"] = frame[
            "hourly_labor_hours"
        ].fillna(0.0)
        frame["ot_hours"] = frame["ot_hours"].fillna(0.0)

    frame["hourly_percent"] = _divide(
        frame["hourly_wages"],
        frame["net_sales"],
    ) * 100
    frame["splh"] = _divide(
        frame["net_sales"],
        frame["hourly_labor_hours"],
    )
    if annualize:
        frame["annualized"] = frame["net_sales"] * WEEKS_IN_YEAR
    else:
        frame["annualized"] = float("nan")
    frame["lane_label"] = frame["lane_total"].map(format_duration)
    frame["row_key"] = frame["location_key"]

    total_sales = _total(frame["net_sales"])
    total_row = {
        "row_key": "ALL",
        "location_key": "ALL",
        "location": "All stores",
        "net_sales": total_sales,
        "hourly_wages": _total(frame["hourly_wages"]),
        "hourly_labor_hours": _total(frame["hourly_labor_hours"]),
        "ot_hours": _total(frame["ot_hours"]),
        "lane_total": overall_lane,
        "annualized": (
            total_sales * WEEKS_IN_YEAR
            if annualize and pd.notna(total_sales)
            else float("nan")
        ),
    }
    total_row["hourly_percent"] = (
        total_row["hourly_wages"] / total_row["net_sales"] * 100
        if pd.notna(total_row["hourly_wages"])
        and pd.notna(total_row["net_sales"])
        and total_row["net_sales"] > 0
        else float("nan")
    )
    total_row["splh"] = (
        total_row["net_sales"] / total_row["hourly_labor_hours"]
        if pd.notna(total_row["net_sales"])
        and pd.notna(total_row["hourly_labor_hours"])
        and total_row["hourly_labor_hours"] > 0
        else float("nan")
    )
    total_row["lane_label"] = format_duration(overall_lane)
    return pd.concat(
        [frame, pd.DataFrame([total_row])],
        ignore_index=True,
    )


def combine_periods(
    day: pd.DataFrame,
    week: pd.DataFrame,
) -> pd.DataFrame:
    """Place day and week measures side by side for each store."""

    day_part = day.rename(
        columns={
            "net_sales": "day_net_sales",
            "hourly_percent": "day_labor_percent",
            "splh": "day_splh",
            "lane_label": "day_lane",
            "ot_hours": "day_ot_hours",
        }
    )
    week_part = week.rename(
        columns={
            "net_sales": "week_net_sales",
            "annualized": "annualized",
            "hourly_percent": "week_labor_percent",
            "splh": "week_splh",
            "lane_label": "week_lane",
            "ot_hours": "week_ot_hours",
        }
    )
    merged = day_part[
        [
            "row_key",
            "location",
            "day_net_sales",
            "day_labor_percent",
            "day_splh",
            "day_lane",
            "day_ot_hours",
        ]
    ].merge(
        week_part[
            [
                "row_key",
                "week_net_sales",
                "annualized",
                "week_labor_percent",
                "week_splh",
                "week_lane",
                "week_ot_hours",
            ]
        ],
        on="row_key",
        how="outer",
    )
    merged["_all"] = merged["row_key"].eq("ALL").astype(int)
    return (
        merged
        .sort_values(
            ["_all", "location"],
            key=lambda column: (
                column.str.lower()
                if column.dtype == object
                else column
            ),
        )
        .drop(columns="_all")
        .reset_index(drop=True)
    )


def sales_pie(week: pd.DataFrame) -> alt.Chart | None:
    """Week net sales, one slice per store, labeled with its share."""

    plotted = week.loc[
        (week["row_key"] != "ALL")
        & week["net_sales"].fillna(0).gt(0)
    ].copy()
    if plotted.empty:
        return None

    total = float(plotted["net_sales"].sum())
    if total <= 0:
        return None

    plotted["share"] = plotted["net_sales"] / total
    plotted["percent_label"] = plotted["share"].map(
        lambda share: f"{share * 100:.1f}%"
    )
    plotted["legend_label"] = (
        plotted["location"] + " · " + plotted["percent_label"]
    )
    plotted = plotted.sort_values("net_sales", ascending=False)
    plotted["slice_label"] = plotted["percent_label"].where(
        plotted["share"] >= SLICE_LABEL_MIN_SHARE,
        "",
    )
    order = plotted["legend_label"].tolist()

    base = alt.Chart(plotted).encode(
        theta=alt.Theta("net_sales:Q", stack=True),
        order=alt.Order("net_sales:Q", sort="descending"),
        tooltip=[
            alt.Tooltip("location:N", title="Store"),
            alt.Tooltip(
                "net_sales:Q",
                title="Net sales",
                format="$,.0f",
            ),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    arcs = base.mark_arc(innerRadius=68, outerRadius=148).encode(
        color=alt.Color(
            "legend_label:N",
            title="Share of week net sales",
            sort=order,
            scale=alt.Scale(domain=order, scheme="category20"),
            legend=alt.Legend(
                orient="right",
                columns=1,
                labelLimit=280,
            ),
        ),
    )
    labels = base.mark_text(radius=176, fontSize=12).encode(
        text="slice_label:N",
        color=alt.value("#1f2933"),
    )
    return (arcs + labels).properties(width=420, height=420)


def review_leaderboard(comparison: pd.DataFrame) -> pd.DataFrame:
    """Stores ranked by this week's average Google review score.

    A store with no reviews this week stays at the bottom.
    The pooled All stores row is left out of the ranking.
    """

    columns = [
        "location",
        "week_review_score",
        "week_review_count",
        "day_review_score",
        "day_review_count",
        "current_review_score",
    ]
    stores = comparison.loc[comparison["row_key"] != "ALL"].copy()
    if stores.empty:
        return pd.DataFrame(columns=["rank", *columns])

    stores = stores.sort_values(
        ["week_review_score", "week_review_count", "location"],
        ascending=[False, False, True],
        na_position="last",
    )
    stores.insert(0, "rank", range(1, len(stores) + 1))
    return stores[["rank", *columns]].reset_index(drop=True)


def _review_rollup(frame: pd.DataFrame) -> pd.DataFrame:
    """Review count and average rating for each store."""

    empty = pd.DataFrame(
        columns=["location_key", "review_count", "review_score"]
    )
    if frame.empty:
        return empty

    working = frame.copy()
    working["rating"] = pd.to_numeric(working["rating"], errors="coerce")
    return (
        working
        .groupby("location_key", as_index=False)
        .agg(
            review_count=("location_key", "size"),
            review_score=("rating", "mean"),
        )
    )


def _review_average(frame: pd.DataFrame) -> float:
    """Average rating across reviews. Blank when none are scored."""

    if frame.empty or "rating" not in frame.columns:
        return float("nan")
    ratings = pd.to_numeric(frame["rating"], errors="coerce").dropna()
    if ratings.empty:
        return float("nan")
    return float(ratings.mean())


def review_metrics(
    reviews: pd.DataFrame,
    stores: pd.DataFrame,
    day,
    week_start,
    week_end,
    reviews_loaded: bool,
) -> pd.DataFrame:
    """Day, week, and lifetime Google review figures for each store."""

    result = pd.DataFrame(
        {
            "row_key": pd.concat(
                [stores["location_key"], pd.Series(["ALL"])],
                ignore_index=True,
            )
        }
    )
    if not reviews_loaded:
        for column in (
            "day_review_count",
            "week_review_count",
            "day_review_score",
            "week_review_score",
            "current_review_score",
        ):
            result[column] = float("nan")
        return result

    day_rows = in_window(reviews, day, day)
    week_rows = in_window(reviews, week_start, week_end)
    pieces = {
        "day": _review_rollup(day_rows),
        "week": _review_rollup(week_rows),
        "current": _review_rollup(reviews),
    }
    for label, rolled in pieces.items():
        count_name = f"{label}_review_count"
        score_name = f"{label}_review_score"
        if label == "current":
            score_name = "current_review_score"
        if rolled.empty:
            result[score_name] = float("nan")
            if label != "current":
                result[count_name] = 0
            continue
        renamed = rolled.rename(
            columns={
                "location_key": "row_key",
                "review_count": count_name,
                "review_score": score_name,
            }
        )
        keep = ["row_key", score_name]
        if label != "current":
            keep.insert(1, count_name)
        result = result.merge(renamed[keep], on="row_key", how="left")

    all_rows = result["row_key"].eq("ALL")
    result.loc[all_rows, "day_review_count"] = float(len(day_rows))
    result.loc[all_rows, "day_review_score"] = _review_average(day_rows)
    result.loc[all_rows, "week_review_count"] = float(len(week_rows))
    result.loc[all_rows, "week_review_score"] = _review_average(week_rows)
    result.loc[all_rows, "current_review_score"] = _review_average(
        reviews
    )
    result["day_review_count"] = result["day_review_count"].fillna(0)
    result["week_review_count"] = result["week_review_count"].fillna(0)
    return result


def leaderboard_column_config() -> dict:
    """Headers for the review leaderboard. Columns size to their content."""

    return {
        "rank": st.column_config.NumberColumn(
            "#",
            format="%d",
        ),
        "location": st.column_config.TextColumn("Store"),
        "week_review_score": st.column_config.NumberColumn(
            "Week avg",
            format="%.2f",
            help=(
                "Average star rating of the reviews "
                "posted during those 7 days."
            ),
        ),
        "week_review_count": st.column_config.NumberColumn(
            "Week",
            format="%.0f",
            help="Google reviews posted during those 7 days.",
        ),
        "day_review_score": st.column_config.NumberColumn(
            "Day avg",
            format="%.2f",
            help=(
                "Average star rating of the reviews posted "
                "on the selected date."
            ),
        ),
        "day_review_count": st.column_config.NumberColumn(
            "Day",
            format="%.0f",
            help="Google reviews posted on the selected business date.",
        ),
        "current_review_score": st.column_config.NumberColumn(
            "Current",
            format="%.2f",
            help=(
                "Average star rating of every Google review "
                "stored for this store."
            ),
        ),
    }


def table_height(row_count: int) -> int:
    """Fit the stores on screen, then scroll once the list passes 16."""

    visible = min(max(row_count, 1), 16)
    return 48 + 36 * visible


def summary_column_config() -> dict:
    """Headers for the store comparison table."""

    return {
        "rank": st.column_config.NumberColumn(
            "Rank",
            format="%d",
        ),
        "location": st.column_config.TextColumn("Store"),
        "day_net_sales": st.column_config.NumberColumn(
            "Net sales (day)",
            format="$%,.0f",
            help="Net sales on the selected business date.",
        ),
        "week_net_sales": st.column_config.NumberColumn(
            "Net sales (week)",
            format="$%,.0f",
            help=(
                "Net sales for the 7 days ending on "
                "the selected date."
            ),
        ),
        "annualized": st.column_config.NumberColumn(
            "Annualized",
            format="$%,.0f",
            help=(
                "Net sales for those 7 days, multiplied by 52."
            ),
        ),
        "day_labor_percent": st.column_config.NumberColumn(
            "Labor % (day)",
            format="%.1f%%",
            help="Shift wages divided by that day's net sales.",
        ),
        "week_labor_percent": st.column_config.NumberColumn(
            "Labor % (week)",
            format="%.1f%%",
            help=(
                "Shift wages divided by net sales for those 7 days."
            ),
        ),
        "day_splh": st.column_config.NumberColumn(
            "SpLH (day)",
            format="$%,.2f",
            help=(
                "That day's net sales divided by hourly labor "
                "hours. Salaried roles, including GM, are left "
                "out of the hours."
            ),
        ),
        "week_splh": st.column_config.NumberColumn(
            "SpLH (week)",
            format="$%,.2f",
            help=(
                "Net sales for those 7 days divided by hourly "
                "labor hours. Salaried roles, including GM, "
                "are left out of the hours."
            ),
        ),
        "day_lane": st.column_config.TextColumn(
            "Drive-thru (day)",
            help="Car-weighted average lane time that day.",
        ),
        "week_lane": st.column_config.TextColumn(
            "Drive-thru (week)",
            help=(
                "Car-weighted average lane time over those 7 days."
            ),
        ),
        "day_ot_hours": st.column_config.NumberColumn(
            "OT hours (day)",
            format="%.1f",
            help="Overtime hours that day. The GM role is left out.",
        ),
        "week_ot_hours": st.column_config.NumberColumn(
            "OT hours (week)",
            format="%.1f",
            help=(
                "Overtime hours over those 7 days. "
                "The GM role is left out."
            ),
        ),
        "day_review_count": st.column_config.NumberColumn(
            "Reviews (day)",
            format="%.0f",
            help="Google reviews posted on the selected business date.",
        ),
        "day_review_score": st.column_config.NumberColumn(
            "Avg score (day)",
            format="%.2f",
            help=(
                "Average star rating of the reviews posted "
                "on that date."
            ),
        ),
        "week_review_count": st.column_config.NumberColumn(
            "Reviews (week)",
            format="%.0f",
            help="Google reviews posted during those 7 days.",
        ),
        "week_review_score": st.column_config.NumberColumn(
            "Avg score (week)",
            format="%.2f",
            help=(
                "Average star rating of the reviews "
                "posted during those 7 days."
            ),
        ),
        "current_review_score": st.column_config.NumberColumn(
            "Current score",
            format="%.2f",
            help=(
                "Average star rating of every Google review "
                "stored for this store."
            ),
        ),
    }


def default_summary_date(min_date, max_date):
    """Yesterday in Central Time, kept inside the loaded dates.

    Streamlit Community Cloud runs in UTC. The restaurant's
    business date follows America/Chicago. The picker rejects
    a value outside min and max, so a yesterday with no rows
    falls back to the nearest loaded date.
    """

    yesterday = datetime.now(STORE_TIMEZONE).date() - timedelta(days=1)
    return min(max(yesterday, min_date), max_date)


def available_dates(*frames: pd.DataFrame) -> pd.Series:
    """Business dates present on any loaded table."""

    parts = []
    for frame in frames:
        if frame.empty or "business_date" not in frame.columns:
            continue
        parts.append(pd.to_datetime(frame["business_date"], errors="coerce"))
    if not parts:
        return pd.Series(dtype="datetime64[ns]")
    return pd.concat(parts, ignore_index=True).dropna()


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.title("Summary (Weekly)")
st.caption(
    "Every store in Supabase, for the selected day and the "
    "7 days ending on that date. A new store shows up here as "
    "soon as it has sales, shifts, drive-thru cars, or "
    "Google reviews."
)

sales_error = None
shift_error = None
salary_error = None
hme_error = None
review_error = None

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
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    shift_df = pd.DataFrame()
    shift_error = str(error)

try:
    salary_df = load_salary_data()
except Exception as error:
    salary_df = pd.DataFrame()
    salary_error = str(error)

try:
    hme_df = load_hme_departures()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    hme_df = pd.DataFrame()
    hme_error = str(error)

try:
    review_df = load_google_reviews()
except MissingSecretError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    review_df = pd.DataFrame()
    review_error = str(error)

sales_daily = sales_by_day(sales_df)
wage_daily = wages_by_day(shift_df)
hours_daily = hours_by_day(shift_df, salaried_role_keys(salary_df))
ot_daily = ot_by_day(shift_df)
stores = store_catalog(
    sales_daily,
    dated(shift_df),
    dated(hme_df),
    review_df,
)

if sales_error:
    st.warning(f"Sales are unavailable: {sales_error}")
if shift_error:
    st.warning(f"Shift wages and hours are unavailable: {shift_error}")
if salary_error:
    st.warning(f"Salary roles are unavailable: {salary_error}")
if hme_error:
    st.warning(f"Drive-thru times are unavailable: {hme_error}")
if review_error:
    st.warning(f"Google reviews are unavailable: {review_error}")

if stores.empty:
    st.warning(
        "No stores were returned from Supabase. Check the "
        "sales, shift, and drive-thru tables."
    )
    st.stop()

dates = available_dates(
    sales_daily,
    wage_daily,
    dated(hme_df),
    review_df,
)
if dates.empty:
    st.warning("No business dates were returned from Supabase.")
    st.stop()

min_date = dates.min().date()
max_date = dates.max().date()
date_key = "summary_business_date"
if date_key not in st.session_state:
    st.session_state[date_key] = default_summary_date(
        min_date,
        max_date,
    )
else:
    chosen_date = st.session_state[date_key]
    if chosen_date < min_date or chosen_date > max_date:
        st.session_state[date_key] = min(
            max(chosen_date, min_date),
            max_date,
        )

st.sidebar.header("Filters")
business_date = st.sidebar.date_input(
    "Business date",
    min_value=min_date,
    max_value=max_date,
    key=date_key,
    help=(
        "Day columns use this business date. Week columns "
        "and the pie use the 7 days ending on it."
    ),
)
if business_date is None:
    st.warning("Pick a business date.")
    st.stop()

day = pd.Timestamp(business_date).normalize()
week_end = day
week_start = day - pd.Timedelta(days=6)
st.sidebar.caption(
    f"Trailing 7 days: {format_week_range(week_start, week_end)}."
)

day_metrics = store_window(
    stores,
    sales_daily,
    wage_daily,
    hours_daily,
    ot_daily,
    hme_df,
    day,
    day,
    annualize=False,
    shifts_loaded=shift_error is None,
)
week_metrics = store_window(
    stores,
    sales_daily,
    wage_daily,
    hours_daily,
    ot_daily,
    hme_df,
    week_start,
    week_end,
    annualize=True,
    shifts_loaded=shift_error is None,
)
comparison = combine_periods(day_metrics, week_metrics)
comparison = comparison.merge(
    review_metrics(
        review_df,
        stores,
        day,
        week_start,
        week_end,
        reviews_loaded=review_error is None,
    ),
    on="row_key",
    how="left",
)

st.subheader(
    f"{day:%A, %b %d, %Y}  ·  "
    f"Trailing 7 days, {format_week_range(week_start, week_end)}"
)
pie = sales_pie(week_metrics)
if pie is None:
    st.info("No store recorded net sales in this week.")
else:
    st.altair_chart(pie, use_container_width=False)
    st.caption(
        "Each slice is one store's net sales for these 7 days. "
        "The percentage is drawn on the slice, and again "
        "beside the store name. A slice under "
        f"{SLICE_LABEL_MIN_SHARE * 100:.1f}% keeps the "
        "percentage in the legend so the labels stay apart."
    )

display_columns = [
    "location",
    "day_net_sales",
    "week_net_sales",
    "annualized",
    "day_labor_percent",
    "week_labor_percent",
    "day_splh",
    "week_splh",
    "day_lane",
    "week_lane",
    "day_ot_hours",
    "week_ot_hours",
]
st.subheader("Sales and Labor Performance")
st.dataframe(
    comparison[display_columns],
    column_config=summary_column_config(),
    hide_index=True,
    width="content",
    height=table_height(len(comparison)),
)

week_sales = in_window(sales_daily, week_start, week_end)
recorded_days = (
    int(week_sales["business_date"].nunique())
    if not week_sales.empty
    else 0
)
if recorded_days < 7:
    st.caption(
        f"Sales are recorded on {recorded_days} of 7 days "
        "in this window. Annualized net sales is the net "
        "sales from these 7 days, multiplied by 52."
    )

st.caption(
    "Hourly labor % is shift wages divided by net sales. "
    "SpLH is net sales divided by hourly labor hours, with "
    "salaried roles such as GM left out of the hours. "
    "Drive-thru time is the car-weighted average lane time. "
    "OT hours leave out the GM role. All stores adds the "
    "stores together, then divides."
)

leaderboard = review_leaderboard(comparison)
leaderboard_columns = [
    "rank",
    "location",
    "week_review_score",
    "week_review_count",
    "day_review_score",
    "day_review_count",
    "current_review_score",
]
st.subheader("Review leaderboard")
st.dataframe(
    leaderboard[leaderboard_columns],
    column_config=leaderboard_column_config(),
    hide_index=True,
    width="content",
    height=table_height(len(leaderboard)),
)
st.caption(
    "Ranked by average score for these 7 days, highest "
    "first. A store with no reviews this week is listed "
    "at the bottom. Reviews (day) use the selected date. "
    "Current score is the average of every Google review "
    "stored for the store."
)
