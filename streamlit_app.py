import streamlit as st


st.set_page_config(
    page_title="Revel Operations Dashboard",
    page_icon="📈",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page(
            "views/week_over_week.py",
            title="Sales: Week Over Week",
            icon="📈",
            default=True,
        ),
        st.Page(
            "views/labor_percentage.py",
            title="Labor % of Net Sales",
            icon="🧾",
        ),
        st.Page(
            "views/employee_hours.py",
            title="Hourly Employee Hours",
            icon="🕒",
        ),
        st.Page(
            "views/sales_per_labor_hour.py",
            title="SpLH per Hour",
            icon="💵",
        ),
        st.Page(
            "views/drive_thru_times.py",
            title="Drive-Thru Times",
            icon="🚗",
        ),
        st.Page(
            "views/combo_mix.py",
            title="Combo Mix",
            icon="🍗",
        ),
        st.Page(
            "views/cash_variance.py",
            title="Cash Variance",
            icon="💰",
        ),
    ]
)

navigation.run()
