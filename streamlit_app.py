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
    ]
)

navigation.run()
