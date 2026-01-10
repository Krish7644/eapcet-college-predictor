import streamlit as st
import pandas as pd

# ----------------------------------
# IMPORT LOGIC MODULES
# ----------------------------------
from logic.college_predictor import (
    recommend_colleges,
    check_aspiration_college
)

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="AI College Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/apeamcet_long_clean.csv")

df = load_data()

# ----------------------------------
# SIDEBAR INPUTS
# ----------------------------------
st.sidebar.title("🎯 Student Profile")

user_rank = st.sidebar.number_input(
    "EAPCET Rank",
    min_value=1,
    step=1,
    value=20000
)

user_category = st.sidebar.selectbox(
    "Category",
    sorted(df["category"].unique())
)

user_gender = st.sidebar.selectbox(
    "Gender",
    sorted(df["gender"].unique())
)

user_region = st.sidebar.selectbox(
    "Region",
    ["Andhra University", "Sri Venkateswara University", "NON-LOCAL"]
)

preferred_district = st.sidebar.selectbox(
    "Preferred District",
    ["All"] + sorted(df["DIST"].dropna().unique())
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)

# ----------------------------------
# MAIN TITLE
# ----------------------------------
st.title("🎓 AI College & Branch Recommendation System")
st.markdown(
    "This system recommends **B.Tech colleges** based on "
    "rank, category, gender, and region using intelligent logic."
)

# ----------------------------------
# TABS
# ----------------------------------
tab1, tab2 = st.tabs([
    "🏛️ College Recommendations",
    "🎯 Aspiration College Check"
])

# ----------------------------------
# TAB 1: COLLEGE RECOMMENDATION
# ----------------------------------
with tab1:
    if st.button("🚀 Get College Recommendations"):
        result = recommend_colleges(
            df=df,
            user_rank=user_rank,
            user_category=user_category,
            user_gender=user_gender,
            user_region=user_region,
            preferred_district=preferred_district,
            top_n=top_n
        )

        if result.empty:
            st.error("No colleges found for the given criteria.")
        else:
            st.success("✅ Recommended Colleges")

            st.dataframe(
                result[
                    [
                        "NAME OF THE INSTITUTION",
                        "branch_code",
                        "DIST",
                        "cutoff_rank",
                        "Suitability %",
                        "Chance"
                    ]
                ],
                use_container_width=True
            )

            st.info(
                "📌 **How to read Suitability %**\n"
                "- 80–100% → Very strong match\n"
                "- 50–80% → Good option\n"
                "- 20–50% → Possible but less preferred\n"
                "- Below 20% → Last-option colleges"
            )

# ----------------------------------
# TAB 2: ASPIRATION COLLEGE CHECK
# ----------------------------------
with tab2:
    st.subheader("Check Your Chances for a Specific College")

    college_list = sorted(df["NAME OF THE INSTITUTION"].unique())
    selected_college = st.selectbox(
        "Select College",
        college_list
    )

    branch_list = sorted(
        df[df["NAME OF THE INSTITUTION"] == selected_college]["branch_code"].unique()
    )
    selected_branch = st.selectbox(
        "Select Branch",
        branch_list
    )

    if st.button("🔍 Check Aspiration College"):
        result = check_aspiration_college(
            df=df,
            college_name=selected_college,
            branch_code=selected_branch,
            user_rank=user_rank,
            user_category=user_category,
            user_gender=user_gender
        )

        if result is None:
            st.warning("No historical cutoff data found for this combination.")
        else:
            cutoff = result["cutoff"]
            gap = result["rank_gap"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Year Cutoff Rank", cutoff)
            with col2:
                st.metric("Your Rank", user_rank, delta=gap)

            if gap < 0:
                st.error(
                    f"❌ Tough chance. You need to improve by {abs(gap)} ranks."
                )
            else:
                st.success(
                    f"✅ Good chance! You are safer by {gap} ranks."
                )

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption(
    "Final Year Project | AI-based College & Branch Recommendation System"
)
