import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="AI College & Branch Recommendation",
    layout="wide"
)

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("apeamcet_long_with_demand.csv")
    return df

df = load_data()

# ----------------------------------
# CORE LOGIC FUNCTIONS
# ----------------------------------
def calculate_suitability(user_rank, cutoff_rank, max_gap=60000):
    rank_gap = cutoff_rank - user_rank
    if rank_gap < 0:
        return 0.0
    gap_ratio = min(rank_gap / max_gap, 1)
    suitability = 85 * (1 - gap_ratio) + 15
    return round(min(max(suitability, 5), 95), 2)

def classify_risk(rank_gap):
    if rank_gap <= 2000:
        return "🎯 Ambitious"
    elif rank_gap <= 10000:
        return "⚖️ Moderate"
    else:
        return "✅ Safe"

def recommend_colleges(
    df,
    user_rank,
    user_category,
    user_gender,
    user_region,
    preferred_district="All",
    top_n=10
):
    eligible = df[
        (df["category"] == user_category) &
        (df["gender"] == user_gender) &
        (df["cutoff_rank"] >= user_rank)
    ].copy()

    if user_region != "NON-LOCAL":
        eligible = eligible[eligible["A_REG"] == user_region]

    if preferred_district != "All":
        eligible = eligible[eligible["DIST"] == preferred_district]

    if eligible.empty:
        return pd.DataFrame()

    eligible["rank_gap"] = eligible["cutoff_rank"] - user_rank
    eligible = eligible[eligible["rank_gap"] <= 30000]

    if eligible.empty:
        return pd.DataFrame()

    eligible["Suitability %"] = eligible["cutoff_rank"].apply(
        lambda c: calculate_suitability(user_rank, c)
    )
    eligible["Risk Level"] = eligible["rank_gap"].apply(classify_risk)

    result = eligible.sort_values(
        by=["Suitability %", "rank_gap"],
        ascending=[False, True]
    ).head(top_n)

    return result[
        [
            "NAME OF THE INSTITUTION",
            "branch_code",
            "DIST",
            "cutoff_rank",
            "Suitability %",
            "Risk Level"
        ]
    ]

# ----------------------------------
# SIDEBAR INPUT
# ----------------------------------
st.sidebar.title("🎓 Student Details")

user_rank = st.sidebar.number_input(
    "Enter EAPCET Rank",
    min_value=1,
    step=1
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
    ["AU", "SVU", "NON-LOCAL"]
)

preferred_district = st.sidebar.selectbox(
    "Preferred District",
    ["All"] + sorted(df["DIST"].dropna().unique())
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    5, 20, 10
)

# ----------------------------------
# MAIN UI
# ----------------------------------
st.title("🎓 AI-Based College & Branch Recommendation System")
st.markdown(
    "This system predicts **eligible B.Tech colleges and branches** based on "
    "EAPCET rank, reservation category, gender, and regional eligibility."
)

tab1, tab2 = st.tabs(
    ["🏛️ College Recommendations", "🎯 Aspiration College Check"]
)

# ----------------------------------
# TAB 1: RECOMMENDATIONS
# ----------------------------------
with tab1:
    if st.button("🚀 Generate Recommendations"):
        result = recommend_colleges(
            df,
            user_rank,
            user_category,
            user_gender,
            user_region,
            preferred_district,
            top_n
        )

        if result.empty:
            st.error("No colleges found for the given inputs.")
        else:
            st.success("Recommended Colleges & Branches")

            st.dataframe(
                result,
                use_container_width=True
            )

            st.info(
                "📌 **How to read Suitability %**\n"
                "• 80–100% → Very strong & competitive match\n"
                "• 50–80% → Good and realistic option\n"
                "• 20–50% → Safe option, but lower preference\n"
                "• Below 20% → Last-option colleges\n\n"
                "Suitability % is based on historical cutoff trends."
            )

# ----------------------------------
# TAB 2: ASPIRATION CHECK
# ----------------------------------
with tab2:
    st.subheader("Check Your Aspiration College")

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

    if st.button("Check Admission Chance"):
        check_df = df[
            (df["NAME OF THE INSTITUTION"] == selected_college) &
            (df["branch_code"] == selected_branch) &
            (df["category"] == user_category) &
            (df["gender"] == user_gender)
        ]

        if check_df.empty:
            st.warning("No data available for this combination.")
        else:
            cutoff = check_df.iloc[0]["cutoff_rank"]
            gap = cutoff - user_rank

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Year Cutoff", cutoff)
            with col2:
                st.metric("Your Rank", user_rank, delta=gap)

            if gap < 0:
                st.error(
                    f"❌ Tough chance. You need {abs(gap)} better rank."
                )
            else:
                st.success(
                    f"✅ You have a good chance! You are within {gap} ranks."
                )

# ----------------------------------
# FOOTER
# ----------------------------------
