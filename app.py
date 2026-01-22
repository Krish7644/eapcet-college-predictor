import streamlit as st
import pandas as pd
import numpy as np

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
    return pd.read_excel("apeamcet_long_with_demand.xls", engine="xlrd")


df = load_data()

# ----------------------------------
# SUITABILITY & RISK LOGIC (ML LOGIC)
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

# ----------------------------------
# COLLEGE RECOMMENDATION FUNCTION
# ----------------------------------
def recommend_colleges(
    df,
    user_rank,
    user_category,
    user_gender,
    user_region,
    preferred_district="All",
    top_n=15
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

    eligible["Chance"] = eligible["rank_gap"].apply(classify_risk)

    result = eligible.sort_values(
        by="Suitability %",
        ascending=False
    ).head(top_n)

    return result

# ----------------------------------
# ASPIRATION COLLEGE CHECK
# ----------------------------------
def check_aspiration_college(
    df,
    college_name,
    branch_code,
    user_rank,
    user_category,
    user_gender
):
    check_df = df[
        (df["NAME OF THE INSTITUTION"] == college_name) &
        (df["branch_code"] == branch_code) &
        (df["category"] == user_category) &
        (df["gender"] == user_gender)
    ]

    if check_df.empty:
        return None

    cutoff = check_df.iloc[0]["cutoff_rank"]
    gap = cutoff - user_rank

    return {
        "cutoff": cutoff,
        "rank_gap": gap
    }

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
# MAIN UI
# ----------------------------------
st.title("🎓 AI College & Branch Recommendation System")
st.markdown(
    "This system recommends **B.Tech colleges and branches** using "
    "**rank-based machine learning logic** and historical AP EAPCET data."
)

tab1, tab2 = st.tabs([
    "🏛️ College Recommendations",
    "🎯 Aspiration College Check"
])

# ----------------------------------
# TAB 1
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
                "📌 **Suitability % Explanation**\n"
                "- 80–100% → Very strong match\n"
                "- 50–80% → Good option\n"
                "- 20–50% → Possible\n"
                "- Below 20% → Risky"
            )

# ----------------------------------
# TAB 2
# ----------------------------------
with tab2:
    st.subheader("Check Your Chances for a Specific College")

    college_list = sorted(df["NAME OF THE INSTITUTION"].unique())
    selected_college = st.selectbox("Select College", college_list)

    branch_list = sorted(
        df[df["NAME OF THE INSTITUTION"] == selected_college]["branch_code"].unique()
    )
    selected_branch = st.selectbox("Select Branch", branch_list)

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
            st.warning("No historical cutoff data found.")
        else:
            cutoff = result["cutoff"]
            gap = result["rank_gap"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Last Year Cutoff Rank", cutoff)
            with col2:
                st.metric("Your Rank", user_rank, delta=gap)

            if gap < 0:
                st.error(f"❌ Tough chance. Improve by {abs(gap)} ranks.")
            else:
                st.success(f"✅ Good chance! Safe by {gap} ranks.")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption("AI-based College & Branch Recommendation System | Streamlit + ML")
