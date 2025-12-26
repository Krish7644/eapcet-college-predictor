import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("apeamcet_long_with_demand.csv")

df = load_data()

# Create display-friendly college name (College + District)
df["college_display"] = df["NAME OF THE INSTITUTION"] + " (" + df["DIST"] + ")"

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(page_title="AI Career & College Guidance", layout="wide")

st.title("🎓 AI-Powered EAPCET College & Branch Recommendation System")

st.markdown(
    "This application helps students explore **B.Tech colleges and branches** using "
    "historical EAPCET cutoff data and intelligent suitability scoring."
)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("🔎 Student Details")

user_rank = st.sidebar.number_input("EAPCET Rank", min_value=1, step=1)

user_category = st.sidebar.selectbox(
    "Category", sorted(df["category"].unique())
)

user_gender = st.sidebar.selectbox(
    "Gender", sorted(df["gender"].unique())
)

user_region = st.sidebar.selectbox(
    "Region", ["AU", "SVU", "NON-LOCAL"]
)

preferred_district = st.sidebar.selectbox(
    "Preferred District (Optional)",
    ["All"] + sorted(df["DIST"].dropna().unique())
)

top_n = st.sidebar.slider(
    "Number of recommendations", min_value=5, max_value=20, value=10
)

# ----------------------------
# SUITABILITY & RISK LOGIC
# ----------------------------
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

# ----------------------------
# GENERAL RECOMMENDATION FUNCTION
# ----------------------------
def recommend_colleges():
    eligible = df[
        (df["category"] == user_category) &
        (df["gender"] == user_gender) &
        (df["cutoff_rank"] >= user_rank)
    ].copy()

    # Region logic
    if user_region != "NON-LOCAL":
        eligible = eligible[eligible["A_REG"] == user_region]

    # District filter
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

    eligible["College Type"] = eligible["rank_gap"].apply(classify_risk)

    result = eligible.sort_values(
        by=["Suitability %", "rank_gap"],
        ascending=[False, True]
    ).head(top_n)

    result = result[
        [
            "NAME OF THE INSTITUTION",
            "branch_code",
            "DIST",
            "cutoff_rank",
            "Suitability %",
            "College Type"
        ]
    ]

    result.reset_index(drop=True, inplace=True)
    return result

# ----------------------------
# ASPIRATION COLLEGE CHECK
# ----------------------------
def check_specific_college(college_name, branch):
    filtered = df[
        (df["NAME OF THE INSTITUTION"] == college_name) &
        (df["branch_code"] == branch) &
        (df["category"] == user_category) &
        (df["gender"] == user_gender)
    ]

    if user_region != "NON-LOCAL":
        filtered = filtered[filtered["A_REG"] == user_region]

    if filtered.empty:
        return None

    cutoff = filtered["cutoff_rank"].values[0]
    rank_gap = cutoff - user_rank

    if rank_gap < 0:
        suitability = 8.0
        remark = "❌ Very low chance based on last year's cutoff"
    else:
        suitability = calculate_suitability(user_rank, cutoff)
        remark = "✅ Possible based on historical cutoff"

    risk = classify_risk(max(rank_gap, 0))

    return {
        "College": college_name,
        "Branch": branch,
        "District": filtered["DIST"].values[0],
        "Last Year Cutoff Rank": cutoff,
        "Your Rank": user_rank,
        "Suitability %": suitability,
        "Assessment": remark,
        "College Type": risk
    }

# ----------------------------
# MAIN TABS
# ----------------------------
tab1, tab2 = st.tabs(["📋 College Recommendations", "🎯 Aspiration College Check"])

# ----------------------------
# TAB 1: GENERAL RECOMMENDATIONS
# ----------------------------
with tab1:
    if st.button("🚀 Get College Recommendations"):
        if user_rank <= 0:
            st.warning("Please enter a valid rank.")
        else:
            result = recommend_colleges()
            if result.empty:
                st.error("No colleges found for the given inputs.")
            else:
                st.success("✅ Recommended Colleges & Branches")
                st.dataframe(result, use_container_width=True)

                st.info(
                    "📌 **How to read Suitability %**\n"
                    "• 80–100% → ⭐ Very strong & competitive match\n"
                    "• 50–80% → ✅ Good and realistic option\n"
                    "• 20–50% → ⚖️ Safe option, lower preference\n"
                    "• Below 20% → ❗ Last-option colleges"
                )

# ----------------------------
# TAB 2: ASPIRATION COLLEGE CHECK
# ----------------------------
with tab2:
    st.subheader("🎯 Check Your Chances for a Specific College")

    selected_college_display = st.selectbox(
        "Select College",
        sorted(df["college_display"].unique())
    )

    # Extract actual college name
    selected_college = selected_college_display.rsplit(" (", 1)[0]

    available_branches = sorted(
        df[df["NAME OF THE INSTITUTION"] == selected_college]["branch_code"].unique()
    )

    selected_branch = st.selectbox(
        "Select Branch",
        available_branches
    )

    if st.button("🔍 Check This College"):
        result = check_specific_college(selected_college, selected_branch)

        if result is None:
            st.warning(
                "⚠️ This college–branch–category combination was not allotted "
                "in the previous counselling data. Please try another branch or category."
            )
        else:
            st.success("📊 Admission Feasibility Result")

            st.table(pd.DataFrame({
                "Parameter": result.keys(),
                "Value": result.values()
            }))

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
