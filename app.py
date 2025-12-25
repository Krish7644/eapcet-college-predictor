import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# LOAD ARTIFACTS
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("apeamcet_long_clean.csv")

@st.cache_resource
def load_model():
    return joblib.load("best_seat_prediction_model.pkl")

df = load_data()
model = load_model()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="AI Career & College Guidance", layout="wide")

st.title("🎓 AI-Powered EAPCET College & Branch Recommendation System")
st.markdown(
    "This system recommends **B.Tech colleges and branches** based on "
    "EAPCET rank, category, gender, and region, along with **admission suitability**."
)

st.sidebar.header("🔎 Enter Your Details")

user_rank = st.sidebar.number_input(
    "EAPCET Rank", min_value=1, step=1
)

user_category = st.sidebar.selectbox(
    "Category", sorted(df["category"].unique())
)

user_gender = st.sidebar.selectbox(
    "Gender", sorted(df["gender"].unique())
)

# Region selection (NON-LOCAL handled logically)
user_region = st.sidebar.selectbox(
    "Region",
    ["AU", "SVU", "NON-LOCAL"]
)

top_n = st.sidebar.slider(
    "Number of recommendations", min_value=5, max_value=20, value=10
)

# ----------------------------
# HYBRID SUITABILITY LOGIC
# ----------------------------
def calculate_suitability(user_rank, cutoff_rank, max_gap=60000):
    """
    Suitability score based on rank proximity.
    Indicates recommendation strength, not guarantee.
    """

    rank_gap = cutoff_rank - user_rank

    # Not eligible
    if rank_gap < 0:
        return 0.0

    gap_ratio = min(rank_gap / max_gap, 1)
    suitability = 85 * (1 - gap_ratio) + 15

    return round(min(max(suitability, 5), 95), 2)

# ----------------------------
# CORE RECOMMENDATION LOGIC
# ----------------------------
def recommend_colleges(
    df, user_rank, user_category, user_gender, user_region, top_n
):
    eligible = df[
        (df["category"] == user_category) &
        (df["gender"] == user_gender) &
        (df["cutoff_rank"] >= user_rank)
    ].copy()

    # NON-LOCAL logic
    if user_region != "NON-LOCAL":
        eligible = eligible[eligible["A_REG"] == user_region]

    if eligible.empty:
        return pd.DataFrame()

    eligible["rank_gap"] = eligible["cutoff_rank"] - user_rank

    # Keep realistic range
    eligible = eligible[eligible["rank_gap"] <= 30000]

    if eligible.empty:
        return pd.DataFrame()

    eligible["Suitability %"] = eligible["cutoff_rank"].apply(
        lambda c: calculate_suitability(user_rank, c)
    )

    result = eligible.sort_values(
        by=["Suitability %", "rank_gap"],
        ascending=[False, True]
    ).head(top_n)

    # ❌ COLLEGE FEE REMOVED
    result = result[
        [
            "NAME OF THE INSTITUTION",
            "branch_code",
            "cutoff_rank",
            "Suitability %"
        ]
    ]

    # Clean index for proper column display
    result.reset_index(drop=True, inplace=True)

    return result

# ----------------------------
# RUN PREDICTION
# ----------------------------
if st.sidebar.button("🚀 Get Recommendations"):
    if user_rank <= 0:
        st.warning("Please enter a valid rank.")
    else:
        result = recommend_colleges(
            df,
            user_rank,
            user_category,
            user_gender,
            user_region,
            top_n
        )

        if result.empty:
            st.error("No colleges found for the given inputs.")
        else:
            st.success("✅ Recommended Colleges & Branches")
            st.dataframe(result, use_container_width=True)

            # ---- CLARITY NOTES ----
            st.markdown(
                """
                ℹ️ **Note:**  
                **Suitability %** indicates how well a college matches your rank based on past trends.  
                Lower values **do not mean rejection**, but indicate **lower preference or competitiveness**.
                """
            )

            st.info(
                "📌 **How to read Suitability %**\n"
                "• **80–100%** → ⭐ Very strong & competitive match\n"
                "• **50–80%** → ✅ Good and realistic option\n"
                "• **20–50%** → ⚠️ Safe option, but lower preference\n"
                "• **Below 20%** → ❗ Last-option colleges"
            )

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    "📌 *Developed as part of an AI-based Career & College Guidance System project.*"
)
