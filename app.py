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
    "EAPCET rank, category, gender, and region, along with **chance of admission**."
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

user_region = st.sidebar.selectbox(
    "Region", sorted(df["A_REG"].dropna().unique())
)

top_n = st.sidebar.slider(
    "Number of recommendations", min_value=5, max_value=20, value=10
)

# ----------------------------
# CORE RECOMMENDATION LOGIC
# ----------------------------
def predict_probability(cutoff_rank):
    """
    ML-based probability prediction
    """
    prob = model.predict_proba(np.array([[cutoff_rank]]))[0][1]
    return round(prob * 100, 1)

def recommend_colleges(
    df, user_rank, user_category, user_gender, user_region, top_n
):
    eligible = df[
        (df["category"] == user_category) &
        (df["gender"] == user_gender) &
        (df["cutoff_rank"] >= user_rank)
    ].copy()

    # Region filter
    eligible = eligible[eligible["A_REG"] == user_region]

    if eligible.empty:
        return pd.DataFrame()

    # Rank gap (relevance)
    eligible["rank_gap"] = eligible["cutoff_rank"] - user_rank

    # Remove very far colleges (keeps results realistic)
    eligible = eligible[eligible["rank_gap"] <= 30000]

    if eligible.empty:
        return pd.DataFrame()

    # ML-based probability
    eligible["chance_percent"] = eligible["cutoff_rank"].apply(
        predict_probability
    )

    # Sort by closeness to rank
    result = eligible.sort_values(
        by="rank_gap", ascending=True
    ).head(top_n)

    return result[
        [
            "NAME OF THE INSTITUTION",
            "branch_code",
            "COLLFEE",
            "cutoff_rank",
            "chance_percent"
        ]
    ]

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

            st.markdown(
                """
                **Note:**  
                *Chance % is estimated using a machine learning model trained on historical cutoff data.*  
                *Final seat allotment depends on counselling rounds and official rules.*
                """
            )

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    "📌 *Developed as part of an AI-based Career & College Guidance System project.*"
)
