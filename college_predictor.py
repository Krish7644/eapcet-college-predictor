import pandas as pd
import numpy as np

from logic.suitability import calculate_suitability, classify_risk


def recommend_colleges(
    df,
    user_rank,
    user_category,
    user_gender,
    user_region,
    preferred_district="All",
    top_n=15
):
    """
    Core college recommendation logic.
    Returns a DataFrame of recommended colleges.
    """

    eligible = df[
        (df["category"] == user_category) &
        (df["gender"] == user_gender) &
        (df["cutoff_rank"] >= user_rank)
    ].copy()

    # Region filter
    if user_region != "NON-LOCAL":
        eligible = eligible[eligible["A_REG"] == user_region]

    # District filter
    if preferred_district != "All":
        eligible = eligible[eligible["DIST"] == preferred_district]

    if eligible.empty:
        return pd.DataFrame()

    # Rank gap
    eligible["rank_gap"] = eligible["cutoff_rank"] - user_rank

    # Keep realistic colleges only
    eligible = eligible[eligible["rank_gap"] <= 30000]

    if eligible.empty:
        return pd.DataFrame()

    # Suitability & Risk
    eligible["Suitability %"] = eligible["cutoff_rank"].apply(
        lambda c: calculate_suitability(user_rank, c)
    )

    eligible["Chance"] = eligible["rank_gap"].apply(classify_risk)

    # Sort by suitability
    result = eligible.sort_values(
        by=["Suitability %"],
        ascending=False
    ).head(top_n)

    return result


def check_aspiration_college(
    df,
    college_name,
    branch_code,
    user_rank,
    user_category,
    user_gender
):
    """
    Checks feasibility of a specific college & branch.
    """

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
