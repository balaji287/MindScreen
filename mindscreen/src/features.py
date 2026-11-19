"""Shared feature engineering. Used by training AND the web app so both always match."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Raw behavioural inputs (what a student / college system would supply) ---
RAW_NUM = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]
# Relationship status lifts accuracy by ~3 points (5-fold CV) but flagging students as "at risk"
# partly because of their relationship status is ethically questionable, so it is OFF by default.
INCLUDE_RELATIONSHIP = False
RAW_CAT = ["Gender", "Academic_Level", "Most_Used_Platform"] + (["Relationship_Status"] if INCLUDE_RELATIONSHIP else [])

# Self-reported items of the original survey. NOT used by the model (only in src/ablation.py,
# to show that adding them is unnecessary).
SELF_REPORTED_NUM = ["Conflicts_Over_Social_Media"]
SELF_REPORTED_CAT = ["Affects_Academic_Performance"] + ([] if INCLUDE_RELATIONSHIP else ["Relationship_Status"])

# --- Engineered "digital balance" features (our small novelty) ---
ENG_NUM = ["sleep_deficit", "usage_sleep_ratio", "screen_share_of_waking"]

NUM = RAW_NUM + ENG_NUM
CAT = RAW_CAT
RAW_COLUMNS = RAW_NUM + RAW_CAT

# NOT used on purpose: Addicted_Score (a questionnaire-derived score, r = -0.95 with the
# target -> it would be target leakage AND would bring back the questionnaire bias).
EXCLUDED = ["Student_ID", "Country", "Mental_Health_Score", "Addicted_Score"]  # + all self-reported items

# Features a student can actually change -> used for the "what-if" advice
CONTROLLABLE = {
    "Avg_Daily_Usage_Hours": "Daily social-media time",
    "Sleep_Hours_Per_Night": "Sleep per night",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sleep_deficit"] = (8 - d["Sleep_Hours_Per_Night"]).clip(lower=0)
    d["usage_sleep_ratio"] = d["Avg_Daily_Usage_Hours"] / d["Sleep_Hours_Per_Night"]
    d["screen_share_of_waking"] = d["Avg_Daily_Usage_Hours"] / (24 - d["Sleep_Hours_Per_Night"])
    return d


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
