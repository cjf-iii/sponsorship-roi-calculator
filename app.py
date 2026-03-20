"""
Sports Sponsorship ROI Calculator — Streamlit App
===================================================
Interactive web application that predicts brand lift, media value, and ROI
from sports sponsorship deals using trained XGBoost models with SHAP explainability.

This is the main entry point for the portfolio project. It loads a pre-trained
model artifact and provides a sidebar-driven interface for exploring predictions.

Sections:
  1. Sidebar: Deal parameter inputs (sport, market, spend, etc.)
  2. Prediction Results: Three metric cards with predicted outcomes
  3. SHAP Explainability: Waterfall chart showing feature contributions
  4. Comparable Deals: Table of similar historical deals from the dataset
  5. Under the Hood: Model performance metrics, feature importance, residuals

Usage:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from xgboost import XGBRegressor

# ──────────────────────────────────────────────────────────────────────
# Force matplotlib to use a non-interactive backend so it doesn't try
# to open GUI windows — required for Streamlit's server environment
# ──────────────────────────────────────────────────────────────────────
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────
# Project paths — resolved relative to this script so the app works
# regardless of the working directory when launched
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.joblib"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "sponsorship_deals.csv"

# ──────────────────────────────────────────────────────────────────────
# Categorical options — must match the values used during data generation
# and training so that one-hot encoding produces the same columns
# ──────────────────────────────────────────────────────────────────────
SPORTS = ["NFL", "NBA", "MLB", "MLS", "NCAA", "Tennis", "Golf", "Boxing/MMA"]
MARKETS = [
    "Los Angeles", "New York", "Chicago", "Miami", "Dallas",
    "San Francisco", "Boston", "Atlanta", "Phoenix", "Denver"
]
DEAL_TYPES = [
    "Jersey patch", "Venue signage", "Broadcast integration",
    "Digital/social", "Experiential", "Naming rights"
]
BRAND_CATEGORIES = [
    "Sports betting", "Insurance", "Fintech", "CPG",
    "Telecom", "Auto", "Alcohol", "Tech"
]
ACTIVATION_CHANNELS = ["on_air", "digital", "social", "experiential", "dooh"]

# ──────────────────────────────────────────────────────────────────────
# Human-readable labels for the activation channel checkboxes
# ──────────────────────────────────────────────────────────────────────
CHANNEL_LABELS = {
    "on_air": "On-Air / Broadcast",
    "digital": "Digital",
    "social": "Social Media",
    "experiential": "Experiential / In-Venue",
    "dooh": "DOOH (Digital Out-of-Home)",
}


# ══════════════════════════════════════════════════════════════════════
# CACHING: Load model artifacts and data once, then reuse across reruns
# st.cache_resource is used for ML models (non-serializable objects)
# st.cache_data is used for DataFrames and JSON (serializable data)
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """
    Load the trained model artifact from disk.
    Returns the full artifact dict containing models, feature columns,
    and preprocessing metadata.
    """
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Run `python model/train.py` first "
            "(or use `bash run.sh` to generate data and train)."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """
    Load the synthetic sponsorship deals dataset for the comparable deals
    table and residual analysis. Returns a pandas DataFrame.
    """
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Run `python data/generate_data.py` first "
            "(or use `bash run.sh`)."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    """
    Load model evaluation metrics (MAE, R², MAPE) saved during training.
    Returns a dict keyed by target variable name.
    """
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def build_input_dataframe(
    sport: str,
    market: str,
    deal_type: str,
    annual_spend: float,
    deal_length: int,
    brand_category: str,
    channels: dict[str, int],
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame from the user's sidebar inputs.

    This mimics the raw data format before feature engineering, with one row
    containing all the same columns as the training data. The feature
    engineering step (below) will then transform it identically to how
    the training data was processed.

    Args:
        sport: Selected sport league/organization
        market: Selected market/city
        deal_type: Type of sponsorship deal
        annual_spend: Annual spend in dollars
        deal_length: Deal length in years (1-5)
        brand_category: Brand's industry category
        channels: Dict of channel_name → 0/1 flags

    Returns:
        Single-row DataFrame ready for feature engineering
    """
    # ──────────────────────────────────────────────────────────────────
    # Compute derived numeric features the same way the data generator does:
    # audience_reach and social_following are estimated from sport + market
    # ──────────────────────────────────────────────────────────────────
    sport_reach_mult = {
        "NFL": 1.0, "NBA": 0.85, "MLB": 0.70, "MLS": 0.35,
        "NCAA": 0.55, "Tennis": 0.30, "Golf": 0.25, "Boxing/MMA": 0.40,
    }
    sport_social_mult = {
        "NFL": 1.0, "NBA": 1.2, "MLB": 0.7, "MLS": 0.9,
        "NCAA": 0.6, "Tennis": 0.8, "Golf": 0.5, "Boxing/MMA": 1.1,
    }
    market_pop_scale = {
        "Los Angeles": 1.10, "New York": 1.15, "Chicago": 0.95,
        "Miami": 0.85, "Dallas": 0.90, "San Francisco": 0.88,
        "Boston": 0.82, "Atlanta": 0.80, "Phoenix": 0.75, "Denver": 0.72,
    }

    # Audience reach: base × sport popularity × market size × spend scaling
    base_reach = 2_000_000 * sport_reach_mult[sport] * market_pop_scale[market]
    spend_boost = np.log10(annual_spend / 50_000 + 1) * 0.3
    audience_reach = int(base_reach * (1 + spend_boost))

    # Social following: base × sport social engagement × market size
    base_social = 500_000 * sport_social_mult[sport] * market_pop_scale[market]
    social_following = int(base_social)

    row = {
        "sport": sport,
        "market": market,
        "deal_type": deal_type,
        "annual_spend": annual_spend,
        "deal_length_years": deal_length,
        "audience_reach": audience_reach,
        "social_following": social_following,
        "brand_category": brand_category,
    }
    # Add activation channel flags
    row.update(channels)

    return pd.DataFrame([row])


def engineer_features_for_prediction(
    input_df: pd.DataFrame,
    feature_cols: list[str],
    log_transform_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training to a single input row.

    This must produce a DataFrame with exactly the same columns (in the same order)
    as the training feature matrix. Columns that don't exist in the input
    (e.g., one-hot columns for other categories) are filled with 0.

    Args:
        input_df: Single-row DataFrame from build_input_dataframe()
        feature_cols: List of column names the model expects
        log_transform_cols: Columns to log-transform
        categorical_cols: Columns to one-hot encode

    Returns:
        Single-row DataFrame with all expected feature columns
    """
    df = input_df.copy()

    # --- Log-transform numeric features ---
    for col in log_transform_cols:
        df[f"log_{col}"] = np.log1p(df[col])

    # --- Interaction feature: sport × market ---
    df["sport_market"] = df["sport"] + "_" + df["market"]

    # --- Activation count ---
    binary_cols = ["on_air", "digital", "social", "experiential", "dooh"]
    df["activation_count"] = df[binary_cols].sum(axis=1)

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=categorical_cols + ["sport_market"], drop_first=True)

    # --- Drop raw numeric columns (keep log versions) ---
    df = df.drop(columns=log_transform_cols, errors="ignore")

    # ──────────────────────────────────────────────────────────────────
    # Align columns with what the model expects:
    # - Add missing columns (set to 0 — means "this category is not active")
    # - Remove extra columns that the model doesn't use
    # - Reorder to match the training column order exactly
    # Using pd.concat to add all missing columns at once avoids DataFrame
    # fragmentation warnings from repeated single-column inserts
    # ──────────────────────────────────────────────────────────────────
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, missing_df], axis=1)

    # Select only the columns the model expects, in the right order
    df = df[feature_cols]

    return df


def find_comparable_deals(
    data: pd.DataFrame,
    sport: str,
    market: str,
    deal_type: str,
    annual_spend: float,
    n: int = 10,
) -> pd.DataFrame:
    """
    Find the most similar deals in the historical dataset.

    Similarity is defined by:
    1. Same sport (required — hard filter)
    2. Same market OR same deal type (soft filter — at least one must match)
    3. Annual spend within 2x of the input (order of magnitude proximity)

    If fewer than n deals match the strict criteria, the filters are relaxed
    progressively until we have enough results.

    Args:
        data: Full historical dataset
        sport: Selected sport
        market: Selected market
        deal_type: Selected deal type
        annual_spend: Selected annual spend
        n: Number of comparable deals to return

    Returns:
        DataFrame of the top n most similar deals, sorted by spend proximity
    """
    # Start with same-sport filter (most important for comparability)
    mask = data["sport"] == sport

    # Add spend proximity: within 0.5x to 2x of the input spend
    spend_low = annual_spend * 0.5
    spend_high = annual_spend * 2.0
    spend_mask = (data["annual_spend"] >= spend_low) & (data["annual_spend"] <= spend_high)

    # Try strict filter first: same sport + similar spend + (same market OR same deal type)
    strict_mask = mask & spend_mask & ((data["market"] == market) | (data["deal_type"] == deal_type))
    result = data[strict_mask]

    # If not enough results, relax to just same sport + similar spend
    if len(result) < n:
        result = data[mask & spend_mask]

    # If still not enough, relax to just same sport
    if len(result) < n:
        result = data[mask]

    # Sort by spend proximity (closest spend first) and return top n
    result = result.copy()
    result["_spend_dist"] = np.abs(result["annual_spend"] - annual_spend)
    result = result.sort_values("_spend_dist").head(n)
    result = result.drop(columns=["_spend_dist"])

    return result


def format_currency(value: float) -> str:
    """Format a dollar amount into human-readable shorthand (e.g., $1.5M, $500K)."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.0f}K"
    else:
        return f"${value:,.0f}"


# ══════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    """
    Main Streamlit application entry point.
    Sets up the page config, sidebar inputs, and all display sections.
    """

    # ──────────────────────────────────────────────────────────────────
    # Page configuration — must be the first Streamlit call
    # ──────────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Sponsorship ROI Calculator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ──────────────────────────────────────────────────────────────────
    # Custom CSS for professional styling and dark-mode compatibility
    # The metric cards use a subtle background to stand out without
    # clashing with either light or dark Streamlit themes
    # ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        /* Metric card container styling */
        div[data-testid="stMetric"] {
            background-color: rgba(28, 131, 225, 0.08);
            border: 1px solid rgba(28, 131, 225, 0.15);
            padding: 16px 20px;
            border-radius: 10px;
        }
        /* Make metric values larger and more prominent */
        div[data-testid="stMetric"] label {
            font-size: 0.9rem !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
        }
        /* Style the sidebar header */
        section[data-testid="stSidebar"] h1 {
            font-size: 1.3rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(28, 131, 225, 0.3);
        }
        /* Comparable deals table styling */
        .dataframe {
            font-size: 0.85rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # Header — project title and brief description
    # ──────────────────────────────────────────────────────────────────
    st.title("Sports Sponsorship ROI Calculator")
    st.markdown(
        "Predict **media value**, **brand lift**, and **ROI** for sports sponsorship deals "
        "using XGBoost with SHAP explainability. Configure a deal in the sidebar to see predictions."
    )

    # ──────────────────────────────────────────────────────────────────
    # Load model artifacts and data (cached after first load)
    # ──────────────────────────────────────────────────────────────────
    artifact = load_model()
    data = load_data()
    metrics = load_metrics()

    models = artifact["models"]
    feature_cols = artifact["feature_cols"]
    categorical_cols = artifact["categorical_cols"]
    log_transform_cols = artifact["log_transform_cols"]

    # ══════════════════════════════════════════════════════════════════
    # SIDEBAR: Deal parameter inputs
    # Each input corresponds to a feature in the model. The sidebar
    # provides an intuitive interface for configuring a hypothetical deal.
    # ══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("Deal Parameters")

        # --- Sport selection ---
        sport = st.selectbox(
            "Sport / League",
            SPORTS,
            index=0,
            help="Select the sport or league for the sponsorship deal",
        )

        # --- Market selection ---
        market = st.selectbox(
            "Market",
            MARKETS,
            index=1,  # Default to New York
            help="Primary market where the sponsorship will be activated",
        )

        # --- Deal type ---
        deal_type = st.selectbox(
            "Deal Type",
            DEAL_TYPES,
            index=0,
            help="Primary type of sponsorship inventory",
        )

        # --- Annual spend with log-scale slider ---
        # Log scale is critical here because sponsorship deals span $50K to $50M
        # A linear slider would make it impossible to select values below $1M
        st.markdown("**Annual Spend**")
        log_spend = st.slider(
            "Annual Spend (log scale)",
            min_value=np.log10(50_000),
            max_value=np.log10(50_000_000),
            value=np.log10(2_000_000),  # Default: $2M
            step=0.05,
            format="%.2f",
            label_visibility="collapsed",
        )
        annual_spend = 10 ** log_spend
        st.caption(f"**{format_currency(annual_spend)}** / year")

        # --- Deal length ---
        deal_length = st.slider(
            "Deal Length (years)",
            min_value=1,
            max_value=5,
            value=3,
            help="Contract duration in years",
        )

        # --- Brand category ---
        brand_category = st.selectbox(
            "Brand Category",
            BRAND_CATEGORIES,
            index=0,
            help="Industry category of the sponsoring brand",
        )

        # --- Activation channels (multiselect checkboxes) ---
        st.markdown("**Activation Channels**")
        channels = {}
        for ch in ACTIVATION_CHANNELS:
            # Default: digital and social are checked (most common baseline)
            default = ch in ("digital", "social")
            channels[ch] = int(st.checkbox(CHANNEL_LABELS[ch], value=default))

        st.divider()
        st.caption("Built by CJ Fleming | Columbia AI '24")

    # ══════════════════════════════════════════════════════════════════
    # PREDICTION: Build input, engineer features, run through models
    # ══════════════════════════════════════════════════════════════════

    # Build the raw input row from sidebar selections
    input_df = build_input_dataframe(
        sport=sport,
        market=market,
        deal_type=deal_type,
        annual_spend=annual_spend,
        deal_length=deal_length,
        brand_category=brand_category,
        channels=channels,
    )

    # Apply the same feature engineering pipeline used during training
    X_pred = engineer_features_for_prediction(
        input_df, feature_cols, log_transform_cols, categorical_cols
    )

    # Run predictions through each target-specific model
    predictions = {}
    for target, model in models.items():
        pred = model.predict(X_pred)[0]
        predictions[target] = pred

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: Prediction Results — Three big metric cards
    # These are the headline numbers that immediately communicate the
    # predicted deal outcomes to the user
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        mvr = predictions["media_value_ratio"]
        # Show the implied total media value as context
        implied_value = format_currency(annual_spend * mvr)
        st.metric(
            label="Media Value Ratio",
            value=f"{mvr:.2f}x",
            delta=f"{implied_value} implied value",
        )

    with col2:
        lift = predictions["brand_lift_pct"]
        # Classify the lift level for quick interpretation
        lift_level = "High" if lift > 7 else "Medium" if lift > 4 else "Moderate"
        st.metric(
            label="Brand Lift",
            value=f"{lift:.1f}%",
            delta=f"{lift_level} impact",
        )

    with col3:
        roi = predictions["roi_score"]
        # Color-code ROI classification for intuitive understanding
        roi_label = "Excellent" if roi > 70 else "Good" if roi > 50 else "Average" if roi > 35 else "Below Avg"
        st.metric(
            label="ROI Score",
            value=f"{roi:.0f}/100",
            delta=roi_label,
        )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: SHAP Explainability — The "interview money shot"
    # SHAP (SHapley Additive exPlanations) shows exactly which features
    # pushed the prediction up or down from the baseline. This is the
    # key differentiator that demonstrates ML literacy.
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("SHAP Explainability")
    st.markdown(
        "How each feature contributes to the predicted **ROI Score**. "
        "Red bars push the score up; blue bars push it down."
    )

    try:
        # Create a SHAP TreeExplainer for the ROI model
        # TreeExplainer is optimized for tree-based models like XGBoost
        # and computes exact SHAP values (not approximations)
        roi_model = models["roi_score"]
        explainer = shap.TreeExplainer(roi_model)
        shap_values = explainer.shap_values(X_pred)

        # ──────────────────────────────────────────────────────────────
        # Build a SHAP Explanation object for the waterfall plot
        # We create human-readable feature names by cleaning up the
        # one-hot encoded column names (e.g., "sport_NBA" → "Sport: NBA")
        # ──────────────────────────────────────────────────────────────
        feature_names = []
        for col in feature_cols:
            # Clean up one-hot encoded column names for readability
            name = col.replace("_", " ").title()
            # Shorten overly long sport_market interaction names
            if "Sport Market" in name:
                parts = col.split("_", 2)
                if len(parts) >= 3:
                    name = f"Sport×Market: {parts[1]}_{parts[2]}"
                    name = name.replace("_", " ")
            elif "Log " in name:
                name = name.replace("Log ", "Log(") + ")"
            feature_names.append(name)

        # Create the SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_pred.values[0],
            feature_names=feature_names,
        )

        # ──────────────────────────────────────────────────────────────
        # Render the SHAP waterfall chart
        # max_display=15 shows the top 15 most impactful features
        # to keep the chart readable without overwhelming the viewer
        # ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        # Adjust layout to prevent label clipping
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    except Exception as e:
        # Graceful degradation: if SHAP fails, show a helpful message
        # rather than crashing the entire app
        st.warning(
            f"SHAP visualization could not be generated: {str(e)}. "
            "This can happen with certain feature configurations. "
            "The predictions above are still valid."
        )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: Comparable Deals
    # Shows similar historical deals from the dataset so the user can
    # see how their hypothetical deal compares to "real" (synthetic) data
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Comparable Deals")
    st.markdown(
        f"Historical deals similar to **{sport}** in **{market}** "
        f"with **{format_currency(annual_spend)}/yr** spend."
    )

    comparables = find_comparable_deals(data, sport, market, deal_type, annual_spend, n=10)

    if len(comparables) > 0:
        # Format the display DataFrame with readable column names and currency formatting
        display_df = comparables[
            ["sport", "market", "deal_type", "annual_spend", "deal_length_years",
             "brand_category", "media_value_ratio", "brand_lift_pct", "roi_score"]
        ].copy()

        display_df.columns = [
            "Sport", "Market", "Deal Type", "Annual Spend", "Length (yrs)",
            "Brand Category", "Media Value Ratio", "Brand Lift %", "ROI Score"
        ]

        # Format the spend column as readable currency
        display_df["Annual Spend"] = display_df["Annual Spend"].apply(format_currency)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No comparable deals found with the current parameters.")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: Under the Hood (Expander)
    # Technical details for the interviewer who wants to dig deeper:
    # model performance metrics, feature importance, and residual analysis
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("Under the Hood — Model Performance & Diagnostics", expanded=False):

        # ──────────────────────────────────────────────────────────────
        # 4a. Model Evaluation Metrics
        # Shows MAE, R², and MAPE for each target variable
        # These demonstrate that the model is well-calibrated
        # ──────────────────────────────────────────────────────────────
        st.markdown("#### Model Evaluation Metrics")
        if metrics:
            metric_cols = st.columns(3)
            target_labels = {
                "media_value_ratio": "Media Value Ratio",
                "brand_lift_pct": "Brand Lift %",
                "roi_score": "ROI Score",
            }

            for idx, (target, label) in enumerate(target_labels.items()):
                with metric_cols[idx]:
                    m = metrics[target]
                    st.markdown(f"**{label}**")
                    st.markdown(f"- **R²:** {m['r2']:.4f}")
                    st.markdown(f"- **MAE:** {m['mae']:.4f}")
                    st.markdown(f"- **MAPE:** {m['mape']:.2f}%")
        else:
            st.info("Metrics file not found. Re-run training to generate.")

        st.markdown("---")

        # ──────────────────────────────────────────────────────────────
        # 4b. Feature Importance Chart
        # XGBoost's built-in feature importance (gain-based) shows which
        # features contribute most to reducing prediction error across
        # all trees in the ensemble
        # ──────────────────────────────────────────────────────────────
        st.markdown("#### Feature Importance (ROI Score Model)")

        roi_model = models["roi_score"]
        importance = roi_model.feature_importances_

        # Get top 20 features by importance for a readable chart
        top_n = 20
        top_indices = np.argsort(importance)[-top_n:]
        top_features = [feature_cols[i] for i in top_indices]
        top_importance = importance[top_indices]

        # Clean up feature names for display
        clean_names = []
        for name in top_features:
            clean = name.replace("_", " ").title()
            if len(clean) > 30:
                clean = clean[:27] + "..."
            clean_names.append(clean)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        bars = ax_imp.barh(range(len(clean_names)), top_importance, color="#1c83e1", alpha=0.85)
        ax_imp.set_yticks(range(len(clean_names)))
        ax_imp.set_yticklabels(clean_names, fontsize=9)
        ax_imp.set_xlabel("Feature Importance (Gain)", fontsize=10)
        ax_imp.set_title(f"Top {top_n} Features by Importance", fontsize=12)
        ax_imp.spines["top"].set_visible(False)
        ax_imp.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

        st.markdown("---")

        # ──────────────────────────────────────────────────────────────
        # 4c. Residual Plot
        # Shows prediction errors on the test set to diagnose whether
        # the model has systematic biases (e.g., under-predicting high
        # ROI deals or over-predicting low ROI deals)
        # ──────────────────────────────────────────────────────────────
        st.markdown("#### Residual Analysis (ROI Score)")

        # Re-run prediction on the full dataset to get residuals
        # This is a simplification — ideally we'd use only test set,
        # but for a portfolio demo this shows the overall fit quality
        from model.train import feature_engineering, TARGETS
        df_full = data.copy()
        y_true = df_full["roi_score"].values

        df_eng = feature_engineering(df_full)
        eng_feature_cols = [c for c in df_eng.columns if c not in TARGETS]

        # Align columns with what the model expects
        for col in feature_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0
        X_full = df_eng[feature_cols]

        y_pred_full = roi_model.predict(X_full)
        residuals = y_true - y_pred_full

        fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel: Predicted vs Actual scatter plot
        ax1.scatter(y_pred_full, y_true, alpha=0.15, s=8, color="#1c83e1")
        # Perfect prediction line (45-degree reference)
        min_val = min(y_true.min(), y_pred_full.min())
        max_val = max(y_true.max(), y_pred_full.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction")
        ax1.set_xlabel("Predicted ROI Score", fontsize=10)
        ax1.set_ylabel("Actual ROI Score", fontsize=10)
        ax1.set_title("Predicted vs Actual", fontsize=12)
        ax1.legend(fontsize=9)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Right panel: Residual distribution (should be centered near 0)
        ax2.hist(residuals, bins=50, color="#1c83e1", alpha=0.75, edgecolor="white")
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
        ax2.set_xlabel("Residual (Actual - Predicted)", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title("Residual Distribution", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_res)
        plt.close()

        # Summary statistics for the residuals
        st.markdown(
            f"**Residual Stats:** Mean = {residuals.mean():.3f}, "
            f"Std = {residuals.std():.3f}, "
            f"Median = {np.median(residuals):.3f}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Footer with project context
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.85rem;'>"
        "Sports Sponsorship ROI Calculator | Built by CJ Fleming | "
        "XGBoost + SHAP | Columbia AI Certificate '24"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
