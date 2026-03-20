"""
Synthetic Sponsorship Deal Data Generator
==========================================
Generates ~5,000 realistic sponsorship deal records with correlated features
based on industry benchmarks from IEG, Nielsen Sports, and Two Circles research.

The generated dataset mimics real-world sponsorship economics:
- NFL/NBA deals command higher media value ratios (bigger audiences, more broadcast time)
- Experiential + social activation combos produce stronger brand lift
- Naming rights have the highest spend ceilings but lower per-dollar ROI
- Market size amplifies audience reach and social following
- Brand categories have different baseline affinities for sports sponsorship

Output: data/sponsorship_deals.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Seed for reproducibility — changing this regenerates the entire dataset
# but maintains the same statistical distributions and correlations
# ──────────────────────────────────────────────────────────────────────
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# Configuration: categorical values and their relative weights
# Weights approximate real-world sponsorship deal volume by category
# ──────────────────────────────────────────────────────────────────────
SPORTS = ["NFL", "NBA", "MLB", "MLS", "NCAA", "Tennis", "Golf", "Boxing/MMA"]
SPORT_WEIGHTS = [0.22, 0.20, 0.15, 0.10, 0.12, 0.07, 0.07, 0.07]

MARKETS = [
    "Los Angeles", "New York", "Chicago", "Miami", "Dallas",
    "San Francisco", "Boston", "Atlanta", "Phoenix", "Denver"
]
MARKET_WEIGHTS = [0.15, 0.18, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.06, 0.06]

DEAL_TYPES = [
    "Jersey patch", "Venue signage", "Broadcast integration",
    "Digital/social", "Experiential", "Naming rights"
]
DEAL_TYPE_WEIGHTS = [0.15, 0.20, 0.18, 0.22, 0.15, 0.10]

BRAND_CATEGORIES = [
    "Sports betting", "Insurance", "Fintech", "CPG",
    "Telecom", "Auto", "Alcohol", "Tech"
]
BRAND_WEIGHTS = [0.18, 0.12, 0.14, 0.12, 0.10, 0.10, 0.12, 0.12]

# ──────────────────────────────────────────────────────────────────────
# Sport-level parameters: these drive the baseline economics for each sport.
# media_mult: how much earned media value a sport typically generates per dollar
# lift_base: baseline brand awareness lift (%) for an average deal in that sport
# reach_mult: relative audience reach multiplier (NFL=1.0 as reference)
# social_mult: relative social media engagement multiplier
# ──────────────────────────────────────────────────────────────────────
SPORT_PARAMS = {
    "NFL":        {"media_mult": 3.2, "lift_base": 6.5, "reach_mult": 1.0,  "social_mult": 1.0},
    "NBA":        {"media_mult": 3.0, "lift_base": 6.0, "reach_mult": 0.85, "social_mult": 1.2},
    "MLB":        {"media_mult": 2.5, "lift_base": 5.0, "reach_mult": 0.70, "social_mult": 0.7},
    "MLS":        {"media_mult": 2.0, "lift_base": 4.0, "reach_mult": 0.35, "social_mult": 0.9},
    "NCAA":       {"media_mult": 2.3, "lift_base": 4.5, "reach_mult": 0.55, "social_mult": 0.6},
    "Tennis":     {"media_mult": 2.2, "lift_base": 3.5, "reach_mult": 0.30, "social_mult": 0.8},
    "Golf":       {"media_mult": 2.0, "lift_base": 3.0, "reach_mult": 0.25, "social_mult": 0.5},
    "Boxing/MMA": {"media_mult": 2.4, "lift_base": 5.5, "reach_mult": 0.40, "social_mult": 1.1},
}

# ──────────────────────────────────────────────────────────────────────
# Market-level parameters: population-based scaling for reach/social
# Larger markets amplify both audience reach and social following
# ──────────────────────────────────────────────────────────────────────
MARKET_PARAMS = {
    "Los Angeles":   {"pop_scale": 1.10},
    "New York":      {"pop_scale": 1.15},
    "Chicago":       {"pop_scale": 0.95},
    "Miami":         {"pop_scale": 0.85},
    "Dallas":        {"pop_scale": 0.90},
    "San Francisco": {"pop_scale": 0.88},
    "Boston":        {"pop_scale": 0.82},
    "Atlanta":       {"pop_scale": 0.80},
    "Phoenix":       {"pop_scale": 0.75},
    "Denver":        {"pop_scale": 0.72},
}

# ──────────────────────────────────────────────────────────────────────
# Deal type modifiers: each deal type has different ROI and brand lift profiles
# roi_mod: multiplier applied to the final ROI score
# lift_mod: additive modifier to brand lift percentage
# spend_range: (min, max) typical annual spend in dollars for this deal type
# ──────────────────────────────────────────────────────────────────────
DEAL_TYPE_PARAMS = {
    "Jersey patch":          {"roi_mod": 1.15, "lift_mod": 1.5,  "spend_range": (500_000, 30_000_000)},
    "Venue signage":         {"roi_mod": 0.90, "lift_mod": 0.5,  "spend_range": (100_000, 10_000_000)},
    "Broadcast integration": {"roi_mod": 1.10, "lift_mod": 1.0,  "spend_range": (200_000, 15_000_000)},
    "Digital/social":        {"roi_mod": 1.20, "lift_mod": 2.0,  "spend_range": (50_000,  5_000_000)},
    "Experiential":          {"roi_mod": 1.05, "lift_mod": 2.5,  "spend_range": (100_000, 8_000_000)},
    "Naming rights":         {"roi_mod": 0.85, "lift_mod": 0.8,  "spend_range": (5_000_000, 50_000_000)},
}

N_RECORDS = 5000


def generate_dataset() -> pd.DataFrame:
    """
    Generate the full synthetic dataset with realistic feature correlations.

    The generation process works in stages:
    1. Sample categorical features (sport, market, deal type, brand category)
    2. Generate continuous features (spend, reach, social) with sport/market correlations
    3. Generate activation channel flags with deal-type-appropriate probabilities
    4. Compute target variables (media value ratio, brand lift, ROI score) with noise
    """

    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Sample categorical features using weighted distributions
    # Weights ensure the dataset reflects real-world deal volume patterns
    # (e.g., more NFL/NBA deals than Golf/Tennis deals)
    # ──────────────────────────────────────────────────────────────────
    sports = np.random.choice(SPORTS, size=N_RECORDS, p=SPORT_WEIGHTS)
    markets = np.random.choice(MARKETS, size=N_RECORDS, p=MARKET_WEIGHTS)
    deal_types = np.random.choice(DEAL_TYPES, size=N_RECORDS, p=DEAL_TYPE_WEIGHTS)
    brand_cats = np.random.choice(BRAND_CATEGORIES, size=N_RECORDS, p=BRAND_WEIGHTS)
    deal_lengths = np.random.choice([1, 2, 3, 4, 5], size=N_RECORDS, p=[0.15, 0.25, 0.30, 0.20, 0.10])

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Generate continuous features with correlations
    # Annual spend follows a log-normal distribution clipped to each deal type's
    # realistic range. Audience reach and social following are derived from
    # spend magnitude, sport popularity, and market size.
    # ──────────────────────────────────────────────────────────────────
    annual_spends = []
    audience_reaches = []
    social_followings = []

    for i in range(N_RECORDS):
        sport = sports[i]
        market = markets[i]
        deal_type = deal_types[i]

        # --- Annual Spend ---
        # Log-normal centered within each deal type's spend range
        # This creates the heavy right tail seen in real sponsorship markets
        dt_params = DEAL_TYPE_PARAMS[deal_type]
        spend_min, spend_max = dt_params["spend_range"]
        log_mean = (np.log(spend_min) + np.log(spend_max)) / 2
        log_std = (np.log(spend_max) - np.log(spend_min)) / 4  # 95% within range
        raw_spend = np.random.lognormal(mean=log_mean, sigma=log_std)
        spend = np.clip(raw_spend, spend_min, spend_max)
        annual_spends.append(round(spend, -3))  # Round to nearest thousand

        # --- Audience Reach ---
        # Base reach scales with sport popularity and market size
        # A random noise factor (±30%) accounts for team-level variation
        sp = SPORT_PARAMS[sport]
        mp = MARKET_PARAMS[market]
        base_reach = 2_000_000 * sp["reach_mult"] * mp["pop_scale"]
        # Bigger deals in bigger sports reach more people
        spend_boost = np.log10(spend / 50_000 + 1) * 0.3
        noise = np.random.uniform(0.7, 1.3)
        reach = base_reach * (1 + spend_boost) * noise
        audience_reaches.append(int(reach))

        # --- Social Following ---
        # Social following correlates with sport's social multiplier,
        # market size, and has a heavy tail (some athletes/teams go viral)
        base_social = 500_000 * sp["social_mult"] * mp["pop_scale"]
        social_noise = np.random.lognormal(mean=0, sigma=0.5)
        social = base_social * social_noise
        social_followings.append(int(np.clip(social, 10_000, 50_000_000)))

    # ──────────────────────────────────────────────────────────────────
    # Stage 3: Generate activation channel flags
    # Each deal has a set of activation channels (on_air, digital, social,
    # experiential, dooh). Probabilities vary by deal type — e.g., "Digital/social"
    # deals almost always have digital and social channels active, while
    # "Venue signage" deals are more likely to include DOOH.
    # ──────────────────────────────────────────────────────────────────
    channel_probs = {
        "Jersey patch":          {"on_air": 0.8, "digital": 0.7, "social": 0.7, "experiential": 0.3, "dooh": 0.4},
        "Venue signage":         {"on_air": 0.4, "digital": 0.3, "social": 0.3, "experiential": 0.2, "dooh": 0.8},
        "Broadcast integration": {"on_air": 0.95, "digital": 0.6, "social": 0.5, "experiential": 0.2, "dooh": 0.3},
        "Digital/social":        {"on_air": 0.2, "digital": 0.95, "social": 0.95, "experiential": 0.4, "dooh": 0.2},
        "Experiential":          {"on_air": 0.3, "digital": 0.6, "social": 0.7, "experiential": 0.95, "dooh": 0.5},
        "Naming rights":         {"on_air": 0.7, "digital": 0.5, "social": 0.4, "experiential": 0.6, "dooh": 0.9},
    }

    activation_flags = {ch: [] for ch in ["on_air", "digital", "social", "experiential", "dooh"]}

    for i in range(N_RECORDS):
        probs = channel_probs[deal_types[i]]
        for ch in activation_flags:
            activation_flags[ch].append(int(np.random.random() < probs[ch]))

    # ──────────────────────────────────────────────────────────────────
    # Stage 4: Compute target variables
    # These are the values the ML model will learn to predict.
    # Each target is computed from a combination of features with
    # controlled noise to make the prediction task non-trivial.
    # ──────────────────────────────────────────────────────────────────
    media_value_ratios = []
    brand_lift_pcts = []
    roi_scores = []

    for i in range(N_RECORDS):
        sport = sports[i]
        deal_type = deal_types[i]
        spend = annual_spends[i]
        sp = SPORT_PARAMS[sport]
        dt = DEAL_TYPE_PARAMS[deal_type]

        # --- Media Value Ratio ---
        # Base comes from sport's media multiplier (NFL ~3.2x, Golf ~2.0x)
        # Smaller deals get slightly higher ratios (diminishing returns at scale)
        # Digital/social channels boost earned media; Gaussian noise adds uncertainty
        base_mvr = sp["media_mult"]
        spend_effect = -0.15 * np.log10(spend / 1_000_000)  # Diminishing returns
        channel_boost = 0.2 * activation_flags["digital"][i] + 0.15 * activation_flags["social"][i]
        noise = np.random.normal(0, 0.3)
        mvr = max(0.5, base_mvr + spend_effect + channel_boost + noise)
        media_value_ratios.append(round(mvr, 2))

        # --- Brand Lift Percentage ---
        # Base from sport's lift parameter + deal type modifier
        # Experiential + social combo gives a synergy bonus (real-world finding)
        # Longer deals accumulate more lift; noise reflects measurement uncertainty
        base_lift = sp["lift_base"] + dt["lift_mod"]
        synergy = 1.5 if (activation_flags["experiential"][i] and activation_flags["social"][i]) else 0
        length_bonus = 0.3 * (deal_lengths[i] - 1)  # Each extra year adds ~0.3%
        noise = np.random.normal(0, 1.0)
        lift = max(0.5, base_lift + synergy + length_bonus + noise)
        brand_lift_pcts.append(round(lift, 2))

        # --- ROI Score (1-100) ---
        # Composite score combining media value efficiency, brand lift achieved,
        # deal type modifier, and activation breadth (more channels = higher ROI)
        # Scaled to 1-100 range and clipped
        media_component = mvr * 10  # Higher media multiplier → higher ROI
        lift_component = lift * 3    # Brand lift contributes to ROI
        deal_mod = dt["roi_mod"]
        channel_count = sum(activation_flags[ch][i] for ch in activation_flags)
        activation_bonus = channel_count * 2  # Each active channel adds ~2 points
        raw_roi = (media_component + lift_component + activation_bonus) * deal_mod
        noise = np.random.normal(0, 5)
        roi = np.clip(raw_roi + noise, 1, 100)
        roi_scores.append(round(roi, 1))

    # ──────────────────────────────────────────────────────────────────
    # Assemble the final DataFrame and save to CSV
    # Column order: categoricals → continuous features → binary flags → targets
    # ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "sport": sports,
        "market": markets,
        "deal_type": deal_types,
        "annual_spend": annual_spends,
        "deal_length_years": deal_lengths,
        "audience_reach": audience_reaches,
        "social_following": social_followings,
        "brand_category": brand_cats,
        "on_air": activation_flags["on_air"],
        "digital": activation_flags["digital"],
        "social": activation_flags["social"],
        "experiential": activation_flags["experiential"],
        "dooh": activation_flags["dooh"],
        "media_value_ratio": media_value_ratios,
        "brand_lift_pct": brand_lift_pcts,
        "roi_score": roi_scores,
    })

    return df


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────
    # Main execution: generate, validate, save, and print summary stats
    # ──────────────────────────────────────────────────────────────────
    print("Generating synthetic sponsorship deal data...")

    df = generate_dataset()

    # Save to CSV in the data/ directory (same directory as this script)
    output_path = Path(__file__).parent / "sponsorship_deals.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} records → {output_path}")
    print(f"\nTarget variable summary:")
    print(f"  media_value_ratio: mean={df['media_value_ratio'].mean():.2f}, "
          f"std={df['media_value_ratio'].std():.2f}, "
          f"range=[{df['media_value_ratio'].min():.2f}, {df['media_value_ratio'].max():.2f}]")
    print(f"  brand_lift_pct:    mean={df['brand_lift_pct'].mean():.2f}, "
          f"std={df['brand_lift_pct'].std():.2f}, "
          f"range=[{df['brand_lift_pct'].min():.2f}, {df['brand_lift_pct'].max():.2f}]")
    print(f"  roi_score:         mean={df['roi_score'].mean():.1f}, "
          f"std={df['roi_score'].std():.1f}, "
          f"range=[{df['roi_score'].min():.1f}, {df['roi_score'].max():.1f}]")
    print(f"\nSport distribution:")
    print(df["sport"].value_counts().to_string())
