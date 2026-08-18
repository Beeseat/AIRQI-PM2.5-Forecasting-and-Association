"""
association_rules.py
Association rule mining over discretized Beijing PM2.5 / weather conditions.

Why association rules instead of hour-to-hour classification:
At hourly resolution PM2.5 barely changes between consecutive readings
(see feature_importance_shap.png: lag_1h dominates every other feature).
Framing this as "classify this hour's AQI band" would mostly reduce to a
trivial persistence problem (predict = last hour's class), which doesn't
show much beyond what the regression benchmark already shows. Association
rule mining asks a different question that doesn't care about temporal
resolution at all: "which combinations of weather/time conditions tend to
co-occur with a given pollution level?" Each hourly row is one transaction
in a market-basket sense; no forecasting horizon is involved.

No third-party frequent-itemset library (mlxtend etc.) was available in
this environment, so Apriori-style candidate generation and rule
extraction are implemented directly below, following the classic
level-wise algorithm (Agrawal & Srikant, 1994): build frequent 1-itemsets,
then repeatedly join + prune to get frequent k-itemsets, then derive
antecedent -> consequent rules with support / confidence / lift.
"""

import itertools
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Step 1: discretize continuous columns into categorical "items"
# ---------------------------------------------------------------------

def discretize_for_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin the lightly-cleaned raw dataframe (see data_loader.load_for_association)
    into categorical items suitable for market-basket-style transactions.
    """
    df = raw_df.copy()

    # PM2.5 severity bins (simplified AQI-style bands, ug/m3)
    df["PM25_LEVEL"] = pd.cut(
        df["pm2.5"],
        bins=[-0.1, 35, 75, 150, 250, np.inf],
        labels=["PM25=Good", "PM25=Moderate", "PM25=Unhealthy",
                "PM25=VeryUnhealthy", "PM25=Hazardous"],
    )

    # Wind direction: already categorical in the raw data (cbwd)
    df["WIND_DIR"] = "WIND=" + df["cbwd"].astype(str)

    # Cumulated wind speed (Iws) -> tertiles (heavily right-skewed, so quantile bins)
    df["WIND_SPD"] = pd.qcut(
        df["Iws"], q=3, labels=["WIND_SPD=Low", "WIND_SPD=Med", "WIND_SPD=High"]
    )

    # Temperature (TEMP, deg C) -> tertiles
    df["TEMP_BIN"] = pd.qcut(
        df["TEMP"], q=3, labels=["TEMP=Low", "TEMP=Med", "TEMP=High"]
    )

    # Pressure (PRES, hPa) -> tertiles
    df["PRES_BIN"] = pd.qcut(
        df["PRES"], q=3, labels=["PRES=Low", "PRES=Med", "PRES=High"]
    )

    # Dew point (DEWP, deg C) -> tertiles
    df["DEWP_BIN"] = pd.qcut(
        df["DEWP"], q=3, labels=["DEWP=Low", "DEWP=Med", "DEWP=High"]
    )

    # Season, from month (Beijing pollution is strongly seasonal - winter heating)
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["SEASON"] = "SEASON=" + df["month"].map(season_map)

    # Time of day, from hour
    def hour_bucket(h):
        if 5 <= h < 11:
            return "TIME=Morning"
        elif 11 <= h < 17:
            return "TIME=Afternoon"
        elif 17 <= h < 22:
            return "TIME=Evening"
        else:
            return "TIME=Night"
    df["TIME_BIN"] = df["hour"].apply(hour_bucket)

    # Precipitation flags (Is/Ir are almost always 0, so binary is more
    # informative here than a tertile split would be)
    df["SNOW_FLAG"] = np.where(df["Is"] > 0, "SNOW=Yes", "SNOW=No")
    df["RAIN_FLAG"] = np.where(df["Ir"] > 0, "RAIN=Yes", "RAIN=No")

    items_df = df[[
        "PM25_LEVEL", "WIND_DIR", "WIND_SPD", "TEMP_BIN",
        "PRES_BIN", "DEWP_BIN", "SEASON", "TIME_BIN", "SNOW_FLAG", "RAIN_FLAG"
    ]].astype(str)

    return items_df


def build_onehot_transactions(items_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the binned items into a boolean transaction matrix."""
    onehot = pd.get_dummies(items_df, prefix="", prefix_sep="")
    return onehot.astype(bool)


# ---------------------------------------------------------------------
# Step 2: Apriori frequent itemset mining (implemented from scratch)
# ---------------------------------------------------------------------

def _generate_candidates(prev_itemsets, k):
    """Join frequent (k-1)-itemsets and prune any candidate whose subsets
    aren't all already frequent (the core Apriori pruning rule)."""
    prev_set = set(prev_itemsets)
    items = sorted(set(itertools.chain.from_iterable(prev_itemsets)))
    candidates = set()

    for combo in itertools.combinations(items, k):
        cand = frozenset(combo)
        if all(frozenset(sub) in prev_set for sub in itertools.combinations(combo, k - 1)):
            candidates.add(cand)
    return candidates


def apriori(onehot: pd.DataFrame, min_support: float = 0.05, max_len: int = 3) -> pd.DataFrame:
    """
    Level-wise Apriori frequent itemset mining (Agrawal & Srikant, 1994).
    Returns a DataFrame with columns: itemsets (frozenset), support (float).
    """
    n = len(onehot)

    support_counts = onehot.sum(axis=0)
    current = {
        frozenset([item]): count / n
        for item, count in support_counts.items()
        if count / n >= min_support
    }

    all_frequent = dict(current)
    k = 2

    while current and k <= max_len:
        candidates = _generate_candidates(list(current.keys()), k)
        next_level = {}
        for cand in candidates:
            cols = list(cand)
            mask = onehot[cols].all(axis=1)
            support = mask.sum() / n
            if support >= min_support:
                next_level[cand] = support
        all_frequent.update(next_level)
        current = next_level
        k += 1

    result = pd.DataFrame(
        [{"itemsets": k, "support": v} for k, v in all_frequent.items()]
    )
    return result.sort_values("support", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Step 3: rule generation (support / confidence / lift)
# ---------------------------------------------------------------------

def generate_rules(frequent_itemsets: pd.DataFrame, min_confidence: float = 0.5) -> pd.DataFrame:
    """Derive antecedent -> consequent rules from frequent itemsets."""
    support_lookup = dict(zip(frequent_itemsets["itemsets"], frequent_itemsets["support"]))
    rules = []

    for itemset, support in support_lookup.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, r):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if antecedent not in support_lookup:
                    continue
                ant_support = support_lookup[antecedent]
                cons_support = support_lookup.get(consequent)
                if cons_support is None or ant_support == 0:
                    continue
                confidence = support / ant_support
                if confidence < min_confidence:
                    continue
                lift = confidence / cons_support
                rules.append({
                    "antecedent": ", ".join(sorted(antecedent)),
                    "consequent": ", ".join(sorted(consequent)),
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                })

    rules_df = pd.DataFrame(rules)
    if len(rules_df):
        rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules_df


# ---------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------

def mine_pollution_rules(raw_df: pd.DataFrame, min_support: float = 0.03,
                          min_confidence: float = 0.5, max_len: int = 3):
    """raw df -> binned items -> transactions -> frequent itemsets -> rules."""
    items_df = discretize_for_transactions(raw_df)
    onehot = build_onehot_transactions(items_df)
    frequent = apriori(onehot, min_support=min_support, max_len=max_len)
    rules = generate_rules(frequent, min_confidence=min_confidence)
    return frequent, rules


def rules_involving_pm25(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rules whose consequent is a PM2.5 severity level - the
    rules most directly relevant to the project's forecasting question."""
    if len(rules_df) == 0:
        return rules_df
    mask = rules_df["consequent"].str.startswith("PM25=")
    return rules_df[mask].reset_index(drop=True)
