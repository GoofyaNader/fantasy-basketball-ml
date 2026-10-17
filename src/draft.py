"""Season-long draft projections: fantasy points per game x games played.

One row per player per season. For each season, the features come only
from the seasons before it (stats from up to 3 prior years, age, usage,
late-season form), and the targets are that season's FP per game and
games played.

Usage:
    python -m src.draft    # backtest, train final models, print top 30
"""
import json

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerbiostats, playerindex
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CURRENT_SEASON, MODELS_DIR, RAW_DIR, REPORTS_DIR, SEASONS
from src.data import load_season
from src.features import fantasy_points

SEASON_GAMES = 82
# More history than the weekly model: this dataset has one row per player-season,
# so every extra season matters. 2019-20 and 2020-21 were shortened by COVID,
# so games played is measured as a share of the team's games.
DRAFT_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21"] + SEASONS
VALID_SEASON = "2024-25"   # choose the model type here...
TEST_SEASON = "2025-26"    # ...and report results here
FEATURES = [
    "age",
    "fp_pg_1", "fp_pg_2", "fp_pg_3",
    "gp_share_1", "gp_share_2", "gp_share_3",
    "min_pg_1", "min_pg_2", "min_pg_3",
    "fp_per_min_1", "fp_late_1", "usg_1", "ts_1",
    "seasons_played",
]


# ---------- data ----------

def load_bio(season: str) -> pd.DataFrame:
    """Age, usage and shooting efficiency per player for a season (cached)."""
    path = RAW_DIR / f"bio_{season}.parquet"
    if not path.exists():
        df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=season, timeout=60).get_data_frames()[0]
        df.to_parquet(path, index=False)
    df = pd.read_parquet(path)
    return df[["PLAYER_ID", "AGE", "USG_PCT", "TS_PCT"]].assign(SEASON=season)


def load_player_index() -> pd.DataFrame:
    """Current rosters with positions."""
    df = playerindex.PlayerIndex(season=CURRENT_SEASON, timeout=60).get_data_frames()[0]
    df = df[df["TEAM_ABBREVIATION"].fillna("") != ""]
    return pd.DataFrame({
        "PLAYER_ID": df["PERSON_ID"],
        "PLAYER_NAME": df["PLAYER_FIRST_NAME"] + " " + df["PLAYER_LAST_NAME"],
        "TEAM": df["TEAM_ABBREVIATION"],
        "POSITION": df["POSITION"].fillna(""),
    })


def season_stats(logs: pd.DataFrame) -> pd.DataFrame:
    logs = logs.assign(FP=fantasy_points(logs)).sort_values("GAME_DATE")
    g = logs.groupby(["PLAYER_ID", "SEASON"])
    stats = g.agg(
        PLAYER_NAME=("PLAYER_NAME", "last"), team=("TEAM_ABBREVIATION", "last"),
        gp=("GAME_ID", "count"), fp_pg=("FP", "mean"), fp_total=("FP", "sum"),
        min_pg=("MIN", "mean"), min_total=("MIN", "sum"),
    )
    stats["fp_per_min"] = stats["fp_total"] / stats["min_total"].replace(0, np.nan)
    team_games = logs.groupby(["SEASON", "TEAM_ABBREVIATION"])["GAME_ID"].nunique().groupby("SEASON").median()
    stats["gp_share"] = stats["gp"] / stats.index.get_level_values("SEASON").map(team_games).to_numpy()
    # Form over the last 20 games captures a role that changed mid-season
    stats["fp_late"] = g.tail(20).groupby(["PLAYER_ID", "SEASON"])["FP"].mean()
    return stats.reset_index()


def build_rows(stats: pd.DataFrame, bio: pd.DataFrame, target_seasons) -> pd.DataFrame:
    """Feature rows for each target season, built only from earlier seasons."""
    by_season = {s: d.set_index("PLAYER_ID") for s, d in stats.groupby("SEASON")}
    bio_by = {s: d.set_index("PLAYER_ID") for s, d in bio.groupby("SEASON")}
    rows = []
    for target in target_seasons:
        i = DRAFT_SEASONS.index(target)
        prior = [DRAFT_SEASONS[i - k] if i - k >= 0 else None for k in (1, 2, 3)]
        ids = set().union(*(by_season[s].index for s in prior if s in by_season))
        df = pd.DataFrame(index=sorted(ids))
        df.index.name = "PLAYER_ID"
        for k, s in enumerate(prior, start=1):
            src = by_season.get(s, pd.DataFrame(columns=["fp_pg", "gp", "gp_share", "min_pg"]))
            df[f"fp_pg_{k}"] = src["fp_pg"].reindex(df.index)
            df[f"gp_{k}"] = src["gp"].reindex(df.index).fillna(0)
            df[f"gp_share_{k}"] = src["gp_share"].reindex(df.index).fillna(0)
            df[f"min_pg_{k}"] = src["min_pg"].reindex(df.index)
        # "Last season" features come from the most recent season the player
        # actually played. A player who missed a whole year (e.g. Achilles)
        # would otherwise have blanks, and tree models lump blanks in with
        # fringe players. gp_share_1 = 0 still tells the model he sat out.
        for k in (2, 3):
            for col in ("fp_pg", "min_pg"):
                df[f"{col}_1"] = df[f"{col}_1"].fillna(df[f"{col}_{k}"])
        for col in ("fp_per_min_1", "fp_late_1", "usg_1", "ts_1"):
            df[col] = np.nan
        for s in prior:
            last = by_season.get(s, pd.DataFrame())
            b = bio_by.get(s, pd.DataFrame())
            for col, src, name in (("fp_per_min_1", last, "fp_per_min"), ("fp_late_1", last, "fp_late"),
                                   ("usg_1", b, "USG_PCT"), ("ts_1", b, "TS_PCT")):
                df[col] = df[col].fillna(src.get(name, pd.Series(dtype=float)).reindex(df.index))
        # Age at the target season: latest known age plus years since
        age = pd.Series(np.nan, index=df.index)
        for k, s in enumerate(prior, start=1):
            if s in bio_by:
                age = age.fillna(bio_by[s]["AGE"].reindex(df.index) + k)
        df["age"] = age
        df["seasons_played"] = sum((df[f"gp_{k}"] > 0).astype(int) for k in (1, 2, 3))

        tgt = by_season.get(target)
        if tgt is not None:
            df["fp_pg_target"] = tgt["fp_pg"].reindex(df.index)
            df["gp_target"] = tgt["gp"].reindex(df.index)
            df["gp_share_target"] = tgt["gp_share"].reindex(df.index)
            df["fp_total_target"] = tgt["fp_total"].reindex(df.index)
        df = df.astype(float)
        df["SEASON"] = target
        rows.append(df.reset_index())
    return pd.concat(rows, ignore_index=True)


def load_dataset():
    frames = [load_season(s) for s in DRAFT_SEASONS]
    logs = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    stats = season_stats(logs)
    bio = pd.concat([load_bio(s) for s in DRAFT_SEASONS[:-1]], ignore_index=True)
    return stats, bio


# ---------- models ----------

def make_model(kind: str):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                         min_samples_leaf=30, random_state=42)


class SeasonModel:
    """Two models: FP per game and games played. Projection = product."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, rows: pd.DataFrame):
        played = rows[rows["gp_target"] > 0]
        # Weight per-game targets by games played: 70 games is more reliable than 5
        self.ppg = make_model(self.kind)
        self.ppg.fit(played[FEATURES], played["fp_pg_target"],
                     **self._weight_kwarg(played["gp_target"]))
        self.gp = make_model(self.kind).fit(played[FEATURES], played["gp_share_target"])
        return self

    def _weight_kwarg(self, w):
        key = "ridge__sample_weight" if self.kind == "ridge" else "sample_weight"
        return {key: w}

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        ppg = self.ppg.predict(rows[FEATURES]).clip(min=0)
        gp = (self.gp.predict(rows[FEATURES]) * SEASON_GAMES).clip(0, SEASON_GAMES)
        return pd.DataFrame({"proj_ppg": ppg, "proj_gp": gp, "proj_total": ppg * gp}, index=rows.index)


def evaluate(actual_total: pd.Series, pred_total: pd.Series, top_n: int = 100) -> dict:
    """How good would a draft based on these projections have been?"""
    actual_top = set(actual_total.nlargest(top_n).index)
    pred_top = set(pred_total.nlargest(top_n).index)
    return {
        "mae_total": float((actual_total - pred_total).abs().mean()),
        "rank_corr": float(spearmanr(actual_total, pred_total).statistic),
        f"top{top_n}_hit_rate": len(actual_top & pred_top) / top_n,
    }


def baselines(rows: pd.DataFrame) -> dict:
    last_total = rows["fp_pg_1"].fillna(0) * rows["gp_share_1"] * SEASON_GAMES
    w = np.array([3, 2, 1])
    ppg = rows[["fp_pg_1", "fp_pg_2", "fp_pg_3"]].to_numpy()
    gp = rows[["gp_share_1", "gp_share_2", "gp_share_3"]].to_numpy() * SEASON_GAMES
    weights = np.where(np.isnan(ppg), 0, w * gp)  # weight each season by games and recency
    ppg3 = np.nansum(np.nan_to_num(ppg) * weights, axis=1) / np.where(weights.sum(1) > 0, weights.sum(1), np.nan)
    gp3 = np.where(np.isnan(ppg), np.nan, gp)
    return {
        "Last season's total": last_total,
        "3-year weighted average": pd.Series(ppg3 * np.nanmean(gp3, axis=1), index=rows.index).fillna(0),
    }


def backtest(rows: pd.DataFrame) -> dict:
    seasons = sorted(rows["SEASON"].unique())
    has_history = rows["seasons_played"] > 0

    # 1. Choose the model type on the validation season
    train = rows[rows["SEASON"] < VALID_SEASON]
    valid = rows[(rows["SEASON"] == VALID_SEASON) & has_history & (rows["gp_target"] > 0)]
    valid_scores = {}
    for kind in ("ridge", "gbm"):
        pred = SeasonModel(kind).fit(train).predict(valid)["proj_total"]
        valid_scores[kind] = evaluate(valid["fp_total_target"], pred)["mae_total"]
    best = min(valid_scores, key=valid_scores.get)
    print(f"Validation ({VALID_SEASON}) MAE: {valid_scores} -> using {best}")

    # 2. Report on the untouched test season
    train = rows[rows["SEASON"] < TEST_SEASON]
    test = rows[(rows["SEASON"] == TEST_SEASON) & has_history & (rows["gp_target"] > 0)]
    actual = test["fp_total_target"]
    results = {name: evaluate(actual, pred) for name, pred in baselines(test).items()}
    model = SeasonModel(best).fit(train)
    results["Our model"] = evaluate(actual, model.predict(test)["proj_total"])

    print(f"\nDrafting for {TEST_SEASON} ({len(test)} players):")
    print(f"  {'':<26} {'MAE total':>10} {'rank corr':>10} {'top-100 hit':>12}")
    for name, r in results.items():
        print(f"  {name:<26} {r['mae_total']:10.0f} {r['rank_corr']:10.3f} {r['top100_hit_rate']:12.0%}")
    return {"model_type": best, "test_season": TEST_SEASON, "test_players": len(test),
            "train_seasons": [s for s in seasons if s < TEST_SEASON], "models": results}


# ---------- current season projections ----------

def injury_games_missed(injuries: pd.DataFrame, schedule: pd.DataFrame, today) -> pd.Series:
    """Team games each long-term injured player is expected to miss, by name key."""
    today = pd.Timestamp(today)
    missed = {}
    for r in injuries.itertuples():
        if r.status in ("OUT", "OFS") and (pd.isna(r.return_date) or r.return_date > today):
            missed[r.name_key] = r.return_date
    return pd.Series(missed, dtype="datetime64[ns]")


def project_season() -> pd.DataFrame:
    from src.injuries import get_injuries, normalize_name
    from src.predict_week import get_schedule

    stats, bio = load_dataset()
    rows = build_rows(stats, bio, [CURRENT_SEASON])
    model = joblib.load(MODELS_DIR / "draft_model.joblib")
    roster = load_player_index()
    rows = rows.merge(roster, on="PLAYER_ID")  # only players on a current roster
    rows = rows[rows["seasons_played"] > 0].reset_index(drop=True)
    rows = rows.join(model.predict(rows))

    # Long-term injuries: remove the team games before the expected return date
    schedule = get_schedule()
    injuries = get_injuries()
    inj = injuries.set_index("name_key")
    keys = rows["PLAYER_NAME"].map(normalize_name)
    rows["status"] = keys.map(inj["status"])
    rows["injury_type"] = keys.map(inj["injury"])
    rows["return_date"] = keys.map(inj["return_date"])
    long_term = rows["status"].isin(["OUT", "OFS"])
    team_games = schedule.groupby("TEAM_ABBREVIATION")["GAME_DATE"].apply(sorted)
    missed = []
    for r in rows.itertuples():
        if not long_term[r.Index]:
            missed.append(0)
            continue
        dates = team_games.get(r.TEAM, [])
        missed.append(len(dates) if pd.isna(r.return_date) else sum(d < r.return_date for d in dates))
    rows["games_missed_injury"] = missed
    rows["proj_gp"] = rows["proj_gp"] * (1 - rows["games_missed_injury"] / SEASON_GAMES)
    rows["proj_total"] = rows["proj_ppg"] * rows["proj_gp"]

    # Per-season history for the dashboard trend lines
    hist = stats[stats["SEASON"] < CURRENT_SEASON].sort_values("SEASON")
    history = {pid: [{"season": s.SEASON, "gp": int(s.gp), "fp_pg": round(float(s.fp_pg), 1)}
                     for s in g.itertuples()]
               for pid, g in hist.groupby("PLAYER_ID")}
    rows["history"] = rows["PLAYER_ID"].map(history)
    last_team = stats[stats["SEASON"] == DRAFT_SEASONS[-2]].set_index("PLAYER_ID")["team"]
    rows["new_team"] = rows["PLAYER_ID"].map(last_team).pipe(lambda t: t.notna() & (t != rows["TEAM"]))
    return rows.sort_values("proj_total", ascending=False).reset_index(drop=True)


def main():
    stats, bio = load_dataset()
    rows = build_rows(stats, bio, DRAFT_SEASONS[1:-1])
    metrics = backtest(rows)

    # Final model on every completed season, for this season's draft
    model = SeasonModel(metrics["model_type"]).fit(rows)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "draft_model.joblib")
    (REPORTS_DIR / "draft_metrics.json").write_text(json.dumps(metrics, indent=2))

    proj = project_season()
    cols = ["PLAYER_NAME", "TEAM", "POSITION", "age", "proj_ppg", "proj_gp", "proj_total", "games_missed_injury"]
    out = proj[cols].head(30).round(1)
    out.index += 1
    print(f"\n{CURRENT_SEASON} draft rankings:\n{out.to_string()}")


if __name__ == "__main__":
    # Run via the module's real name so the saved model pickles as
    # src.draft.SeasonModel (not __main__), and can be loaded elsewhere
    from src import draft
    draft.main()
