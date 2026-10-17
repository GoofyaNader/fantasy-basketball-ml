"""Turn raw box scores into model features.

The golden rule: every feature for a game must only use information
available *before* that game tipped off. That's why everything below
uses .shift(1) — otherwise the model would "see" the answer and look
far better in testing than it ever could in real life (data leakage).
"""
import numpy as np
import pandas as pd

from src.config import SCORING

FEATURES = [
    "fp_last3", "fp_last5", "fp_last10", "fp_season", "fp_prev_season",
    "min_last3", "min_last5", "min_last10", "min_season",
    "fp_per_min_last10", "fp_std_last10",
    "games_played", "rest_days", "back_to_back", "home",
    "opp_fp_allowed",
    "missing_fp", "missing_fp_delta",
]
TARGET = "FP"

# A teammate counts as "missing" for this many team games after his last
# appearance. Beyond that, his absence is already reflected in teammates'
# recent stats (and he may have been waived).
MAX_ABSENCE_GAMES = 20
# Only rotation players count. Deep-bench players who sit by coach's decision
# aren't on the injury report, so counting them in training wouldn't match
# what we can know when predicting.
MIN_ROTATION_MINUTES = 15


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    return sum(df[stat] * pts for stat, pts in SCORING.items())


def _add_game_context(df: pd.DataFrame) -> pd.DataFrame:
    # MATCHUP looks like "LAL vs. BOS" (home) or "LAL @ BOS" (away)
    df["home"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["OPP"] = df["MATCHUP"].str.split(" ").str[-1]
    return df


def _add_player_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    g = df.groupby("PLAYER_ID")

    # Previous games' values, so a row never includes its own game
    prev_fp = g["FP"].shift(1)
    prev_min = g["MIN"].shift(1)
    pid = df["PLAYER_ID"]

    for n in (3, 5, 10):
        df[f"fp_last{n}"] = prev_fp.groupby(pid).transform(lambda s: s.rolling(n, min_periods=1).mean())
        df[f"min_last{n}"] = prev_min.groupby(pid).transform(lambda s: s.rolling(n, min_periods=1).mean())
    df["fp_std_last10"] = prev_fp.groupby(pid).transform(lambda s: s.rolling(10, min_periods=2).std())

    fp_sum10 = prev_fp.groupby(pid).transform(lambda s: s.rolling(10, min_periods=1).sum())
    min_sum10 = prev_min.groupby(pid).transform(lambda s: s.rolling(10, min_periods=1).sum())
    df["fp_per_min_last10"] = fp_sum10 / min_sum10.replace(0, np.nan)

    # Season-to-date averages (reset each season)
    gs = df.groupby(["PLAYER_ID", "SEASON"])
    df["fp_season"] = gs["FP"].transform(lambda s: s.shift(1).expanding().mean())
    df["min_season"] = gs["MIN"].transform(lambda s: s.shift(1).expanding().mean())
    df["games_played"] = gs.cumcount()

    # Last season's average helps early in a season when there's little data
    season_avg = df.groupby(["PLAYER_ID", "SEASON"])["FP"].mean().reset_index()
    season_avg["SEASON"] = season_avg["SEASON"].map(_next_season)
    season_avg = season_avg.rename(columns={"FP": "fp_prev_season"})
    df = df.merge(season_avg, on=["PLAYER_ID", "SEASON"], how="left")

    # Before a player's first game of a season there is no season average yet.
    # Fill it with the best available guess rather than leaving it blank: tree
    # models send all blanks down one branch, lumping stars with bench players.
    # games_played still tells the model how much to trust it.
    df["fp_season"] = df["fp_season"].fillna(df["fp_prev_season"]).fillna(df["fp_last10"])
    df["min_season"] = df["min_season"].fillna(df["min_last10"])

    # Rest is measured within a season. A season opener is left blank (NaN) rather
    # than "long rest": otherwise it looks like a return from injury, which
    # usually means limited minutes, and the model under-projects everyone.
    days = df.groupby(["PLAYER_ID", "SEASON"])["GAME_DATE"].diff().dt.days
    df["rest_days"] = days.clip(upper=7)
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)
    return df


def _add_opponent_defense(df: pd.DataFrame) -> pd.DataFrame:
    """How many fantasy points does each team give up per game, recently?"""
    allowed = (
        df.dropna(subset=["FP"])
        .groupby(["OPP", "GAME_ID", "GAME_DATE"], as_index=False)["FP"].sum()
        .sort_values("GAME_DATE")
    )
    allowed["opp_fp_allowed"] = allowed.groupby("OPP")["FP"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    )
    # For future games (no box score yet) use each team's most recent value
    latest = allowed.groupby("OPP")["FP"].apply(lambda s: s.tail(20).mean()).rename("opp_latest")

    df = df.merge(allowed[["OPP", "GAME_ID", "opp_fp_allowed"]], on=["OPP", "GAME_ID"], how="left")
    df = df.merge(latest, on="OPP", how="left")
    df["opp_fp_allowed"] = df["opp_fp_allowed"].fillna(df["opp_latest"])
    return df.drop(columns="opp_latest")


def _add_teammate_absences(df: pd.DataFrame) -> pd.DataFrame:
    """Fantasy points normally produced by teammates who sit out each game.

    Box scores only list players who played, so a regular who is missing
    from his team's box score was out. When a big contributor sits, the
    remaining players pick up his shots and minutes.

    missing_fp:       usual FP of this team's absent regulars tonight
    missing_fp_delta: missing_fp vs. the team's average over its last 10 games
                      (a new absence matters more than a month-long one)
    """
    played = df[df["FP"].notna()]
    team_games = (
        played[["SEASON", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]]
        .drop_duplicates().sort_values("GAME_DATE")
    )
    team_games["game_num"] = team_games.groupby(["SEASON", "TEAM_ABBREVIATION"]).cumcount()

    # Each player's season average *after* each game he played: the "usual"
    # production a teammate would lose if he sat out a later game.
    p = played[["PLAYER_ID", "SEASON", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "FP", "MIN"]]
    p = p.sort_values("GAME_DATE")
    by_player = p.groupby(["PLAYER_ID", "SEASON"])
    p = p.assign(usual_fp=by_player["FP"].transform(lambda s: s.expanding().mean()),
                 usual_min=by_player["MIN"].transform(lambda s: s.expanding().mean()))
    p = p.merge(team_games[["TEAM_ABBREVIATION", "GAME_ID", "game_num"]],
                on=["TEAM_ABBREVIATION", "GAME_ID"])

    # A stint = a player's time on one team in one season. It starts at his
    # first game for the team and ends when he first plays for another team.
    stints = (
        p.groupby(["PLAYER_ID", "SEASON", "TEAM_ABBREVIATION"], as_index=False)
        .agg(start_num=("game_num", "min"), start_date=("GAME_DATE", "min"))
        .sort_values("start_date")
    )
    stints["end_date"] = stints.groupby(["PLAYER_ID", "SEASON"])["start_date"].shift(-1)

    # Every team game during each stint, whether or not the player played
    grid = stints.merge(team_games, on=["SEASON", "TEAM_ABBREVIATION"])
    grid = grid[(grid["game_num"] >= grid["start_num"]) &
                (grid["end_date"].isna() | (grid["GAME_DATE"] < grid["end_date"]))]
    grid = grid.merge(p[["PLAYER_ID", "GAME_ID", "usual_fp", "usual_min"]],
                      on=["PLAYER_ID", "GAME_ID"], how="left")
    grid = grid.sort_values(["PLAYER_ID", "SEASON", "TEAM_ABBREVIATION", "game_num"])

    grid["played"] = grid["usual_fp"].notna()
    stint_key = [grid["PLAYER_ID"], grid["SEASON"], grid["TEAM_ABBREVIATION"]]
    # For missed games, carry forward his averages as of his last appearance
    grid[["usual_fp", "usual_min"]] = grid.groupby(stint_key)[["usual_fp", "usual_min"]].ffill()
    last_played = grid["game_num"].where(grid["played"]).groupby(stint_key).ffill()
    absent = (~grid["played"]
              & (grid["game_num"] - last_played <= MAX_ABSENCE_GAMES)
              & (grid["usual_min"] >= MIN_ROTATION_MINUTES))

    missing = grid[absent].groupby(["TEAM_ABBREVIATION", "GAME_ID"])["usual_fp"].sum()
    team_games = team_games.merge(missing.rename("missing_fp").reset_index(),
                                  on=["TEAM_ABBREVIATION", "GAME_ID"], how="left")
    team_games["missing_fp"] = team_games["missing_fp"].fillna(0)
    team_games = team_games.sort_values("GAME_DATE")
    by_team = team_games.groupby(["SEASON", "TEAM_ABBREVIATION"])["missing_fp"]
    team_games["missing_fp_last10"] = by_team.transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    df = df.merge(team_games[["TEAM_ABBREVIATION", "GAME_ID", "missing_fp", "missing_fp_last10"]],
                  on=["TEAM_ABBREVIATION", "GAME_ID"], how="left")

    # Future games: missing_fp comes from the injury report (see predict_week),
    # and the baseline is the team's most recent 10-game average.
    latest = (by_team.apply(lambda s: s.tail(10).mean())
              .rename("last10_latest").reset_index())
    df = df.merge(latest, on=["SEASON", "TEAM_ABBREVIATION"], how="left")
    future = df["FP"].isna()
    df.loc[future, "missing_fp_last10"] = df.loc[future, "last10_latest"]
    df["missing_fp_delta"] = df["missing_fp"] - df["missing_fp_last10"]
    return df.drop(columns="last10_latest")


def _next_season(season: str) -> str:
    start = int(season[:4]) + 1
    return f"{start}-{str(start + 1)[-2:]}"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: raw game logs (optionally with future rows where stats are NaN)."""
    df = df.copy()
    df["FP"] = fantasy_points(df)
    df = _add_game_context(df)
    df = _add_player_form(df)
    df = _add_opponent_defense(df)
    df = _add_teammate_absences(df)
    return df.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)
