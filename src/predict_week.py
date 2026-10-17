"""Rank players by projected fantasy points for a week.

Usage:
    python -m src.predict_week                     # the coming Mon-Sun week
    python -m src.predict_week --start 2026-10-20  # any week
    python -m src.predict_week --players "LeBron James" "Jalen Brunson"
    python -m src.predict_week --no-injuries       # assume everyone is healthy
"""
import argparse
from datetime import date, timedelta

import joblib
import pandas as pd
from nba_api.stats.endpoints import commonallplayers, scheduleleaguev2

from src.config import CURRENT_SEASON, MODELS_DIR, REPORTS_DIR
from src.data import load_all
from src.features import FEATURES, MAX_ABSENCE_GAMES, MIN_ROTATION_MINUTES, build_features
from src.injuries import apply_injuries, get_injuries


def get_schedule() -> pd.DataFrame:
    """Every regular-season game this season: one row per team per game."""
    games = scheduleleaguev2.ScheduleLeagueV2(season=CURRENT_SEASON, timeout=60).get_data_frames()[0]
    games = games[games["gameId"].str.startswith("002")]  # 002 = regular season
    games = games.assign(GAME_DATE=pd.to_datetime(games["gameDate"], format="%m/%d/%Y %H:%M:%S"))

    home, away = games["homeTeam_teamTricode"], games["awayTeam_teamTricode"]
    base = games[["gameId", "GAME_DATE"]].rename(columns={"gameId": "GAME_ID"})
    return pd.concat([
        base.assign(TEAM_ABBREVIATION=home, MATCHUP=home + " vs. " + away),
        base.assign(TEAM_ABBREVIATION=away, MATCHUP=away + " @ " + home),
    ], ignore_index=True)


def get_rosters() -> pd.DataFrame:
    """Current team for every active player (reflects offseason moves)."""
    df = commonallplayers.CommonAllPlayers(
        is_only_current_season=1, season=CURRENT_SEASON, timeout=60
    ).get_data_frames()[0]
    df = df[df["TEAM_ABBREVIATION"].fillna("") != ""]
    return df.rename(columns={"PERSON_ID": "PLAYER_ID", "DISPLAY_FIRST_LAST": "PLAYER_NAME"})[
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
    ]


def next_monday(today: date) -> date:
    return today + timedelta(days=(7 - today.weekday()) % 7)


def project_week(start: date, use_injuries: bool = True) -> pd.DataFrame:
    end = start + timedelta(days=7)
    history = load_all()
    schedule = get_schedule()
    week_games = schedule[(schedule["GAME_DATE"] >= pd.Timestamp(start)) &
                          (schedule["GAME_DATE"] < pd.Timestamp(end))]
    if week_games.empty:
        raise SystemExit(f"No regular-season games between {start} and {end}.")

    # One placeholder row per player per game this week (stats unknown = NaN)
    rosters = get_rosters()
    rosters = rosters[rosters["PLAYER_ID"].isin(history["PLAYER_ID"])]  # need history to predict
    future = rosters.merge(week_games, on="TEAM_ABBREVIATION")
    future["SEASON"] = CURRENT_SEASON

    feats = build_features(pd.concat([history, future], ignore_index=True))
    upcoming = feats[feats["GAME_ID"].isin(week_games["GAME_ID"]) & feats["FP"].isna()].copy()

    # Form features for later games this week would be computed from the earlier
    # (not yet played) games, so give every game the form as of the week's start.
    # Game context (rest, home/away, opponent) still varies per game.
    upcoming = upcoming.sort_values(["PLAYER_ID", "GAME_DATE"])
    form_cols = [c for c in FEATURES if c.startswith(("fp_", "min_")) or c == "games_played"]
    upcoming[form_cols] = upcoming.groupby("PLAYER_ID")[form_cols].transform("first")

    if use_injuries:
        upcoming = apply_injuries(upcoming, get_injuries(), date.today())
    else:
        upcoming = upcoming.assign(status=None, return_date=pd.NaT, injury="", play_prob=1.0)
    upcoming = _add_missing_teammates(upcoming, history)

    model = joblib.load(MODELS_DIR / "fp_model.joblib")
    upcoming["proj_fp"] = model.predict(upcoming[FEATURES])

    # Weight each game by the chance the player actually plays it
    upcoming["exp_fp"] = upcoming["proj_fp"] * upcoming["play_prob"]

    out = (
        upcoming.groupby(["PLAYER_NAME", "TEAM_ABBREVIATION"])
        .agg(games=("GAME_ID", "count"), exp_games=("play_prob", "sum"),
             proj_per_game=("proj_fp", "mean"), proj_week=("exp_fp", "sum"),
             last10_avg=("fp_last10", "first"),
             status=("status", "first"), return_date=("return_date", "first"),
             injury_type=("injury", "first"))
        .reset_index()
        .sort_values("proj_week", ascending=False)
        .reset_index(drop=True)
    )
    out["injury"] = [_injury_note(*row) for row in zip(out["status"], out["injury_type"], out["return_date"])]
    out = out.drop(columns=["status", "return_date", "injury_type"])
    out.index += 1
    return out.round(1)


def _add_missing_teammates(upcoming: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Fill the teammate-absence features for future games from the injury report.

    Mirrors the training definition: rotation players who have played within
    the team's last MAX_ABSENCE_GAMES games. A player with a 25% chance of
    sitting counts as 25% of his usual production missing.
    """
    last_game = history.groupby("PLAYER_ID")["GAME_DATE"].max()
    team_dates = history[["TEAM_ABBREVIATION", "GAME_DATE"]].drop_duplicates()
    cutoff = team_dates.groupby("TEAM_ABBREVIATION")["GAME_DATE"].apply(
        lambda s: s.nlargest(MAX_ABSENCE_GAMES).min())

    counts = (
        (upcoming["PLAYER_ID"].map(last_game) >= upcoming["TEAM_ABBREVIATION"].map(cutoff))
        & (upcoming["min_season"] >= MIN_ROTATION_MINUTES)
    )
    lost = (1 - upcoming["play_prob"]) * upcoming["fp_season"] * counts
    upcoming["missing_fp"] = lost.groupby([upcoming["TEAM_ABBREVIATION"], upcoming["GAME_ID"]]).transform("sum")
    upcoming["missing_fp_delta"] = upcoming["missing_fp"] - upcoming["missing_fp_last10"]
    return upcoming


def _injury_note(status, injury_type, return_date) -> str:
    if pd.isna(status):
        return ""
    note = f"{status} - {injury_type}" if injury_type else status
    return f"{note} (back ~{return_date:%b %d})" if pd.notna(return_date) else note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=next_monday(date.today()),
                        help="first day of the week (YYYY-MM-DD), default: next Monday")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--players", nargs="*", help="only show these players (e.g. your roster)")
    parser.add_argument("--no-injuries", action="store_true", help="assume every player is healthy")
    args = parser.parse_args()

    table = project_week(args.start, use_injuries=not args.no_injuries)
    REPORTS_DIR.mkdir(exist_ok=True)
    csv_path = REPORTS_DIR / f"projections_{args.start}.csv"
    table.to_csv(csv_path, index_label="rank")

    if args.players:
        wanted = [p.lower() for p in args.players]
        table = table[table["PLAYER_NAME"].str.lower().isin(wanted)]
    else:
        table = table.head(args.top)

    print(f"\nProjected fantasy points, week of {args.start}\n")
    print(table.to_string())
    print(f"\nFull rankings saved to {csv_path}")


if __name__ == "__main__":
    main()
