"""Download and cache NBA player game logs from stats.nba.com.

Usage:
    python -m src.data            # download any seasons not cached yet
    python -m src.data --refresh  # also re-download the current season
"""
import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.config import CURRENT_SEASON, RAW_DIR, SEASONS


def fetch_season(season: str) -> pd.DataFrame:
    """One row per player per game for a regular season."""
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
        timeout=60,
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    return df


def load_season(season: str, refresh: bool = False) -> pd.DataFrame:
    """Return a season's game logs, downloading only if not cached."""
    path = RAW_DIR / f"gamelogs_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    print(f"Downloading {season}...")
    df = fetch_season(season)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    time.sleep(1)  # be polite to the NBA's servers
    return df


def load_all(refresh_current: bool = False) -> pd.DataFrame:
    frames = [load_season(s, refresh=refresh_current and s == CURRENT_SEASON) for s in SEASONS]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the current season")
    args = parser.parse_args()

    df = load_all(refresh_current=args.refresh)
    print(df.groupby("SEASON").size().rename("player-games").to_string())
