"""Current NBA injury report (from ESPN) and per-game play probabilities.

Usage:
    python -m src.injuries    # print the current injury report
"""
import re
import unicodedata
from datetime import date

import pandas as pd
import requests

from src.config import GTD_PLAY_PROB, STALE_OUT_PLAY_PROB

INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
OUT_STATUSES = {"OUT", "OFS"}  # OFS = out for season


def normalize_name(name: str) -> str:
    """'Luka Dončić' -> 'luka doncic', 'Trey Murphy III' -> 'trey murphy'.

    ESPN and NBA.com spell some names differently, so match on a simplified form.
    """
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z ]", "", name.lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return " ".join(name.split())


def get_injuries() -> pd.DataFrame:
    data = requests.get(INJURIES_URL, timeout=30).json()
    rows = []
    for team in data["injuries"]:
        for inj in team["injuries"]:
            details = inj.get("details", {})
            rows.append({
                "player": inj["athlete"]["displayName"],
                "status": details.get("fantasyStatus", {}).get("abbreviation", inj["status"]),
                "return_date": details.get("returnDate"),
                "injury": details.get("type", ""),
                "comment": inj.get("shortComment", ""),
            })
    df = pd.DataFrame(rows)
    df["return_date"] = pd.to_datetime(df["return_date"])
    df["name_key"] = df["player"].map(normalize_name)
    return df.drop_duplicates("name_key")


def play_probability(status, injury, return_date, game_date, today, first_game_back) -> float:
    """Chance a player on the injury report plays in a given game."""
    if pd.isna(status):
        return 1.0  # not on the injury report
    if pd.notna(return_date) and game_date < return_date:
        return 0.0
    if status in OUT_STATUSES:
        if pd.isna(return_date):
            return 0.0
        if return_date <= today:
            return STALE_OUT_PLAY_PROB  # report not updated since expected return
        return 1.0
    # Day-to-day. Scheduled rest isn't an injury, and the uncertainty only
    # applies to the first game back, not the rest of the week.
    if injury == "Rest" or not first_game_back:
        return 1.0
    return GTD_PLAY_PROB


def apply_injuries(games: pd.DataFrame, injuries: pd.DataFrame, today: date) -> pd.DataFrame:
    """Add status, return_date and play_prob columns to per-player-per-game rows."""
    games = games.assign(name_key=games["PLAYER_NAME"].map(normalize_name))
    games = games.merge(injuries[["name_key", "status", "return_date", "injury"]],
                        on="name_key", how="left")
    games = games.sort_values(["PLAYER_ID", "GAME_DATE"])

    # Each player's first game on or after their expected return date
    eligible = games["return_date"].isna() | (games["GAME_DATE"] >= games["return_date"])
    first_back = eligible & (eligible.astype(int).groupby(games["PLAYER_ID"]).cumsum() == 1)

    today = pd.Timestamp(today)
    games["play_prob"] = [
        play_probability(s, i, r, g, today, f)
        for s, i, r, g, f in zip(games["status"], games["injury"], games["return_date"],
                                 games["GAME_DATE"], first_back)
    ]
    return games.drop(columns="name_key")


if __name__ == "__main__":
    inj = get_injuries().sort_values(["status", "return_date"])
    print(inj[["player", "status", "return_date", "injury"]].to_string(index=False))
