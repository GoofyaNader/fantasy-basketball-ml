"""Project-wide settings. Change SCORING to match your league."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# ESPN default points-league scoring. Edit these to match your league.
SCORING = {
    "PTS": 1,
    "FG3M": 1,
    "FGA": -1,
    "FGM": 2,
    "FTA": -1,
    "FTM": 1,
    "REB": 1,
    "AST": 2,
    "STL": 4,
    "BLK": 4,
    "TOV": -2,
}

# Seasons used for training/testing. The last one is the "current" season.
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT_SEASON = SEASONS[-1]

# Injury filter: chance a player suits up for a game, by situation.
# Games before a player's expected return date always count as 0.
GTD_PLAY_PROB = 0.75         # day-to-day / game-time decision, on or after return date
STALE_OUT_PLAY_PROB = 0.5    # listed "Out" but the return date has already passed

# Time-based split: never train on games that happen after the test games.
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"
