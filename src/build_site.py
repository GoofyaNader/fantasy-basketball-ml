"""Build the dashboard (docs/index.html) from this week's projections.

GitHub Pages serves the docs/ folder, so publishing is just a git push.

Usage:
    python -m src.build_site                       # build for the coming week
    python -m src.build_site --refresh --publish   # weekly: new games, rebuild, push
"""
import argparse
import json
import subprocess
from datetime import date, datetime, timedelta

import pandas as pd

from src.config import CURRENT_SEASON, REPORTS_DIR, ROOT, SCORING
from src.data import load_all
from src.predict_week import next_monday, project_games, summarize_week

TEMPLATE = ROOT / "site" / "template.html"
OUTPUT = ROOT / "docs" / "index.html"


def build_payload(start: date) -> dict:
    games = project_games(start)
    table = summarize_week(games)

    games = games.sort_values("GAME_DATE")
    games["opp"] = games["MATCHUP"].str.split(" ").str[-1]
    detail = {
        name: [
            {"date": str(g.GAME_DATE.date()), "opp": g.opp, "home": bool(g.home),
             "proj": round(float(g.proj_fp), 1), "prob": round(float(g.play_prob), 2)}
            for g in rows.itertuples()
        ]
        for name, rows in games.groupby("PLAYER_NAME")
    }

    inj = games.groupby("PLAYER_NAME")[["status", "injury", "return_date"]].first()
    players = []
    for rank, r in table.iterrows():
        i = inj.loc[r.PLAYER_NAME]
        players.append({
            "rank": int(rank), "name": r.PLAYER_NAME, "team": r.TEAM_ABBREVIATION,
            "games": int(r.games), "exp_games": float(r.exp_games),
            "ppg": float(r.proj_per_game), "week": float(r.proj_week),
            "last10": None if pd.isna(r.last10_avg) else float(r.last10_avg),
            "status": None if pd.isna(i.status) else i.status,
            "injury_type": i.injury or None if isinstance(i.injury, str) else None,
            "return_date": None if pd.isna(i.return_date) else str(i.return_date.date()),
            "injury": r.injury,
            "detail": detail.get(r.PLAYER_NAME, []),
        })

    team_games = games.drop_duplicates(["TEAM_ABBREVIATION", "GAME_ID"])
    teams = [
        {"team": team, "games": [{"date": str(g.GAME_DATE.date()), "opp": g.opp, "home": bool(g.home)}
                                 for g in rows.itertuples()]}
        for team, rows in team_games.groupby("TEAM_ABBREVIATION")
    ]

    return {
        "meta": {
            "week_start": str(start),
            "week_end": str(start + timedelta(days=6)),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "season": CURRENT_SEASON,
            "scoring": SCORING,
            "repo_url": _repo_url(),
        },
        "players": players,
        "teams": teams,
        "metrics": json.loads((REPORTS_DIR / "metrics.json").read_text()),
    }


def _repo_url():
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return url.removesuffix(".git").replace("git@github.com:", "https://github.com/")


def write_site(payload: dict):
    # Escape "</" so player data can never close the <script> tag early
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", data)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(html)
    (OUTPUT.parent / ".nojekyll").touch()  # serve files as-is on GitHub Pages


def publish(start: date):
    subprocess.run(["git", "add", "docs", "reports"], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("Nothing changed; skipping commit.")
        return
    subprocess.run(["git", "commit", "-q", "-m", f"Update dashboard for week of {start}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)
    print("Pushed. The live site updates in about a minute.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=next_monday(date.today()))
    parser.add_argument("--refresh", action="store_true", help="download the latest games first")
    parser.add_argument("--publish", action="store_true", help="commit and push to GitHub")
    args = parser.parse_args()

    if args.refresh:
        load_all(refresh_current=True)
    write_site(build_payload(args.start))
    print(f"Built {OUTPUT}")
    if args.publish:
        publish(args.start)


if __name__ == "__main__":
    main()
