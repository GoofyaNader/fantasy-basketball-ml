# Fantasy Basketball Weekly Projections

A machine learning model that projects how many fantasy points each NBA player
will score in an upcoming week, so you can decide who to start and who to pick
up off waivers.

Scoring follows ESPN's default points league (editable in `src/config.py`).

## How it works

1. **Data**: every player's box score from the last five NBA regular seasons,
   pulled from NBA.com with [`nba_api`](https://github.com/swar/nba_api).
2. **Features**: for each game, only information known *before* tip-off:
   - recent form (fantasy points and minutes over the last 3/5/10 games)
   - season-to-date and previous-season averages
   - efficiency (fantasy points per minute) and consistency (std. dev.)
   - schedule context: rest days, back-to-backs, home/away
   - opponent defense: fantasy points the opponent allowed over its last 20 games
3. **Model**: a gradient-boosted tree regressor (scikit-learn) predicts
   fantasy points for each game.
4. **Weekly projection**: per-game predictions summed over each team's games that
   week. A player with 4 games often beats a better player with 2.
5. **Injury filter**: each game is weighted by the chance the player suits up,
   using ESPN's live injury report and each player's expected return date:

   | Situation | Chance of playing |
   |---|---|
   | Game before expected return date | 0% |
   | Out, no return date | 0% |
   | Out, but return date already passed (stale report) | 50% |
   | Day-to-day, first game on/after return date | 75% |
   | Day-to-day, later games / scheduled rest / healthy | 100% |

   An early version applied the day-to-day discount to every game in the week.
   In preseason, ESPN lists dozens of healthy stars as "day-to-day" because
   they're resting, and that version pushed Giannis from 14th to 52nd. The
   discount now applies only to real injuries, and only to the first game back.

## Evaluation

The model trains on 2021-22 through 2024-25 and is tested on the full 2025-26
season, which it never saw. It's compared against the baselines a fantasy
manager would use by eye:

| Approach | MAE per game (FP) |
|---|---|
| Last 10 games average | 10.04 |
| Season average | 10.03 |
| Ridge regression | 9.77 |
| **Gradient boosting** | **9.65** |

Single games are noisy (a star can score 30 or 90), so per-game error stays
high for any method. What matters for fantasy is ranking players over a week,
and there the model reaches an average **Spearman rank correlation of 0.89**
between projected and actual weekly totals.

![Evaluation](reports/evaluation.png)

Ranking accuracy dips around the February trade deadline, when players change
teams and roles faster than recent-form features can catch up.

### Bugs found along the way

- **Season openers looked like injury returns.** Rest days were capped at 7, so
  a player's first game after the offseason looked the same as a comeback from
  injury (usually limited minutes). Stars were under-projected by ~10 points on
  opening night. Fixed by measuring rest within a season only.
- **Missing values in tree models.** Before a player's first game, their season
  average is blank. Gradient-boosted trees send all blanks down one branch, which
  lumped stars in with bench players. Filling it with last season's average
  brought opening-night projections for stars from 41.8 to 46.2 FP (actual: 47.3).

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data            # download game logs (cached in data/raw/)
python -m src.train           # train + evaluate, saves models/fp_model.joblib
python -m src.predict_week    # rankings for the coming Monday-Sunday week

# During the season: refresh this season's games, then re-project
python -m src.data --refresh
python -m src.predict_week --players "Jalen Brunson" "Scottie Barnes"
python -m src.predict_week --no-injuries   # assume everyone is healthy
python -m src.injuries                     # view the current injury report
```

## Limitations and next steps

- **Injury report quality**: ESPN's return dates are estimates and are
  sometimes stale. Check the notes (`python -m src.injuries`) for key players.
- **Teammate absences**: when a star sits, teammates' usage rises. This isn't
  modelled yet.
- **Early season**: with few games played, the model leans on last season's
  numbers, so it's slow to catch new roles (trades, rookies).
