# League of Legends Esports Match Predictor

A logistic regression model that predicts professional League of Legends match outcomes using historical team performance data from OraclesElixir.

## How It Works

1. **Team profiles** — Raw match data is aggregated into per-team stat profiles (gold diff, XP diff, CS diff, kill diff, objective rates, game length, CKPM). Recent games are weighted more heavily using exponential decay with a 180-day half-life, and international games are boosted 3x when predicting cross-region matchups.

2. **Region weights** — Each region's win rate at international events (Worlds, MSI, EWC) is used to derive a normalized strength multiplier applied to every prediction.

3. **Prediction** — For each matchup, 10,000 Monte Carlo simulations sample from each team's stat distributions (mean ± std), scale the features, and feed them into the logistic regression model via `predict_proba`. Simulations are run twice (swapping blue/red side) and averaged to remove side bias. The result is a win probability for each team.

## Data

- **Source:** [OraclesElixir](https://oracleselixir.com/)
- **Coverage:** January 2025 – June 21, 2026
- **Refresh:** Re-download the yearly CSVs from OraclesElixir and replace the files in `data/` to update the model with newer results.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single custom matchup (edit the predict_matchup call at the bottom of main.py)
python main.py

# Run the MSI 2026 Play-in bracket predictor
python predict_playin.py
```

## MSI 2026 Play-in Predictions

### Round 1 — Upper Bracket

| Matchup | Predicted Winner | Confidence | Actual Result |
|---|---|---|---|
| T1 vs Team Liquid | T1 | 54.8% | T1 |
| Karmine Corp vs Deep Cross Gaming | Karmine Corp | 54.3% | Karmine Corp |

### Upper Bracket Finals

| Matchup | Predicted Winner | Confidence | Actual Result |
|---|---|---|---|
| T1 vs Karmine Corp | T1 | 59.7% | T1 |

### Lower Bracket Semifinals

| Matchup | Predicted Winner | Confidence | Actual Result |
|---|---|---|---|
| Team Liquid vs Deep Cross Gaming | Team Liquid | 59.4% | Team Liquid |

### Lower Bracket Finals

| Matchup | Predicted Winner | Confidence | Actual Result |
|---|---|---|---|
| Karmine Corp vs Team Liquid | Team Liquid | 55.2% | Team Liquid |

### Grand Finals

| Matchup | Predicted Winner | Confidence | Actual Result |
|---|---|---|---|
| Team Liquid vs T1 | T1 | 54.6% | T1 |
