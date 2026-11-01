import warnings

import numpy as np

warnings.filterwarnings('ignore')

from src.preprocess import build_team_df, build_team_profiles, build_region_weights
from src.features import build_matchups
from src.model import train_model, predict_matchup

# Worlds 2026 knockout draw. Each match is best of five.
QUARTERFINALS = [
    ("Dplus Kia", "Bilibili Gaming"),
    ("Karmine Corp", "T1"),
    ("Top Esports", "Invictus Gaming"),
    ("Gen.G", "LYON"),
]

filenames = [
    "data/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "data/2026_LoL_esports_match_data_from_OraclesElixir.csv",
]
team_df = build_team_df(filenames)
team_profiles = build_team_profiles(team_df, intl_boost=3.0)
matchups = build_matchups(team_df)
region_weights = build_region_weights(team_df)
model, scaler, feature_cols, x_test_scaled, y_test = train_model(matchups)

np.random.seed(2026)


def series_win_prob(game_prob):
    '''Best-of-five win probability from an independent game win probability.'''
    q = 1 - game_prob
    return game_prob ** 3 * (1 + 3 * q + 6 * q ** 2)


def predict_series(team_a, team_b):
    game_prob = predict_matchup(
        team_a, team_b, team_profiles, model, scaler, feature_cols, region_weights
    )
    prob_a = series_win_prob(game_prob)
    if prob_a >= 0.5:
        return team_a, team_b, prob_a
    return team_b, team_a, 1 - prob_a


def play_round(title, pairs):
    print(f"\n--- {title} ---")
    winners = []
    for team_a, team_b in pairs:
        winner, loser, conf = predict_series(team_a, team_b)
        winners.append(winner)
        print(f"  {team_a} vs {team_b} (Bo5)")
        print(f"  >>> {winner} advances ({conf:.1%} confidence), {loser} is ELIMINATED")
    return winners


known = set(team_profiles['teamname'])
missing = sorted({team for pair in QUARTERFINALS for team in pair if team not in known})
if missing:
    raise SystemExit(f"teams missing from profiles: {missing}")

print("\n" + "=" * 50)
print("  WORLDS 2026 KNOCKOUT PREDICTION")
print("=" * 50)

qf_winners = play_round("QUARTERFINALS", QUARTERFINALS)
semis = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
sf_winners = play_round("SEMIFINALS", semis)
champion, runner_up, conf = predict_series(*sf_winners)

print("\n--- FINALS ---")
print(f"  {sf_winners[0]} vs {sf_winners[1]} (Bo5)")
print(f"  >>> {champion} wins ({conf:.1%} confidence)")
print(f"  >>> {runner_up} is ELIMINATED")

print("\n" + "=" * 50)
print("  WORLDS 2026 CHAMPION")
print("=" * 50)
print(f"  {champion}")
print("=" * 50)
