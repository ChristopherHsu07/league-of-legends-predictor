import warnings
warnings.filterwarnings('ignore')

from src.preprocess import build_team_df, build_team_profiles, build_region_weights
from src.features import build_matchups
from src.model import train_model, predict_matchup

# --- Setup ---
filenames = [
    "data/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "data/2026_LoL_esports_match_data_from_OraclesElixir.csv",
]
team_df = build_team_df(filenames)
team_profiles = build_team_profiles(team_df, intl_boost=3.0)
matchups = build_matchups(team_df)
region_weights = build_region_weights(team_df)
model, scaler, feature_cols, x_test_scaled, y_test = train_model(matchups)

def predict_game(team_a, team_b):
    prob_a = predict_matchup(
        team_a, team_b, team_profiles, model, scaler, feature_cols, region_weights
    )
    if prob_a >= 0.5:
        winner, loser, conf = team_a, team_b, prob_a
    else:
        winner, loser, conf = team_b, team_a, 1 - prob_a
    return winner, loser, conf

# --- Bracket ---
print("\n" + "=" * 50)
print("  MSI PLAY-INS BRACKET PREDICTION")
print("=" * 50)

print("\n--- UPPER BRACKET ROUND 1 ---")
ub_winner_1, ub_loser_1, conf_1 = predict_game("T1", "Team Liquid")
print(f"  >>> {ub_winner_1} advances ({conf_1:.1%} confidence), {ub_loser_1} drops to lower bracket")

ub_winner_2, ub_loser_2, conf_2 = predict_game("Karmine Corp", "Deep Cross Gaming")
print(f"  >>> {ub_winner_2} advances ({conf_2:.1%} confidence), {ub_loser_2} drops to lower bracket")

print("\n--- UPPER BRACKET FINALS ---")
ub_finalist, ub_finalist_loser, conf_3 = predict_game(ub_winner_1, ub_winner_2)
print(f"  >>> {ub_finalist} wins ({conf_3:.1%} confidence) — ADVANCES TO MAIN EVENT")

print("\n--- LOWER BRACKET SEMIFINALS ---")
lb_winner, lb_loser, conf_4 = predict_game(ub_loser_1, ub_loser_2)
print(f"  >>> {lb_winner} advances ({conf_4:.1%} confidence), {lb_loser} is ELIMINATED")

print("\n--- LOWER BRACKET FINALS ---")
lb_finalist, eliminated, conf_5 = predict_game(ub_finalist_loser, lb_winner)
print(f"  >>> {lb_finalist} wins ({conf_5:.1%} confidence) — ADVANCES TO MAIN EVENT")
print(f"  >>> {eliminated} is ELIMINATED")

print("\n" + "=" * 50)
print("  MAIN EVENT QUALIFIERS")
print("=" * 50)
print(f"  1. {ub_finalist}")
print(f"  2. {lb_finalist}")
print("=" * 50)
