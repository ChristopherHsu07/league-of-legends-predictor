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
print("  MSI 2026 MAIN EVENT BRACKET PREDICTION")
print("=" * 50)

# Upper Bracket Round 1
print("\n--- UPPER BRACKET ROUND 1 ---")
ub_r1_w1, ub_r1_l1, conf_1 = predict_game("Bilibili Gaming", "T1")
print(f"  >>> {ub_r1_w1} advances ({conf_1:.1%} confidence), {ub_r1_l1} drops to lower bracket")

ub_r1_w2, ub_r1_l2, conf_2 = predict_game("LYON", "FURIA")
print(f"  >>> {ub_r1_w2} advances ({conf_2:.1%} confidence), {ub_r1_l2} drops to lower bracket")

ub_r1_w3, ub_r1_l3, conf_3 = predict_game("Hanwha Life Esports", "Team Secret Whales")
print(f"  >>> {ub_r1_w3} advances ({conf_3:.1%} confidence), {ub_r1_l3} drops to lower bracket")

ub_r1_w4, ub_r1_l4, conf_4 = predict_game("G2 Esports", "Top Esports")
print(f"  >>> {ub_r1_w4} advances ({conf_4:.1%} confidence), {ub_r1_l4} drops to lower bracket")

# Lower Bracket Round 1 (mirror-seeded: R1 losers play same-half)
print("\n--- LOWER BRACKET ROUND 1 ---")
lb_r1_w1, lb_r1_l1, conf_5 = predict_game(ub_r1_l1, ub_r1_l2)
print(f"  >>> {lb_r1_w1} advances ({conf_5:.1%} confidence), {lb_r1_l1} is ELIMINATED")

lb_r1_w2, lb_r1_l2, conf_6 = predict_game(ub_r1_l3, ub_r1_l4)
print(f"  >>> {lb_r1_w2} advances ({conf_6:.1%} confidence), {lb_r1_l2} is ELIMINATED")

# Upper Bracket Round 2
print("\n--- UPPER BRACKET ROUND 2 ---")
ub_r2_w1, ub_r2_l1, conf_7 = predict_game(ub_r1_w1, ub_r1_w2)
print(f"  >>> {ub_r2_w1} advances ({conf_7:.1%} confidence), {ub_r2_l1} drops to lower bracket")

ub_r2_w2, ub_r2_l2, conf_8 = predict_game(ub_r1_w3, ub_r1_w4)
print(f"  >>> {ub_r2_w2} advances ({conf_8:.1%} confidence), {ub_r2_l2} drops to lower bracket")

# Lower Bracket Round 2 (UB R2 losers vs LB R1 winners, cross-matched)
print("\n--- LOWER BRACKET ROUND 2 ---")
lb_r2_w1, lb_r2_l1, conf_9 = predict_game(ub_r2_l1, lb_r1_w2)
print(f"  >>> {lb_r2_w1} advances ({conf_9:.1%} confidence), {lb_r2_l1} is ELIMINATED")

lb_r2_w2, lb_r2_l2, conf_10 = predict_game(ub_r2_l2, lb_r1_w1)
print(f"  >>> {lb_r2_w2} advances ({conf_10:.1%} confidence), {lb_r2_l2} is ELIMINATED")

# Upper Bracket Round 3 (semifinals)
print("\n--- UPPER BRACKET ROUND 3 ---")
ub_finalist, ub_r3_loser, conf_11 = predict_game(ub_r2_w1, ub_r2_w2)
print(f"  >>> {ub_finalist} advances ({conf_11:.1%} confidence), {ub_r3_loser} drops to lower bracket")

# Lower Bracket Round 3
print("\n--- LOWER BRACKET ROUND 3 ---")
lb_r3_winner, lb_r3_loser, conf_12 = predict_game(lb_r2_w1, lb_r2_w2)
print(f"  >>> {lb_r3_winner} advances ({conf_12:.1%} confidence), {lb_r3_loser} is ELIMINATED")

# Lower Bracket Round 4 / LB Finals
print("\n--- LOWER BRACKET FINALS ---")
lb_finalist, lb_finals_loser, conf_13 = predict_game(ub_r3_loser, lb_r3_winner)
print(f"  >>> {lb_finalist} advances ({conf_13:.1%} confidence), {lb_finals_loser} is ELIMINATED")

# Grand Finals
print("\n--- GRAND FINALS ---")
champion, runner_up, conf_14 = predict_game(ub_finalist, lb_finalist)
print(f"  >>> {champion} wins ({conf_14:.1%} confidence)")
print(f"  >>> {runner_up} is ELIMINATED")

print("\n" + "=" * 50)
print("  MSI 2026 CHAMPION")
print("=" * 50)
print(f"  {champion}")
print("=" * 50)
