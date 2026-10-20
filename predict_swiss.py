import random
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from src.preprocess import build_team_df, build_team_profiles, build_region_weights
from src.features import build_matchups
from src.model import train_model, predict_matchup

DRAW_SEED = 2026

# Adjacent teams from the Worlds 2026 Swiss Round 1 bracket.
ROUND_1 = [
    ("Gen.G", "Invictus Gaming"),
    ("Team Liquid", "CTBC Flying Oyster"),
    ("Anyone's Legend", "Karmine Corp"),
    ("G2 Esports", "Dplus Kia"),
    ("Hanwha Life Esports", "Top Esports"),
    ("LØS", "T1"),
    ("Team Secret Whales", "LYON"),
    ("Bilibili Gaming", "Movistar KOI"),
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

# Seed after training so only the draw and the match simulations are fixed.
random.seed(DRAW_SEED)
np.random.seed(DRAW_SEED)


def series_win_prob(game_prob):
    '''Best-of-three win probability from an independent game win probability.'''
    return game_prob ** 2 * (3 - 2 * game_prob)


def is_decisive(wins, losses):
    '''A win would advance, or a loss would eliminate.'''
    return wins == 2 or losses == 2


def predict_series(team_a, team_b, decisive):
    game_prob = predict_matchup(
        team_a, team_b, team_profiles, model, scaler, feature_cols, region_weights
    )
    prob_a = series_win_prob(game_prob) if decisive else game_prob
    if prob_a >= 0.5:
        return team_a, team_b, prob_a
    return team_b, team_a, 1 - prob_a


def pair_group(names, opponents):
    '''
    Random pairing inside one record group. Reject draws that rematch
    two teams, then fall back to a search if shuffling does not find one.
    '''
    names = list(names)
    if len(names) % 2 != 0:
        raise RuntimeError(f"record group has an odd number of teams: {names}")

    for _ in range(5000):
        order = names[:]
        random.shuffle(order)
        pairs = list(zip(order[0::2], order[1::2]))
        if all(b not in opponents[a] for a, b in pairs):
            return pairs

    def search(remaining):
        if not remaining:
            return []
        first = remaining[0]
        candidates = remaining[1:]
        random.shuffle(candidates)
        for opp in candidates:
            if opp in opponents[first]:
                continue
            rest = [team for team in remaining[1:] if team != opp]
            tail = search(rest)
            if tail is not None:
                return [(first, opp)] + tail
        return None

    for _ in range(20):
        random.shuffle(names)
        found = search(list(names))
        if found is not None:
            return found
    raise RuntimeError(f"no pairing without a rematch for {names}")


def play_match(records, opponents, team_a, team_b):
    wins_a, losses_a = records[team_a]
    decisive = is_decisive(wins_a, losses_a)
    winner, loser, conf = predict_series(team_a, team_b, decisive)
    w_wins, w_losses = records[winner]
    l_wins, l_losses = records[loser]
    records[winner] = (w_wins + 1, w_losses)
    records[loser] = (l_wins, l_losses + 1)
    opponents[team_a].add(team_b)
    opponents[team_b].add(team_a)

    series = "Bo3" if decisive else "Bo1"
    new_w = records[winner]
    new_l = records[loser]
    if new_w[0] == 3:
        outcome = f"ADVANCES {new_w[0]}-{new_w[1]}"
    else:
        outcome = f"moves to {new_w[0]}-{new_w[1]}"
    if new_l[1] == 3:
        loser_outcome = "is ELIMINATED"
    else:
        loser_outcome = f"falls to {new_l[0]}-{new_l[1]}"
    print(f"  {team_a} vs {team_b} ({series})")
    print(f"  >>> {winner} {outcome} ({conf:.1%} confidence), {loser} {loser_outcome}")


def active_by_record(records):
    groups = {}
    for team, (wins, losses) in records.items():
        if wins >= 3 or losses >= 3:
            continue
        groups.setdefault((wins, losses), []).append(team)
    return sorted(groups.items(), key=lambda item: (-item[0][0], item[0][1]))


known = set(team_profiles['teamname'])
missing = sorted({team for pair in ROUND_1 for team in pair if team not in known})
if missing:
    raise SystemExit(f"teams missing from profiles: {missing}")

records = {team: (0, 0) for pair in ROUND_1 for team in pair}
opponents = {team: set() for team in records}

print("\n" + "=" * 50)
print("  WORLDS 2026 SWISS STAGE PREDICTION")
print(f"  draw seed: {DRAW_SEED}")
print("=" * 50)

for round_num in range(1, 6):
    print(f"\n--- ROUND {round_num} ---")
    if round_num == 1:
        for team_a, team_b in ROUND_1:
            play_match(records, opponents, team_a, team_b)
        continue

    for (wins, losses), teams in active_by_record(records):
        print(f"\n  [{wins}-{losses}]")
        for team_a, team_b in pair_group(teams, opponents):
            play_match(records, opponents, team_a, team_b)

advanced = []
for team, (wins, losses) in records.items():
    if wins == 3:
        advanced.append((losses, team))
    elif losses != 3:
        raise RuntimeError(f"{team} still active at {wins}-{losses}")

by_losses = {0: [], 1: [], 2: []}
for losses, team in advanced:
    by_losses[losses].append(team)
for losses in by_losses:
    by_losses[losses].sort()

print("\n" + "=" * 50)
print("  SWISS RESULTS")
print("=" * 50)
print(f"  3-0: {', '.join(by_losses[0])}")
print(f"  3-1: {', '.join(by_losses[1])}")
print(f"  3-2: {', '.join(by_losses[2])}")
advancers = by_losses[0] + by_losses[1] + by_losses[2]
print(f"\n  Advance ({len(advancers)}): {', '.join(advancers)}")
print("=" * 50)
