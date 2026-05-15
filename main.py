from pathlib import Path
from src.preprocess import build_team_df
from src.preprocess import build_team_profiles
from src.features import build_matchups

filename = "data/2025_LoL_esports_match_data_from_OraclesElixir.csv"
team_df = build_team_df(filename)
team_profiles = build_team_profiles(team_df)
matchups = build_matchups(team_df)

print(team_df.head())
print(team_profiles.head())
print(matchups.head())