from pathlib import Path
import warnings
from src.preprocess import build_team_df
from src.preprocess import build_team_profiles
from src.preprocess import build_region_weights
from src.features import build_matchups
from src.model import train_model
from src.model import evaluate_model
from src.model import predict_matchup

warnings.filterwarnings('ignore')
filenames = [
    "data/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "data/2026_LoL_esports_match_data_from_OraclesElixir.csv",
]
team_df = build_team_df(filenames)

INTERNATIONAL_MODE = True  # set to False for domestic predictions

team_profiles = build_team_profiles(team_df, intl_boost=3.0 if INTERNATIONAL_MODE else 1.0)
matchups = build_matchups(team_df)

region_weights = build_region_weights(team_df)

model, scaler, feature_cols, x_test_scaled, y_test = train_model(matchups)

print(predict_matchup("T1", "Cloud9", team_profiles, model, scaler, feature_cols, region_weights))