import pandas as pd

def build_matchups(team_df):
    '''
    train model on matchups: start by blue/red matchup for 
    each matchup and merge to differentiate
    
    Args: 
        team_df, filtered team data with desired features
    
    Returns: 
        matchups, differentials in features in each matchup
    '''

    blue = team_df[team_df['side'] == 'Blue'].copy()
    red  = team_df[team_df['side'] == 'Red'].copy()

    matchups = blue.merge(red, on='gameid', suffixes=('_blue', '_red'))

    # differentiate features to compare opponents
    diff_features = ['golddiffat15', 'golddiffat25', 'killdiffat25', 'gamelength', 'ckpm']
    for f in diff_features:
        matchups[f'diff_{f}'] = matchups[f'{f}_blue'] - matchups[f'{f}_red']
    
    return matchups