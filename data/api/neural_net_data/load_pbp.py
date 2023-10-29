# -----------------------------------------
# import libraries
# -----------------------------------------
import numpy as np

# -----------------------------------------
# import scripts
# -----------------------------------------
from data.api import api_load

# -----------------------------------------
# load and clean data for
# -----------------------------------------
years = [year for year in range(2001, api_load.curr_year + 1)]

# use the nfl api to load play by play data for
# years 2001 - current_year
pbp_curr_year = api_load.nfl.import_pbp_data(years)

# create a list of cols to keep
keep_col_list = [
    'yardline_100', 'quarter_seconds_remaining', 'game_seconds_remaining',
    'qtr', 'down', 'ydstogo', 'play_type', 'posteam_timeouts_remaining',
    'defteam_timeouts_remaining', 'posteam_score','defteam_score', 
    'home_coach', 'away_coach', 'posteam_type'
]

# delete unwanted cols
pbp_curr_year = pbp_curr_year[keep_col_list]

# create an off_coach and def_coach column
pbp_curr_year['off_coach'] = np.where(
    pbp_curr_year['posteam_type'] == 'home', 
    pbp_curr_year['home_coach'], 
    np.where(pbp_curr_year['posteam_type'] == 'away', pbp_curr_year['away_coach'],
             ''))
pbp_curr_year['def_coach'] = np.where(
    pbp_curr_year['posteam_type'] == 'away', 
    pbp_curr_year['home_coach'], 
    np.where(pbp_curr_year['posteam_type'] == 'home', pbp_curr_year['away_coach'],
             ''))

# drop unwanted cols
pbp_curr_year = pbp_curr_year.drop(comlumns = ['posteam_type','home_coach','away_coach'])
