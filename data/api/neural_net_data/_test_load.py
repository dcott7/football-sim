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

# use the nfl api to load play by play data for
# years 2001 - current_year
pbp_curr_year = api_load.nfl.import_pbp_data([2023])

pbp_curr_year.to_csv('data/api/neural_net_data/pbp_2023.csv')