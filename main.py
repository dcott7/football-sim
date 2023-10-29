# run the neural network
# from simulation.nn.create_models import offensive_play_type
import sys
print(sys.version)

# -----------
from tensorflow import keras
import random
# -----------

# # load model
model = keras.models.load_model('simulation/nn/usable_models/offensive_playtype_model.h5')

for _ in range(100):
    # yardline, game seconds remaining, down, yards to go, possession team timeouts, defensive team timeouts, possession score, defense team score
    yard_line = random.randint(1,99)
    game_sec = random.randint(1,120)
    down = random.randint(1,4)
    ytg = random.randint(1,12)
    pos_to = random.randint(1,3)
    def_to = random.randint(1,3)
    pos_score = random.choice([3,7,10,13,14,17,21])
    def_score = random.choice([3,7,10,13,14,17,21])
    situation = [yard_line,game_sec,down,ytg,pos_to,def_to,pos_score,def_score]
    model.predict(situation)