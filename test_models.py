# notes
# run / test neural networks

#imports 
from tensorflow import keras
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from simulation.nn.create_models.offensive_preprocess import *
import os

# load the already 
oc = keras.models.load_model("simulation/nn/usable_models/offensive_playtype_model.h5")
if os.path.exists('simulation/nn/output/offensive_playtype_test.txt'):
    os.remove('simulation/nn/output/offensive_playtype_test.txt')
if os.path.exists('simulation/nn/output/offensive_playtype_test_2.txt'):
    os.remove('simulation/nn/output/offensive_playtype_test_2.txt')
for _ in range(100):
    # yardline, game seconds remaining, down, yards to go, possession team timeouts, defensive team timeouts, possession score, defense team score
    yard_line = random.randint(1,99)
    game_sec = random.randint(1,120)
    down = random.randint(1,4)
    ytg = random.randint(1,12)
    pos_timeo = random.randint(1,3)
    def_timeo = random.randint(1,3)
    pos_score = random.choice([3,7,10,13,14,17,21])
    def_score = random.choice([3,7,10,13,14,17,21])
    situation = [yard_line,game_sec,down,ytg,pos_timeo,def_timeo,pos_score,def_score]
    # Ensure 'situation' has the same number of features as your training data
    situation = np.array(situation).reshape(1, -1)
    # Scale the input data using the same scaler
    situation_scaled = scaler.transform(situation)
    # Make the prediction
    prediction = oc.predict(situation_scaled)
    oc.predict(situation)
    # Decode the prediction to get the play type
    predicted_play_type = list(play_type_map.keys())[np.argmax(prediction)]
    with open('simulation/nn/output/offensive_playtype_test.txt','a') as w:
        w.write(f"Yard Line: {yard_line}\n")
        w.write(f"Game Seconds Remaining: {game_sec}\n")
        w.write(f"Down: {down}\n")
        w.write(f"Yards to Go: {ytg}\n")
        w.write(f"Possession Team Timeouts: {pos_timeo}\n")
        w.write(f"Defensive Team Timeouts: {def_timeo}\n")
        w.write(f"Possession Score: {pos_score}\n")
        w.write(f"Defense Score: {def_score}\n")
        w.write(f"Prediction: {predicted_play_type}\n----------\n")
    # print("Predicted Play Type:", predicted_play_type)


for _ in range(100):
    # yardline, game seconds remaining, down, yards to go, possession team timeouts, defensive team timeouts, possession score, defense team score
    yard_line = random.randint(1,99)
    game_sec = random.randint(1,120)
    down = random.randint(4,4)
    ytg = random.randint(1,25)
    pos_timeo = random.randint(0,2)
    def_timeo = random.randint(0,3)
    pos_score = random.choice([3,7,10,13,14,17,21])
    def_score = random.choice([3,7,10,13,14,17,21])
    situation = [yard_line,game_sec,down,ytg,pos_timeo,def_timeo,pos_score,def_score]
    # Ensure 'situation' has the same number of features as your training data
    situation = np.array(situation).reshape(1, -1)
    # Scale the input data using the same scaler
    situation_scaled = scaler.transform(situation)
    # Make the prediction
    prediction = oc.predict(situation_scaled)
    oc.predict(situation)
    # Decode the prediction to get the play type
    predicted_play_type = list(play_type_map.keys())[np.argmax(prediction)]
    with open('simulation/nn/output/offensive_playtype_test_2.txt','a') as w:
        w.write(f"Yard Line: {yard_line}\n")
        w.write(f"Game Seconds Remaining: {game_sec}\n")
        w.write(f"Down: {down}\n")
        w.write(f"Yards to Go: {ytg}\n")
        w.write(f"Possession Team Timeouts: {pos_timeo}\n")
        w.write(f"Defensive Team Timeouts: {def_timeo}\n")
        w.write(f"Possession Score: {pos_score}\n")
        w.write(f"Defense Score: {def_score}\n")
        w.write(f"Prediction: {predicted_play_type}\n----------\n")
    # print("Predicted Play Type:", predicted_play_type)


