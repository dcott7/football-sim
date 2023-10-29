# -----------------------------------------
# import libraries
# -----------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers

# -----------------------------------------
# import scripts
# -----------------------------------------
from data.api.neural_net_data import load_pbp

# -----------------------------------------
# pull and clean data
# -----------------------------------------
df = load_pbp.pbp_curr_year
df = df.drop(columns = [
    'off_coach', 
    'def_coach', 
    'quarter_seconds_remaining', 
    'qtr'])

df = df[~df['play_type'].isin([None, 'kickoff', 'no_play', 'extra_point'])]
df = df.dropna()
df = df.reset_index(drop = True)
# df.to_csv('pbp_cleaned.csv')
print(df)
# # remove play_type from training data
# X = df.drop(columns = ['play_type'])
# # create a numerical dictionary for each play_type
# # This is done because we need the output to be a 
# # numerical output and not categorical. We create
# # this dictionary so that we can use reference it
# # once our nn output a number. 
# # Ex. if the number 1 is output it is actually 'pass'
# play_type_map = {
#     play_type: index for index, play_type in enumerate(df['play_type'].unique())
# }
# y = df['play_type'].map(play_type_map)

# # split the data into training and testing df
# X_train, X_test, y_train, y_test = train_test_split(X, 
#                                                     y, 
#                                                     test_size = 0.2,
#                                                     random_state = 42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = keras.Sequential([
#     layers.Input(shape = (X_train_scaled.shape[1],)),
#     layers.Dense(64, activation = 'relu'),
#     layers.Dense(32, activation = 'relu'),
#     layers.Dense(len(play_type_map), activation='softmax') # output layer w/ soft-max for multi-class classification
# ])

# model.compile(optimizer = 'adam',
#               loss = 'sparse_categorical_crossentropy',
#               metrics = ['accuracy'])

# model.fit(X_train_scaled, 
#           y_train,
#           epochs = 20,
#           batch_size = 32,
#           validation_split = 0.2)

# test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
# print("Test accuracy: ", test_acc)

# model.save('simulation/nn/output/offensive_playtype_model.h5')