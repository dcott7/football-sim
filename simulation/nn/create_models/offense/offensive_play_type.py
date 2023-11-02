from simulation.nn.create_models.offense.offensive_play_preprocess import *

model = keras.Sequential([
    layers.Input(shape = (X_train_scaled.shape[1],)),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(len(play_type_map), activation='softmax') # output layer w/ soft-max for multi-class classification
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train_scaled, 
          y_train,
          epochs = 20,
          batch_size = 32,
          validation_split = 0.2)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print("Test accuracy: ", test_acc)

print(X)