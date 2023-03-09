import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

state_size = 39
action_size = 5


model = Sequential()
model.add(Dense(40, input_dim=state_size, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(action_size, activation = 'linear'))
model.summary()

import os

x = os.getcwd()

print(x)

reconstructed_model = keras.models.load_model("/home/soohyoen/Reinforcement_study/Grid_world(custom)/custom_env(deep_sarsa)/custom(layer, map)/sixth/save_model/deep_sarsa(10 env).h5")


