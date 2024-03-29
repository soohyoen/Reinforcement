import copy
import pylab
import random
import numpy as np
from environment7 import Env7
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dropout
from keras import backend as K
import time

EPISODES = 610


# this is DeepSARSA Agent for the GridWorld
# Utilize Neural Network as q function approximator
class DeepSARSAgent:
    def __init__(self):
        self.load_model = True
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 39
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.99
            self.model.load_weights('./save_model/deep_sarsa.h5')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(20, activation ='relu'))
        model.add(Dense(10, activation ='relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, 5])
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(state, target, epochs=1, verbose=0)
       
if __name__ == "__main__":

    env = Env7()
    agent = DeepSARSAgent()

    global_step = 0
    local_step = 0
    scores, episodes, local_steps = [], [], []
    x = 0

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 39])
  
        while not done:
            # fresh env
            global_step += 1
            local_step += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 39])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action,
                              done)
            state = next_state
            # every time step we do training
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                local_steps.append(local_step)
                pylab.plot(episodes, scores, 'b', label='scores')
                pylab.plot(episodes, local_steps, 'r', label = 'local_step')
                pylab.savefig("./save_graph/env case 10/7.png")
                print("episode:", e, "  score:", score, "global_step",
                      global_step, " epsilon:", agent.epsilon)
                local_step = 0

            if local_step >= 50 and e >= 200:
               done = True
               local_step = 0

        if e % 100 == 0:
            agent.model.save_weights("./save_model/deep_sarsa.h5")
