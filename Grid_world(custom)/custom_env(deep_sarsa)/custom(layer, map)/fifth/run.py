import copy
import pylab
import random
import numpy as np
from environment_val import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES =10

class DeepSARSAgent:
    def __init__(self):
        self.load_model = True
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 39
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        
        if self.load_model:
            self.epsilon = 0.00
            self.model.load_weights('./save_model/deep_sarsa_custom.h5')
            
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(60, input_dim=self.state_size, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def get_action(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
        
    def train_model(self,state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            state = np.float32(state)
            next_state = np.float32(next_state)
            target = self.model.predict(state)[0]
            
            if done:
                target[action] = reward
            
            else:
                target[action] = (reward+ self.discount_factor *
                                 self.model.predict(next_state[0][next_action]))
                
            target = np.reshape(target, [1,5])
            self.model.fit(state, target, epochs=1, verbose=0)

if __name__ == "__main__":
    env = Env()
    agent = DeepSARSAgent()
  
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 39])

        while not done:
            global_step += 1

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1,39])
            next_action = agent.get_action(next_state)
            state = next_state
            score += reward

            if done:
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

