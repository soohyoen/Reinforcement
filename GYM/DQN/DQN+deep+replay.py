# 1st: setup
import numpy as np
import random
from collections import deque
import dqn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import gym
env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis=0.9
REPLAY_MEMORY = 50000


# 2nd : Network setting
# dqn.py
    
# 3nd: Train from Replay Buffer
def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)
    
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
        
        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + dis * np.max(DQN.predict(next_state))
            
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
        
        
    return DQN.update(x_stack, y_stack)


# 4th : bot play
def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break
# 5th : main
def main():
    max_episodes = 5000
    
    replay_buffer =deque()
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size)
        tf.global_variables_initializer().run()
        
        for episode in range(max_episodes):
            e = 1. / ((episode/10) + 1)
            done = False
            step_count =0
            
            state = env.reset()
            
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                    
                next_state,reward,done, _ = env.step(action)
                if done:
                    reward = -100

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break
            print("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000:
                pass
            
            if episode % 10 == 1:
                for _ in range(50):
                    
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = simple_replay_train(mainDQN, minibatch)
                print("Loss :", loss)
                

        bot_play(mainDQN)
if __name__ == "__main__":
    main()
