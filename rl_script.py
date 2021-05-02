import tensorflow as tf
import numpy as np

import sys
import os


# Now, we will import the pygame simulation of the flappy bird game

sys.path.append("game/")
import wrapped_flappy_bird as game
#
# class flappy():
#     def play():


for i in range(1):
    # This is how we initiate one instance of the game
    game_state = game.GameState()
    total_steps = 0
    max_steps = 1000 # Can change this number according to yourself

    for j in range(max_steps):
        # Here we are just taking a uniform random action
        # 0 denotes no action
        # 1 denotes flap
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        # sum of input_actions[0] + input_actions[1] must be equal to 1
        # temp = np.random.randint(0,1)
        # action = np.zeros([2])
        # action[temp] = 1
        action = np.array([0, 1])
        new_state, reward, done = game_state.frame_step(action)

        total_steps += 1
        if done:
            break
    print("Total Steps ", str(total_steps))