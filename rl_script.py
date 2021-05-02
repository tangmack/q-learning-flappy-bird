import tensorflow as tf
import numpy as np

import sys
import os

EPISODES = 1

# Now, we will import the pygame simulation of the flappy bird game

sys.path.append("game/")
import wrapped_flappy_bird as game


for episode in range(EPISODES):
    # This is how we initiate one instance of the game
    game_state = game.GameState()
    total_frames = 0
    max_steps = 1000 # Can change this number according to yourself

    for j in range(max_steps):
        action = 1 # 0 for no action, 1 for flap
        new_state, reward, done = game_state.frame_step(action)

        total_frames += 1
        if done:
            break
    print("Total Steps ", str(total_frames))