import wrapped_flappy_bird as game
import numpy as np
import time
t0 = time.time()

EPISODES = 1

bin_count = [20, 20] # [20, 20]
env_state_high = np.array([234, 200])
env_state_low = np.array([-60, -200])
# bin_size = ([234 - -60, 200 - -200 ]) / bin_count
bin_size = (env_state_high - env_state_low) / bin_count

q_table = np.random.uniform(low=-2, high=0, size=(20,20,2))

def discretize_state(state):
    # print(state)
    # print(state - env.observation_space.low)
    discrete_state = (state - env_state_low) / bin_size
    # print(discrete_state)
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    game_state = game.GameState()
    total_frames = 0
    max_frames = 10000 # Can change this number according to yourself

    for frame in range(max_frames):
        action = 1 # 0 for no action, 1 for flap
        new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=10)
        print(new_state)

        total_frames += 1
        if done:
            break
    print("Total Frames ", str(total_frames))

t1 = time.time()
print("total time: ", t1-t0) # 9.764827251434326, 20,000 episodes, completely headless, 16000 FPS