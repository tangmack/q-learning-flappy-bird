import wrapped_flappy_bird as game
import numpy as np
import time
t0 = time.time()

ALPHA = .1 # learning rate
GAMMA = 0.95 # discount factor
EPISODES = 20000
SHOW_EVERY = 2000

bin_count = [20, 20] # [20, 20]
env_state_high = np.array([260, 250])
env_state_low = np.array([-60, -250])
# bin_size = ([234 - -60, 200 - -200 ]) / bin_count
bin_size = (env_state_high - env_state_low) / bin_count

q_table = np.random.uniform(low= -0.2, high=0.0, size=(bin_count[0],bin_count[1],2))

def discretize_state(state):
    # print(state)
    # print(state - env.observation_space.low)
    discrete_state = (state - env_state_low) / bin_size
    # print(discrete_state)
    return tuple(discrete_state.astype(int))

best_frames_survived = 0
for episode in range(EPISODES):
    game_state = game.GameState()
    total_frames = 0
    max_frames = 10000 # Can change this number according to yourself

    action = 0 # first action will always be nothing
    state, reward, done = game_state.frame_step(action, headless=True, desired_fps=16000)
    # print("starting state: ", state)

    discrete_state = discretize_state(state)
    for frame in range(max_frames):
        # action = 1 # 0 for no action, 1 for flap

        try:
            action = np.argmax(q_table[discrete_state])
        except:
            print(state)
        # new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=10)
        if episode % SHOW_EVERY == 0:
            new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=30)
            print(new_state)
        else:
            new_state, reward, done = game_state.frame_step(action, headless=True, desired_fps=16000)

        new_discrete_state = discretize_state(new_state)

        total_frames += 1
        if not done:
            max_future_q = np.max(q_table[discrete_state])
            current_q = q_table[discrete_state]
            new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
            q_table[discrete_state] = new_q
        elif done:
            print("Total Frames ", str(total_frames), " for episode ", episode)
            if total_frames > best_frames_survived:
                best_frames_survived = total_frames
            break

        discrete_state = new_discrete_state

print("best_frames_survived: ", best_frames_survived)
t1 = time.time()
print("total time: ", t1-t0) # 9.764827251434326, 20,000 episodes, completely headless, 16000 FPS