import wrapped_flappy_bird as game
import numpy as np
import matplotlib.pyplot as plt
import time
t0 = time.time()

ALPHA = .1 # learning rate
GAMMA = 0.95 # discount factor
EPISODES = 20000
SHOW_EVERY = 100000

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

FLAP_EVERY = 17

bin_count = [20, 20] # [20, 20]
env_state_high = np.array([260, 250])
env_state_low = np.array([-60, -250])
env_number_of_actions = 2
# bin_size = ([234 - -60, 200 - -200 ]) / bin_count
bin_size = (env_state_high - env_state_low) / bin_count

q_table = np.random.uniform(low= -0.2, high=0.2, size=(bin_count[0],bin_count[1],2))

def discretize_state(state):
    # print(state)
    # print(state - env.observation_space.low)
    discrete_state = (state - env_state_low) / bin_size
    # print(discrete_state)
    return tuple(discrete_state.astype(int))

frames_survived = []
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

        try:
            action = np.argmax(q_table[discrete_state])

            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                roll = np.random.uniform(low=0.0, high=1.0)
                if roll < 0.80: # do random action, with emphasis on doing nothing
                    action = 0
                else:
                    action = 1

            # action = np.argmax(q_table[discrete_state])

            # if frame % FLAP_EVERY == 0: action = 1
            # else: action = 0
        except:
            print(state)
        # new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=10)
        if episode % SHOW_EVERY == 0:
            new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=30)
            print(new_state, action)
        else:
            new_state, reward, done = game_state.frame_step(action, headless=True, desired_fps=16000)

        new_discrete_state = discretize_state(new_state)

        total_frames += 1
        if not done:
            max_future_q = np.max(q_table[discrete_state])
            current_q = q_table[discrete_state][action]
            new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
            q_table[discrete_state][action] = new_q
        elif done:
            print("Total Frames ", str(total_frames), " for episode ", episode)
            if total_frames > best_frames_survived:
                best_frames_survived = total_frames
            break

        discrete_state = new_discrete_state

    frames_survived.append(total_frames)

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

print(" ")
print("best_frames_survived: ", best_frames_survived)
t1 = time.time()
print("total time: ", t1-t0) # 9.764827251434326, 20,000 episodes, completely headless, 16000 FPS

plt.plot(frames_survived)
plt.show()

print("min frames survived: ", min(frames_survived) )
print("average frames survived: ", sum(frames_survived)/len(frames_survived) )
print("max frames survived: ", max(frames_survived))