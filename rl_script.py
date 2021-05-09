import wrapped_flappy_bird as game
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import h5py
t0 = time.time()

ALPHA = .7 # learning rate
GAMMA = 0.95 # discount factor
# EPISODES = 100_000 # 17 minute run time
# EPISODES = 600_000 # 36 minute run time
# EPISODES = 600_000 # 93 minute run time
# EPISODES = 600_000 # 93 minute run time
# EPISODES = 600_000*5.4 # 93 minute run time
EPISODES = 3240000 # 93 minute run time
# EPISODES = 10 # 17 minute run time
# EPISODES = 10_000
# EPISODES = 1000
# SHOW_EVERY = 100_000
SHOW_EVERY = 3240000
# SHOW_EVERY = 1_000
# SHOW_EVERY = 1

# AFTER = 80_000
AFTER = 0

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# FLAP_EVERY = 17

bin_count = [200, 410, 410, 10] # [20, 20]
# bin_count = [220, 451, 451, 380, 10] # [20, 20]
env_state_high = np.array([250, 234, 234, 11])
env_state_low = np.array([30, -217, -217, -9])
env_number_of_actions = 2
# bin_size = ([234 - -60, 200 - -200 ]) / bin_count
bin_size = (env_state_high - env_state_low) / bin_count

# q_table = np.random.uniform(low= -0.2, high=0.2, size=(bin_count[0],bin_count[1],2))
# q_table = np.random.uniform(low= -0.1, high=0.0, size=(bin_count + [env_number_of_actions]))

# q_table = np.random.uniform(low= -0.2, high=0.0, size=(bin_count + [env_number_of_actions]))

# q_table[:,:,1] = np.random.uniform(low=-.5, high=0.0, size=(bin_count[0],bin_count[1])) # de-emphasize flap (avoid hitting ceiling)

# q_table = np.load(f"./qtables/{7078}-qtable.npy")

# hfr = h5py.File(f"qtables/{6640}-qtable.h5", 'r')
# q_table = np.array(hfr.get('dataset_1'))
# hfr.close()

hfr = h5py.File(f"qtables/qtable_long.h5", 'r')
q_table = np.array(hfr.get('dataset_1'))
hfr.close()


def discretize_state(state):
    # print(state)
    # print(state - env.observation_space.low)
    discrete_state = (state - env_state_low) / bin_size
    # print(discrete_state)
    return tuple(discrete_state.astype(int))

episode_state_action_new_states = []
frames_survived = []
env_max_measured_values = [-999, -999, -999, -999, -999]
env_min_measured_values = [999, 999, 999, 999, 999]
best_frames_survived = 0
for episode in range(EPISODES):
    game_state = game.GameState()
    total_frames = 0
    max_frames = 10000 # Can change this number according to yourself

    action = 0 # first action will always be nothing
    state, reward, done = game_state.frame_step(action, headless=True, desired_fps=16000)
    # print("starting state: ", state)

    action = 0 # first action will always be nothing
    state, reward, done = game_state.frame_step(action, headless=True, desired_fps=16000)
    # print("starting state: ", state)

    discrete_state = discretize_state(state)
    for frame in range(max_frames):

        try:
            action = np.argmax(q_table[discrete_state])

            # if np.random.random() > epsilon:
            #     # Get action from Q table
            #     action = np.argmax(q_table[discrete_state])
            # else:
            #     # Get random action
            #     roll = np.random.uniform(low=0.0, high=1.0)
            #     if roll < 0.80: # do random action, with emphasis on doing nothing
            #         action = 0
            #     else:
            #         action = 1

            # action = np.argmax(q_table[discrete_state])

            # if frame % FLAP_EVERY == 0: action = 1
            # else: action = 0
        except:
            print(state)
        # new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=10)
        if episode % SHOW_EVERY == 0 and episode > AFTER:
            new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=30)
            print(new_state, action)
        else:
            new_state, reward, done = game_state.frame_step(action, headless=True, desired_fps=16000)

        # if new_state[0] == 257.0:
        #     pass
        #     print("stop")


        total_frames += 1
        if not done:
            # if new_state[0] < env_min_measured_values[0]:
            #     env_min_measured_values[0] = new_state[0]
            # if new_state[1] < env_min_measured_values[1]:
            #     env_min_measured_values[1] = new_state[1]
            # if new_state[2] < env_min_measured_values[2]:
            #     env_min_measured_values[2] = new_state[2]
            # if new_state[3] < env_min_measured_values[3]:
            #     env_min_measured_values[3] = new_state[3]
            #
            # if new_state[0] > env_max_measured_values[0]:
            #     env_max_measured_values[0] = new_state[0]
            # if new_state[1] > env_max_measured_values[1]:
            #     env_max_measured_values[1] = new_state[1]
            # if new_state[2] > env_max_measured_values[2]:
            #     env_max_measured_values[2] = new_state[2]
            # if new_state[3] > env_max_measured_values[3]:
            #     env_max_measured_values[3] = new_state[3]


            new_discrete_state = discretize_state(new_state)

            episode_state_action_new_states.append((discrete_state, action, new_discrete_state))

            # # max_future_q = np.max(q_table[discrete_state]) # big mistake
            # max_future_q = np.max(q_table[new_discrete_state])
            # current_q = q_table[discrete_state][action]
            # new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
            # q_table[discrete_state][action] = new_q
        elif done:
            # new_q = (1 - ALPHA) * current_q + ALPHA * (reward)
            # q_table[discrete_state][action] = new_q

            episode_state_action_new_states.reverse() # already not appending very very last faulty state (don't reach if not done above)
            last_flap_dealt_with = False

            if episode_state_action_new_states[0][0][1] > 0: upper_pipe_death = True
            else: upper_pipe_death = False

            # bird has died, update q values
            for idx, state_action_new_state in enumerate(episode_state_action_new_states):
                discrete_state_ = state_action_new_state[0]
                action_ = state_action_new_state[1]
                new_discrete_state_ = state_action_new_state[2]


                # idea behind this: if there was an upper pipe death, it was ACTION that caused that, versus no action, if lower pipe death
                # if upper_pipe_death == True:
                if last_flap_dealt_with == False and upper_pipe_death == True and action_ == 1: # deal with last flap if we haven't before and action = 1 = flap and we had upper_pipe_death
                    max_future_q = np.max(q_table[new_discrete_state_])
                    current_q = q_table[discrete_state_][action_]
                    new_q = (1 - ALPHA) * current_q + ALPHA * (-1000 + GAMMA * max_future_q) # -1000 reward
                    q_table[discrete_state_][action_] = new_q
                    last_flap_dealt_with = True
                elif idx == 0 or idx == 1: # punish anything near ceiling, floor, or pipes
                    max_future_q = np.max(q_table[new_discrete_state_])
                    current_q = q_table[discrete_state_][action_]
                    new_q = (1 - ALPHA) * current_q + ALPHA * (-1000 + GAMMA * max_future_q)  # -1000 reward
                    q_table[discrete_state_][action_] = new_q
                else: # else, normal case, just give +1 reward
                    max_future_q = np.max(q_table[new_discrete_state_])
                    current_q = q_table[discrete_state_][action_]
                    new_q = (1 - ALPHA) * current_q + ALPHA * (1 + GAMMA * max_future_q) # +1 reward
                    q_table[discrete_state_][action_] = new_q


            episode_state_action_new_states = [] # empty out saved states action state tuples

            print("Total Frames ", str(total_frames), " for episode ", episode)
            if total_frames > best_frames_survived:
                best_frames_survived = total_frames
                # if total_frames > 4000: # save hard drive space
                #     # np.save(f"qtables/{total_frames}-qtable.npy", q_table)
                #     hfw = h5py.File(f"qtables/{total_frames}-qtable.h5", 'w')
                #     hfw.create_dataset('dataset_1', data=q_table)
                #     hfw.close()
            if total_frames >= 10000: # save hard drive space
                print("saving q table over 4000")
                # np.save(f"qtables/{total_frames}-qtable.npy", q_table)
                # hfw = h5py.File(f"qtables/{11111111}-qtable.h5", 'w')
                hfw = h5py.File(f"qtables/{total_frames}-qtable_long.h5", 'w')
                hfw.create_dataset('dataset_1', data=q_table)
                hfw.close()
                print("q table done saving over 4000")
            if episode == EPISODES-1: # save hard drive space
                print("saving q table")
                # np.save(f"qtables/{total_frames}-qtable.npy", q_table)
                # hfw = h5py.File(f"qtables/{11111111}-qtable.h5", 'w')
                hfw = h5py.File(f"qtables/qtable_long.h5", 'w')
                hfw.create_dataset('dataset_1', data=q_table)
                hfw.close()
                print("q table done saving")

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

plt.plot(range(len(frames_survived)), frames_survived, linestyle='', marker='.')
plt.show()

print("total frames survived = ", sum(frames_survived))

print("min frames survived: ", min(frames_survived) )
print("average frames survived: ", sum(frames_survived)/len(frames_survived) )
print("max frames survived: ", max(frames_survived))

print(" ")
print("env_min_measured_values: ", env_min_measured_values)
print("env_max_measured_values: ", env_max_measured_values)