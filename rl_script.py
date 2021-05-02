import wrapped_flappy_bird as game
import time
t0 = time.time()

EPISODES = 1

bin_count = [20, 20] # [20, 20]
# bin_size = ([env.observation_space.high - env.observation_space.low]) / bin_count

for episode in range(EPISODES):
    # This is how we initiate one instance of the game
    game_state = game.GameState()
    total_frames = 0
    max_frames = 10000 # Can change this number according to yourself

    for j in range(max_frames):
        action = 1 # 0 for no action, 1 for flap
        new_state, reward, done = game_state.frame_step(action, headless=False, desired_fps=1)
        print(new_state)

        total_frames += 1
        if done:
            break
    print("Total Frames ", str(total_frames))

t1 = time.time()
print("total time: ", t1-t0) # 9.764827251434326, 20,000 episodes, completely headless, 16000 FPS