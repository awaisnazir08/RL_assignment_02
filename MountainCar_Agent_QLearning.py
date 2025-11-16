import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
from config import LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_FACTOR, MOUNTAIN_CAR_EPISODES

env = gym.make("MountainCar-v0")

# Discretize the continuous state space
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

# Initialize Q-table
q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

def get_state(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return (pos_bin, vel_bin)

# Epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Q-Learning Training
rewards_per_episode = []

for episode in range(MOUNTAIN_CAR_EPISODES):
    state = get_state(env.reset()[0])
    total_reward = 0
    terminated = False
    
    while not terminated:
        action = choose_action(state, EXPLORATION_FACTOR)
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(observation)
        
        total_reward += reward
        
        # Q-Learning Update Rule (off-policy: uses max Q-value for next state)
        q_table[state][action] = q_table[state][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

    rewards_per_episode.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()

# Plotting the results
plt.plot(rewards_per_episode)
plt.title('Episodic Returns for Mountain Car with Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('mountain_car_qlearning_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# Simulating Learned Strategy
env = gym.make("MountainCar-v0", render_mode="rgb_array")
state = get_state(env.reset()[0])
terminated = False
time_to_goal = 0
frames = []

start_time = time.time()
while not terminated:
    action = choose_action(state, 0)  # Use learned policy (epsilon=0)
    observation, reward, terminated, truncated, info = env.step(action)
    state = get_state(observation)
    time_to_goal += 1
    frame = env.render()
    frames.append(frame)

end_time = time.time()
env.close()

imageio.mimsave("mountain_car_qlearning.gif", frames, fps=30)

imageio.mimsave("mountain_car_qlearning.mp4", frames, fps=30)

print(f"\nTime taken to achieve the goal using the learned Q-Learning strategy: {time_to_goal} steps.")
print(f"Simulation wall-clock time: {end_time - start_time:.2f} seconds.")
