import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
from config import LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_FACTOR, LUNAR_LANDER_EPISODES

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)

# Discretization bins
position_bins = np.linspace(-1.5, 1.5, 10)
velocity_bins = np.linspace(-3.0, 3.0, 10)
angle_bins = np.linspace(-0.4, 0.4, 10)
angular_velocity_bins = np.linspace(-3.0, 3.0, 10)
leg_contact_bins = [0, 1]

# Q-table shape: (state_space) + (action_space,)
state_space_size = (11, 11, 11, 11, 11, 11, 2, 2)
action_space_size = env.action_space.n
q_table = np.zeros(state_space_size + (action_space_size,))

# Discretize Lunar Lander state (8D → reduced for simplicity)
def get_state(state):
    pos_x = np.digitize(state[0], position_bins)
    pos_y = np.digitize(state[1], position_bins)
    vel_x = np.digitize(state[2], velocity_bins)
    vel_y = np.digitize(state[3], velocity_bins)
    angle = np.digitize(state[4], angle_bins)
    ang_vel = np.digitize(state[5], angular_velocity_bins)
    left_leg = int(state[6])
    right_leg = int(state[7])

    return (pos_x, pos_y, vel_x, vel_y, angle, ang_vel, left_leg, right_leg)

# Epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# SARSA(0) Training
rewards_per_episode = []

for episode in range(LUNAR_LANDER_EPISODES):
    state = get_state(env.reset()[0])
    action = choose_action(state, EXPLORATION_FACTOR)
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = get_state(observation)
        next_action = choose_action(next_state, EXPLORATION_FACTOR)
        
        total_reward += reward
        done = terminated or truncated
        
        # SARSA Update Rule (with terminal state handling)
        td_target = reward + DISCOUNT_FACTOR * q_table[next_state][next_action] * (not done)
        td_error = td_target - q_table[state][action]
        q_table[state][action] += LEARNING_RATE * td_error
        
        state = next_state
        action = next_action

    rewards_per_episode.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()

#  Plotting
plt.plot(rewards_per_episode)
plt.title('Episodic Returns for Lunar Lander with SARSA(0)')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('lunar_lander_sarsa_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# Simulate Learned Strategy
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
state = get_state(env.reset()[0])
terminated = False
truncated = False
time_to_goal = 0
frames = []

start_time = time.time()
while not (terminated or truncated):
    action = choose_action(state, 0)  # Use learned policy (epsilon=0)
    observation, reward, terminated, truncated, info = env.step(action)
    state = get_state(observation)
    time_to_goal += 1
    frame = env.render()
    frames.append(frame)

end_time = time.time()
env.close()

imageio.mimsave("lunar_lander_sarsa.gif", frames, fps=30)

imageio.mimsave("lunar_lander_sarsa.mp4", frames, fps=30)

print(f"\nTime taken to achieve the goal using the learned SARSA(0) strategy: {time_to_goal} steps.")
print(f"Simulation wall-clock time: {end_time - start_time:.2f} seconds.")
