import gymnasium as gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
from config import LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_FACTOR, LUNAR_LANDER_EPISODES

env = gym.make("LunarLander-v3")

# Discretization bins
position_bins = np.linspace(-1.5, 1.5, 10)
velocity_bins = np.linspace(-3.0, 3.0, 10)
angle_bins = np.linspace(-0.4, 0.4, 10)
angular_velocity_bins = np.linspace(-3.0, 3.0, 10)

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

# Initialize Q-table
state_space_size = (11, 11, 11, 11, 11, 11, 2, 2)
action_space_size = env.action_space.n
q_table = np.zeros(state_space_size + (action_space_size,))

# Epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

#  Q-Learning Training
rewards_per_episode = []

for episode in range(LUNAR_LANDER_EPISODES):
    state = get_state(env.reset()[0])
    total_reward = 0
    terminated = False
    truncated = False
    
    done = False
    
    while not done:
        action = choose_action(state, EXPLORATION_FACTOR)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = get_state(observation)
        
        # Q-Learning Update Rule (off-policy: uses max Q-value for next state)
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + DISCOUNT_FACTOR * q_table[next_state][best_next_action] * (not done)
        td_error = td_target - q_table[state][action]
        q_table[state][action] += LEARNING_RATE * td_error
            
        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()

# Plotting
plt.plot(rewards_per_episode)
plt.title('Episodic Returns for Lunar Lander with Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('lunar_lander_qlearning_returns.png', dpi=300, bbox_inches='tight')
plt.show()


# Simulate Learned Strategy
env = gym.make("LunarLander-v3", render_mode="rgb_array")
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

imageio.mimsave("lunar_lander_qlearning0.gif", frames, fps=30)

imageio.mimsave("lunar_lander_qlearning0.mp4", frames, fps=30)

print(f"\nTime taken to achieve the goal using the learned Q-Learning strategy: {time_to_goal} steps.")
print(f"Simulation wall-clock time: {end_time - start_time:.2f} seconds.")
