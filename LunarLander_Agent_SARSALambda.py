import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from config import LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_FACTOR, LUNAR_LANDER_EPISODES, LAMBDA

EPS_MIN = 0.01  # Lower minimum for more exploitation
EPS_DECAY = 0.9995  # Slower decay

env = gym.make("LunarLander-v3", continuous=False)

# Simplified discretization (focus on most important features)
bins = {
    "pos_x": np.linspace(-1.5, 1.5, 8),
    "pos_y": np.linspace(-0.5, 1.5, 8),
    "vel_x": np.linspace(-2, 2, 8),
    "vel_y": np.linspace(-2, 2, 8),
    "angle": np.linspace(-0.5, 0.5, 8),
    "ang_vel": np.linspace(-2, 2, 8)
}

def discretize(state):
    # Clip values to reasonable ranges before discretization
    pos_x = np.digitize(np.clip(state[0], -1.5, 1.5), bins["pos_x"])
    pos_y = np.digitize(np.clip(state[1], -0.5, 1.5), bins["pos_y"])
    vel_x = np.digitize(np.clip(state[2], -2, 2), bins["vel_x"])
    vel_y = np.digitize(np.clip(state[3], -2, 2), bins["vel_y"])
    angle = np.digitize(np.clip(state[4], -0.5, 0.5), bins["angle"])
    ang_vel = np.digitize(np.clip(state[5], -2, 2), bins["ang_vel"])
    leg_l = int(state[6])
    leg_r = int(state[7])
    return (pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg_l, leg_r)

state_sizes = (9, 9, 9, 9, 9, 9, 2, 2)
action_size = env.action_space.n
Q = np.zeros(state_sizes + (action_size,))

def choose_action(state, eps):
    if np.random.rand() < eps:
        return np.random.randint(action_size)
    return np.argmax(Q[state])

returns = []
avg_returns = []  # Track moving average
eps = EXPLORATION_FACTOR

for episode in range(LUNAR_LANDER_EPISODES):
    state = discretize(env.reset()[0])
    action = choose_action(state, eps)
    total_reward = 0

    # Sparse eligibility traces: dictionary
    E = {}

    terminated = truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(obs)
        next_action = choose_action(next_state, eps)

        total_reward += reward
        done = terminated or truncated

        td_target = reward + DISCOUNT_FACTOR * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]

        # Update eligibility trace for current (s,a)
        E[(state, action)] = E.get((state, action), 0) + 1

        for (s, a), trace_value in list(E.items()):
            Q[s][a] += LEARNING_RATE * td_error * trace_value
            E[(s, a)] = DISCOUNT_FACTOR * LAMBDA * trace_value

            # Remove small traces to keep it fast
            if E[(s, a)] < 1e-5:
                del E[(s, a)]

        state, action = next_state, next_action

    returns.append(total_reward)
    
    # Calculate moving average (last 100 episodes)
    if len(returns) >= 100:
        avg_returns.append(np.mean(returns[-100:]))
    
    # Decay epsilon
    eps = max(EPS_MIN, eps * EPS_DECAY)

    if (episode + 1) % 100 == 0:
        avg_100 = np.mean(returns[-100:])
        print(f"Episode {episode + 1}/{LUNAR_LANDER_EPISODES}, Return = {total_reward:.1f}, "
              f"Avg(100) = {avg_100:.1f}, Eps = {eps:.3f}")

env.close()

# Plotting training curve
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

ax1.plot(returns, alpha=0.3, label='Episode Return')
if avg_returns:
    ax1.plot(range(99, len(returns)), avg_returns, label='Moving Avg (100)', linewidth=2)
ax1.axhline(y=200, color='g', linestyle='--', label='Solved Threshold')
ax1.set_title(f"SARSA(λ={LAMBDA}) - LunarLander Returns")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Return")
ax1.legend()
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sarsa_lambda_lunar_lander.png', dpi=150)
plt.show()

print("\nGenerating simulation...")
env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
state = discretize(env.reset()[0])
frames = []
terminated = truncated = False
steps = 0
sim_reward = 0

start_time = time.time()
while not (terminated or truncated):
    action = choose_action(state, 0.0)  # Greedy policy
    obs, reward, terminated, truncated, _ = env.step(action)
    state = discretize(obs)
    frames.append(env.render())
    sim_reward += reward
    steps += 1

end_time = time.time()
env.close()

imageio.mimsave("sarsa_lambda_lander.gif", frames, fps=30)
imageio.mimsave("sarsa_lambda_lander.mp4", frames, fps=30)

print(f"Simulation time: {end_time - start_time:.2f} seconds")
print(f"Simulation: {steps} steps, Total Reward: {sim_reward:.1f}")
print(f"Final average (last 100 episodes): {np.mean(returns[-100:]):.1f}")