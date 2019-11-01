import gym
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
from rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3 -- DONE
# from dqn_agent import Agent as DQNAgent  # Use for Task 4 -- DONE
from utils import plot_rewards
import torch

env_name = "CartPole-v0"
# env_name = "LunarLander-v2"
env = gym.make(env_name)
env.reset()

# Reasonable values for Cartpole discretization
num_of_actions = 2
discr = 32
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = np.zeros((discr, discr))  # Cartpole

def discretize_position_velocity(x0, th0):
    return (
        np.argmin(abs(x0-x_grid)),
        np.argmin(abs(th0-th_grid))
    )


def discretize_state(x0):
    return np.array([
        np.argmin(abs(x0[0]-x_grid)),
        np.argmin(abs(x0[1]-v_grid)),
        np.argmin(abs(x0[2]-th_grid)),
        np.argmin(abs(x0[3]-av_grid))
    ])


# Set hyperparameters
# Values for RBF (Tasks 1-3) -- DONE
glie_a = 50
num_episodes = 1000

# Values for DQN  (Task 4) -- DONE
# if "CartPole" in env_name:
#     TARGET_UPDATE = 20
#     glie_a = 200
#     num_episodes = 5001
#     hidden = 12
#     gamma = 0.98
#     replay_buffer_size = 50000
#     batch_size = 32
# elif "LunarLander" in env_name:
#     TARGET_UPDATE = 20
#     glie_a = 5000
#     num_episodes = 15001
#     hidden = 64
#     gamma = 0.95
#     replay_buffer_size = 50000
#     batch_size = 128
# else:
#     raise ValueError("Please provide hyperparameters for %s" % env_name)


# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]

# Tasks 1-3 - RBF -- DONE
agent = RBFAgent(n_actions)

# Task 4 - DQN -- DONE
# agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size, hidden, gamma)

# Training loop
cumulative_rewards = []
policy = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 1: TODO: Update the Q-values -- DONE
        # agent.single_update(state, action, next_state, reward, done)
        # Task 2: TODO: Store transition and batch-update Q-values -- DONE
        agent.store_transition(state, action, next_state, reward, done)
        agent.update_estimator()
        # Task 4: Update the DQN TODO: Call agent update -- DONE
        # agent.update_network()

        # Move to the next state
        state = next_state
    cumulative_rewards.append(cum_reward)

    # Update the target network, copying all weights and biases in DQN
    # TODO: Uncomment for Task 4 -- DONE
    # if ep % TARGET_UPDATE == 0:
    #     agent.update_target_network()

    # Save the policy
    # TODO: Uncomment for Task 4 -- DONE
    # if ep % 1000 == 0:
    #     torch.save(agent.policy_net.state_dict(), "weights_%s_%d.mdl" % (env_name, ep))

print('Complete')
plot_rewards(cumulative_rewards)
plt.ioff()
plt.show()

# TODO: Task 3 - plot the policy -- DONE

for x in x_grid:
    for th in th_grid:
        state = np.array([x, 0, th, 0])
        action = agent.get_action(state)
        x, th = discretize_position_velocity(x, th)
        q_grid[x][th] = action
sb.heatmap(q_grid, cbar=False, xticklabels=x_grid.round(1), yticklabels=th_grid.round(1))
plt.show()
