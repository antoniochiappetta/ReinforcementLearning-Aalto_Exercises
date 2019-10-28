import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import sys

from policies.egpolicy import EGPolicy
from policies.gliepolicy import GLIEPolicy
from policies.greedypolicy import GreedyPolicy

np.random.seed(123)

env = gym.make('CartPole-v0')
# env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2  # CartPole
# num_of_actions = 4  # LunarLander

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]
ll_x_min, ll_x_max = -1.2, 1.2
ll_y_min, ll_y_max = -0.3, 1.2
ll_xdot_min, ll_xdot_max = -2.4, 2.4
ll_ydot_min, ll_ydot_max = -2, 2
ll_theta_min, ll_theta_max = -6.28, 6.28
ll_thetadot_min, ll_thetadot_max = -8, 8
ll_cl_min, ll_cl_max = 0, 1
ll_cr_min, ll_cr_max = 0, 1


# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2222
initial_q = 0  # TODO Use initial Q value = 0 or = 50 depending on the task -- DONE

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

ll_x_grid = np.linspace(ll_x_min, ll_x_max, discr)
ll_y_grid = np.linspace(ll_y_min, ll_y_max, discr)
ll_xdot_grid = np.linspace(ll_xdot_min, ll_xdot_max, discr)
ll_ydot_grid = np.linspace(ll_ydot_min, ll_ydot_max, discr)
ll_theta_grid = np.linspace(ll_theta_min, ll_theta_max, discr)
ll_thetadot_grid = np.linspace(ll_thetadot_min, ll_thetadot_max, discr)
ll_cl_grid = np.linspace(ll_cl_min, ll_cl_max, 2)
ll_cr_grid = np.linspace(ll_cr_min, ll_cr_max, 2)

# q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, num_of_actions)) + initial_q  # LunarLander
q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q  # Cartpole

ep_lengths, epl_avg = [], []
total_rewards, total_rewards_avg = [], []

def discretize_state(x0):
    return (
        np.argmin(abs(x0[0]-x_grid)),
        np.argmin(abs(x0[1]-v_grid)),
        np.argmin(abs(x0[2]-th_grid)),
        np.argmin(abs(x0[3]-av_grid))
    )

def discretize_state_lunar_lander(x0):
    return (
        np.argmin(abs(x0[0]-ll_x_grid)),
        np.argmin(abs(x0[1]-ll_y_grid)),
        np.argmin(abs(x0[2]-ll_xdot_grid)),
        np.argmin(abs(x0[3]-ll_ydot_grid)),
        np.argmin(abs(x0[4]-ll_theta_grid)),
        np.argmin(abs(x0[5]-ll_thetadot_grid)),
        np.argmin(abs(x0[6]-ll_cl_grid)),
        np.argmin(abs(x0[7]-ll_cr_grid))
    )

def compute_value_function():
    return np.amax(q_grid, axis=4)  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID

def plot_value_function(value_function):
    sb.heatmap(np.mean(value_function, axis=(1, 3)))
    plt.show()

def plot_episode_stats():
    # Draw plots
    plt.plot(ep_lengths)
    plt.plot(epl_avg)
    plt.legend(["Episode length", "500 episode average"])
    plt.title("Episode lengths")
    plt.show()


# Value function before the training
print("Compute and plot value function")
values = compute_value_function()
plot_value_function(values)

# Training loop
for ep in range(episodes+test_episodes):
    test = ep >= episodes
    state, total_reward, done, steps = env.reset(), 0, False, 0
    epsilon = 0.2  # TODO: Use 0.2 or a/(a+k) or 0.0 depending on the task -- DONE
    while not done:
        policy = GLIEPolicy(q_grid, a, num_of_actions)  # TODO: USE GLIE POLICY -- DONE
        action = policy.get_action(discretize_state(state), ep)
        # policy = EGPolicy(q_grid, epsilon, num_of_actions)  # TODO: USE CONSTANT EPSILON POLICY -- DONE
        # policy = GreedyPolicy(q_grid, num_of_actions)  # TODO: USE GREEDY POLICY -- DONE
        # action = policy.get_action(discretize_state(state))
        new_state, reward, done, _ = env.step(action)
        if not test:
            greedy_policy = GreedyPolicy(q_grid, num_of_actions)
            new_action = greedy_policy.get_action(discretize_state(new_state))
            # if done:  # LunarLander
            if done and steps != 199:  # Cartpole
                td_target = reward - q_grid[discretize_state(state)][action]
                q_grid[discretize_state(state)][action] += alpha * td_target
            else:
                td_target = reward + gamma * q_grid[discretize_state(new_state)][new_action] - q_grid[discretize_state(state)][action]
                q_grid[discretize_state(state)][action] += alpha * td_target
            pass
        else:
            env.render()
        state = new_state
        steps += 1
        total_reward += reward
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    total_rewards.append(total_reward)
    total_rewards_avg.append(np.mean(total_rewards[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}, average reward {:.2f}"
              .format(ep, np.mean(ep_lengths[max(0, ep-200):]), np.mean(total_rewards[max(0, ep-200)])))
    if ep == 1 or ep == 10000:
        # Value function before the training
        print("Compute and plot value function")
        values = compute_value_function()
        plot_value_function(values)

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY -- DONE

# Calculate the value function
print("Compute and plot value function")
values = compute_value_function()  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY -- DONE

# Plots
# TODO: Plot the heatmap here using Seaborn or Matplotlib -- DONE
plot_value_function(values)
plot_episode_stats()

# Terminate script
sys.exit()


