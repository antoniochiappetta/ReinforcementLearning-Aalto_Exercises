import numpy as np


class EGPolicy:

    def __init__(self, q_function, eps, num_actions):
        self.q_function = q_function
        self.eps = eps
        self.num_actions = num_actions

    def get_action(self, current_state):
        # Calculate probabilities of actions from current state
        probabilities = np.ones(self.num_actions, dtype=float) * self.eps / self.num_actions
        best_action = np.argmax(self.q_function[current_state])
        # Assign high probability to the best action
        probabilities[best_action] += 1.0 - self.eps
        # Choose action according to an epsilon-greedy fashion
        return np.random.choice(np.arange(len(probabilities)), p=probabilities)
