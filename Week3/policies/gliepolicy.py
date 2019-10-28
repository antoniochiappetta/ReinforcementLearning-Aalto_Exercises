import numpy as np


class GLIEPolicy:

    def __init__(self, q_function, a, num_actions):
        self.q_function = q_function
        self.a = a
        self.num_actions = num_actions

    def get_action(self, current_state, episode):
        # Calculate epsilon based on current episode
        epsilon = self.a / (self.a + episode)
        # Calculate probabilities of actions from current state
        probabilities = np.ones(self.num_actions, dtype=float) * epsilon / self.num_actions
        best_action = np.argmax(self.q_function[current_state])
        # Assign high probability to the best action
        probabilities[best_action] += 1.0 - epsilon
        # Choose action according to an epsilon-greedy fashion
        return np.random.choice(np.arange(len(probabilities)), p=probabilities)
