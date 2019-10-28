import numpy as np


class GreedyPolicy:

    def __init__(self, q_function, num_actions):
        self.q_function = q_function
        self.num_actions = num_actions

    def get_action(self, current_state):
        # Choose action according to a greedy fashion
        return np.argmax(self.q_function[current_state])
