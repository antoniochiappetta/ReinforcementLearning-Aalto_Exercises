import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = torch.nn.Parameter(torch.tensor([1000.]))  # torch.tensor([5.])  # TODO: Implement accordingly (T1, T2) -- DONE T1
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, variance):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = variance  # TODO: Is it a good idea to leave it like this? -- DONE

        # TODO: Instantiate and return a normal distribution -- DONE
        # with mean mu and std of sigma (T1)
        return Normal(mu, torch.sqrt(sigma))

        # TODO: Add a layer for state value calculation (T3)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.baseline = 20
        self.variance = self.policy.sigma
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Update policy variance (T2) -- DONE
        c = 5e-4
        # self.variance = self.policy.sigma * np.exp(-c * episode_number)  # Exponentially decaying variance

        # TODO: Compute discounted rewards (use the discount_rewards function) -- DONE
        rewards = discount_rewards(rewards, gamma=self.gamma)
        rewards = (rewards - torch.mean(rewards))/torch.std(rewards)  # REINFORCE with normalized rewards

        # TODO: Compute critic loss and advantages (T3)

        # TODO: Compute the optimization term (T1, T3) -- DONE
        loss = torch.sum(-rewards * action_probs)  # REINFORCE
        # loss = torch.sum(-(rewards - self.baseline) * action_probs)  # REINFORCE with baseline

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1) -- DONE
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1) -- DONE
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1) -- DONE
        actions_distribution = self.policy.forward(x, self.variance)

        # TODO: Return mean if evaluation, else sample from the distribution returned by the policy (T1) -- DONE
        if evaluation:
            action = actions_distribution.mean
        else:
            action = actions_distribution.sample((1,))[0]

        # TODO: Calculate the log probability of the action (T1) -- DONE
        act_log_prob = actions_distribution.log_prob(action)

        # TODO: Return state value prediction, and/or save it somewhere (T3)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

