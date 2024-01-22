from abc import ABC, abstractmethod
from typing import List
import numpy as np
from policy import Policy

class ValueBasedAgent(ABC):
    def __init__(self, env, gamma=0.9, infinite_horizon=False):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))
        self.V = np.zeros(self.env.n_states)
        self.infinite_horizon = infinite_horizon
        self.pi = Policy(self.env, self.gamma)

    @abstractmethod
    def train(self, n_episodes=1000, max_steps=1000, verbose=False) -> List[float]:
        """
        Train the agent.

        Parameters:
        - n_episodes (int): The number of episodes (default: 1000).
        - max_steps (int): The maximum number of steps per episode (default: 1000).
        - verbose (bool): If True, displays a progress bar; if False, no progress bar is shown (default: False).

        Returns:
        - List[float]: Reward per step.
        """
        pass

    def test(self, n_episodes=1000, max_steps=1000) -> List[float]:
        """
        Test the trained agent.

        Parameters:
        - n_episodes (int): The number of episodes (default: 1000).
        - max_steps (int): The maximum number of steps per episode (default: 1000).

        Returns:
        - List[float]: Rewards per episode.
        """
        self.env.reset()
        rewards = []
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            for _ in range(max_steps):
                action = np.argmax(self.Q[state])
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done and self.infinite_horizon == False:
                    break
            rewards.append(episode_reward)
        return rewards

    def reset(self):
        """
        Reset the Q-values, value function, and policy.
        """
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))
        self.V = np.zeros(self.env.n_states)
        self.pi = Policy(self.env, self.gamma)
