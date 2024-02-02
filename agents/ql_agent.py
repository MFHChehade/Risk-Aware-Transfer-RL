import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from action_selection_policies.random_search import RandomSearch
from action_selection_policies.softmax import Softmax
from action_selection_policies.epsilon_greedy import EpsilonGreedy
from action_selection_policies.greedy_search import GreedySearch
from action_selection_policies.adaptive_epsilon_greedy import AdaptiveEpsilonGreedy as AdaptiveEpsilonGreedy
from action_selection_policies.adaptive_softmax import AdaptiveSoftmax as AdaptiveSoftmax
from policy import Policy
from .value_based_agent import ValueBasedAgent

class QLearningAgent(ValueBasedAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, infinite_horizon=False, action_selection_policy=None, action_selection_policy_params=None):
        """
        Q-learning algorithm.

        Parameters:
        - env: The environment.
        - alpha (float): The learning rate (default: 0.1).
        - gamma (float): The discount factor (default: 0.9).
        - infinite_horizon (bool): Whether the problem is infinite horizon (default: False).
        - action_selection_policy (str): The action selection policy (default: None).
        - action_selection_policy_params (dict): The parameters of the action selection policy (default: None).
        """
        super().__init__(env, gamma, infinite_horizon)
        self.alpha = alpha
        self.action_selection_policy = self._select_action_selection_policy(action_selection_policy, action_selection_policy_params)

    def _select_action_selection_policy(self, policy, params):

        valid_policies = ["random_search", "softmax", "epsilon_greedy", "greedy_search", "adaptive_epsilon_greedy", "adaptive_softmax"]
    
        if policy not in valid_policies:
            raise ValueError(f"Invalid policy '{policy}'. Choose one of: {valid_policies}")
        
        if policy == "random_search":
            return RandomSearch()
        elif policy == "softmax":
            return Softmax(temperature=params["temperature"]) if params else Softmax()
        elif policy == "epsilon_greedy":
            return EpsilonGreedy(epsilon=params["epsilon"]) if params else EpsilonGreedy()
        elif policy == "greedy_search":
            return GreedySearch()
        elif policy == "adaptive_epsilon_greedy":
            return AdaptiveEpsilonGreedy(
                initial_epsilon=params["initial_epsilon"],
                decay_rate=params["decay_rate"]
            ) if params else AdaptiveEpsilonGreedy()
        elif policy == "adaptive_softmax":
            return AdaptiveSoftmax(
                initial_temp=params["initial_temp"],
                decay_rate=params["decay_rate"]
            ) if params else AdaptiveSoftmax()
        else:
            return EpsilonGreedy()  # Default to EpsilonGreedy


    # Method to choose an action based on epsilon-greedy policy
    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.

        Parameters:
        - state: The current state.

        Returns:
        - action: The chosen action.
        """

        return self.action_selection_policy.select_action(self.Q[state])
    
    # # Method to update Q-values based on experience
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-values.

        Parameters:
        - state: The current state.
        - action: The chosen action.
        - reward: The reward received.
        - next_state: The next state.
        """
        if not self.infinite_horizon and done:
            self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
        else:
            self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    # def update(self, state, action, reward, next_state, done, episode):
    #     """
    #     Update the Q-values.

    #     Parameters:
    #     - state: The current state.
    #     - action: The chosen action.
    #     - reward: The reward received.
    #     - next_state: The next state.
    #     - done: Whether the episode is done.
    #     - episode: The current episode number.
    #     """
    #     current_alpha = 1.0 / (episode + 1)  # Adding 1 to avoid division by zero for the first episode

    #     if not self.infinite_horizon and done:
    #         self.Q[state, action] += current_alpha * (reward - self.Q[state, action])
    #     else:
    #         self.Q[state, action] += current_alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
    
    # Method to train the Q-learning agent
    def train(self, n_episodes=1000, max_steps=1000, verbose=True):
        """
        Train the agent.

        Parameters:
        - n_episodes (int): The number of episodes (default: 1000).
        - max_steps (int): The maximum number of steps per episode (default: 1000).
        - verbose (bool): If True, displays a progress bar; if False, no progress bar is shown (default: True).
        """
        reward_per_episode = []
        steps_per_episode = []
        episodes = range(n_episodes)
        self.reset()

        if verbose:
            tqdm.write(f"Training for {n_episodes} episodes...")
            episodes = tqdm(episodes, desc='Episodes', unit=' episodes')

        for episode in episodes:
            state = self.env.reset()
            total_reward = 0
            steps = 0 
            while True:
                steps +=1
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                # self.update(state, action, reward, next_state, done, episode)
                state = next_state
                total_reward += reward
                if done and self.infinite_horizon == False:
                    break
                if steps % max_steps == 0:
                    print(steps)
            # print(steps, total_reward)
            # self.env.render()
            reward_per_episode.append(total_reward)
            steps_per_episode.append(steps)
        
        self.update_policy()
        self.set_value_function()

        return [reward / steps for reward, steps in zip(reward_per_episode, steps_per_episode)]

 
    # Method to plot the moving average of rewards per episode
    def plot_moving_average_rewards(self, rewards_per_episode, window_size=200):
        """
        Plot the moving average of rewards per episode.

        Parameters:
        - rewards_per_episode (list): The rewards per episode.
        - window_size (int): The size of the moving average window (default: 200).
        """

        # Calculate moving average
        moving_average = pd.Series(rewards_per_episode).rolling(window=window_size, min_periods=1).mean()

        # Plot the moving average
        plt.figure(figsize=(8, 6))
        plt.plot(moving_average)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per step')
        plt.grid(True)
        plt.show()

    def update_policy(self):
        """
        Updates the policy.
        """
        for state in range(self.env.n_states):
            max_actions = np.where(self.Q[state] == np.max(self.Q[state]))[0]
            optimal_action = np.random.choice(max_actions)
            self.pi[state] = np.zeros(self.env.n_actions)
            self.pi[state][optimal_action] = 1

    
    def set_value_function(self):
        """
        Get the value function for an optimal policy.

        Returns:
        - value_function (list): The value function.
        """
        for state in range(self.env.n_states):
            self.V = np.max(self.Q, axis=1)
    
    def get_q_values(self):
        """
        Get the Q-values.

        Returns:
        - Q (list): The Q-values.
        """
        return self.Q
