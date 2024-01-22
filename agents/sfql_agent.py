import numpy as np
from .ql_agent import QLearningAgent

class SFQLearningAgent(QLearningAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, infinite_horizon=False, action_selection_policy=None, action_selection_policy_params=None):
        """
        Q-learning agent that extends QLearningAgent.

        Parameters:
        - env: The environment.
        - alpha (float): The learning rate (default: 0.1).
        - gamma (float): The discount factor (default: 0.9).
        - infinite_horizon (bool): Whether the problem is infinite horizon (default: False).
        - action_selection_policy (str): The action selection policy (default: None).
        - action_selection_policy_params (dict): Parameters of the action selection policy (default: None).
        """
        # Initializing the SFQ-learning agent by calling the parent class constructor
        super().__init__(env, alpha, gamma, infinite_horizon, action_selection_policy, action_selection_policy_params)
        
        # Initializing the additional parameter for SFQ-learning
        self.psi = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_weights))

    def update_q_values(self):
        """
        Update the Q-values based on the SFQ-learning approach.
        """
        # Computing Q-values using the psi matrix and environment weights
        self.Q = self.psi @ self.env.get_weights()
    
    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-values based on the received experience.

        Parameters:
        - state: The current state.
        - action: The chosen action.
        - reward: The reward received.
        - next_state: The next state.
        - done: Indicates if the episode is finished.
        """
        # Calculating the reward function indicator vector
        phi = self.env.get_feature_vector(self.env.decode_state(state), action, self.env.decode_state(next_state))
        
        if not self.infinite_horizon and done:
            # Update psi for non-infinite horizon scenarios at termination
            self.psi[state, action] += self.alpha * (phi - self.psi[state, action])
        else:
            max_action = np.argmax(self.Q[next_state])
            self.psi[state, action] += self.alpha * (phi + self.gamma * self.psi[next_state, max_action] - self.psi[state, action])
        
        # Update Q-values based on updated psi
        self.update_q_values()
