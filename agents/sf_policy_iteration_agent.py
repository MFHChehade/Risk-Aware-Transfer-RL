from .policy_iteration_agent import PolicyIterationAgent
import numpy as np
from typing import Tuple

class SFPolicyIterationAgent(PolicyIterationAgent):
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        beta: float = 0,
        epsilon: float = 1e-12,
        mu0: np.ndarray = None,
        infinite_horizon: bool = False
    ):
        """
        Initializes the SFPolicyIterationAgent class.

        Args:
            - env (Environment): The environment object.
            - gamma (float): Discount rate for future rewards.
            - beta (float): Entropic utility coefficient:
                - beta = 0: Risk-neutral behavior.
                - beta < 0: Indicates risk-aware behavior.
            - epsilon (float): Threshold for convergence in policy evaluation and iteration.
            - mu0 (np.ndarray): Initial state distribution.
            - infinite_horizon (bool): Whether the problem is infinite horizon.

        Raises:
            AssertionError: If beta is not 0 for SF Policy Iteration.
        """
        assert beta == 0, "Entropy cannot be used with SF Policy Iteration, please set beta to 0."

        super().__init__(env, gamma, beta, epsilon, mu0, infinite_horizon)

        # Initializing the additional parameter for SF Policy Iteration
        self.psi = np.zeros((self.env.n_states, self.env.n_actions, self.env.n_weights))

    def greedy_action(self, state: int) -> Tuple[float, int]:
        """
        Get the best action for a given state based on the current policy.

        Args:
            - state (int): The current state.

        Returns:
            - Tuple[float, int]: Value and action pair.
        """
        return np.max(self.Q[state]), np.argmax(self.Q[state])

    def calculate_action_transition(self, state: int, input_action: int) -> float:
        """
        Calculates the transition for a specific action given a state.

        Args:
            - state (int): Current state information.
            - input_action (int): The action to evaluate.

        Returns:
            - float: Weighted sum representing the transition for the given action.
        """
        probabilities = self.env.set_action_probabilities(input_action)
        weighted_sum = 0
        self.env.state = self.env.decode_state(state)
        for action in self.env.actions:
            next_state = self.env.move_agent(action)
            reward = self.env.get_reward(state, action, next_state)
            next_state_encoded = self.env.encode_state(next_state)
            phi = self.env.get_feature_vector(state, action, next_state)
            next_action = np.argmax(self.pi[next_state_encoded])
            weighted_sum += probabilities[action] * (phi + self.gamma * self.psi[next_state_encoded, next_action])
        return weighted_sum

    def policy_evaluation(self) -> None:
        """
        Evaluates the policy and updates the value array until convergence.
        """
        delta = self.epsilon + 1

        sweeps = 0

        while delta > self.epsilon:
            delta = 0

            for state in range(self.env.n_states):
                for action in range(self.env.n_actions):
                    if not self.infinite_horizon and self.env.is_terminal_state(self.env.decode_state(state)):
                        self.psi[state, action] = np.zeros(self.env.n_weights)
                    else:
                        prev_value = self.psi[state, action]
                        self.psi[state, action] = self.calculate_action_transition(state, action)
                        delta = max(delta, max(abs(prev_value - self.psi[state, action])))
                sweeps += 1
        self.update_q_values() 
        self.update_V_values()

    def update_q_values(self):
        """
        Update the Q-values based on the SFQ-learning approach.
        """
        self.Q = self.psi @ self.env.get_weights()

    def update_V_values(self):
        """
        Update the V-values based on the SFQ-learning approach.
        """
        for state in range(self.env.n_states):
            self.V[state] = self.Q[state, np.argmax(self.pi[state])]
