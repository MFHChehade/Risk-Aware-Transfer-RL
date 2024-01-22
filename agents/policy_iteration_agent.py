import numpy as np
from typing import Union, List, Tuple
from pulp import LpVariable, LpProblem, LpMaximize
from policy import Policy  
from .value_based_agent import ValueBasedAgent

class PolicyIterationAgent(ValueBasedAgent):
    def __init__(self, env, gamma: float = 0.99, beta: float = 0, epsilon: float = 1e-12, mu0: np.ndarray = None, infinite_horizon: bool = False):
        """
        Initializes the class with environment parameters and initializes value, action-value, and policy arrays.

        Args:
        - env (Environment): The environment object.
        - gamma (float): Discount rate for future rewards.
        - beta (float): Entropic utility coefficient:
            - beta = 0: Risk-neutral behavior.
            - beta < 0: Indicates risk-aware behavior.
        - epsilon (float): Threshold for convergence in policy evaluation and iteration.
        - mu0 (np.ndarray): Initial state distribution.
        - infinite_horizon (bool): Whether the problem is infinite horizon.
        """

        super().__init__(env, gamma, infinite_horizon)
        self.beta = beta
        self.epsilon = epsilon

        # Warn about risk-seeking behavior when beta is positive
        if beta > 0:
            print("Note: With positive beta, the agent becomes risk-seeking rather than risk-aware.")


    def calculate_action_transition(self, state: int, input_action: int) -> float:
            """
            Calculates the transition for a specific action given a state.

            Args:
            - state (Any): Current state information.
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
                if self.beta == 0 :
                    weighted_sum += probabilities[action] * (reward + self.gamma * self.V[next_state_encoded])
                else:
                    weighted_sum += probabilities[action] * ( np.exp ( self.beta * (reward + self.gamma * self.V[next_state_encoded]) ) )
            if self.beta == 0:
                return weighted_sum
            else:
                return np.log(weighted_sum) / self.beta
                

    def calculate_transition(self, state: int) -> float:
        """
        Calculates the total transition for a state across all actions.

        Args:
        - state (Any): Current state information.

        Returns:
        - float: Total weighted sum representing transitions for all actions.
        """
        total_sum = 0
        
        for action in self.env.actions:
            total_sum += self.pi[state][action] * self.calculate_action_transition(state, action)
        
        return total_sum
    
    
    def greedy_action(self, state: int) -> Tuple[float, int]:
        """
        Finds the greedy action with the highest transition value for a given state.

        Args:
        - state (int): Current state information.

        Returns:
        - Tuple[float, int]: Tuple containing the maximum transition value and the corresponding action index.
        """
        transition_values = np.zeros(self.env.n_actions)

        # Calculate transition values for each action in the state
        for action in self.env.actions:
            transition_values[action] = self.calculate_action_transition(state, action)

        # Find the action with the highest transition value
        max_transition_value = np.max(transition_values)
        best_action_index = np.argmax(transition_values)

        return max_transition_value, best_action_index
    

    def policy_evaluation(self) -> None:
        """
        Evaluates the policy and updates the value array until convergence.

        Iteratively updates the value function until the maximum change (delta) in
        the value function across states is smaller than the convergence threshold (epsilon).
        """
        delta = self.epsilon + 1

        sweeps = 0

        while delta > self.epsilon:
            delta = 0

            for state in range(self.env.n_states):
                if not self.infinite_horizon and self.env.is_terminal_state(self.env.decode_state(state)):
                    # If not using infinite horizon and the state is terminal, its value is 0
                    self.V[state] = 0
                else:
                    prev_value = self.V[state]
                    self.V[state] = self.calculate_transition(state)

                    # Measure the maximum change in value function across states
                    delta = max(delta, abs(prev_value - self.V[state]))
            sweeps += 1
        # print(f"Policy evaluation converged after {sweeps} sweeps.")
        # Update Q-values after convergence
        self.update_q_values()
    

    def update_q_values(self) -> None:
        """
        Updates the Q-values based on the current value function and policy.
        """
        for state in range(self.env.n_states):
            for action in range(self.env.n_actions):
                if not self.infinite_horizon and self.env.is_terminal_state(self.env.decode_state(state)):
                    self.Q[state][action] = 0
                else:
                    self.Q[state][action] = self.calculate_action_transition(state, action)
    
    def train(self) -> None:
        """
        Performs policy iteration to find the optimal policy.
        """
        policy_stable = False
        sweeps = 0

        while not policy_stable:
            self.policy_evaluation()
            policy_stable = True
            sweeps += 1

            for state in range(self.env.n_states):
                # Find the greedy action for the current state
                _, best_action_index = self.greedy_action(state)

                # Update the policy if the greedy action is not the current policy
                if self.pi[state][best_action_index] != 1:
                    policy_stable = False
                    self.pi[state, :] = 0
                    self.pi[state][best_action_index] = 1
            
        self.pi.update_visitation_distribution()



    