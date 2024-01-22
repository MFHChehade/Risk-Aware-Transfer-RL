import numpy as np
from typing import Union, List, Tuple
from pulp import LpVariable, LpProblem, LpMaximize
import copy

class Policy(np.ndarray):
    def __new__(cls, env, gamma=0.99, mu0=None):
        """
        Initializes a Policy object as a NumPy array subclass.

        Args:
        - env: The environment object.
        - gamma (float, optional): The discount factor. Defaults to 0.99.
        - mu0: Initial state distribution. Defaults to None.

        Returns:
        - Policy: An instance of the Policy class.
        """

        # Creates a new instance as a NumPy array representing the policy
        obj = np.asarray((1 / env.n_actions) * np.ones(
            (env.n_states, env.n_actions), dtype=np.float64
        )).view(cls)

        return obj


    def __init__(self, env, gamma=0.99, mu0=None):
        """
        Initializes the Policy object.

        Args:
        - env: The environment object.
        - gamma (float): The discount factor.
        - mu0: Initial state distribution.

        Additional setup:
        - Initializes d values for state-action pairs.
        - Sets the default initial state distribution if not provided.
        """
        # Sets attributes for the Policy object
        self.env = env  # Holds the environment object
        self.gamma = gamma  # Stores the discount factor
        self.d = np.zeros((env.n_states, env.n_actions))  # Initializes d values
        
        # Sets default value for mu0 if not provided
        if mu0 is None:
            mu0 = (
                np.zeros(env.n_states) if env.initial_state is None else
                np.eye(env.n_states)[env.encode_state(env.initial_state)]
            ) if env.initial_state is not None else (
                (1 / env.n_states) * np.ones(env.n_states)
            )
        
        self.mu0 = mu0  # Stores the initial state distribution
        self.transition_matrix = self.calculate_transition_matrix()  # Stores the transition matrix

    
    def calculate_transition_matrix(self) -> List[np.ndarray]:
        """
        Creates a list of transition matrices for each action.
        """

        transition_matrix = []

        for input_action in range(self.env.n_actions):

            P_a = np.zeros((self.env.n_states, self.env.n_states))
            probabilities = self.env.set_action_probabilities(input_action)

            for state in range(self.env.n_states):
                for action in range(self.env.n_actions):
                    self.env.state = self.env.decode_state(state)
                    next_state = self.env.move_agent(action)
                    next_state_encoded = self.env.encode_state(next_state)
                    P_a[state][next_state_encoded] += probabilities[action]

            transition_matrix.append(P_a)

        return transition_matrix
    
    def expected_reward(self, state, action) -> float:
        """
        Calculates the expected reward for a given state and action.

        Args:
        - state (int): Current state information.
        - action (int): Action to evaluate.

        Returns:
        - float: Expected reward for the given state and action.
        """
        reward = 0

        for next_state in range(self.env.n_states):
            reward +=self.transition_matrix[action][state][next_state] * self.env.get_reward(self.env.decode_state(state), action, self.env.decode_state(next_state))
        return reward
    
    def expected_reward_squared(self, state, action) -> float:
        """
        Calculates the expected squared reward for a given state and action.
        
        Args:
        - state (int): Current state information.
        - action (int): Action to evaluate.

        Returns:
        - float: Expected squared reward for the given state and action.
        """

        reward = 0

        for next_state in range(self.env.n_states):
            reward += self.transition_matrix[action][state][next_state] * self.env.get_reward(self.env.decode_state(state), action, self.env.decode_state(next_state))**2

        return reward

    
    def update_visitation_distribution(self) -> np.ndarray:
        """
        Calculates the visitation distribution for each state.

        Returns:
        - np.ndarray: Visitation distribution for each state.
        """
        
        d = [[LpVariable(f"d_{state}_{action}", lowBound=0) for action in range(self.env.n_actions)] for state in range(self.env.n_states)]
        objective = 0 
        prob = LpProblem("Dual_Q_Problem", LpMaximize)

        for state in range(self.env.n_states):
            for action in range(self.env.n_actions):
                objective += d[state][action] * self.expected_reward(state, action)   
            
        prob += objective

        for state in range(self.env.n_states):
            for action in range(self.env.n_actions):
                P_star_pi = 0
                for previous_state in range(self.env.n_states):
                    for previous_action in range(self.env.n_actions):
                        P_star_pi += self[state][action] * self.transition_matrix[previous_action][previous_state][state] * d[previous_state][previous_action]
                prob += d[state][action] == (1 - self.gamma) * self.mu0[state] * self[state][action] + self.gamma * P_star_pi
        
        # Solve the problem
        prob.solve()

        # Check if the problem is feasible
        if prob.status != 1:
            print("The problem is infeasible or did not converge.")

        d_soln = np.zeros((self.env.n_states, self.env.n_actions))
        for state in range(self.env.n_states):
            for action in range(self.env.n_actions):
                soln = d[state][action].varValue
                d_soln[state, action] = soln

        self.d = d_soln

    def variance_expression(self) -> float:
        """
        Calculates the variance expression for the dual problem.

        Returns:
        - float: Variance expression for the dual problem.
        """
        term1, term2 = 0, 0

        for state in range(self.env.n_states):
            for action in range(self.env.n_actions):
                term1 += self.d[state][action] * self.expected_reward(state, action)
                term2 += self.d[state][action] * self.expected_reward_squared(state, action)
        variance = term2 - term1**2
        return variance
    
    def deterministic_policy(self) -> List[int]:
        """
        Returns the deterministic policy based on the optimal policy.

        Returns:
        - List[int]: List of actions for each state.
        """
        policy = np.argmax(self, axis=1)
        return policy

