from agents.policy_iteration_agent import PolicyIterationAgent
import numpy as np
from typing import Union, List, Tuple
from policy import Policy
import copy  

class Transfer_MDP(PolicyIterationAgent):
    def __init__(self, env, policies=None, gamma=0.99, beta=0, epsilon=1e-12, dual = True, c = 0.1,  mu0=None, infinite_horizon=False):
        """
        Initialize Transfer_M0DP class inheriting from MDP
        
        Arguments:
        - env: the environment for which the Transfer_MDP is defined
        - policies: a list of policies used in the Transfer_MDP (default is None, meaning no transfer)
        - gamma: discount factor for future rewards (default is 0.99)
        - beta: weighting factor for regularization (default is 0)
        - epsilon: small value for convergence threshold (default is 1e-12)
        - dual: flag indicating if the dual RL is used in the GPI (default is True)
        - c: regularization parameter (default is 0.1)
        - mu0: initial distribution over states (default is None)
        - infinite_horizon: flag indicating if the problem is infinite horizon (default is False)
        """

        # Warn about risk-seeking behavior when beta is positive
        if beta > 0:
            print("Note: With positive beta, the agent becomes risk-seeking rather than risk-aware.")
        
        assert policies is None or (isinstance(policies, list) and all(isinstance(policy, Policy) for policy in policies)), "Policies should be a list of Policy instances or None"
        self.policies = policies if policies is not None else []

        # Ensure beta and c are not nonzero simultaneously
        assert not(beta != 0 and c != 0), "Cannot have both beta and c nonzero at the same time"


        super().__init__(env, gamma, beta, epsilon, mu0, infinite_horizon)
        self.policies = policies # list of policies
        self.Q_list = np.zeros((len(self.policies), self.env.n_states, self.env.n_actions), dtype=np.float64)
        self.dual = dual
        self.variances = np.zeros(len(self.policies))
        self.c = c 


    def policy_evaluation_transfer(self) -> None:
        """
        Evaluate the policies in the new environment
        """
        for i, policy in enumerate(self.policies):
            self.pi = policy
            self.policy_evaluation()
            self.Q_list[i] = self.Q

            if self.dual:
                self.pi.update_visitation_distribution()
                self.variances[i] = self.pi.variance_expression()
        self.pi = Policy(self.env, self.gamma) # reset the policy to the original one

    def train(self) -> None:
        """
        Perform generalized policy improvement for the new environment
        """
        if self.policies is None or len(self.policies) == 0:
            print("No transfer policies provided")
            super().train()
            return

        self.policy_evaluation_transfer()
        for state in range(self.env.n_states):
            # Initialize the state-action distribution
            self.pi[state, :] = 0

            # Choose action based on the Q values and variances (if dual)
            if self.dual:
                action = np.argmax(np.max(self.Q_list[:, state, :] - self.c * self.variances[:, np.newaxis], axis=0))
            else:
                action = np.argmax(np.max(self.Q_list[:, state, :], axis=0))

            # Update the state-action distribution for the current state
            self.pi[state, action] = 1

        # Evaluate policy and update visitation distribution
        self.policy_evaluation()
        self.pi.update_visitation_distribution()
        

        
        
