from .transfer_mdp import Transfer_MDP
from policy import Policy

class SFTransfer_MDP(Transfer_MDP):

    def __init__(self, env, policies=None, psi_list = None, gamma=0.99, beta=0, epsilon=1e-12, dual=True, c=0.1, mu0=None, infinite_horizon=False):
        """
        Initializes the SFTransfer_MDP class inheriting from Transfer_MDP.

        Args:
            - env: The environment for which the SFTransfer_MDP is defined.
            - policies: A list of policies used in the SFTransfer_MDP (default is None, meaning no transfer).
            - gamma: Discount factor for future rewards (default is 0.99).
            - beta: Weighting factor for regularization (default is 0).
            - epsilon: Small value for convergence threshold (default is 1e-12).
            - dual: Flag indicating if the dual RL is used in the GPI (default is True).
            - c: Regularization parameter (default is 0.1).
            - mu0: Initial distribution over states (default is None).
            - infinite_horizon: Flag indicating if the problem is infinite horizon (default is False).
        """
        super().__init__(env, policies, gamma, beta, epsilon, dual, c, mu0, infinite_horizon)
        self.psi_list = psi_list if psi_list is not None else []

        # assert psi_list and policies must be of same size
        assert len(self.policies) == len(self.psi_list), "psi_list and policies must be of same size"
        


    def policy_evaluation_transfer(self) -> None:
        """
        Evaluate the policies in the new environment
        """
        for i, policy in enumerate(self.policies):
            self.pi = policy
            self.policy_evaluation()
            self.Q_list[i] = self.psi_list[i] @ self.env.get_weights()

            if self.dual:
                self.pi.update_visitation_distribution()
                self.variances[i] = self.pi.variance_expression()
        self.pi = Policy(self.env, self.gamma) # reset the policy to the original one
