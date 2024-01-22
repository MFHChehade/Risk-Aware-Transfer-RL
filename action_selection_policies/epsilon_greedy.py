from .action_selection_policy import ActionSelectionPolicy  # Import the base class
import numpy as np

class EpsilonGreedy(ActionSelectionPolicy):
    """
    Epsilon-Greedy action selection policy.
    """

    def __init__(self, epsilon=0.1):
        """
        Initialize the Epsilon-Greedy policy.

        Args:
        - epsilon (float): The probability of choosing a random action (default: 0.1).
        """
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, values):
        """
        Select an action based on the Epsilon-Greedy policy.

        Args:
        - values: Values associated with each action.

        Returns:
        - int: Selected action.
        """
        if np.random.uniform() < self.epsilon:
            return np.random.choice(len(values))
        else:
            return np.random.choice(np.flatnonzero(values == values.max()))
