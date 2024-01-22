from .action_selection_policy import ActionSelectionPolicy  # Import the base class
import numpy as np

class RandomSearch(ActionSelectionPolicy):
    """
    Random search action selection policy.
    """

    def __init__(self):
        """
        Initialize the Random Search policy.
        """
        super().__init__()

    def select_action(self, values):
        """
        Select an action based on the Random Search policy.

        Args:
        - values: Values associated with each action.

        Returns:
        - int: Selected action.
        """
        return np.random.choice(len(values))
