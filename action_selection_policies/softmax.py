from .action_selection_policy import ActionSelectionPolicy  # Import the base class
import numpy as np

class Softmax(ActionSelectionPolicy):
    """
    Softmax action selection policy.
    """

    def __init__(self, temperature=1.0):
        """
        Initialize the Softmax policy.

        Args:
        - temperature (float): Controls the degree of exploration (default: 1.0).
        """
        super().__init__()
        self.temperature = temperature

    def select_action(self, values):
        """
        Selects an action using softmax probabilities computed with the given temperature.

        Args:
        - values (numpy.ndarray): The action values.

        Returns:
        - numpy.int64: The selected action index or None if an exception occurs.
        """
        with np.errstate(all='raise'):
            try:
                exp_values = np.exp(values / self.temperature)
                probabilities = exp_values / np.sum(exp_values)
                return np.random.choice(len(values), p=probabilities)
            except Exception as e:
                return np.argmax(values)
