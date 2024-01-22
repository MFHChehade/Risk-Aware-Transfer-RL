import numpy as np
from .epsilon_greedy import EpsilonGreedy

class AdaptiveEpsilonGreedy(EpsilonGreedy):
    def __init__(self, initial_epsilon=1.0, decay_rate=0.99999):
        """
        Initializes the AdaptiveEpsilonGreedy object.

        Args:
        - initial_epsilon (float): The initial epsilon value (default: 1.0).
        - decay_rate (float): The rate at which epsilon decays (default: 0.99999).
        """
        super().__init__(initial_epsilon)
        self.decay_rate = decay_rate

    def update_epsilon(self):
        """
        Updates the epsilon value based on the decay rate.
        If the epsilon value is greater than 0.01, it decays by the specified decay rate.
        """
        if self.epsilon > 0.01:
            self.epsilon *= self.decay_rate  # Decay the epsilon value if it's above 0.01

    def select_action(self, values):
        """
        Selects an action based on the epsilon-greedy policy after updating epsilon.

        Args:
        - values (array-like): The values used for action selection.

        Returns:
        - int: The index of the selected action.
        """
        self.update_epsilon()  # Update epsilon before action selection
        return super().select_action(values)
