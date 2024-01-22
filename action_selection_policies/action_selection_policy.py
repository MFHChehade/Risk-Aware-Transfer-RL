from abc import ABC, abstractmethod
import numpy as np

class ActionSelectionPolicy(ABC):
    """
    Abstract base class for action selection policies.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def select_action(self, values):
        """
        Abstract method to select an action based on a given set of values.

        Args:
        - values: Values associated with each action.

        Returns:
        - int: Selected action.
        """
        pass
