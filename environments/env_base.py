from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class Env(ABC):
    """
    Abstract base class defining the interface for reinforcement learning environments.
    """

    @abstractmethod
    def reset(self) -> int:
        """
        Reset the environment to its initial state.

        Returns:
        int: Encoded representation of the initial state.
        """
        raise NotImplementedError("Method 'reset' must be implemented in subclasses.")

    @abstractmethod
    def step(self, input_action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute one time step within the environment.

        Parameters:
        input_action (int): The action taken by the agent.

        Returns:
        Tuple[int, float, bool, dict]: Encoded next state, reward, flag indicating terminal state, and additional information.
        """
        raise NotImplementedError("Method 'step' must be implemented in subclasses.")

    @abstractmethod
    def render(self):
        """
        Render the current state of the environment.
        """
        raise NotImplementedError("Method 'render' must be implemented in subclasses.")

    @abstractmethod
    def is_terminal_state(self, state: Tuple[int, int]) -> bool:
        """
        Check if a given state is a terminal state.

        Parameters:
        state (Tuple[int, int]): State coordinates.

        Returns:
        bool: True if the state is terminal, False otherwise.
        """
        raise NotImplementedError("Method 'is_terminal_state' must be implemented in subclasses.")

    @abstractmethod
    def get_reward(self, state, action, next_state) -> float:
        """
        Get the reward for transitioning from a state to another based on an action.

        Parameters:
        state: Current state.
        action: Action taken.
        next_state: Next state.

        Returns:
        float: Reward value.
        """
        raise NotImplementedError("Method 'get_reward' must be implemented in subclasses.")
    
    @abstractmethod
    def get_weights(self) -> List[float]:
        """
        Abstract method to get weights for the task.
        Returns a list of weights.
        """
        pass

    @abstractmethod
    def get_feature_vector(self, state, action, next_state) -> np.ndarray:
        """
        Abstract method to get the feature vector for a given state-action-state transition.
        Returns a NumPy array representing the feature vector.
        """
        pass

    @abstractmethod
    def encode_state(self, state: Tuple[int, int]) -> int:
        """
        Encode a state represented by coordinates into a single integer.

        Parameters:
        state (Tuple[int, int]): State coordinates.

        Returns:
        int: Encoded state.
        """
        raise NotImplementedError("Method 'encode_state' must be implemented in subclasses.")

    @abstractmethod
    def decode_state(self, encoded_state: int) -> Tuple[int, int]:
        """
        Decode an encoded state into its original coordinates.

        Parameters:
        encoded_state (int): Encoded state.

        Returns:
        Tuple[int, int]: Decoded state coordinates.
        """
        raise NotImplementedError("Method 'decode_state' must be implemented in subclasses.")

    @abstractmethod
    def close(self):
        """
        Close the environment.
        """
        raise NotImplementedError("Method 'close' must be implemented in subclasses.")

    @abstractmethod
    def __str__(self):
        """
        Provide a string representation of the Env object.
        """
        raise NotImplementedError("Method '__str__' must be implemented in subclasses.")
