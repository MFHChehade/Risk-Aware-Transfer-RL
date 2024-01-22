from .softmax import Softmax

class AdaptiveSoftmax(Softmax):
    def __init__(self, initial_temp=1.0, decay_rate=0.99999):
        """
        Initializes the AdaptiveSoftmax object.

        Args:
        - initial_temp (float): The initial temperature value (default: 1.0).
        - decay_rate (float): The rate at which temperature decays (default: 0.99999).
        """
        super().__init__(initial_temp)
        self.decay_rate = decay_rate

    def update_temperature(self):
        """
        Updates the temperature value based on the decay rate.
        If the temperature value is greater than 0.01, it decays by the specified decay rate.
        """
        if self.temperature > 0.01:
            self.temperature *= self.decay_rate  # Decay the temperature value if it's above 0.01

    def select_action(self, values):
        """
        Selects an action using the softmax policy with the updated temperature.

        Args:
        - values (numpy.ndarray): The action values.

        Returns:
        - int: The selected action index.
        """
        self.update_temperature()  # Update temperature before action selection
        return super().select_action(values)  # Use Softmax's select_action method after temperature update
