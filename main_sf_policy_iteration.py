import numpy as np
from environments.grid_world import GridWorld 
from agents.sf_policy_iteration_agent import SFPolicyIterationAgent
from transfer.sf_transfer_mdp import SFTransfer_MDP

# Function to create reward matrices
def create_reward_matrix(n, x, y, g, scenario=None):
    rewards = x * np.ones((n, n))
    rewards[0, 9] = g

    if scenario == 2:
        for row in range(6):
            for column in range(4, 6):
                rewards[row, column] = -y

    elif scenario == 3:
        for row in range(3):
            for column in range(4, 6):
                rewards[row, column] = -y

    return rewards

# Function to create GridWorld environment
def create_grid_world(rewards):
    return GridWorld(height=10, terminal_states=[(9, 0)], rewards=rewards, initial_state=(0, 0), proba=prob)

# Function to create and train PolicyIterationAgent
def create_and_train_sf_policy_iteration_agent(grid_env, gamma=0.9):
    agent = SFPolicyIterationAgent(grid_env, gamma=gamma)
    agent.train()
    return agent

# Function to plot policies and value functions
def plot_policies_and_values(grid_env, agent, title = " "):
    grid_env.plot_policy(agent.pi, title = title)
    grid_env.plot_policy(agent.pi, np.transpose(agent.V.reshape(grid_env.height, grid_env.width)), title = title + " - Value Function")
    grid_env.plot_trajectory(agent.pi.deterministic_policy(), title = title)

# Define parameters
n = 10
x = x
y = y


g = 10

# Create reward matrices for different scenarios
rewards1 = create_reward_matrix(n, x, y, g)
rewards2 = create_reward_matrix(n, x, y, g, scenario=2)
rewards3 = create_reward_matrix(n, x, y, g, scenario=3)

# Create GridWorld environments
grid_env_1 = create_grid_world(rewards1)
grid_env_2 = create_grid_world(rewards2)
grid_env_3 = create_grid_world(rewards3)

# Create and train PolicyIterationAgent for each environment
sf_policy_iteration_agent1 = create_and_train_sf_policy_iteration_agent(grid_env_1)
sf_policy_iteration_agent2 = create_and_train_sf_policy_iteration_agent(grid_env_2)
sf_policy_iteration_agent3 = create_and_train_sf_policy_iteration_agent(grid_env_3)

# Plot policies and value functions for each MDP
plot_policies_and_values(grid_env_1, sf_policy_iteration_agent1)
plot_policies_and_values(grid_env_2, sf_policy_iteration_agent2)
plot_policies_and_values(grid_env_3, sf_policy_iteration_agent3)

# Create and train SFTransfer_MDP with different configurations
sf_transfer_mdp1 = SFTransfer_MDP(grid_env_3, [sf_policy_iteration_agent1.pi, sf_policy_iteration_agent2.pi],[sf_policy_iteration_agent1.psi, sf_policy_iteration_agent2.psi], gamma=0.9, beta=0, dual=True, c=0)
sf_transfer_mdp1.train()

# Plot policies and value functions for Transfer_MDP1
plot_policies_and_values(grid_env_3, sf_transfer_mdp1, title = f"c = {sf_transfer_mdp1.c}")

# Create and train another SFTransfer_MDP with different configurations
sf_transfer_mdp2 = SFTransfer_MDP(grid_env_3, [sf_policy_iteration_agent1.pi, sf_policy_iteration_agent2.pi], [sf_policy_iteration_agent1.psi, sf_policy_iteration_agent2.psi], gamma=0.9, beta=0, dual=True, c=0.01)
sf_transfer_mdp2.train()

# Plot policies and value functions for Transfer_MDP2
plot_policies_and_values(grid_env_3, sf_transfer_mdp2, title = f"c = {sf_transfer_mdp2.c}")

# Create and train another SFTransfer_MDP with different configurations
sf_transfer_mdp3 = SFTransfer_MDP(grid_env_3, [sf_policy_iteration_agent1.pi, sf_policy_iteration_agent2.pi], [sf_policy_iteration_agent1.psi, sf_policy_iteration_agent2.psi], gamma=0.9, beta=0, dual=True, c=0.1)
sf_transfer_mdp3.train()

# Plot policies and value functions for Transfer_MDP2
plot_policies_and_values(grid_env_3, sf_transfer_mdp3, title = f"c = {sf_transfer_mdp3.c}")
