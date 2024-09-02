'''
Deep Reinforcement Learning for mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com
'''

import tensorflow as tf
# Enable eager execution
tf.config.run_functions_eagerly(True)

# Import Ray Reinforcement Learning Library
from ray.rllib.algorithms.algorithm import Algorithm

# Import supporting libraries
import time
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Assuming the NeuralNetworksModel class is defined as provided
from MiniGreenhouse import MiniGreenhouse

# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:

ppo_model_from_checkpoint = Algorithm.from_checkpoint('trained-drl-models/model-calibrator-config-1')

# Make the calibratorModel instance
env = MiniGreenhouse({"flag_run": True,
                        "first_day_gl": 1,
                        "first_day_nn": 0,
                        "season_length_gl": 1/72,
                        "season_length_nn": 0,
                        "online_measurements": False,
                        "action_from_drl": True,
                        "flag_run_nn": True,
                        "flag_run_gl": True,
                        "flag_run_combined_models": True,
                        "max_steps": 3
                        })

# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
terminated = truncated = False
total_rewards = 0.0
total_rewards_list = [] # List to collect rewards

# Play one episode
while not terminated and not truncated:    
    # Make some delay, so it is easier to see
    time.sleep(1.0)
    
    # Compute a single action, given the current observation
    # from the environment.
    action = ppo_model_from_checkpoint.compute_single_action(obs)
    
    # Apply the computed action in the environment.
    obs, reward, terminated, _, info = env.step(action)

    # Print obs
    print(obs)

    # sum up rewards for reporting purposes
    total_rewards += reward
    total_rewards_list.append(total_rewards)  # Append the total reward to the list

# Report results.
print(f"Played 1 episode; total-reward={total_rewards}")

# Plot the cumulative rewards
plt.plot(total_rewards_list)
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward per Step')
plt.show()
    