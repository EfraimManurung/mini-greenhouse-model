'''
Deep Reinforcement Learning for mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.manurung@gmail.com

Read more information in the MiniGreenhouse class.
'''

# Import tensorflow 
import tensorflow as tf

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Import Ray Reinforcement Learning Library
from ray.rllib.algorithms.algorithm import Algorithm

# Import supporting libraries
import time

# Assuming the NeuralNetworksModel class is defined as provided
from MiniGreenhouse import MiniGreenhouse

# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:

ppo_model_from_checkpoint = Algorithm.from_checkpoint('trained-drl-models/model-calibrator-config-3-checkpoint')

# Make the calibratorModel instance
env = MiniGreenhouse({"flag_run": True,
                    "first_day_gl": 1,
                    "first_day_dnn": 0,
                    "season_length_gl": 1/72,
                    "season_length_dnn": 0,
                    "online_measurements": False,
                    "action_from_drl": True,
                    "flag_run_dnn": True,
                    "flag_run_gl": True,
                    "flag_run_combined_models": True,
                    "is_mature": False,
                    "max_steps": 5 * 72
                    })

# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
terminated = truncated = False
total_rewards = 0.0
total_rewards_list = [] # List to collect rewards

# Play episodes
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