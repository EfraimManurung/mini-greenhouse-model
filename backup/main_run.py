'''
Deep Reinforcement Learning for mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com
'''

# Import Ray Reinforcement Learning Library
from ray.rllib.algorithms.algorithm import Algorithm

# Import supporting libraries
import time
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Import libraries needed for PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from envs.MiniGreenhouse import MiniGreenhouse

# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:

my_new_ppo = Algorithm.from_checkpoint('physics-model/model-minigreenhouse-config-3')

# Call the MiniGreenhouse instance
env = MiniGreenhouse({"flag_run": True,
                        "first_day": 1,
                        "season_length": 1/72, 
                        "online_measurements": False,
                        "max_steps": 12
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
    action = my_new_ppo.compute_single_action(obs)
    
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