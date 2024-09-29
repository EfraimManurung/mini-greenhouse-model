'''
Training Deep Reinforcement Learning (DRL) for a mini-greenhouse environment

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.manurung@gmail.com

Read more information in the MiniGreenhouse class.
'''

# Import tensorflow 
import tensorflow as tf

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Import RLlib algorithms
from ray.rllib.algorithms.ppo import PPOConfig

# Import the custom environment
from MiniGreenhouse import MiniGreenhouse

# Import support libraries
from tqdm import tqdm
import os

# Configure the RLlib PPO algorithm
config = PPOConfig()

# Algorithm config to run the env_runners
config.env_runners(num_env_runners=1)
config.env_runners(num_cpus_per_env_runner=1)
config.environment(
    env=MiniGreenhouse,
        env_config={"flag_run": False,
                    "first_day_gl": 1,
                    "first_day_dnn": 0,
                    "season_length_gl": 1/72,
                    "season_length_dnn": 0,
                    "online_measurements": False,
                    "action_from_drl": True,
                    "flag_run_dnn": False,
                    "flag_run_gl": True,
                    "flag_run_combined_models": False,
                    "is_mature": False,
                    "max_steps": 3  # 1 step equal to 20 minutes in real-time, thus, 3 steps = 60 minutes
                    })

# Config for training 
config.training(
        gamma=0.9,  # Discount factor
        lr=0.001,     # Learning rate
        kl_coeff=0.3,  # KL divergence coefficient
        train_batch_size=2, 
        sgd_minibatch_size=1
)

# Build the algorithm object
try:
    algo = config.build()
except Exception as e:
    raise RuntimeError(f"Failed to build the PPO algorithm: {e}")

# Train the algorithm
for episode in tqdm(range(2700)):  # Train for 2700 episodes or around 76 days in real-time
    result = algo.train()  # Perform training
    if episode % 20 == 0:
        # Save a checkpoint every 20 episodes 
        save_result = algo.save('trained-drl-models/model-calibrator-config-4-checkpoint')
        path_to_checkpoint = save_result
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )

# Save it again after it finished
save_result = algo.save('trained-drl-models/model-calibrator-config-3')
    
path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
    
# Remove unnecessary files for the matlab files
os.remove('controls.mat') # controls file
os.remove('drl-env.mat')  # simulation file
os.remove('indoor.mat')   # indoor measurements
os.remove('fruit.mat')    # fruit growth
#os.remove('outdoor.mat')  # outdoor measurements
