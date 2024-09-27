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
                    "max_steps": 9
                    })

config.training(
        gamma=0.9,  # Discount factor
        lr=0.001, #lr = 0.1,  # Learning rate
        kl_coeff=0.3,  # KL divergence coefficient
        # model={
        #     "fcnet_hiddens": [256, 256, 256],  # Hidden layer configuration
        #     "fcnet_activation": "relu",  # Activation function
        #     "use_lstm": True,  # Whether to use LSTM
        #     "max_seq_len": 48,  # Maximum sequence length
        # }, 
        train_batch_size=2, 
        sgd_minibatch_size=1
)

# Build the algorithm object
try:
    algo = config.build()
except Exception as e:
    raise RuntimeError(f"Failed to build the PPO algorithm: {e}")

# Train the algorithm
for episode in tqdm(range(414)):  # Train for 414 episodes
    result = algo.train()  # Perform training
        
# Save the model checkpoint
save_result = algo.save('trained-drl-models/model-calibrator-config-1')

path_to_checkpoint = save_result
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
    
# Remove unnecessary variables
os.remove('controls.mat') # controls file
os.remove('drl-env.mat')  # simulation file
os.remove('indoor.mat')   # indoor measurements
os.remove('fruit.mat')    # fruit growth
#os.remove('outdoor.mat')  # outdoor measurements
