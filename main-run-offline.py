# Assuming the NeuralNetworksModel class is defined as provided
from MiniGreenhouse import MiniGreenhouse

calibrator_model = MiniGreenhouse({"flag_run": True,
                        "first_day_gl": 1, #20,   # Start at day 15 as an example
                        "first_day_dnn": 0, #5760, # 20 days * 72 steps * 4 [-] data / steps
                        "season_length_gl": 1/72,
                        "season_length_dnn": 0,
                        "online_measurements": False,
                        "action_from_drl": False,
                        "flag_run_dnn": True,
                        "flag_run_gl": True,
                        "flag_run_combined_models": True,
                        "max_steps": 3 #72 * 5 # 3 steps = 1 hour or 1 episode, so 24 hours = 24 * 3 = 72 steps, 72 steps (24 hours) * day
                        })

terminated = truncated = False

while not terminated and not truncated: 
    
    obs, reward, terminated, _, info = calibrator_model.step(None)