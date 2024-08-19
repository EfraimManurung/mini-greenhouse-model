# Assuming the NeuralNetworksModel class is defined as provided
from CalibratorModel import CalibratorModel

calibrator_model = CalibratorModel({"flag_run": True,
                        "first_day_gl": 15,
                        "first_day_nn": 4320, # 15 days * 72 steps * 4 [-] data / steps
                        "season_length_gl": 1/72,
                        "online_measurements": False,
                        "action_from_drl": False,
                        "flag_run_nn": True,
                        "flag_run_gl": True,
                        "max_steps": 6 #72 * 5 # 3 steps = 1 hour or 1 episode, so 24 hours = 24 * 3 = 72 steps, 72 steps (24 hours) * day
                        })

terminated = truncated = False

#_action_drl = np.array([0.7, 0.8, 0.7])

while not terminated and not truncated: 
    
    #obs, reward, terminated, _, info = calibrator_model.step(_action_drl)
    obs, reward, terminated, _, info = calibrator_model.step(None)