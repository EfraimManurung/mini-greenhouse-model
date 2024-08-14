# Assuming the NeuralNetworksModel class is defined as provided
from CalibratorModel import CalibratorModel

calibrator_model = CalibratorModel({"flag_run": True,
                        "first_day": 1,
                        "season_length_gl": 1/72,
                        "season_length_nn": 0, 
                        "online_measurements": False,
                        "action_from_drl": False,
                        "flag_run_nn": True,
                        "flag_run_gl": True,
                        "max_steps": 72 * 4
                        })

terminated = truncated = False

#_action_drl = np.array([0.7, 0.8, 0.7])

while not terminated and not truncated: 
    
    #obs, reward, terminated, _, info = calibrator_model.step(_action_drl)
    obs, reward, terminated, _, info = calibrator_model.step(None)