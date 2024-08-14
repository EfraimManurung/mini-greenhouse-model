# Assuming the NeuralNetworksModel class is defined as provided
from CalibratorModel import CalibratorModel

calibrator_model = CalibratorModel({"flag_run": True,
                        "first_day": 1,
                        "season_length_gl": 1/72,
                        "season_length_nn": 0, 
                        "online_measurements": False,
                        "flag_action_from_drl": False,
                        "max_steps": 6
                        })

terminated = truncated = False

while not terminated and not truncated: 
    obs, reward, terminated, _, info = calibrator_model.step(None)
    