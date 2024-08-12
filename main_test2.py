from NeuralNetworksModel import NeuralNetworksModel

nn_model = NeuralNetworksModel({"flag_run": True,
                        "first_day": 0,
                        "season_length": 4, 
                        "max_steps": 6
                        })

for i in range(1, 10):
    obs = nn_model.step()
    print("observation_measurements_in : ", obs)
