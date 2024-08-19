# Assuming the NeuralNetworksModel class is defined as provided
from backup.NeuralNetworksModel import NeuralNetworksModel

nn_model = NeuralNetworksModel({"flag_run": True,
                                "first_day": 0,
                                "season_length": 0, 
                                "max_steps": 6
                               })

terminated = truncated = False

while not terminated and not truncated: 
    obs, terminated = nn_model.step()
    
    # # Extracting the relevant indices from the obs array
    # co2_in = obs[0]
    # temp_in = obs[1]
    # rh_in = obs[2]
    # par_in = obs[3]
    
    # par_in_predicted = obs[10]
    # temp_in_predicted = obs[11]
    # rh_in_predicted = obs[12]
    # co2_in_predicted = obs[13]
    
    # # Printing the actual and predicted values
    # print(f"Step {i+1}:")
    # print(f"  PAR In       -> Actual: {par_in:.4f}, Predicted: {par_in_predicted:.4f}")
    # print(f"  Temp In      -> Actual: {temp_in:.4f}, Predicted: {temp_in_predicted:.4f}")
    # print(f"  RH In        -> Actual: {rh_in:.4f}, Predicted: {rh_in_predicted:.4f}")
    # print(f"  CO2 In       -> Actual: {co2_in:.4f}, Predicted: {co2_in_predicted:.4f}")
    # print("--------------------------------------------------")

# After all steps, evaluate the predictions
# nn_model.evaluate_predictions()
