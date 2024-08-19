# Import CalibratorModel

from backup.NeuralNetworksModel import NeuralNetworksModel
from utils.LoadMiniGreenhouseData import LoadMiniGreenhouseData

nn_model = NeuralNetworksModel()
load_mgh_data = LoadMiniGreenhouseData()

data_input = {
    'time': [1200, 1500, 1800],
    'global out': [0, 0.029625, 0.032943],
    'temp out': [22.8, 22.7, 22.8],
    'rh out': [51.65, 51.05, 48.05],
    'co2 out': [613, 619, 623],
    'ventilation': [0, 0, 0],
    'toplights': [1, 1, 1],
    'heater': [1, 1, 1]
}

target_variable = 'temp in'
predictions = nn_model.predict_measurements(target_variable, data_input)
print(predictions)

# List of target variables
# target_variables = ['global in', 'temp in', 'rh in', 'co2 in']

# # Iterate through each target variable and call the function
# for target in target_variables:
#     predictions = nn_model.predict_measurements(target, data_input)
#     print(predictions)

outdoor, indoor, controls, start_time = load_mgh_data.loadData(1, 1)

# Display first few rows of the output
print(outdoor.head())
print(indoor.head())
print(controls.head())

outdoor, indoor, controls, start_time = load_mgh_data.loadData(1 + 1/72, 1/72)

# Display first few rows of the output
print(outdoor.head())
print(indoor.head())
print(controls.head())
