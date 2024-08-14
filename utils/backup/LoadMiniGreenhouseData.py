import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class LoadMiniGreenhouseData():
    def __init__(self):
        print("Initialized LoadMiniGreenhouseData!")
        
        # Load the datasets from separate files
        self.file_path = r"C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\mini-greenhouse-greenlight-model\Code\inputs\Mini Greenhouse\dataset7.xlsx"
        
    def loadData(self, first_day, season_length):
        
        # Local variables for seconds in day
        SECONDS_IN_DAY = 24 * 60 * 60
        
        # Load the Excel file
        minigreenhouse = pd.read_excel(self.file_path)
        
        # Ensure the first column is in datetime format, if applicable
        if pd.api.types.is_numeric_dtype(minigreenhouse.iloc[:, 0]):
            # If the first column is numeric, assume it's in seconds or some time unit
            interval = minigreenhouse.iloc[1, 0] - minigreenhouse.iloc[0, 0]
        else:
            # Convert to datetime if it's not already in that format
            minigreenhouse.iloc[:, 0] = pd.to_datetime(minigreenhouse.iloc[:, 0])
            interval = (minigreenhouse.iloc[1, 0] - minigreenhouse.iloc[0, 0]).total_seconds()
        
        # Adjust firstDay to within a year
        first_day = first_day % 365
        
        # Calculate start and end points in the data array
        start_point = 1 + round((first_day - 1) * SECONDS_IN_DAY / interval)
        end_point = start_point - 1 + round(season_length * SECONDS_IN_DAY / interval)
        
        # Calculate the starting time
        start_time = pd.to_datetime(minigreenhouse.iloc[0, 0])
        
        # Determine the length of the dataset
        data_length = len(minigreenhouse)
        new_years = (end_point - end_point % data_length) // data_length
        
        # Extract the required season of data
        if end_point <= data_length:
            input_data = minigreenhouse.iloc[start_point:end_point, :]
        else:
            input_data = minigreenhouse.iloc[start_point:, :]
            for _ in range(new_years - 1):
                input_data = pd.concat([input_data, minigreenhouse], ignore_index=True)
            end_point = end_point % data_length
            input_data = pd.concat([input_data, minigreenhouse.iloc[:end_point, :]], ignore_index=True)
        
        # Reformat the weather data
        outdoor = pd.DataFrame()
        outdoor['time'] = interval * np.arange(len(input_data))
        outdoor['radiation'] = input_data.iloc[:, 1]  # Radiation outside
        outdoor['temperature'] = input_data.iloc[:, 4] + 1.5  # Temperature outside (adjusted by 1.5°C)
        outdoor['humidity'] = input_data.iloc[:, 6]  # relative humidity
        outdoor['co2'] = input_data.iloc[:, 8] # CO2 outdoor measurements
        outdoor['wind'] = 0  # Assuming no wind data in this example
        
        # Reformat the indoor data
        indoor = pd.DataFrame()
        indoor['time'] = outdoor['time']
        indoor['temperature'] = input_data.iloc[:, 3]  # Indoor temperature
        indoor['humidity'] = input_data.iloc[:, 5]  # Indoor relative humidity
        indoor['co2'] = input_data.iloc[:, 7]  # Indoor CO2 concentration
        indoor['radiation'] = input_data.iloc[:, 2]  # Radiation inside
        
        # Reformat the controls data
        controls = pd.DataFrame()
        controls['time'] = outdoor['time']
        controls['ventilation'] = input_data.iloc[:, 10]  # Ventilation aperture
        controls['toplights'] = input_data.iloc[:, 9]  # Toplights on/off
        controls['heater'] = input_data.iloc[:, 11] # Heater
        
        return outdoor, indoor, controls, start_time


