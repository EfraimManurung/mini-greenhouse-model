'''
Calibrator model

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com
'''

import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

import os

'''
TO-DO:
- Make the class of Neural Networks look alike MiniGreenhouse
- Can give the output of observation
- Can iterate the excel
-

'''

class NeuralNetworksModel():
    def __init__(self, env_config):
        print("Initialized CalibratorModel!")
        
        # Initiliaze the variables from env_config
        self.flag_run = env_config.get("flag_run", True) # The simulation is for running (other option is False for training)
        
        self.first_day = env_config.get("first_day", 0) # The first day of the simulation
        
        self.season_length = env_config.get("season_length", 1 * 4) # 1 / 72 in matlab is 1 step in this NN model
        
        self.max_steps = env_config.get("max_steps", 4) # How many iteration the program run
        
        self.current_step = 0
        
        # Load the datasets from separate files
        # file_path = r"datasets\dataset7.xlsx"
        file_path = r"C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\mini-greenhouse-greenlight-model\Code\inputs\Mini Greenhouse\dataset7.xlsx"

        # Load and clean the training dataset
        self.mgh_data = pd.read_excel(file_path)

        # Display the first few rows of the dataframe
        print("MiniGreenhouse DATA Columns / Variables (DEBUG): \n")
        print(self.mgh_data.head())
    
    def r2_score_metric(self, y_true, y_pred):
        '''
        Custom R2 score metric
        
        Parameters:
        y_true: tf.Tensor - Ground truth values.
        y_pred: tf.Tensor - Predicted values.
        
        Returns: 
        float: R2 score metric 
        '''
        SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
        return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))
    
    def load_excel_data(self):
        '''
        Load data from .xlsx file.
        
        Returns:
        Array: 
        '''

        # Slice the dataframe to get the rows for the current step
        step_data = self.mgh_data.iloc[self.season_length:self.season_length + 4]
        
        # Create a DataFrame with the required columns
        data = {
            'time': step_data['Time'].values,
            'global out': step_data['global out'].values,
            'global in': step_data['global in'].values,
            'temp in': step_data['temp in'].values,
            'temp out': step_data['temp out'].values,
            'rh in': step_data['rh in'].values,
            'rh out': step_data['rh out'].values,
            'co2 in': step_data['co2 in'].values,
            'co2 out': step_data['co2 out'].values,
            'toplights': step_data['toplights'].values,
            'ventilation': step_data['ventilation'].values,
            'heater': step_data['heater'].values
        }
        
        return pd.DataFrame(data)
        
    def predict_measurements(self, target_variable, data_input):
        '''
        Predict the measurements or state variables inside mini-greenhouse 
        
        Parameters:
        target_variable: str - The target variable to predict.
        data_input: dict or pd.DataFrame - The input features for the prediction.

        Features (inputs):
            Outside measurements information
                - time
                - global out
                - temp out
                - rh out
                - co2 out
            Control(s) input
                - ventilation
                - toplights
                - heater
        
        Return: 
        np.array: predicted measurements inside mini-greenhouse
        '''
        if isinstance(data_input, dict):
            data_input = pd.DataFrame(data_input)
        
        # Need to be fixed
        features = ['time', 'global out', 'temp out', 'temp out', 'rh out', 'co2 out', 'ventilation', 'toplights', 'heater']
    
        # Ensure the data_input has the required features
        for feature in features:
            if feature not in data_input.columns:
                raise ValueError(f"Missing feature '{feature}' in the input data.")
        
        X_features = data_input[features]
        
        # Load the model using the native Keras format
        loaded_model = load_model(f'nn-model/{target_variable}_model.keras', custom_objects={'r2_score_metric': self.r2_score_metric})
        
        # Load the scalerc:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\mini-greenhouse-drl-model\main_run.py
        scaler = joblib.load(f'nn-model/{target_variable}_scaler.pkl')
            
        # Scale the input features
        X_features_scaled = scaler.transform(X_features)
        
        # Predict the measurements
        y_hat_measurements = loaded_model.predict(X_features_scaled)
        
        return y_hat_measurements
    
    def step(self):
        '''
        Iterate the step
        
        '''
        
        self.season_length += 4
        self.current_step += 1
        print("")
        print("")
        print("----------------------------------")
        print("CURRENT STEPS: ", self.current_step)
        
        par_in = self.predict_measurements('global in', self.load_excel_data())
        temp_in = self.predict_measurements('temp in', self.load_excel_data())
        
        return par_in, temp_in