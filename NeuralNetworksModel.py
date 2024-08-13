'''
Calibrator model with Artifical Neural Networks algorithm

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

ANN Concept:

Table 1 Meaning of the state x(t), measurement y(t), control signal u(t) and disturbance d(t).
----------------------------------------------------------------------------------------------------------------------------------
 x1(t) Dry-weight (m2 / m-2)					 y1(t) Dry-weight (m2 / m-2) 
 x2(t) Indoor CO2 (ppm)							 y2(t) Indoor CO2 (ppm)
 x3(t) Indoor temperature (◦C)					 y3(t) Indoor temperature (◦C)
 x4(t) Indoor humidity (%)						 y4(t) Indoor humidity (%)
 x5(t) PAR Inside (W / m2)					     x5(t) PAR Inside (W / m2)
----------------------------------------------------------------------------------------------------------------------------------
 d1(t) Outside Radiation (W / m2)				 u1(t) Fan (-)
 d2(t) Outdoor CO2 (ppm)						 u2(t) Toplighting status (-)
 d3(t) Outdoor temperature (◦C)					 u3(t) Heating (-) 
 
 based on Table 1, we want to predict the state variable x(t) with control signal u(t) and disturbance d(t)
 
 Project sources:
    - Tensor flow
    - test
'''

# Import supporting libraries
from utils.ServiceFunctions import ServiceFunctions

import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np

import sklearn.metrics as metrics

import os

class NeuralNetworksModel():
    '''
    A MiniGreenhouse model, a environtment based on offline datasets that trained and used to predict.
    
    Will be used and combined with physics based model (the GreenLight model)
    '''
    
    def __init__(self, env_config):
        '''
        Initialize the NeuralNetworksModel.
        
        Parameters:
        env_config(dict): Configuration dictionary for the environment.
        '''
        
        print("Initialized CalibratorModel!")
        
        # Initialize service functions
        self.service_functions = ServiceFunctions()
        
        # Initiliaze the variables from env_config
        self.flag_run = env_config.get("flag_run", True) # The simulation is for running (other option is False for training)
        
        self.first_day = env_config.get("first_day", 0) # The first day of the simulation
        
        self.season_length = env_config.get("season_length", 1 * 4) # 1 / 72 in matlab is 1 step in this NN model, 20 minutes
        
        self.max_steps = env_config.get("max_steps", 4) # How many iteration the program run
        
        self.current_step = 0
        
        # Load the datasets from separate files
        file_path = r"C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\mini-greenhouse-greenlight-model\Code\inputs\Mini Greenhouse\dataset7.xlsx"

        # Load the dataset
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
    
    def predict_inside_measurements(self, target_variable, data_input):
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
    
    def load_excel_data(self):
        '''
        Load data from .xlsx file and store in instance variables.
        
        The data is appended to existing variables if they already exist.
        '''

        # Slice the dataframe to get the rows for the current step
        self.step_data = self.mgh_data.iloc[self.season_length:self.season_length + 4]

        # Extract the required columns and flatten them
        new_time = self.step_data['time'].values
        new_global_out = self.step_data['global out'].values
        new_global_in = self.step_data['global in'].values
        new_temp_in = self.step_data['temp in'].values
        new_temp_out = self.step_data['temp out'].values
        new_rh_in = self.step_data['rh in'].values
        new_rh_out = self.step_data['rh out'].values
        new_co2_in = self.step_data['co2 in'].values
        new_co2_out = self.step_data['co2 out'].values
        new_toplights = self.step_data['toplights'].values
        new_ventilation = self.step_data['ventilation'].values
        new_heater = self.step_data['heater'].values

        # Check if instance variables already exist; if not, initialize them
        if not hasattr(self, 'time'):
            self.time = new_time
            self.global_out = new_global_out
            self.global_in = new_global_in
            self.temp_in = new_temp_in
            self.temp_out = new_temp_out
            self.rh_in = new_rh_in
            self.rh_out = new_rh_out
            self.co2_in = new_co2_in
            self.co2_out = new_co2_out
            self.toplights = new_toplights
            self.ventilation = new_ventilation
            self.heater = new_heater
        else:
            # Concatenate new data with existing data
            self.time = np.concatenate((self.time, new_time))
            self.global_out = np.concatenate((self.global_out, new_global_out))
            self.global_in = np.concatenate((self.global_in, new_global_in))
            self.temp_in = np.concatenate((self.temp_in, new_temp_in))
            self.temp_out = np.concatenate((self.temp_out, new_temp_out))
            self.rh_in = np.concatenate((self.rh_in, new_rh_in))
            self.rh_out = np.concatenate((self.rh_out, new_rh_out))
            self.co2_in = np.concatenate((self.co2_in, new_co2_in))
            self.co2_out = np.concatenate((self.co2_out, new_co2_out))
            self.toplights = np.concatenate((self.toplights, new_toplights))
            self.ventilation = np.concatenate((self.ventilation, new_ventilation))
            self.heater = np.concatenate((self.heater, new_heater))
    
    def predicted_inside_measurements(self):
        '''
        Predicted inside measurements
        
        '''
        # Load the updated data from the excel file
        self.load_excel_data()
        
        # Predict the inside measurements (the state variable inside the mini-greenhouse)
        new_par_in_predicted = self.predict_inside_measurements('global in', self.step_data)
        new_temp_in_predicted = self.predict_inside_measurements('temp in', self.step_data)
        new_rh_in_predicted = self.predict_inside_measurements('rh in', self.step_data)
        new_co2_in_predicted = self.predict_inside_measurements('co2 in', self.step_data)
    
        # Check if instance variables already exist; if not, initialize them
        if not hasattr(self, 'par_in_predicted'):
            self.par_in_predicted = new_par_in_predicted
            self.temp_in_predicted = new_temp_in_predicted
            self.rh_in_predicted = new_rh_in_predicted
            self.co2_in_predicted = new_co2_in_predicted
        else:
            # Concatenate new data with existing data
            self.par_in_predicted = np.concatenate((self.par_in_predicted, new_par_in_predicted))
            self.temp_in_predicted = np.concatenate((self.temp_in_predicted, new_temp_in_predicted))
            self.rh_in_predicted = np.concatenate((self.rh_in_predicted, new_rh_in_predicted))
            self.co2_in_predicted = np.concatenate((self.co2_in_predicted, new_co2_in_predicted))
            
            # print("self.par_in_predicted", self.par_in_predicted)
            # print("self.temp_in_predicted", self.temp_in_predicted)
            # print("self.rh_in_predicted", self.rh_in_predicted)
            # print("self.co2_in_predicted", self.co2_in_predicted)
            
            print("LENGTH self.par_in_predicted", len(self.par_in_predicted))
            print("LENGTH self.temp_in_predicted", len(self.temp_in_predicted))
            print("LENGTH self.rh_in_predicted", len(self.rh_in_predicted))
            print("LENGTH self.co2_in_predicted", len(self.co2_in_predicted))
            
    
    def observation(self):
        '''
        Get the observation of the environment for every state based on the Excel data.
        
        Returns:
        array: The observation space of the environment.
        '''
        
        return np.array([
            self.co2_in[-1],
            self.temp_in[-1],
            self.rh_in[-1],
            self.global_in[-1],
            self.global_out[-1],
            self.temp_out[-1],
            self.rh_out[-1],
            self.toplights[-1],
            self.ventilation[-1],
            self.heater[-1],
            self.par_in_predicted[-1][0],  # Ensure this is a scalar value
            self.temp_in_predicted[-1][0], # Ensure this is a scalar value
            self.rh_in_predicted[-1][0],   # Ensure this is a scalar value
            self.co2_in_predicted[-1][0]   # Ensure this is a scalar value
        ], np.float32)
        
    def done(self):
        '''
        Check if the episode is done.
        
        Returns:
        bool: True if the episode is done, otherwise False.
        '''
        
        if self.current_step >= self.max_steps:

                # Evaluate the R² and MAE
                self.evaluate_predictions()
                
                # Save and plot the data
                self.print_and_save_all_data('output/output_simulated_data_offline.xlsx')
                
                return True

        return False
    
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
        
        # Call the predicted inside measurements
        self.predicted_inside_measurements()
        
        return self.observation(), self.done()
    
    def evaluate_predictions(self):
        '''
        Evaluate the R² and MAE of the predicted vs actual values for `par_in`, `temp_in`, `rh_in`, and `co2_in`.
        '''
        
        # Extract actual values (entire series)
        y_true_par_in = self.global_in
        y_true_temp_in = self.temp_in
        y_true_rh_in = self.rh_in
        y_true_co2_in = self.co2_in

        # Extract predicted values
        y_pred_par_in = self.par_in_predicted.flatten()  
        y_pred_temp_in = self.temp_in_predicted.flatten()  
        y_pred_rh_in = self.rh_in_predicted.flatten()  
        y_pred_co2_in = self.co2_in_predicted.flatten()  

        # Calculate R² and MAE for each variable
        r2_par_in = metrics.r2_score(y_true_par_in, y_pred_par_in)
        r2_temp_in = metrics.r2_score(y_true_temp_in, y_pred_temp_in)
        r2_rh_in = metrics.r2_score(y_true_rh_in, y_pred_rh_in)
        r2_co2_in = metrics.r2_score(y_true_co2_in, y_pred_co2_in)
        
        r2_par_in_1 = self.r2_score_metric(y_true_par_in, y_pred_par_in)
        r2_temp_in_1 = self.r2_score_metric(y_true_temp_in, y_pred_temp_in)
        r2_rh_in_1 = self.r2_score_metric(y_true_rh_in, y_pred_rh_in)
        r2_co2_in_1 = self.r2_score_metric(y_true_co2_in, y_pred_co2_in)

        mae_par_in = metrics.mean_absolute_error(y_true_par_in, y_pred_par_in)
        mae_temp_in = metrics.mean_absolute_error(y_true_temp_in, y_pred_temp_in)
        mae_rh_in = metrics.mean_absolute_error(y_true_rh_in, y_pred_rh_in)
        mae_co2_in = metrics.mean_absolute_error(y_true_co2_in, y_pred_co2_in)

        # Print the results
        print("Evaluation Results:")
        print(f"PAR In: R² (metrics) = {r2_par_in:.4f}, R² (custom) = {r2_par_in_1:.4f}, MAE = {mae_par_in:.4f}")
        print(f"Temperature In: R² (metrics) = {r2_temp_in:.4f}, R² (custom) = {r2_temp_in_1:.4f}, MAE = {mae_temp_in:.4f}")
        print(f"Humidity In: R² (metrics) = {r2_rh_in:.4f}, R² (custom) = {r2_rh_in_1:.4f}, MAE = {mae_rh_in:.4f}")
        print(f"CO2 In: R² (metrics) = {r2_co2_in:.4f}, R² (custom) = {r2_co2_in_1:.4f}, MAE = {mae_co2_in:.4f}")

    def print_and_save_all_data(self, _file_name):
        '''
        Print all the appended data, export it to an Excel file, and plot the data.
        '''
        print("\n\n-------------------------------------------------------------------------------------")
        print("Print all the appended data.")
        print(f"Length of Time: {len(self.time)}")
        print(f"Length of CO2 In: {len(self.co2_in)}")
        print(f"Length of Temperature In: {len(self.temp_in)}")
        print(f"Length of RH In: {len(self.rh_in)}")
        print(f"Length of PAR In: {len(self.global_in)}")
        print(f"Length of Predicted PAR In: {len(self.par_in_predicted)}")
        print(f"Length of Predicted Temperature In: {len(self.temp_in_predicted)}")
        print(f"Length of Predicted RH In: {len(self.rh_in_predicted)}")
        print(f"Length of Predicted CO2 In: {len(self.co2_in_predicted)}")
        
        time_steps_formatted = range(0, self.season_length)
        print("time_steps_formatted : ", len(time_steps_formatted))
       
        # Save all the data in an excel file
        self.service_functions.export_to_excel_nn(_file_name, time_steps_formatted, self.co2_in, self.temp_in, self.rh_in,
                             self.global_in, self.par_in_predicted[:, 0], self.temp_in_predicted[:, 0],
                             self.rh_in_predicted[:, 0], self.co2_in_predicted[:, 0])

        # Plot the data
        self.service_functions.plot_all_data_nn(time_steps_formatted, self.co2_in, self.temp_in, self.rh_in,
                           self.global_in, self.par_in_predicted[:, 0], self.temp_in_predicted[:, 0],
                           self.rh_in_predicted[:, 0], self.co2_in_predicted[:, 0])
