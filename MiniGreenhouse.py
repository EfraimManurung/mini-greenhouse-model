'''
Mini-greenhouse environment with calibration model (Deep Neural Networks algorithm and physics based model (the GreenLight model)). 
In this class we will combine the DNN model with the GreenLight model.

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.manurung@gmail.com

Table - General structure of the attributes, the predicted state variables of DNN x(t) and GreenLight x'(t), 
real measurement from the sensors ŷ(t), combined predicted measurement from LSTM \(\hat{y}(t)\), actuators control signal 
u(t) and disturbance d(t).
-------------------------------------------------------------------------------------------------------------------
x1(t), x'1(t)  Indoor CO2 [ppm]             y1(t)  Indoor CO2 [ppm]                 ŷ1(t)  Indoor CO2 [ppm] 
x2(t), x'2(t)  Indoor temperature [°C]      y2(t)  Indoor temperature [°C]          ŷ2(t)  Indoor temperature [°C]
x3(t), x'3(t)  Indoor humidity [%]          y3(t)  Indoor humidity [%]              ŷ3(t)  Indoor humidity [%]   
x4(t), x'4(t)  PAR Inside [W/m²]            y4(t)  PAR Inside [W/m²]                ŷ4(t)  PAR Inside [W/m²]
x7(t), x'7(t)  Leaf Temperature [°C]        y5(t)  Leaf Temperature [°C]            ŷ5(t)  Leaf Temperature [°C]
-------------------------------------------------------------------------------------------------------------------
d1(t)  Outside Radiation [W/m²]             u1(t)  Fan [0/1] (1 is on)
d2(t)  Outdoor CO2 [ppm]                    u2(t)  Toplighting status [0/1] (1 is on)
d3(t)  Outdoor temperature [°C]             u3(t)  Heating [0/1] (1 is on)
d4(t)  Outdoor humidity [%]
-------------------------------------------------------------------------------------------------------------------

based on Table above, we want to predict the state variable x(t) with control signal u(t) and disturbance d(t)
 
Project sources:
    - https://github.com/davkat1/GreenLight
    - 
Other sources:
    - 
    -
References:
    - David Katzin, Simon van Mourik, Frank Kempkes, and Eldert J. Van Henten. 2020. “GreenLight - An Open Source Model for 
      Greenhouses with Supplemental Lighting: Evaluation of Heat Requirements under LED and HPS Lamps.” 
      Biosystems Engineering 194: 61–81. https://doi.org/10.1016/j.biosystemseng.2020.03.010
    - Literature used:
        [1] Vanthoor, B., Stanghellini, C., van Henten, E. J. & de Visser, P. H. B. 
            A methodology for model-based greenhouse design: Part 1, a greenhouse climate 
            model for a broad range of designs and climates. Biosyst. Eng. 110, 363-377 (2011).
        [2] Vanthoor, B., de Visser, P. H. B., Stanghellini, C. & van Henten, E. J. 
            A methodology for model-based greenhouse design: Part 2, description and 
            validation of a tomato yield model. Biosyst. Eng. 110, 378-395 (2011).
        [3] Vanthoor, B. A model based greenhouse design method. (Wageningen University, 2011).
    - 
'''

# IMPORT LIBRARIES for DRL model 
# Import Farama foundation's gymnasium
import gymnasium as gym
from gymnasium.spaces import Box

# Import supporting libraries
import numpy as np
import scipy.io as sio
import os
import pandas as pd

# IMPORT LIBRARIES for the matlab file
import matlab.engine

# Import service functions
from utils.ServiceFunctions import ServiceFunctions

# IMPORT LIBRARIES for DNN and LSTM models
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Suppress specific TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Set TensorFlow logging level to ERROR
tf.get_logger().setLevel('ERROR')

# For the tf.data.Dataset only supports Python-style environment
tf.compat.v1.enable_eager_execution()

class MiniGreenhouse(gym.Env):
    '''
    Calibrator model that combine a DNN model and physics based model.
    
    Link the Python code to matlab program with related methods. We can link it with the .mat file.
    '''
    
    def __init__(self, env_config):
        '''
        Initialize the environment.
        
        Parameters:
        env_config(dict): Configuration dictionary for the environment.
        '''  
        
        print("Initialized MiniGreenhouse Environment!")
        
        # Initialize if the main program for training or running
        self.flag_run  = env_config.get("flag_run", True) # The simulation is for running (other option is False for training)
        self.first_day_gl = env_config.get("first_day_gl", 1) # The first day of the simulation
        self.first_day_dnn = env_config.get("first_day_dnn", 0) # 1 / 72 in matlab is 1 step in this DNN model, 20 minutes
        
        # Define the season length parameter
        self.season_length_gl = env_config.get("season_length_gl", 1 / 72) # 1 / 72 * 24 [hours] * 60 [minutes / hours] = 20 minutes  
        self.season_length_dnn = self.first_day_dnn # so we can substract the timesteps, that is why we used the season_length_dnn from the first_day
        
        self.online_measurements = env_config.get("online_measurements", False) # Use online measurements or not from the real-time measurements from IoT system 
        self.action_from_drl = env_config.get("action_from_drl", False) # Default is false, and we will use the action from offline datasets
        self.flag_run_dnn = env_config.get("flag_run_dnn", False) # Default is false, flag to run the Neural Networks model
        self.flag_run_gl = env_config.get("flag_run_gl", True) # Default is true, flag to run the green light model
        self.flag_run_combined_models = env_config.get("flag_run_combined_models", True) # Default is true, flag to run the LSTM model
        
        is_mature = env_config.get("is_mature", False) # The crops are mature or not
        
        # Convert Python boolean to integer (1 for True, 0 for False)
        self.is_mature_matlab = int(is_mature)
        
        # Initiate and max steps
        self.max_steps = env_config.get("max_steps", 3) # One episode = 3 steps = 1 hour, because 1 step = 20 minutes
    
        # Start MATLAB engine
        self.eng = matlab.engine.start_matlab()

        # Path to MATLAB script
        if self.flag_run_gl == True:
            self.matlab_script_path = r'matlab\DrlGlEnvironment.m'
        
        # No matter if the flag_run_dnn True or not we still need to load the files for the offline training
        # Load the datasets from separate files for the DNN model
        if is_mature == True and self.flag_run == True:
            print("IS MATURE - TRUE, USING MATURE CROPS DATASETS")
            # file_path = r"matlab\Mini Greenhouse\september-iot-datasets-test-mature-crops.xlsx"
            file_path = r"matlab\Mini Greenhouse\october-iot-datasets-test-mature-crops.xlsx"
        else:
            if is_mature == False and self.flag_run == False:
                print("IS MATURE - FALSE, FLAG RUN - FALSE, USING IOT DATASETS TO TRAIN DRL MODEL")
                file_path = r"matlab\Mini Greenhouse\iot-datasets-train-drl.xlsx"
            elif is_mature == False and self.flag_run == True:
                print("IS MATURE - FALSE, FLAG RUN - TRUE, USING SMALL CROPS DATASETS")
                # file_path = r"matlab\Mini Greenhouse\june-iot-datasets-test-small-crops.xlsx"
                file_path = r"matlab\Mini Greenhouse\august-iot-datasets-test-small-crops.xlsx"
                
        # Load the dataset
        self.mgh_data = pd.read_excel(file_path)
                        
        # Initialize a list to store rewards
        self.rewards_list = []
        
        # Initialize reward
        reward = 0
        
        # Record the reward for the first time
        self.rewards_list.extend([reward] * 4)

        # Initialize ServiceFunctions
        self.service_functions = ServiceFunctions()
        
        # Load the updated data from the excel or from mqtt 
        self.load_excel_or_mqtt_data(None)
        
        # Check if MATLAB script exists
        if os.path.isfile(self.matlab_script_path):
            
            # Initialize lists to store control values and initialize control variables to zero 
            self.init_controls()
            
            # Call the MATLAB function 
            if self.online_measurements == True:
                
                # Run the script with the updated outdoor measurements for the first time
                self.run_matlab_script('outdoor-indoor.mat', None, None)
            else:
                                
                # Run the script with empty parameter
                self.run_matlab_script()
        else:
            print(f"MATLAB script not found: {self.matlab_script_path}")

        if self.flag_run_gl == True:
            #Predict from the GL model
            self.predicted_inside_measurements_gl()
            
        if self.flag_run_dnn == True:
            
            # Predict from the DNN model
            self.predicted_inside_measurements_dnn()
            self.season_length_dnn += 4
        
        # Combine the predicted results from the GL and DNN models
        time_steps_formatted= list(range(0, int(self.season_length_dnn - self.first_day_dnn)))
        self.format_time_steps(time_steps_formatted)
            
        if self.flag_run_combined_models == True:
            # print("time_steps_formatted 2 : ", time_steps_formatted)
            self.predicted_combined_models(time_steps_formatted)
        
        # Define the observation and action space
        self.define_spaces()
        
        # Initialize the state
        self.reset()
    
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
    
    def define_spaces(self):
        '''
        Define the observation and action spaces.
        
        Based on the observation of the mini-greenhouse system                          Explanation 
            - co2_in: CO2 inside the mini-greenhouse [ppm]
            - temp_in: Temperature inside the mini-greenhouse [°C]
            - rh_in: Relative humidity in percentage in the mini-greenhouse [%]
            - PAR_in: Global radiation inside the mini-greenhouse [W m^{-2}]
            - fruit_dw: Carbohydrates in fruit dry weight [mg{CH2O} m^{-2}]             Equation 2, 3 [2], Equation A44 [5]
            - fruit_tcansum: Crop development stage [°C day]                            Equation 8 [2]
            - leaf_temp: Crop temperature [°C]
            
        Not included:
            - fruit_leaf: Carbohydrates in leaves [mg{CH2O} m^{-2}]                     Equation 4, 5 [2]
            - fruit_stem: Carbohydrates in stem [mg{CH2O} m^{-2}]                       Equation 6, 7 [2]
            - fruit_cbuf: Carbohydrates in buffer [mg{CH2O} m^{-2}]                     Equation 1, 2 [2]
        '''
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=np.array([0.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),  #in order: co2_in, temp_in, rh_in, PAR_in, fruit_dw, fruit_tcansum and leaf_temp
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), 
            dtype=np.float64
        )
    
        self.action_space = Box(
            low=np.array([0, 0, 0], dtype=np.float32), 
            high=np.array([1, 1, 1], dtype=np.float32), 
            dtype=np.float32
        )
        
    def init_controls(self):
        '''
        Initialize control variables.
        '''
        
        # Initialize for the first time 
        time_steps = np.linspace(300, 1200, 4) # 20 minutes (1200 seconds)
        self.controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': np.zeros(4).reshape(-1, 1),
            'toplights': np.zeros(4).reshape(-1, 1),
            'heater': np.zeros(4).reshape(-1, 1)
        }
        
        sio.savemat('controls.mat', self.controls)
        
    def run_matlab_script(self, outdoor_file = None, indoor_file=None, fruit_file=None):
        '''
        Run the MATLAB script.
        '''
        # Check if the outdoor_file or indoor_file or fruit_file is None
        if indoor_file is None:
            indoor_file = []
        
        if fruit_file is None:
            fruit_file = []
        
        if outdoor_file is None:
            outdoor_file = []
        
        self.eng.DrlGlEnvironment(self.season_length_gl, self.first_day_gl, 'controls.mat', outdoor_file, indoor_file, fruit_file, self.is_mature_matlab, nargout=0)

    def load_excel_or_mqtt_data(self, _action_drl):
        '''
        Load data from .xlsx file or mqtt data and store in instance variables.
        
        The data is appended to existing variables if they already exist.
        '''

        if self.online_measurements == True:
            print("load_excel_or_mqtt_data from ONLINE MEASUREMENTS")
            
            if self.action_from_drl == True and _action_drl is not None:
                # Use the actions from the DRL model and convert actions to discrete values
                ventilation = 1 if _action_drl[0] >= 0.5 else 0
                toplights = 1 if _action_drl[1] >= 0.5 else 0
                heater = 1 if _action_drl[2] >= 0.5 else 0
                
                ventilation = np.full(4, ventilation)
                toplights = np.full(4, toplights)
                heater = np.full(4, heater)
                
                # Update the step_data with the DRL model's actions using .loc
                self.step_data.loc[:, 'toplights'] = toplights
                self.step_data.loc[:, 'ventilation'] = ventilation
                self.step_data.loc[:, 'heater'] = heater
                
                # Add new data
                new_toplights = self.step_data['toplights'].values
                new_ventilation = self.step_data['ventilation'].values
                new_heater = self.step_data['heater'].values

            else:
                # Handle if the _action_drl is None for the first time 
                ventilation = np.full(4, 0)
                toplights = np.full(4, 0)
                heater = np.full(4, 0)
                
                # Update the step_data with the DRL model's actions using .loc
                self.step_data.loc[:, 'toplights'] = toplights
                self.step_data.loc[:, 'ventilation'] = ventilation
                self.step_data.loc[:, 'heater'] = heater
                
                # Use the actions from the offline dataset and add new data
                new_toplights = self.step_data['toplights'].values
                new_ventilation = self.step_data['ventilation'].values
                new_heater = self.step_data['heater'].values
            
            '''TO-DO: Server send the actions with all zero as the initializiation'''
            # Get the actions from the excel or drl from the load_excel_or_mqtt_data, for online or offline measurement
            time_steps = np.linspace(300, 1200, 4)  # Time steps in seconds
            ventilation = self.ventilation[-4:]
            toplights = self.toplights[-4:]
            heater = self.heater[-4:]
                                    
            print("CONVERTED ACTION REAL-TIME MEASUREMENTS")
            print("ventilation: ", ventilation)
            print("toplights: ", toplights)
            print("heater: ", heater)

            # Format data controls in JSON format
            json_data = self.service_functions.format_data_in_JSON(time_steps, \
                                                ventilation, toplights, \
                                                heater)
            
            # Publish controls to the raspberry pi (IoT system client)
            self.service_functions.publish_mqtt_data(json_data, broker="192.168.1.56", port=1883, topic="greenhouse-iot-system/drl-controls")

            '''TO-DO: The firmware or client or raspberry pi run the command for 20 minutes, then send again the data.
               So, the server can process the receive data and determine the action again.
            '''
            # Initialize outdoor measurements, to get the outdoor measurements
            outdoor_indoor_measurements = self.service_functions.get_outdoor_indoor_measurements(broker="192.168.1.56", port=1883, topic="greenhouse-iot-system/outdoor-indoor-measurements")
            
            # Convert outdoor_indoor_measurements to a DataFrame
            self.step_data = pd.DataFrame({
                'time': outdoor_indoor_measurements['time'].flatten(),
                'global out': outdoor_indoor_measurements['par_out'].flatten(),
                'global in': outdoor_indoor_measurements['par_in'].flatten(),
                'temp out': outdoor_indoor_measurements['temp_out'].flatten(),
                'temp in': outdoor_indoor_measurements['temp_in'].flatten(),
                'rh out': outdoor_indoor_measurements['hum_out'].flatten(),
                'rh in': outdoor_indoor_measurements['hum_in'].flatten(),
                'co2 out': outdoor_indoor_measurements['co2_out'].flatten(),
                'co2 in': outdoor_indoor_measurements['co2_in'].flatten(),
                'leaf temp': outdoor_indoor_measurements['leaf_temp'].flatten()
            })
            
            # Add empty columns for toplights, ventilation, and heater
            self.step_data['toplights'] = np.nan       # or None
            self.step_data['ventilation'] = np.nan     # or None
            self.step_data['heater'] = np.nan          # or None
            
            # Map the outdoor measurements to the corresponding variables
            new_time_excel_mqtt = outdoor_indoor_measurements['time'].flatten()
            new_global_out_excel_mqtt = outdoor_indoor_measurements['par_out'].flatten()
            new_global_in_excel_mqtt = outdoor_indoor_measurements['par_in'].flatten()  
            new_temp_out_excel_mqtt = outdoor_indoor_measurements['temp_out'].flatten()  
            new_temp_in_excel_mqtt = outdoor_indoor_measurements['temp_in'].flatten()
            new_rh_out_excel_mqtt = outdoor_indoor_measurements['hum_out'].flatten()  
            new_rh_in_excel_mqtt = outdoor_indoor_measurements['hum_in'].flatten()
            new_co2_out_excel_mqtt = outdoor_indoor_measurements['co2_out'].flatten()  
            new_co2_in_excel_mqtt = outdoor_indoor_measurements['co2_in'].flatten()
            new_leaf_temp_excel_mqtt = outdoor_indoor_measurements['leaf_temp'].flatten()
            
            # Check if instance variables already exist; if not, initialize them
            if not hasattr(self, 'time_excel_mqtt'):
                self.time_excel_mqtt = new_time_excel_mqtt
                self.global_out_excel_mqtt = new_global_out_excel_mqtt
                self.global_in_excel_mqtt = new_global_in_excel_mqtt
                self.temp_in_excel_mqtt = new_temp_in_excel_mqtt
                self.temp_out_excel_mqtt = new_temp_out_excel_mqtt
                self.rh_in_excel_mqtt = new_rh_in_excel_mqtt
                self.rh_out_excel_mqtt = new_rh_out_excel_mqtt
                self.co2_in_excel_mqtt = new_co2_in_excel_mqtt
                self.co2_out_excel_mqtt = new_co2_out_excel_mqtt
                self.leaf_temp_excel_mqtt = new_leaf_temp_excel_mqtt
                self.toplights = new_toplights
                self.ventilation = new_ventilation
                self.heater = new_heater
            else:
                # Concatenate new data with existing data
                self.time_excel_mqtt = np.concatenate((self.time_excel_mqtt, new_time_excel_mqtt))
                self.global_out_excel_mqtt = np.concatenate((self.global_out_excel_mqtt, new_global_out_excel_mqtt))
                self.global_in_excel_mqtt = np.concatenate((self.global_in_excel_mqtt, new_global_in_excel_mqtt))
                self.temp_in_excel_mqtt = np.concatenate((self.temp_in_excel_mqtt, new_temp_in_excel_mqtt))
                self.temp_out_excel_mqtt = np.concatenate((self.temp_out_excel_mqtt, new_temp_out_excel_mqtt))
                self.rh_in_excel_mqtt = np.concatenate((self.rh_in_excel_mqtt, new_rh_in_excel_mqtt))
                self.rh_out_excel_mqtt = np.concatenate((self.rh_out_excel_mqtt, new_rh_out_excel_mqtt))
                self.co2_in_excel_mqtt = np.concatenate((self.co2_in_excel_mqtt, new_co2_in_excel_mqtt))
                self.co2_out_excel_mqtt = np.concatenate((self.co2_out_excel_mqtt, new_co2_out_excel_mqtt))
                self.leaf_temp_excel_mqtt = np.concatenate((self.leaf_temp_excel_mqtt, new_leaf_temp_excel_mqtt))
                self.toplights = np.concatenate((self.toplights, new_toplights))
                self.ventilation = np.concatenate((self.ventilation, new_ventilation))
                self.heater = np.concatenate((self.heater, new_heater))
            
            '''TO-DO: Get the newest data from the online measurements and save it as outdoor-indoor.mat'''
            if self.flag_run_gl == True:
            
                print("USE OUTDOOR ONLINE MEASUREMENTS")

                # Update the MATLAB environment with the 4 latest data                
                # In the MATLAB, need to follow this format
                # outdoor_drl = [outdoor_file.time, outdoor_file.par_out, outdoor_file.temp_out, outdoor_file.hum_out, outdoor_file.co2_out];
                outdoor_indoor = {
                    'time': self.time_excel_mqtt[-4:].astype(float).reshape(-1, 1),
                    'par_out': self.global_out_excel_mqtt[-4:].astype(float).reshape(-1, 1),
                    'temp_out': self.temp_out_excel_mqtt[-4:].astype(float).reshape(-1, 1),
                    'hum_out': self.rh_out_excel_mqtt[-4:].astype(float).reshape(-1, 1),
                    'co2_out': self.co2_out_excel_mqtt[-4:].astype(float).reshape(-1, 1)
                }
            
                # Save control variables to .mat file
                sio.savemat('outdoor-indoor.mat', outdoor_indoor)
        
            # Optionally: Check or print the step_data structure to ensure it's correct
            print("Step Data (online):", self.step_data.head())
        
        elif self.online_measurements == False:
            # Use offline dataset 
            print("load_excel_or_mqtt_data from OFFLINE MEASUREMENTS")
            
            # Slice the dataframe to get the rows for the current step
            self.step_data = self.mgh_data.iloc[self.season_length_dnn:self.season_length_dnn + 4]
            
            if self.action_from_drl == True and _action_drl is not None:
                # Use the actions from the DRL model
                # Convert actions to discrete values
                ventilation = 1 if _action_drl[0] >= 0.5 else 0
                toplights = 1 if _action_drl[1] >= 0.5 else 0
                heater = 1 if _action_drl[2] >= 0.5 else 0
                
                ventilation = np.full(4, ventilation)
                toplights = np.full(4, toplights)
                heater = np.full(4, heater)
                
                # Update the step_data with the DRL model's actions using .loc
                self.step_data.loc[:, 'toplights'] = toplights
                self.step_data.loc[:, 'ventilation'] = ventilation
                self.step_data.loc[:, 'heater'] = heater
                
                # Add new data
                new_toplights = self.step_data['toplights'].values
                new_ventilation = self.step_data['ventilation'].values
                new_heater = self.step_data['heater'].values

            else:
                # Use the actions from the offline dataset and add new data
                new_toplights = self.step_data['toplights'].values
                new_ventilation = self.step_data['ventilation'].values
                new_heater = self.step_data['heater'].values
            
            # Extract the required columns and flatten them
            new_time_excel_mqtt = self.step_data['time'].values
            new_global_out_excel_mqtt = self.step_data['global out'].values
            new_global_in_excel_mqtt = self.step_data['global in'].values
            new_temp_in_excel_mqtt = self.step_data['temp in'].values
            new_temp_out_excel_mqtt = self.step_data['temp out'].values
            new_rh_in_excel_mqtt = self.step_data['rh in'].values
            new_rh_out_excel_mqtt = self.step_data['rh out'].values
            new_co2_in_excel_mqtt = self.step_data['co2 in'].values
            new_co2_out_excel_mqtt = self.step_data['co2 out'].values
            new_leaf_temp_excel_mqtt = self.step_data['leaf temp'].values
            
            # Check if instance variables already exist; if not, initialize them
            if not hasattr(self, 'time_excel_mqtt'):
                self.time_excel_mqtt = new_time_excel_mqtt
                self.global_out_excel_mqtt = new_global_out_excel_mqtt
                self.global_in_excel_mqtt = new_global_in_excel_mqtt
                self.temp_in_excel_mqtt = new_temp_in_excel_mqtt
                self.temp_out_excel_mqtt = new_temp_out_excel_mqtt
                self.rh_in_excel_mqtt = new_rh_in_excel_mqtt
                self.rh_out_excel_mqtt = new_rh_out_excel_mqtt
                self.co2_in_excel_mqtt = new_co2_in_excel_mqtt
                self.co2_out_excel_mqtt = new_co2_out_excel_mqtt
                self.leaf_temp_excel_mqtt = new_leaf_temp_excel_mqtt
                self.toplights = new_toplights
                self.ventilation = new_ventilation
                self.heater = new_heater
            else:
                # Concatenate new data with existing data
                self.time_excel_mqtt = np.concatenate((self.time_excel_mqtt, new_time_excel_mqtt))
                self.global_out_excel_mqtt = np.concatenate((self.global_out_excel_mqtt, new_global_out_excel_mqtt))
                self.global_in_excel_mqtt = np.concatenate((self.global_in_excel_mqtt, new_global_in_excel_mqtt))
                self.temp_in_excel_mqtt = np.concatenate((self.temp_in_excel_mqtt, new_temp_in_excel_mqtt))
                self.temp_out_excel_mqtt = np.concatenate((self.temp_out_excel_mqtt, new_temp_out_excel_mqtt))
                self.rh_in_excel_mqtt = np.concatenate((self.rh_in_excel_mqtt, new_rh_in_excel_mqtt))
                self.rh_out_excel_mqtt = np.concatenate((self.rh_out_excel_mqtt, new_rh_out_excel_mqtt))
                self.co2_in_excel_mqtt = np.concatenate((self.co2_in_excel_mqtt, new_co2_in_excel_mqtt))
                self.co2_out_excel_mqtt = np.concatenate((self.co2_out_excel_mqtt, new_co2_out_excel_mqtt))
                self.leaf_temp_excel_mqtt = np.concatenate((self.leaf_temp_excel_mqtt, new_leaf_temp_excel_mqtt))
                self.toplights = np.concatenate((self.toplights, new_toplights))
                self.ventilation = np.concatenate((self.ventilation, new_ventilation))
                self.heater = np.concatenate((self.heater, new_heater))

            # Get the actions from the excel or drl from the load_excel_or_mqtt_data, for online or offline measurement
            time_steps = np.linspace(300, 1200, 4)  # Time steps in seconds
            ventilation = self.ventilation[-4:]
            toplights = self.toplights[-4:]
            heater = self.heater[-4:]
                                    
            print("CONVERTED ACTION OFFLINE")
            print("ventilation: ", ventilation)
            print("toplights: ", toplights)
            print("heater: ", heater)
            
            # Debugging
            print("Step Data (offline):", self.step_data.head())
    
    # @tf.function
    def predict_inside_measurements_dnn(self, target_variable, data_input):
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
        features = ['time', 'global out', 'temp out', 'rh out', 'co2 out', 'ventilation', 'toplights', 'heater']

        # Ensure the data_input has the required features
        for feature in features:
            if feature not in data_input.columns:
                raise ValueError(f"Missing feature '{feature}' in the input data.")
        
        X_features = data_input[features]
        
        # Load the model and scaler with error handling
        try:
            loaded_model = load_model(f'trained-dnn-models/{target_variable}_model.keras', custom_objects={'r2_score_metric': self.r2_score_metric})
        except Exception as e:
            raise ValueError(f"Failed to load the model: {e}")
        
        try:
            scaler = joblib.load(f'trained-dnn-models/{target_variable}_scaler.pkl')
        except Exception as e:
            raise ValueError(f"Failed to load the scaler: {e}")
                    
        # Scale the input features
        X_features_scaled = scaler.transform(X_features)
        
        # Predict the measurements
        y_hat_measurements = loaded_model.predict(X_features_scaled)
        
        # Return the predicted measurements inside the mini-greenhouse
        return y_hat_measurements
    
    def predicted_inside_measurements_dnn(self):
        '''
        Predicted inside measurements
        
        '''
    
        # Predict the inside measurements (the state variable inside the mini-greenhouse)
        new_par_in_predicted_dnn = self.predict_inside_measurements_dnn('global in', self.step_data)
        new_temp_in_predicted_dnn = self.predict_inside_measurements_dnn('temp in', self.step_data)
        new_rh_in_predicted_dnn = self.predict_inside_measurements_dnn('rh in', self.step_data)
        new_co2_in_predicted_dnn = self.predict_inside_measurements_dnn('co2 in', self.step_data)
        new_leaf_temp_predicted_dnn = self.predict_inside_measurements_dnn('leaf temp', self.step_data)
    
        # Check if instance variables already exist; if not, initialize them
        if not hasattr(self, 'temp_in_predicted_dnn'):
            self.par_in_predicted_dnn = new_par_in_predicted_dnn
            self.temp_in_predicted_dnn = new_temp_in_predicted_dnn
            self.rh_in_predicted_dnn = new_rh_in_predicted_dnn
            self.co2_in_predicted_dnn = new_co2_in_predicted_dnn
            self.leaf_temp_predicted_dnn = new_leaf_temp_predicted_dnn
    
        else:
            # Concatenate new data with existing data
            self.par_in_predicted_dnn = np.concatenate((self.par_in_predicted_dnn, new_par_in_predicted_dnn))
            self.temp_in_predicted_dnn = np.concatenate((self.temp_in_predicted_dnn, new_temp_in_predicted_dnn))
            self.rh_in_predicted_dnn = np.concatenate((self.rh_in_predicted_dnn, new_rh_in_predicted_dnn))
            self.co2_in_predicted_dnn = np.concatenate((self.co2_in_predicted_dnn, new_co2_in_predicted_dnn))
            self.leaf_temp_predicted_dnn = np.concatenate((self.leaf_temp_predicted_dnn, new_leaf_temp_predicted_dnn))
                
    def predicted_inside_measurements_gl(self):
        '''
        Load data from the .mat file.
        
        From matlab, the structure is:
        
        % Save the extracted data to a .mat file
        save('drl-env.mat', 'time', 'temp_in', 'rh_in', 'co2_in', 'PAR_in', 'fruit_leaf', 'fruit_stem', 'fruit_dw', 'fruit_tcansum', 'leaf_temp');
        '''
        
        # Read the drl-env mat from the initialization 
        # Read the 3 values and append it
        # get the prediction from the matlab results
        data = sio.loadmat("drl-env.mat")
        
        new_time_gl = data['time'].flatten()[-4:]
        new_co2_in_predicted_gl = data['co2_in'].flatten()[-4:]
        new_temp_in_predicted_gl = data['temp_in'].flatten()[-4:]
        new_rh_in_predicted_gl = data['rh_in'].flatten()[-4:]
        new_par_in_predicted_gl = data['PAR_in'].flatten()[-4:]
        new_fruit_leaf_predicted_gl = data['fruit_leaf'].flatten()[-4:]
        new_fruit_stem_predicted_gl = data['fruit_stem'].flatten()[-4:]
        new_fruit_dw_predicted_gl = data['fruit_dw'].flatten()[-4:]
        new_fruit_cbuf_predicted_gl = data['fruit_cbuf'].flatten()[-4:]
        new_fruit_tcansum_predicted_gl = data['fruit_tcansum'].flatten()[-4:]
        new_leaf_temp_predicted_gl = data['leaf_temp'].flatten()[-4:]

        if not hasattr(self, 'time_gl'):
            self.time_gl = new_time_gl
            self.co2_in_predicted_gl = new_co2_in_predicted_gl
            self.temp_in_predicted_gl = new_temp_in_predicted_gl
            self.rh_in_predicted_gl = new_rh_in_predicted_gl
            self.par_in_predicted_gl = new_par_in_predicted_gl
            self.fruit_leaf_predicted_gl = new_fruit_leaf_predicted_gl
            self.fruit_stem_predicted_gl = new_fruit_stem_predicted_gl
            self.fruit_dw_predicted_gl = new_fruit_dw_predicted_gl
            self.fruit_cbuf_predicted_gl = new_fruit_cbuf_predicted_gl
            self.fruit_tcansum_predicted_gl = new_fruit_tcansum_predicted_gl
            self.leaf_temp_predicted_gl = new_leaf_temp_predicted_gl
        else:
            self.time_gl = np.concatenate((self.time_gl, new_time_gl))
            self.co2_in_predicted_gl = np.concatenate((self.co2_in_predicted_gl, new_co2_in_predicted_gl))
            self.temp_in_predicted_gl = np.concatenate((self.temp_in_predicted_gl, new_temp_in_predicted_gl))
            self.rh_in_predicted_gl= np.concatenate((self.rh_in_predicted_gl, new_rh_in_predicted_gl))
            self.par_in_predicted_gl = np.concatenate((self.par_in_predicted_gl, new_par_in_predicted_gl))
            self.fruit_leaf_predicted_gl = np.concatenate((self.fruit_leaf_predicted_gl, new_fruit_leaf_predicted_gl))
            self.fruit_stem_predicted_gl = np.concatenate((self.fruit_stem_predicted_gl, new_fruit_stem_predicted_gl))
            self.fruit_dw_predicted_gl = np.concatenate((self.fruit_dw_predicted_gl, new_fruit_dw_predicted_gl))
            self.fruit_cbuf_predicted_gl = np.concatenate((self.fruit_cbuf_predicted_gl, new_fruit_cbuf_predicted_gl))
            self.fruit_tcansum_predicted_gl = np.concatenate((self.fruit_tcansum_predicted_gl, new_fruit_tcansum_predicted_gl))
            self.leaf_temp_predicted_gl = np.concatenate((self.leaf_temp_predicted_gl, new_leaf_temp_predicted_gl))
            
    def reset(self, *, seed=None, options=None):
        '''
        Reset the environment to the initial state.
        
        Returns:
        int: The initial observation of the environment.
        '''
        self.current_step = 0
        # self.season_length_dnn += 4 # moved it to the intialized class 
        
        return self.observation(), {}

    def observation(self):
        '''
        Get the observation of the environment for every state.
        
        Returns:
        array: The observation space of the environment.
        '''
    
        if self.flag_run_combined_models == True:
                
            # print the predict measurements using the LSTM model
            print("PRINT THE OBSERVATION BASED ON THE COMBINED MODELS")
            print("self.co2_in_predicted_combined_models :", self.co2_in_predicted_combined_models[-1])
            print("self.temp_in_predicted_combined_models : ", self.temp_in_predicted_combined_models[-1])
            print("self.rh_in_predicted_combined_models : ", self.rh_in_predicted_combined_models[-1])
            print("self.par_in_predicted_combined_models : ", self.par_in_predicted_combined_models[-1])
            print("self.leaf_temp_predicted_combined_models : ", self.leaf_temp_predicted_combined_models[-1] )
            
            #in order: co2_in, temp_in, rh_in, PAR_in, fruit_dw, fruit_tcansum and leaf_temp
            return np.array([
                self.co2_in_predicted_combined_models[-1],      # use combined models for the observation
                self.temp_in_predicted_combined_models[-1],     # use combined models for the observation
                self.rh_in_predicted_combined_models[-1],       # use combined models for the observation
                self.par_in_predicted_combined_models[-1],      # use combined models for the observation
                self.fruit_dw_predicted_gl[-1],                 # use the predicted from the GL
                self.fruit_tcansum_predicted_gl[-1],            # use the predicted from the GL
                self.leaf_temp_predicted_combined_models[-1]    # use combined models for the observation
            ], np.float32) 
                
        else:
            
            #in order: co2_in, temp_in, rh_in, PAR_in, fruit_dw, fruit_tcansum and leaf_temp
            return np.array([
                self.co2_in_predicted_gl[-1],
                self.temp_in_predicted_gl[-1],
                self.rh_in_predicted_gl[-1],
                self.par_in_predicted_gl[-1],
                self.fruit_dw_predicted_gl[-1],
                self.fruit_tcansum_predicted_gl[-1],
                self.leaf_temp_predicted_gl[-1]
            ], np.float32) 

    def get_reward(self, _ventilation, _toplights, _heater):
        '''
        Get the reward for the current state.
        
        The reward function defines the immediate reward obtained by the agent for its actions in a given state. 
        Changed based on the MiniGreenhouse environment from the equation (4) in the source.
        
        Source: Bridging the reality gap: An Adaptive Framework for Autonomous Greenhouse 
        Control with Deep Reinforcement Learning. George Qi. Wageningen University. 2024.
        
        r(k) = w_r,y1 * Δy1(k) - Σ (from i=1 to 3) w_r,ai * ai(k) 
        
        Details 
        r(k): the imediate reward the agent receives at time step k.
        w_r,y1 * Δy1(k): represents the positive reward for the agent due to increased in the fruit dry weight Δy1(k).
        Σ (from i=1 to 3) w_r,ai * ai(k): represents the negative reward received by the agent due to cost energy with arbitrary 

        Obtained coefficient setting for the reward function.
        Coefficients        Values          Details
        w_r_y1              1               Fruit dry weight 
        w_r_a1              0.005           Ventilation
        w_r_a2              0.010           Toplights
        w_r_a3              0.001           Heater
        
        Returns:
        int: the immediate reward the agent receives at time step k in integer.
        '''
        
        # Initialize variables, based on the equation above
        # Need to be determined to make the r_k unitless
        w_r_y1 = 1          # Fruit dry weight 
        w_r_a1 = 0.005      # Ventilation
        w_r_a2 = 0.010      # Toplights
        w_r_a3 = 0.001      # Heater
        
        # Give initial reward 
        if self.current_step == 0:
            r_k = 0.0
            return r_k # No reward for the initial state 
        
        # In the createCropModel.m in the GreenLight model (mini-greenhouse-model)
        # cFruit or dry weight of fruit is the carbohydrates in fruit, so it is the best variable to count for the reward
        # Calculate the change in fruit dry weight
        delta_fruit_dw = (self.fruit_dw_predicted_gl[-1] - self.fruit_dw_predicted_gl[-2])
        print("delta_fruit_dw: ", delta_fruit_dw)
        
        r_k = w_r_y1 * delta_fruit_dw - ((w_r_a1 * _ventilation) + (w_r_a2 * _toplights) + (w_r_a3 * _heater))
        print("r_k immediate reward: ", r_k)
        
        return r_k
        
    def delete_files(self):
        '''
        delete matlab files after simulation to make it clear.    
        '''
        os.remove('controls.mat') # controls file
        os.remove('drl-env.mat')  # simulation file
        os.remove('indoor.mat')   # indoor measurements
        os.remove('fruit.mat')    # fruit growth

        if self.online_measurements == True:
            os.remove('outdoor-indoor.mat')  # outdoor measurements
        
    def done(self):
        '''
        Check if the episode is done.
        
        Returns:
        bool: True if the episode is done, otherwise False.
        '''
        
        # Episode is done if we have reached the target
        # We print all the physical parameters and controls

        if self.flag_run == True:
            # Terminated when current step is same value with max_steps 
            # In one episode, for example if the max_step = 4, that mean 1 episode is for 1 hour in real-time (real-measurements)
            if self.current_step >= self.max_steps:
                
                # Print and save all data
                if self.online_measurements == True:
                    self.print_and_save_all_data('output/output_simulated_data_online.xlsx')
                else:
                    self.print_and_save_all_data('output/output_simulated_data_offline.xlsx')
                
                # Delete all files
                self.delete_files()
                
                return True
        else:
            if self.current_step >= self.max_steps:

                return True
            
        return False

    def step(self, _action_drl):
        '''
        Take an action in the environment.
        
        Parameters:
        
        Based on the u(t) controls
        
        action (discrete integer):
        -  u1(t) Ventilation (-)               0-1 (1 is fully open) 
        -  u2(t) Toplights (-)                 0/1 (1 is on)
        -  u3(t) Heater (-)                    0/1 (1 is on)

        Returns:
            New observation, reward, terminated-flag (frome done method), truncated-flag, info-dict (empty).
        '''
        
        # Increment the current step
        self.current_step += 1
        print("")
        print("")
        print("----------------------------------")
        print("CURRENT STEPS: ", self.current_step)
        
        # Load the updated data from the excel or from mqtt, for online or offline measurements, we still need to call the data
        # Get the oudoor measurements
        '''TO-DO: The firmware or client or raspberry pi run the command for 20 minutes with the determined actions 
           from the DRL algorithm then send again the data'''
        self.load_excel_or_mqtt_data(_action_drl)
        
        # Get the actions from the excel or drl from the load_excel_or_mqtt_data, for online or offline measurement
        time_steps = np.linspace(300, 1200, 4)  # Time steps in seconds
        ventilation = self.ventilation[-4:]
        toplights = self.toplights[-4:]
        heater = self.heater[-4:]
                                 
        # Create control dictionary
        controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': ventilation.reshape(-1, 1),
            'toplights': toplights.reshape(-1, 1),
            'heater': heater.reshape(-1, 1)
        }
        
        # Save control variables to .mat file
        sio.savemat('controls.mat', controls)
        
        # Update the season_length and first_day for the GL model
        # 1 / 72 is 20 minutes in 24 hours, the calculation look like this
        # 1 / 72 * 24 [hours] * 60 [minutes . hours ^ -1] = 20 minutes 
        self.season_length_gl = 1 / 72 
        self.first_day_gl += 1 / 72 
        
        # Update the season_length for the DNN model
        self.season_length_dnn += 4

        if self.flag_run_gl == True:
            
            print("USE INDOOR GREENLIGHT")
            # Use the data from the GreenLight model
            # Convert co2_in ppm using service functions
            co2_density_gl = self.service_functions.co2ppm_to_dens(self.temp_in_predicted_gl[-4:], self.co2_in_predicted_gl[-4:])
            
            # Convert Relative Humidity (RH) to Pressure in Pa
            vapor_density_gl = self.service_functions.rh_to_vapor_density(self.temp_in_predicted_gl[-4:], self.rh_in_predicted_gl[-4:])
            vapor_pressure_gl = self.service_functions.vapor_density_to_pressure(self.temp_in_predicted_gl[-4:], vapor_density_gl)

            # Update the MATLAB environment with the 3 latest current state
            # It will be used to be simulated in the GreenLight model with mini-greenhouse parameters
            drl_indoor = {
                'time': self.time_gl[-3:].astype(float).reshape(-1, 1),
                'temp_in': self.temp_in_predicted_gl[-3:].astype(float).reshape(-1, 1),
                'rh_in': vapor_pressure_gl[-3:].astype(float).reshape(-1, 1),
                'co2_in': co2_density_gl[-3:].astype(float).reshape(-1, 1)
            }
        
            # Save control variables to .mat file
            sio.savemat('indoor.mat', drl_indoor)
            
        # Update the fruit growth with the 1 latest current state from the GreenLight model - mini-greenhouse parameters
        fruit_growth = {
            'time': self.time_gl[-1:].astype(float).reshape(-1, 1),
            'fruit_leaf': self.fruit_leaf_predicted_gl[-1:].astype(float).reshape(-1, 1),
            'fruit_stem': self.fruit_stem_predicted_gl[-1:].astype(float).reshape(-1, 1),
            'fruit_dw': self.fruit_dw_predicted_gl[-1:].astype(float).reshape(-1, 1),
            'fruit_cbuf': self.fruit_cbuf_predicted_gl[-1:].astype(float).reshape(-1, 1),
            'fruit_tcansum': self.fruit_tcansum_predicted_gl[-1:].astype(float).reshape(-1, 1),
            'leaf_temp': self.leaf_temp_predicted_gl[-1:].astype(float).reshape(-1,1)
        }

        # Save the fruit growth to .mat file
        sio.savemat('fruit.mat', fruit_growth)
        
        # Run the script with the updated state variables
        if self.online_measurements == True:
            self.run_matlab_script('outdoor-indoor.mat', 'indoor.mat', 'fruit.mat')
        else:
            self.run_matlab_script(None, 'indoor.mat', 'fruit.mat')
        
        if self.flag_run_gl == True:
            # Load the updated data from predcited from the greenlight model
            self.predicted_inside_measurements_gl()
        
        if self.flag_run_dnn == True:
            # Call the predicted inside measurements with the DNN model
            self.predicted_inside_measurements_dnn()
        
        time_steps_formatted = list(range(0, int(self.season_length_dnn - self.first_day_dnn)))
        
        self.format_time_steps(time_steps_formatted)
        
        if self.flag_run_combined_models == True:
            # Combine the predicted results from the GL and DNN models
            self.predicted_combined_models(time_steps_formatted)
            
        # Calculate reward
        # Remember that the actions become a list, but we only need the first actions from 15 minutes (all of the list of actions is the same)
        _reward = self.get_reward(ventilation[0], toplights[0], heater[0])
        
        # Record the reward
        self.rewards_list.extend([_reward] * 4)

        # Truncated flag
        truncated = False
    
        return self.observation(), _reward, self.done(), truncated, {}
    
    def print_and_save_all_data(self, file_name):
        '''
        Print all the appended data and save to an Excel file and plot it.

        Parameters:
        - file_name: Name of the output Excel file
        
        - ventilation_list: List of action for fan/ventilation from DRL model or offline datasets
        - toplights_list: List of action for toplights from DRL model or offline datasets
        - heater_list: List of action for heater from DRL model or offline datasets
        - reward_list: List of reward for iterated step
        
        - co2_in_excel_mqtt: List of actual CO2 values
        - temp_in_excel_mqtt: List of actual temperature values
        - rh_in_excel_mqtt: List of actual relative humidity values
        - par_in_excel_mqtt: List of actual PAR values
        
        - co2_in_predicted_dnn: List of predicted CO2 values from Neural Network
        - temp_in_predicted_dnn: List of predicted temperature values from Neural Network
        - rh_in_predicted_dnn: List of predicted relative humidity values from Neural Network
        - par_in_predicted_dnn: List of predicted PAR values from Neural Network
        
        - co2_in_predicted_gl: List of predicted CO2 values from Generalized Linear Model
        - temp_in_predicted_gl: List of predicted temperature values from Generalized Linear Model
        - rh_in_predicted_gl: List of predicted relative humidity values from Generalized Linear Model
        - par_in_predicted_gl: List of predicted PAR values from Generalized Linear Model
        '''
        
        print("\n\n-------------------------------------------------------------------------------------")
        print("Print all the appended data.")
        print(f"Length of Time: {len(self.time_excel_mqtt)}")
        print(f"Length of Action Ventilation: {len(self.ventilation)}")
        print(f"Length of Action Toplights: {len(self.toplights)}")
        print(f"Length of Action Heater: {len(self.heater)}")
        print(f"Length of reward: {len(self.rewards_list)}")
        print(f"Length of CO2 In (Actual): {len(self.co2_in_excel_mqtt)}")
        print(f"Length of Temperature In (Actual): {len(self.temp_in_excel_mqtt)}")
        print(f"Length of RH In (Actual): {len(self.rh_in_excel_mqtt)}")
        print(f"Length of PAR In (Actual): {len(self.global_in_excel_mqtt)}")
        print(f"Length of Predicted CO2 In (DNN): {len(self.co2_in_predicted_dnn)}")
        print(f"Length of Predicted Temperature In (DNN): {len(self.temp_in_predicted_dnn)}")
        print(f"Length of Predicted RH In (DNN): {len(self.rh_in_predicted_dnn)}")
        print(f"Length of Predicted PAR In (DNN): {len(self.par_in_predicted_dnn)}")
        print(f"Length of Predicted CO2 In (GL): {len(self.co2_in_predicted_gl)}")
        print(f"Length of Predicted Temperature In (GL): {len(self.temp_in_predicted_gl)}")
        print(f"Length of Predicted RH In (GL): {len(self.rh_in_predicted_gl)}")
        print(f"Length of Predicted PAR In (GL): {len(self.par_in_predicted_gl)}")
        
        # RUN WITH COMBINED MODELS!
        if self.flag_run_combined_models == True:
            print(f"Length of Predicted CO2 In (LSTM-Combined-models): {len(self.co2_in_predicted_combined_models)}")
            print(f"Length of Predicted Temperature In (LSTM-Combined-models): {len(self.temp_in_predicted_combined_models)}")
            print(f"Length of Predicted RH In (LSTM-Combined-models): {len(self.rh_in_predicted_combined_models)}")
            print(f"Length of Predicted PAR In (LSTM-Combined-models): {len(self.par_in_predicted_combined_models)}")
            
            if self.action_from_drl == True and self.online_measurements == False:
                print("---------------------------------------------------------------")
                print("COMBINED MODELS | ACTION: DRL ON | OFFLINE")
                
                # Save all the data (included the actions) in an Excel file                
                self.service_functions.export_to_excel(
                    file_name, self.time_combined_models, self.ventilation, self.toplights, self.heater, self.rewards,
                    None, None, None, None, None,
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], self.leaf_temp_predicted_dnn[:, 0],
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, self.leaf_temp_predicted_gl,
                    self.co2_in_predicted_combined_models, self.temp_in_predicted_combined_models, self.rh_in_predicted_combined_models, self.par_in_predicted_combined_models, self.leaf_temp_predicted_combined_models
                )
                
                # Save the rewards list 
                self.service_functions.export_rewards_to_excel('output/rewards_list.xlsx', self.time_combined_models, self.rewards_list)
                
                # Plot the data
                self.service_functions.plot_all_data(
                    'output/output_all_data.png', self.time_combined_models, 
                    None, None, None, None, 
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], 
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, 
                    self.co2_in_predicted_combined_models, self.temp_in_predicted_combined_models, self.rh_in_predicted_combined_models, self.par_in_predicted_combined_models, 
                    None, None, None)
                
                # Plot the actions
                self.service_functions.plot_actions('output/output_actions.png', self.time_combined_models, self.ventilation, self.toplights, 
                                                            self.heater)
                        
            else:
                print("---------------------------------------------------------------")
                print("COMBINED MODELS | ACTION: SCHEDULED OR DRL | OFFLINE OR ONLINE")
                
                # Evaluate predictions to get R² and MAE metrics
                metrics_dnn, metrics_gl, metrics_combined = self.evaluate_predictions()

                # Save all the data in an Excel file
                self.service_functions.export_to_excel(
                    file_name, self.time_combined_models, self.ventilation, self.toplights, self.heater, self.rewards_list,
                    self.co2_in_excel_mqtt, self.temp_in_excel_mqtt, self.rh_in_excel_mqtt, self.global_in_excel_mqtt, self.leaf_temp_excel_mqtt,
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], self.leaf_temp_predicted_dnn[:, 0],
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, self.leaf_temp_predicted_gl,
                    self.co2_in_predicted_combined_models, self.temp_in_predicted_combined_models, self.rh_in_predicted_combined_models, self.par_in_predicted_combined_models, self.leaf_temp_predicted_combined_models
                )
                
                # Save the metrics data in an Excel file as table format
                self.service_functions.export_evaluated_data_to_excel_table('output/metrics_table.xlsx', metrics_dnn, metrics_gl, metrics_combined)

                # Save the rewards list 
                self.service_functions.export_rewards_to_excel('output/rewards_list.xlsx', self.time_combined_models, self.rewards_list)
                
                # Plot the leaf temperature
                self.service_functions.plot_leaf_temperature('output/output_leaf_data.png', 
                                                             self.time_combined_models, 
                                                             self.leaf_temp_excel_mqtt,
                                                             self.leaf_temp_predicted_dnn,
                                                             self.leaf_temp_predicted_gl,
                                                             self.leaf_temp_predicted_combined_models) 
            
                # Plot the data
                self.service_functions.plot_all_data(
                    'output/output_all_data.png', self.time_combined_models, 
                    self.co2_in_excel_mqtt, self.temp_in_excel_mqtt, self.rh_in_excel_mqtt, self.global_in_excel_mqtt,
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], 
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, 
                    self.co2_in_predicted_combined_models, self.temp_in_predicted_combined_models, self.rh_in_predicted_combined_models, self.par_in_predicted_combined_models, 
                    metrics_dnn, metrics_gl, metrics_combined)
                
                # Plot the actions
                self.service_functions.plot_actions('output/output_actions.png', self.time_combined_models, self.ventilation, self.toplights, 
                                                            self.heater)
                
                # Plot the rewards
                # self.service_functions.plot_rewards('output/rewars-graphs.png', self.time_combined_models, self.rewards_list)
        
        # RUN WITHOUT COMBINED MODELS!
        elif self.flag_run_combined_models == False:
            
            # Run with dynamics control from the DRL action
            if self.action_from_drl == True and self.online_measurements == False:
                print("---------------------------------------------------------------")
                print("NOT COMBINED MODELS | ACTION: DRL | OFFLINE")
                
                # Save all the data (included the actions) in an Excel file
                self.service_functions.export_to_excel(
                    file_name, self.time_combined_models, self.ventilation, self.toplights, self.heater, self.rewards_list,
                    self.co2_in_excel_mqtt, self.temp_in_excel_mqtt, self.rh_in_excel_mqtt, self.global_in_excel_mqtt, self.leaf_temp_excel_mqtt,
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], self.leaf_temp_predicted_dnn[:, 0],
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, self.leaf_temp_predicted_gl,
                    None, None, None, None, None
                )
                
                # Save the rewards list 
                self.service_functions.export_rewards_to_excel('output/rewards_list.xlsx', self.time_combined_models, self.rewards_list)
                
                # Plot the data
                self.service_functions.plot_all_data(
                    'output/output_all_data.png', self.time_combined_models, 
                    None, None, None, None, None, 
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], self.leaf_temp_predicted_dnn[:, 0],
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, self.leaf_temp_predicted_gl,
                    None, None, None, None, None,
                    None, None, None)
                
                # Plot the actions
                self.service_functions.plot_actions('output/output_actions.png', self.time_combined_models, self.ventilation, self.toplights, 
                                                            self.heater)
            
            # Run with scheduled actions
            else:
                print("-------------------------------------------------------------------")
                print("NOT COMBINED MODELS | ACTION: SCHEDULED OR DRL | OFFLINE OR ONLINE")
                
                # Save all the data in an Excel file                
                self.service_functions.export_to_excel(
                    file_name, self.time_combined_models, self.ventilation, self.toplights, self.heater, self.rewards_list,
                    self.co2_in_excel_mqtt, self.temp_in_excel_mqtt, self.rh_in_excel_mqtt, self.global_in_excel_mqtt, self.leaf_temp_excel_mqtt,
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], self.leaf_temp_predicted_dnn[:, 0],
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, self.leaf_temp_predicted_gl,
                    None, None, None, None, None
                )
                
                # Save the rewards list 
                self.service_functions.export_rewards_to_excel('output/rewards_list.xlsx', self.time_combined_models, self.rewards_list)
                
                # Plot the data
                self.service_functions.plot_all_data(
                    'output/output_all_data.png', self.time_combined_models, 
                    self.co2_in_excel_mqtt, self.temp_in_excel_mqtt, self.rh_in_excel_mqtt, self.global_in_excel_mqtt, 
                    self.co2_in_predicted_dnn[:, 0], self.temp_in_predicted_dnn[:, 0], self.rh_in_predicted_dnn[:, 0], self.par_in_predicted_dnn[:, 0], 
                    self.co2_in_predicted_gl, self.temp_in_predicted_gl, self.rh_in_predicted_gl, self.par_in_predicted_gl, 
                    None, None, None, None, 
                    None, None, None)
                
                self.service_functions.plot_leaf_temperature(
                    'output/output_leaf_data.png', self.time_combined_models,
                    self.leaf_temp_excel_mqtt, self.leaf_temp_predicted_dnn, self.leaf_temp_predicted_gl, None
                )
                
                # Plot the actions
                self.service_functions.plot_actions('output/output_actions.png', self.time_combined_models, self.ventilation, self.toplights, 
                                                            self.heater)
                
    def evaluate_predictions(self):
        '''
        Evaluate the RMSE, RRMSE, and ME of the predicted vs actual values for `par_in`, `temp_in`, `rh_in`, `co2_in`, and `leaf_temp`.
        '''
        
        # Extract actual values
        y_true_par_in = self.global_in_excel_mqtt
        y_true_temp_in = self.temp_in_excel_mqtt
        y_true_rh_in = self.rh_in_excel_mqtt
        y_true_co2_in = self.co2_in_excel_mqtt
        y_true_leaf_temp = self.leaf_temp_excel_mqtt

        # Extract predicted values from Neural Network (DNN)
        y_pred_par_in_dnn = self.par_in_predicted_dnn[:, 0]
        y_pred_temp_in_dnn = self.temp_in_predicted_dnn[:, 0]
        y_pred_rh_in_dnn = self.rh_in_predicted_dnn[:, 0]
        y_pred_co2_in_dnn = self.co2_in_predicted_dnn[:, 0]
        y_pred_leaf_temp_dnn = self.leaf_temp_predicted_dnn[:, 0]

        # Extract predicted values from Generalized Linear Model (GL)
        y_pred_par_in_gl = self.par_in_predicted_gl
        y_pred_temp_in_gl = self.temp_in_predicted_gl
        y_pred_rh_in_gl = self.rh_in_predicted_gl
        y_pred_co2_in_gl = self.co2_in_predicted_gl
        y_pred_leaf_temp_gl = self.leaf_temp_predicted_gl

        # Extract combined model predictions
        y_pred_par_in_combined = self.par_in_predicted_combined_models
        y_pred_temp_in_combined = self.temp_in_predicted_combined_models
        y_pred_rh_in_combined = self.rh_in_predicted_combined_models
        y_pred_co2_in_combined = self.co2_in_predicted_combined_models
        y_pred_leaf_temp_combined = self.leaf_temp_predicted_combined_models

        # Calculate RMSE, RRMSE, and ME for each variable
        def calculate_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rrmse = rmse / np.mean(y_true) * 100  # Remember that it is in percentage
            me = np.mean(y_pred - y_true)
            return rmse, rrmse, me

        # DNN model metrics
        metrics_dnn = {
            'PAR': calculate_metrics(y_true_par_in, y_pred_par_in_dnn),
            'Temperature': calculate_metrics(y_true_temp_in, y_pred_temp_in_dnn),
            'Humidity': calculate_metrics(y_true_rh_in, y_pred_rh_in_dnn),
            'CO2': calculate_metrics(y_true_co2_in, y_pred_co2_in_dnn),
            'Leaf Temperature': calculate_metrics(y_true_leaf_temp, y_pred_leaf_temp_dnn)
        }

        # GL model metrics
        metrics_gl = {
            'PAR': calculate_metrics(y_true_par_in, y_pred_par_in_gl),
            'Temperature': calculate_metrics(y_true_temp_in, y_pred_temp_in_gl),
            'Humidity': calculate_metrics(y_true_rh_in, y_pred_rh_in_gl),
            'CO2': calculate_metrics(y_true_co2_in, y_pred_co2_in_gl),
            'Leaf Temperature': calculate_metrics(y_true_leaf_temp, y_pred_leaf_temp_gl)
        }

        # Combined model metrics
        metrics_combined = {
            'PAR': calculate_metrics(y_true_par_in, y_pred_par_in_combined),
            'Temperature': calculate_metrics(y_true_temp_in, y_pred_temp_in_combined),
            'Humidity': calculate_metrics(y_true_rh_in, y_pred_rh_in_combined),
            'CO2': calculate_metrics(y_true_co2_in, y_pred_co2_in_combined),
            'Leaf Temperature': calculate_metrics(y_true_leaf_temp, y_pred_leaf_temp_combined)
        }

        # Print the results
        print("------------------------------------------------------------------------------------")
        print("EVALUATION RESULTS :")
        for variable in ['PAR', 'Temperature', 'Humidity', 'CO2', 'Leaf Temperature']:
            rmse_dnn, rrmse_dnn, me_dnn = metrics_dnn[variable]
            rmse_gl, rrmse_gl, me_gl = metrics_gl[variable]
            rmse_combined, rrmse_combined, me_combined = metrics_combined[variable]

            if variable == 'Temperature' or variable == 'Leaf Temperature':
                unit_rmse = "°C"
                unit_me = "°C"
            elif variable == 'Humidity':
                unit_rmse = "%"
                unit_me = "%"
            elif variable == 'CO2':
                unit_rmse = "ppm"
                unit_me = "ppm"
            else:
                unit_rmse = "W/m²"  # Assuming PAR is in W/m² (common unit)
                unit_me = "W/m²"

            unit_rrmse = "%"  # RRMSE is always in percentage for all variables
            
            print(f"{variable} (DNN): RMSE = {rmse_dnn:.4f} {unit_rmse}, RRMSE = {rrmse_dnn:.4f} {unit_rrmse}, ME = {me_dnn:.4f} {unit_me}")
            print(f"{variable} (GL): RMSE = {rmse_gl:.4f} {unit_rmse}, RRMSE = {rrmse_gl:.4f} {unit_rrmse}, ME = {me_gl:.4f} {unit_me}")
            print(f"{variable} (Combined): RMSE = {rmse_combined:.4f} {unit_rmse}, RRMSE = {rrmse_combined:.4f} {unit_rrmse}, ME = {me_combined:.4f} {unit_me}")

        return metrics_dnn, metrics_gl, metrics_combined
    
    def predict_inside_measurements_LSTM(self, target_variable, data_input):
        '''
        Predict the measurements or state variables inside mini-greenhouse using a LSTM model to combine both of 
        the GL and DNN models.
        
        Parameters:
        target_variable: str - The target variable to predict.
        data_input: dict or pd.DataFrame - The input features for the prediction.

        Features (inputs):
            - Timesteps [5 minutes]
            - {target_variable} (Predicted GL)
            - {target_variable} (Predicted DNN)
        
        Return: 
        np.array: predicted measurements inside mini-greenhouse
        '''
        # Custom Layer to subtract from 1
        class SubtractFromOne(Layer):
            def call(self, inputs):
                return 1.0 - inputs

        # Custom Layer to extract a specific feature (replacing Lambda layers that slice inputs)
        class ExtractFeature(Layer):
            def __init__(self, index, **kwargs):
                super(ExtractFeature, self).__init__(**kwargs)
                self.index = index

            def call(self, inputs):
                return inputs[:, :, self.index]
        
        if isinstance(data_input, dict):
            data_input = pd.DataFrame(data_input)
        
         # Features required for the LSTM model
        features = ['Timesteps [5 minutes]', f'{target_variable} (Predicted GL)', f'{target_variable} (Predicted DNN)']

        # Ensure the data_input has the required features
        for feature in features:
            if feature not in data_input.columns:
                raise ValueError(f"Missing feature '{feature}' in the input data.")
        
        X_features = data_input[features]
        
        # Extract the underlying NumPy array and reshape
        X_features_values = X_features.values
        X_features_reshaped = X_features_values.reshape((X_features_values.shape[0], -1, X_features_values.shape[1]))
        
        # Load the LSTM model
        with open(f"trained-lstm-models/{target_variable.replace(' ', '_')}_lstm_model.json", "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = tf.keras.models.model_from_json(
                loaded_model_json,
                custom_objects={
                    'r2_score_metric': self.r2_score_metric,
                    'SubtractFromOne': SubtractFromOne,
                    'ExtractFeature': ExtractFeature
                }
            )
        
        # Load the model weights
        loaded_model.load_weights(f"trained-lstm-models/{target_variable.replace(' ', '_')}_lstm_model.weights.h5")
        
        # Compile the loaded model
        loaded_model.compile(optimizer='adam', loss='mse', metrics=['mae', self.r2_score_metric])
        
        # Predict the measurements
        y_hat_measurements = loaded_model.predict(X_features_reshaped)
        
        # Flatten it
        y_hat_measurements_1d = y_hat_measurements.flatten()
        
        return y_hat_measurements_1d
    
    def format_time_steps(self, _timesteps):
        # Create data_input for timesteps
        data_input = pd.DataFrame({
            'Timesteps [5 minutes]': _timesteps
        })
        
        new_time_combined = data_input['Timesteps [5 minutes]'][-4:]
        # Check if instance variables already exist; if not, initialize them
        if not hasattr(self, 'time_combined_models'):
            self.time_combined_models = new_time_combined
        else:
            self.time_combined_models = np.concatenate((self.time_combined_models, new_time_combined))
             
    def predicted_combined_models(self, _timesteps):
        '''
        Combine predictions from the Neural Network (DNN) and Generalized Linear Model (GL), 
        and include predictions from the LSTM model for all variables.
            
        This method updates the following attributes:
        - self.co2_in_combined_models
        - self.temp_in_combined_models
        - self.rh_in_combined_models
        - self.par_in_combined_models
        - self.leaf_temp_combined_models
        '''

        # Create data_input for LSTM prediction
        data_input = pd.DataFrame({
            'Timesteps [5 minutes]': _timesteps,
            'CO2 In (Predicted GL)': self.co2_in_predicted_gl.flatten(),
            'CO2 In (Predicted DNN)': self.co2_in_predicted_dnn[:, 0],
            'Temperature In (Predicted GL)': self.temp_in_predicted_gl.flatten(),
            'Temperature In (Predicted DNN)': self.temp_in_predicted_dnn[:, 0],
            'RH In (Predicted GL)': self.rh_in_predicted_gl.flatten(),
            'RH In (Predicted DNN)': self.rh_in_predicted_dnn[:, 0],
            'PAR In (Predicted GL)': self.par_in_predicted_gl.flatten(),
            'PAR In (Predicted DNN)': self.par_in_predicted_dnn[:, 0],
            'Leaf Temp (Predicted GL)': self.leaf_temp_predicted_gl.flatten(),
            'Leaf Temp (Predicted DNN)': self.leaf_temp_predicted_dnn[:, 0]
        })
        
        # Predict inside measurements using the LSTM model
        new_par_in_predicted_combined = self.predict_inside_measurements_LSTM("PAR In", data_input)[-4:]
        new_temp_in_predicted_combined = self.predict_inside_measurements_LSTM("Temperature In", data_input)[-4:]
        new_rh_in_predicted_combined = self.predict_inside_measurements_LSTM("RH In", data_input)[-4:]
        new_co2_in_predicted_combined = self.predict_inside_measurements_LSTM("CO2 In", data_input)[-4:]
        new_leaf_temp_predicted_combined = self.predict_inside_measurements_LSTM("Leaf Temp", data_input)[-4:]
        
        # Initialize or update combined model predictions
        if not hasattr(self, 'co2_in_predicted_combined_models'):
            self.co2_in_predicted_combined_models = new_co2_in_predicted_combined
            self.temp_in_predicted_combined_models = new_temp_in_predicted_combined
            self.rh_in_predicted_combined_models = new_rh_in_predicted_combined
            self.par_in_predicted_combined_models = new_par_in_predicted_combined
            self.leaf_temp_predicted_combined_models = new_leaf_temp_predicted_combined
        else:
            # Concatenate with existing data if already initialized
            self.co2_in_predicted_combined_models = np.concatenate((self.co2_in_predicted_combined_models, new_co2_in_predicted_combined))
            self.temp_in_predicted_combined_models = np.concatenate((self.temp_in_predicted_combined_models, new_temp_in_predicted_combined))
            self.rh_in_predicted_combined_models = np.concatenate((self.rh_in_predicted_combined_models, new_rh_in_predicted_combined))
            self.par_in_predicted_combined_models = np.concatenate((self.par_in_predicted_combined_models, new_par_in_predicted_combined))
            self.leaf_temp_predicted_combined_models = np.concatenate((self.leaf_temp_predicted_combined_models, new_leaf_temp_predicted_combined))
            
    # Ensure to properly close the MATLAB engine when the environment is no longer used
    def __del__(self):
        self.eng.quit()