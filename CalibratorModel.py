'''
Calibrator model with Artifical Neural Networks and physics based model (the GreenModel) algorithm. 
In this class we will combine the ANN model with the GreenLight model.

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com

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
    - 
Other sources:
    -
    -
'''

# Import Farama foundation's gymnasium
import gymnasium as gym
from gymnasium.spaces import Box

# Import supporting libraries
import numpy as np
import scipy.io as sio
import matlab.engine
import os
from datetime import timedelta
import pandas as pd

# Import service functions
from utils.ServiceFunctions import ServiceFunctions

class MiniGreenhouse(gym.Env):
    '''
    MiniGreenhouse environment, a custom environment based on the GreenLight model
    and real mini-greenhouse.
    
    Link the Python code to matlab program with related methods. We can link it with the .mat file.
    '''
    
    def __init__(self, env_config):
        '''
        Initialize the MiniGreenhouse environment.
        
        Parameters:
        env_config(dict): Configuration dictionary for the environment.
        '''  
        
        print("MiniGreenhouse Environment initiated!")
        
        # Initialize if the main program for training or running
        self.flag_run  = env_config.get("flag_run", True) # The simulation is for running (other option is False for training)
        self.online_measurements = env_config.get("online_measurements", False) # Use online measurements or not from the IoT system 
        # or just only using offline datasets
        self.first_day = env_config.get("first_day", 1) # The first day of the simulation
        
        # Define the season length parameter
        # 20 minutes
        # But remember, the first 5 minutes is the initial values so
        # only count for the 15 minutes
        # The calculation look like this:
        # 1 / 72 * 24 [hours] * 60 [minutes / hours] = 20 minutes  
        self.season_length = env_config.get("season_length", 1 / 72) #* 3/4
        
        # Initiate and max steps
        self.max_steps = env_config.get("max_steps", 4) # How many iteration the program run
    
        # Start MATLAB engine
        self.eng = matlab.engine.start_matlab()

        # Path to MATLAB script
        # Change path based on your directory!
        self.matlab_script_path = r'C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\mini-greenhouse-drl-model\matlab\DrlGlEnvironment.m'

        # Initialize lists to store control values
        self.ventilation_list = []
        self.toplights_list = []
        self.heater_list = []
        
        # Initialize a list to store rewards
        self.rewards_list = []
        
        # Initialize reward
        reward = 0
        
        # Record the reward for the first time
        self.rewards_list.extend([reward] * 4)

        # Initialize ServiceFunctions
        self.service_functions = ServiceFunctions()

        # Check if MATLAB script exists
        if os.path.isfile(self.matlab_script_path):
            
            # Initialize control variables to zero 
            self.init_controls()
            
            # Call the MATLAB function 
            if self.online_measurements == True:
                # Initialize outdoor measurements, to get the outdoor measurements
                self.service_functions.get_outdoor_measurements()
                
                # Run the script with the updated outdoor measurements for the first time
                self.run_matlab_script('outdoor.mat', None, None)
            else:
                # Run the script with empty parameter
                self.run_matlab_script()
        
        else:
            print(f"MATLAB script not found: {self.matlab_script_path}")

        # Load the data from the .mat file
        self.load_mat_data()
        
        # Define the observation and action space
        self.define_spaces()
        
        # Initialize the state
        self.reset()
    
    def define_spaces(self):
        '''
        Define the observation and action spaces.
        
        Based on the observation of the mini-greenhouse system                          Explanation 
            - co2_in: CO2 inside the mini-greenhouse [ppm]
            - temp_in: Temperature inside the mini-greenhouse [°C]
            - rh_in: Relative humidity in percentage in the mini-greenhouse [%]
            - PAR_in: Global radiation inside the mini-greenhouse [W m^{-2}]
            - fruit_leaf: Carbohydrates in leaves [mg{CH2O} m^{-2}]                     Equation 4, 5 [2]
            - fruit_stem: Carbohydrates in stem [mg{CH2O} m^{-2}]                       Equation 6, 7 [2]
            - fruit_dw: Carbohydrates in fruit dry weight [mg{CH2O} m^{-2}]             Equation 2, 3 [2], Equation A44 [5]
            - fruit_cbuf: Carbohydrates in buffer [mg{CH2O} m^{-2}]                     Equation 1, 2 [2]
            - fruit_tcansum: Crop development stage [°C day]                            Equation 8 [2]
        
        The state x(t) variables:
        - Temperature (°C) 
        - Relative Humidity (%) 
        - CO2 Concentration (ppm) 
        - PAR Inside (W/m^2) 
        '''
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=np.array([0.0, 10.00, 0.00, 0.00, 0, 0, 0, -10.00, -10.00]), # In order: CO2, Temperature, Humidity, PAR-in, Fruit Leaf, Fruit Stem, and Fruit Dry Weight
            high=np.array([2000.0, 30.00, 90.00, 25.00, np.inf, np.inf, np.inf, np.inf, np.inf]), 
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
        
        # Append only the latest 3 values from each control variable 
        self.ventilation_list.extend(self.controls['ventilation'].flatten()[-4:])
        self.toplights_list.extend(self.controls['toplights'].flatten()[-4:])
        self.heater_list.extend(self.controls['heater'].flatten()[-4:])
        sio.savemat('controls.mat', self.controls)
        
    def run_matlab_script(self, controls_file = None, outdoor_file = None, indoor_file=None, fruit_file=None):
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
        
        if controls_file is None:
            controls_file = []

        self.eng.DrlGlEnvironment(self.season_length, self.first_day, controls_file, outdoor_file, indoor_file, fruit_file, nargout=0)

    def load_mat_data(self):
        '''
        Load data from the .mat file.
        
        From matlab, the structure is:
        
        save('drl-env.mat', 'time', 'temp_in', 'rh_in', 'co2_in', 'PAR_in', 'fruit_leaf', 'fruit_stem', 'fruit_dw');
        '''
        
        # Read the drl-env mat from the initialization 
        # Read the 3 values and append it
        data = sio.loadmat("drl-env.mat")
        
        new_time = data['time'].flatten()[-4:]
        new_co2_in = data['co2_in'].flatten()[-4:]
        new_temp_in = data['temp_in'].flatten()[-4:]
        new_rh_in = data['rh_in'].flatten()[-4:]
        new_PAR_in = data['PAR_in'].flatten()[-4:]
        new_fruit_leaf = data['fruit_leaf'].flatten()[-4:]
        new_fruit_stem = data['fruit_stem'].flatten()[-4:]
        new_fruit_dw = data['fruit_dw'].flatten()[-4:]
        new_fruit_cbuf = data['fruit_cbuf'].flatten()[-4:]
        new_fruit_tcansum = data['fruit_tcansum'].flatten()[-4:]

        if not hasattr(self, 'time'):
            self.time = new_time
            self.co2_in = new_co2_in
            self.temp_in = new_temp_in
            self.rh_in = new_rh_in
            self.PAR_in = new_PAR_in
            self.fruit_leaf = new_fruit_leaf
            self.fruit_stem = new_fruit_stem
            self.fruit_dw = new_fruit_dw
            self.fruit_cbuf = new_fruit_cbuf
            self.fruit_tcansum = new_fruit_tcansum
        else:
            self.time = np.concatenate((self.time, new_time))
            self.co2_in = np.concatenate((self.co2_in, new_co2_in))
            self.temp_in = np.concatenate((self.temp_in, new_temp_in))
            self.rh_in = np.concatenate((self.rh_in, new_rh_in))
            self.PAR_in = np.concatenate((self.PAR_in, new_PAR_in))
            self.fruit_leaf = np.concatenate((self.fruit_leaf, new_fruit_leaf))
            self.fruit_stem = np.concatenate((self.fruit_stem, new_fruit_stem))
            self.fruit_dw = np.concatenate((self.fruit_dw, new_fruit_dw))
            self.fruit_cbuf = np.concatenate((self.fruit_cbuf, new_fruit_cbuf))
            self.fruit_tcansum = np.concatenate((self.fruit_tcansum, new_fruit_tcansum))
        

    def reset(self, *, seed=None, options=None):
        '''
        Reset the environment to the initial state.
        
        Returns:
        int: The initial observation of the environment.
        '''
        self.current_step = 0
        
        return self.observation(), {}

    def observation(self):
        '''
        Get the observation of the environment for every state.
        
        Returns:
        array: The observation space of the environment.
        '''
        
        return np.array([
            self.co2_in[-1],
            self.temp_in[-1],
            self.rh_in[-1],
            self.PAR_in[-1],
            self.fruit_leaf[-1],
            self.fruit_stem[-1],
            self.fruit_dw[-1],
            self.fruit_cbuf[-1],
            self.fruit_tcansum[-1]
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
        w_r_a2              0.015           Toplights
        w_r_a3              0.001           Heater
        
        Returns:
        int: the immediate reward the agent receives at time step k in integer.
        '''
        
        # Initialize variables, based on the equation above
        # Need to be determined to make the r_k unitless
        w_r_y1 = 1          # Fruit dry weight 
        w_r_a1 = 0.005      # Ventilation
        w_r_a2 = 0.015      # Toplights
        w_r_a3 = 0.001      # Heater
        
        # Give initial reward 
        if self.current_step == 0:
            r_k = 0.0
            return r_k # No reward for the initial state 
        
        # In the createCropModel.m in the GreenLight model (mini-greenhouse-model)
        # cFruit or dry weight of fruit is the carbohydrates in fruit, so it is the best variable to count for the reward
        # Calculate the change in fruit dry weight
        delta_fruit_dw = (self.fruit_dw[-1] - self.fruit_dw[-2])
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
            os.remove('outdoor.mat')  # outdoor measurements
        
        
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
                # Delete all files
                # self.delete_files()
                return True
            
        return False

    def step(self, _action):
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

        print("ACTION: ", _action)
        
        # Convert actions to discrete values
        ventilation = 1 if _action[0] >= 0.5 else 0
        toplights = 1 if _action[1] >= 0.5 else 0
        heater = 1 if _action[2] >= 0.5 else 0
        
        print("CONVERTED ACTION")
        print("ventilation: ", ventilation)
        print("toplights: ", toplights)
        print("heating: ", heater)

        # TO-DO: Refactor this so the time_steps is not reset from beginning but increment it
        
        time_steps = np.linspace(300, 1200, 4)  # Time steps in seconds
        ventilation = np.full(4, ventilation)
        toplights = np.full(4, toplights)
        heater = np.full(4, heater)

        # Keep only the latest 3 data points before appending
        # Append controls to the lists
        self.ventilation_list.extend(ventilation[-4:])
        self.toplights_list.extend(toplights[-4:])
        self.heater_list.extend(heater[-4:])
        
        # Only publish MQTT data for the Raspberry Pi when running not training
        if self.online_measurements == True:
            # Format data controls in JSON format
            json_data = self.service_functions.format_data_in_JSON(time_steps, \
                                                ventilation, toplights, \
                                                heater)
            
            # Publish controls to the raspberry pi (IoT system client)
            self.service_functions.publish_mqtt_data(json_data)

        # Create control dictionary
        controls = {
            'time': time_steps.reshape(-1, 1),
            'ventilation': ventilation.reshape(-1, 1),
            'toplights': toplights.reshape(-1, 1),
            'heater': heater.reshape(-1, 1)
        }
        
        # Save control variables to .mat file
        sio.savemat('controls.mat', controls)
        
        # Update the season_length and first_day
        # 1 / 72 is 20 minutes in 24 hours, the calculation look like this
        # 1 / 72 * 24 [hours] * 60 [minutes . hours ^ -1] = 20 minutes 
        self.season_length = 1 / 72 
        self.first_day += 1 / 72 

        # Convert co2_in ppm
        co2_density = self.service_functions.co2ppm_to_dens(self.temp_in[-4:], self.co2_in[-4:])
        
        # Convert Relative Humidity (RH) to Pressure in Pa
        vapor_density = self.service_functions.rh_to_vapor_density(self.temp_in[-4:], self.rh_in[-4:])
        vapor_pressure = self.service_functions.vapor_density_to_pressure(self.temp_in[-4:], vapor_density)

        # Update the MATLAB environment with the 3 latest current state
        # It will be used to be simulated in the GreenLight model with mini-greenhouse parameters
        drl_indoor = {
            'time': self.time[-3:].astype(float).reshape(-1, 1),
            'temp_in': self.temp_in[-3:].astype(float).reshape(-1, 1),
            'rh_in': vapor_pressure[-3:].astype(float).reshape(-1, 1),
            'co2_in': co2_density[-3:].astype(float).reshape(-1, 1)
        }
        
        # Save control variables to .mat file
        sio.savemat('indoor.mat', drl_indoor)

        # Update the fruit growth with the 1 latest current state from the GreenLight model - mini-greenhouse parameters
        fruit_growth = {
            'time': self.time[-1:].astype(float).reshape(-1, 1),
            'fruit_leaf': self.fruit_leaf[-1:].astype(float).reshape(-1, 1),
            'fruit_stem': self.fruit_stem[-1:].astype(float).reshape(-1, 1),
            'fruit_dw': self.fruit_dw[-1:].astype(float).reshape(-1, 1),
            'fruit_cbuf': self.fruit_cbuf[-1:].astype(float).reshape(-1, 1),
            'fruit_tcansum': self.fruit_tcansum[-1:].astype(float).reshape(-1, 1)
        }

        # Save the fruit growth to .mat file
        sio.savemat('fruit.mat', fruit_growth)
        
        if self.online_measurements == True:
            # Get the outdoor measurements
            self.service_functions.get_outdoor_measurements()

        # Run the script with the updated state variables
        if self.online_measurements == True:
            self.run_matlab_script('controls.mat', 'outdoor.mat', 'indoor.mat', 'fruit.mat')
        else:
            self.run_matlab_script(None, None, 'indoor.mat', 'fruit.mat')
        
        # Load the updated data from the .mat file
        self.load_mat_data()
        
        # Calculate reward
        # Remember that the actions become a list, but we only need the first actions from 15 minutes (all of the is the same)
        _reward = self.get_reward(ventilation[0], toplights[0], heater[0])
        
        # Record the reward
        self.rewards_list.extend([_reward] * 4)

        # Truncated flag
        truncated = False
    
        return self.observation(), _reward, self.done(), truncated, {}
    
    def print_and_save_all_data(self, _file_name):
        '''
        Print all the appended data.
        '''
        print("")
        print("")
        print("-------------------------------------------------------------------------------------")
        print("Print all the appended data.")
        # Print lengths of each list to identify discrepancies
        print(f"Length of Time: {len(self.time)}")
        print(f"Length of CO2 In: {len(self.co2_in)}")
        print(f"Length of Temperature In: {len(self.temp_in)}")
        print(f"Length of RH In: {len(self.rh_in)}")
        print(f"Length of PAR In: {len(self.PAR_in)}")
        #print(f"Length of Fruit leaf: {len(self.fruit_leaf)}")
        #print(f"Length of Fruit stem: {len(self.fruit_stem)}")
        print(f"Length of Fruit Dry Weight: {len(self.fruit_dw)}")
        print(f"Length of Fruit cBuf: {len(self.fruit_cbuf)}")
        print(f"Length of Fruit tCanSum: {len(self.fruit_tcansum)}")
        print(f"Length of Ventilation: {len(self.ventilation_list)}")
        print(f"Length of toplights: {len(self.toplights_list)}")
        print(f"Length of Heater: {len(self.heater_list)}")
        print(f"Length of Rewards: {len(self.rewards_list)}")
        data = {
            'Time': self.time,
            'CO2 In': self.co2_in,
            'Temperature In': self.temp_in,
            'RH In': self.rh_in,
            'PAR In': self.PAR_in,
            #'Fruit leaf': self.fruit_leaf,
            #'Fruit stem': self.fruit_stem,
            'Fruit Dry Weight': self.fruit_dw,
            'Fruit cBuf':self.fruit_cbuf,
            'Fruit tCanSum': self.fruit_tcansum,
            'Ventilation': self.ventilation_list,
            'Toplights': self.toplights_list,
            'Heater': self.heater_list,
            'Rewards': self.rewards_list
        }
        
        df = pd.DataFrame(data)
        print(df)
        
        # Calculate time_steps for plot and save it on the excel file
        time_max = (self.max_steps + 1) * 1200 # for e.g. 3 steps * 1200 (20 minutes) = 60 minutes
        time_steps_seconds = np.linspace(300, time_max, (self.max_steps + 1)  * 4)  # Time steps in seconds
        time_steps_hours = time_steps_seconds / 3600  # Convert seconds to hours
        time_steps_formatted = [str(timedelta(hours=h))[:-3] for h in time_steps_hours]  # Format to HH:MM
        print("time_steps_plot (in HH:MM format):", time_steps_formatted)
        
        # Save all the data in an excel file
        self.service_functions.export_to_excel(_file_name, time_steps_formatted, self.co2_in, self.temp_in, self.rh_in,
                                               self.PAR_in, self.fruit_leaf, self.fruit_stem,
                                               self.fruit_dw, self.fruit_cbuf, self.fruit_tcansum, self.ventilation_list, self.toplights_list,
                                               self.heater_list, self.rewards_list)

        self.service_functions.plot_all_data(self.max_steps, time_steps_formatted, self.co2_in, self.temp_in, self.rh_in,
                                             self.PAR_in, self.fruit_leaf, self.fruit_stem,
                                             self.fruit_dw, self.fruit_cbuf, self.fruit_tcansum, self.ventilation_list, self.toplights_list,
                                             self.heater_list, self.rewards_list)

    # Ensure to properly close the MATLAB engine when the environment is no longer used
    def __del__(self):
        self.eng.quit()
        

