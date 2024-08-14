'''
Service Functions for Deep Reinforcement Learning mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.efraimpartoginahotasi@wur.nl
efraim.manurung@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

import json
import paho.mqtt.client as mqtt

class ServiceFunctions:
    def __init__(self):
        print("Service Functions initiated!")
        
        # Initiate the MQTT client for publishing data
        self.client_pub = mqtt.Client()
        
        # Initialize the MQTT client for subscribing data
        self.client_sub = mqtt.Client(client_id="", protocol=mqtt.MQTTv5)
        
        self.message_received = False  # Initialize message_received flag
        
    def co2ppm_to_dens(self, _temp, _ppm):
        '''
        co2ppm_to_dens Convert CO2 molar concetration [ppm] to density [kg m^{-3}]
        
        Usage:
            co2_density = co2ppm_to_dens(temp, ppm) 
        Inputs:
           temp        given temperatures [°C] (numeric vector)
           ppm         CO2 concetration in air (ppm) (numeric vector)
           Inputs should have identical dimensions
         Outputs:
           co2Dens     CO2 concentration in air [mg m^{-3}] (numeric vector)
        
         Calculation based on ideal gas law pV=nRT, with pressure at 1 atm

        Based on the GreenLight model
        '''
        
        R = 8.3144598; # molar gas constant [J mol^{-1} K^{-1}]
        C2K = 273.15; # conversion from Celsius to Kelvin [K]
        M_CO2 = 44.01e-3; # molar mass of CO2 [kg mol^-{1}]
        P = 101325; # pressure (assumed to be 1 atm) [Pa]
        
        # Ensure inputs are numpy arrays for element-wise operations
        _temp = np.array(_temp)
        _ppm = np.array(_ppm)
        
        # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
        # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm   
        # co2Dens = P*10^-6*ppm*M_CO2./(R*(temp+C2K)); 
        _co2_density = P * 10**-6 * _ppm * M_CO2 / (R * (_temp + C2K)) * 1e6
        
        return _co2_density
    
    def rh_to_vapor_density(self, _temp, _rh):
        '''
        Convert relative humidity [%] to vapor density [kg{H2O} m^{-3}]
        
        Usage:
            vaporDens = rh2vaporDens(temp, rh)
        Inputs:
            temp        given temperatures [°C] (numeric vector)
            rh          relative humidity [%] between 0 and 100 (numeric vector)
            Inputs should have identical dimensions
        Outputs:
            vaporDens   absolute humidity [kg{H2O} m^{-3}] (numeric vector)
        
        Calculation based on 
            http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
        '''
        
        # constants
        R = 8.3144598  # molar gas constant [J mol^{-1} K^{-1}]
        C2K = 273.15  # conversion from Celsius to Kelvin [K]
        Mw = 18.01528e-3  # molar mass of water [kg mol^{-1}]
        
        # parameters used in the conversion
        p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        
        # Ensure inputs are numpy arrays for element-wise operations
        _temp = np.array(_temp)
        _rh = np.array(_rh)
        
        # Saturation vapor pressure of air in given temperature [Pa]
        satP = p[0] * np.exp(p[2] * _temp / (_temp + p[1]))
        
        # Partial pressure of vapor in air [Pa]
        pascals = (_rh / 100.0) * satP
        
        # convert to density using the ideal gas law pV=nRT => n=pV/RT 
        # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
        # number of kg in a m^3, where Mw is the molar mass of water.
        vaporDens = pascals * Mw / (R * (_temp + C2K))
        
        return vaporDens
    
    
    def vapor_density_to_pressure(self, _temp, _vaporDens):
        '''
        Convert vapor density [kg{H2O} m^{-3}] to vapor pressure [Pa]
        
        Usage:
            vaporPres = vaporDens2pres(temp, vaporDens)
        Inputs:
            temp        given temperatures [°C] (numeric vector)
            vaporDens   vapor density [kg{H2O} m^{-3}] (numeric vector)
            Inputs should have identical dimensions
        Outputs:
            vaporPres   vapor pressure [Pa] (numeric vector)
        
        Calculation based on 
            http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
        '''
        
        # parameters used in the conversion
        p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        
        # Ensure inputs are numpy arrays for element-wise operations
        _temp = np.array(_temp)
        _vaporDens = np.array(_vaporDens)
        
        # Convert relative humidity from vapor density
        _rh = _vaporDens / self.rh_to_vapor_density(_temp, 100)  # relative humidity [0-1]
        
        # Saturation vapor pressure of air in given temperature [Pa]
        _satP = p[0] * np.exp(p[2] * _temp / (_temp + p[1]))
        
        vaporPres = _satP * _rh
        
        return vaporPres

    def plot_rewards_actions(self, filename, time, ventilation_list, toplights_list, heater_list, reward_list):
        '''
        Plot the rewards and actions.

        Parameters:
        - filename: Name of the output Excel file
        
        - time: List of time values
        
        - ventilation_list: List of actions for ventilation from DRL model or offline datasets
        - toplights_list: List of actions for toplights from DRL model or offline datasets
        - heater_list: List of actions for heater from DRL model or offline datasets
        
        - reward_list: List of rewards for each iterated step
        '''

        # Create subplots with 2 rows and 2 columns
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Data to be plotted along with their titles
        data = [
            (ventilation_list, 'Ventilation Action [-]'),
            (toplights_list, 'Toplights Action [-]'),
            (heater_list, 'Heater Action [-]'),
            (reward_list, 'Reward [-]')
        ]

        # Plot each dataset in a subplot
        for ax, (y_data, title) in zip(axes.flatten(), data):
            ax.plot(time, y_data, label=title, color='blue')  # Plot the data
            
            ax.set_title(title)  # Set the title for each subplot
            ax.set_xlabel('Timesteps [- / 5 minutes]')  # Set the x-axis label
            ax.set_ylabel(title)  # Set the y-axis label
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.legend()  # Add legend to each subplot

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()

        # Save the plot to a file
        fig.savefig(filename)
    
    def plot_all_data(self, filename, time, co2_actual, temp_actual, rh_actual, par_actual, 
                  co2_predicted_nn, temp_predicted_nn, rh_predicted_nn, par_predicted_nn,
                  co2_predicted_gl, temp_predicted_gl, rh_predicted_gl, par_predicted_gl,
                  metrics_nn, metrics_gl):
        '''
        Plot all the parameters to make it easier to compare predicted vs actual values.

        Parameters:
        - time: List of time values
        
        - co2_actual: List of actual CO2 values
        - temp_actual: List of actual temperature values
        - rh_actual: List of actual relative humidity values
        - par_actual: List of actual PAR values
        
        - co2_predicted_nn: List of predicted CO2 values from Neural Network
        - temp_predicted_nn: List of predicted temperature values from Neural Network
        - rh_predicted_nn: List of predicted relative humidity values from Neural Network
        - par_predicted_nn: List of predicted PAR values from Neural Network
        
        - co2_predicted_gl: List of predicted CO2 values from Generalized Linear Model
        - temp_predicted_gl: List of predicted temperature values from Generalized Linear Model
        - rh_predicted_gl: List of predicted relative humidity values from Generalized Linear Model
        - par_predicted_gl: List of predicted PAR values from Generalized Linear Model
        
        - metrics_nn: Dictionary with R² and MAE for NN predictions
        - metrics_gl: Dictionary with R² and MAE for GL predictions
        '''

        # Create subplots with 2 rows and 2 columns
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

        # Data to be plotted along with their titles and predicted data
        data = [
            (co2_actual, co2_predicted_nn, co2_predicted_gl, 'CO2 In [ppm]', metrics_nn['CO2'], metrics_gl['CO2']),
            (temp_actual, temp_predicted_nn, temp_predicted_gl, 'Temperature In [°C]', metrics_nn['Temperature'], metrics_gl['Temperature']),
            (rh_actual, rh_predicted_nn, rh_predicted_gl, 'RH In [%]', metrics_nn['Humidity'], metrics_gl['Humidity']),
            (par_actual, par_predicted_nn, par_predicted_gl, 'PAR In [W/m2]', metrics_nn['PAR'], metrics_gl['PAR'])
        ]

        # Plot each dataset in a subplot
        for ax, (y_actual, y_pred_nn, y_pred_gl, title, metrics_nn, metrics_gl) in zip(axes.flatten(), data):
            ax.plot(time, y_actual, label='Actual', color='blue')  # Plot actual data
            ax.plot(time, y_pred_nn, label='Predicted NN', color='red', linestyle='--')  # Plot NN predicted data
            ax.plot(time, y_pred_gl, label='Predicted GL', color='green', linestyle=':')  # Plot GL predicted data
            
            # Add R² and MAE to the title
            ax.set_title(f"{title}\nNN R²: {metrics_nn[0]:.4f}, MAE: {metrics_nn[1]:.4f}\nGL R²: {metrics_gl[0]:.4f}, MAE: {metrics_gl[1]:.4f}")
            
            ax.set_xlabel('Timesteps [- / 5 minutes]')  # Set the x-axis label
            ax.set_ylabel(title)  # Set the y-axis label
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.legend()  # Add legend to distinguish between actual and predicted data

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
        # Save the plot to a file
        fig.savefig(filename)

    def export_to_excel(self, filename, time, ventilation_list, toplights_list, heater_list, reward_list,
                    co2_actual, temp_actual, rh_actual, par_actual,
                    co2_predicted_nn, temp_predicted_nn, rh_predicted_nn, par_predicted_nn,
                    co2_predicted_gl, temp_predicted_gl, rh_predicted_gl, par_predicted_gl):
        '''
        Export all the appended data to an Excel file including both actual and predicted values.

        Parameters:
        - file_name: Name of the output Excel file
        
        - ventilation_list: List of action for fan/ventilation from DRL model or offline datasets
        - toplights_list: List of action for toplights from DRL model or offline datasets
        - heater_list: List of action for heater from DRL model or offline datasets
        - reward_list: List of reward for iterated step
        
        - co2_in_excel: List of actual CO2 values
        - temp_in_excel: List of actual temperature values
        - rh_in_excel: List of actual relative humidity values
        - par_in_excel: List of actual PAR values
        
        - co2_in_predicted_nn: List of predicted CO2 values from Neural Network
        - temp_in_predicted_nn: List of predicted temperature values from Neural Network
        - rh_in_predicted_nn: List of predicted relative humidity values from Neural Network
        - par_in_predicted_nn: List of predicted PAR values from Neural Network
        
        - co2_in_predicted_gl: List of predicted CO2 values from Generalized Linear Model
        - temp_in_predicted_gl: List of predicted temperature values from Generalized Linear Model
        - rh_in_predicted_gl: List of predicted relative humidity values from Generalized Linear Model
        - par_in_predicted_gl: List of predicted PAR values from Generalized Linear Model
        '''

        data = {
            'Timesteps [- / 5 minutes]': time,
            'Action Ventilation': ventilation_list,
            'Action Toplights': toplights_list,
            'Action Heater': heater_list,
            'Rewards': reward_list,
            'CO2 In (Actual)': co2_actual,
            'CO2 In (Predicted NN)': co2_predicted_nn,
            'CO2 In (Predicted GL)': co2_predicted_gl,
            'Temperature In (Actual)': temp_actual,
            'Temperature In (Predicted NN)': temp_predicted_nn,
            'Temperature In (Predicted GL)': temp_predicted_gl,
            'RH In (Actual)': rh_actual,
            'RH In (Predicted NN)': rh_predicted_nn,
            'RH In (Predicted GL)': rh_predicted_gl,
            'PAR In (Actual)': par_actual,
            'PAR In (Predicted NN)': par_predicted_nn,
            'PAR In (Predicted GL)': par_predicted_gl,
        }

        # Check if all lists have the same length
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All arrays must be of the same length")

        # Create a DataFrame and export to Excel
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"Data successfully exported to {filename}")

    def format_data_in_JSON(self, time, ventilation, toplights, heater):
        '''
        Convert data to JSON format and print it.
        
        Parameters:
        - time: List of time values
        - ventilation: List of ventilation control values
        - toplights: List of toplights control values
        - heater: List of heater control values
        '''
        
        def convert_to_native(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.int32, np.int64, np.float32, np.float64)):
                return value.item()
            else:
                return value

        data = {
            "time": [convert_to_native(v) for v in time],
            "ventilation": [convert_to_native(v) for v in ventilation],
            "toplights": [convert_to_native(v) for v in toplights],
            "heater": [convert_to_native(v) for v in heater]
        }

        json_data = json.dumps(data, indent=4)
        
        # For debugging JSON data
        # print("JSON DATA: ", json_data)
        return json_data
    
    def publish_mqtt_data(self, json_data, broker="192.168.1.131", port=1883, topic="greenhouse-iot-system/drl-controls"):
        '''
        Publish JSON data to an MQTT broker.
        
        Parameters:
        - json_data: JSON formatted data to publish
        - broker: MQTT broker address
        - port: MQTT broker port
        - topic: MQTT topic to publish data to
        '''
        
        def on_connect(client, userdata, flags, rc):
            print("Connected with result code PUBLISH MQTT " + str(rc))
            client.publish(topic, str(json_data))
        
        self.client_pub.on_connect = on_connect
        
        self.client_pub.connect(broker, port, 60)
        self.client_pub.loop_start()

    def get_outdoor_measurements(self, broker="192.168.1.131", port=1883, topic="greenhouse-iot-system/outdoor-measurements"):
        '''
        Initialize outdoor measurements.
        
        Subscribe JSON data from a MQTT broker.
        
        Parameters:
        - json_data: JSON formatted data to publish
        - broker: MQTT broker address
        - port: MQTT broker port
        - topic: MQTT topic to publish data to
        '''

        def on_connect(client, userdata, flags, reason_code, properties):
            print("Connected with result code SUBSCRIBE MQTT " + str(reason_code))
            client.subscribe(topic)
            
        def on_message(client, userdata, msg):
            # print(msg.topic + " " + str(msg.payload.decode()))
            # Parse the JSON data
            data = json.loads(msg.payload.decode())
            
            # Process the received data
            # Change the matlab file in here
            self.process_received_data(data) 
        
            # Set the flag to indicate a message was received
            self.message_received = True
            self.client_sub.loop_stop()  # Stop the loop
        
        self.message_received = False # Reset message_received flag
        self.client_sub.on_connect = on_connect
        self.client_sub.on_message = on_message

        self.client_sub.connect(broker, port, 60)
        self.client_sub.loop_start()  # Start the loop in a separate thread
    
        # Wait for a message to be received
        while not self.message_received:
            continue
        
        self.client_sub.loop_stop()  # Ensure the loop is stopped
        self.client_sub.disconnect()  # Disconnect the client
        return True

    def process_received_data(self, data):
        '''
        Process the outdoor measurements and .
        
        Outdoor measurements:
        - time: from main loop iteration in 1 s
        - lux: Need to be converted to W / m^2
        - temperature
        - humidity
        - co2
        '''
        
        # Extract variables
        time = data.get("time", [])
        lux = data.get("lux", [])
        temp = data.get("temperature", [])
        hum = data.get("humidity", [])
        co2 = data.get("co2", [])
        
        # Print the extracted variables
        print("Time:", time)
        print("Lux:", lux)
        print("Temperature:", temp)
        print("Humidity:", hum)
        print("CO2:", co2)
        
        # Create outdoor measurements dictionary
        outdoor_measurements = {
            'time': np.array(time).reshape(-1, 1),
            'lux': np.array(lux).reshape(-1, 1),
            'temperature': np.array(temp).reshape(-1, 1),
            'humidity': np.array(hum).reshape(-1, 1),
            'co2': np.array(co2).reshape(-1, 1)
        }
        
        # Save outdoor measurements to .mat file
        sio.savemat('outdoor.mat', outdoor_measurements)