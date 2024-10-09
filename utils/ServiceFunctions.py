'''
Service Functions for Deep Reinforcement Learning mini-greenhouse 

Author: Efraim Manurung
MSc Thesis in Information Technology Group, Wageningen University

efraim.manurung@gmail.com
'''

# Import libraries
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
    
    def plot_all_data(self, filename, time, co2_actual, temp_actual, rh_actual, par_actual, 
                    co2_predicted_dnn, temp_predicted_dnn, rh_predicted_dnn, par_predicted_dnn, 
                    co2_predicted_gl, temp_predicted_gl, rh_predicted_gl, par_predicted_gl, 
                    co2_combined=None, temp_combined=None, rh_combined=None, par_combined=None, 
                    metrics_dnn=None, metrics_gl=None, metrics_combined=None):
        '''
        Plot all the parameters to make it easier to compare predicted vs actual values, including Leaf Temperature.

        Parameters:
        - time: List of time values
        
        - co2_actual: List of actual CO2 values
        - temp_actual: List of actual temperature values
        - rh_actual: List of actual relative humidity values
        - par_actual: List of actual PAR values
        - leaf_temp_actual: List of actual leaf temperature values
        
        - co2_predicted_dnn: List of predicted CO2 values from Neural Network
        - temp_predicted_dnn: List of predicted temperature values from Neural Network
        - rh_predicted_dnn: List of predicted relative humidity values from Neural Network
        - par_predicted_dnn: List of predicted PAR values from Neural Network
        - leaf_temp_predicted_dnn: List of predicted leaf temperature values from Neural Network
        
        - co2_predicted_gl: List of predicted CO2 values from GreenLight Model
        - temp_predicted_gl: List of predicted temperature values from GreenLight Model
        - rh_predicted_gl: List of predicted relative humidity values from GreenLight Model
        - par_predicted_gl: List of predicted PAR values from GreenLight Model
        - leaf_temp_predicted_gl: List of predicted leaf temperature values from GreenLight Model
        
        - co2_combined: List of combined CO2 predictions (optional)
        - temp_combined: List of combined temperature predictions (optional)
        - rh_combined: List of combined relative humidity predictions (optional)
        - par_combined: List of combined PAR predictions (optional)
        - leaf_temp_combined: List of combined leaf temperature predictions (optional)
        
        - metrics_dnn: Dictionary with RMSE, RRMSE, and ME for DNN predictions (optional)
        - metrics_gl: Dictionary with RMSE, RRMSE, and ME for GL predictions (optional)
        - metrics_combined: Dictionary with RMSE, RRMSE, and ME for Combined predictions (optional)
        '''
        
        # Create subplots with 2 rows and 2 columns for 4 parameters (CO2, Temperature, RH, PAR)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

        # Data to be plotted along with their titles and predicted data
        data = [
            (co2_actual, co2_predicted_dnn, co2_predicted_gl, co2_combined, 'CO2 In [ppm]',
            metrics_dnn['CO2'] if metrics_dnn else None,
            metrics_gl['CO2'] if metrics_gl else None,
            metrics_combined['CO2'] if metrics_combined else None),
            (temp_actual, temp_predicted_dnn, temp_predicted_gl, temp_combined, 'Temperature In [°C]',
            metrics_dnn['Temperature'] if metrics_dnn else None,
            metrics_gl['Temperature'] if metrics_gl else None,
            metrics_combined['Temperature'] if metrics_combined else None),
            (rh_actual, rh_predicted_dnn, rh_predicted_gl, rh_combined, 'RH In [%]',
            metrics_dnn['Humidity'] if metrics_dnn else None,
            metrics_gl['Humidity'] if metrics_gl else None,
            metrics_combined['Humidity'] if metrics_combined else None),
            (par_actual, par_predicted_dnn, par_predicted_gl, par_combined, 'PAR In [W/m²]',
            metrics_dnn['PAR'] if metrics_dnn else None,
            metrics_gl['PAR'] if metrics_gl else None,
            metrics_combined['PAR'] if metrics_combined else None)
        ]

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Plot each dataset in a subplot
        for ax, (y_actual, y_pred_dnn, y_pred_gl, y_combined, title, dnn_metrics, gl_metrics, combined_metrics) in zip(axes_flat, data):
            # Plot actual data if it is not None
            if y_actual is not None:
                ax.plot(time, y_actual, label='Actual', color='blue')
            
            # Plot predicted data
            ax.plot(time, y_pred_dnn, label='Predicted dnn', color='purple', linestyle='--')
            ax.plot(time, y_pred_gl, label='Predicted GL', color='green', linestyle=':')
            
            # Only plot combined data if it's not None
            if y_combined is not None:
                ax.plot(time, y_combined, label='Predicted Combined', color='red', linestyle='-.')

            # Add RMSE, RRMSE, and ME to the title if metrics are available
            if dnn_metrics and gl_metrics and combined_metrics:
                ax.set_title(f"{title}\n"
                            f"DNN RMSE: {dnn_metrics[0]:.4f}, RRMSE: {dnn_metrics[1]:.4f}, ME: {dnn_metrics[2]:.4f}\n"
                            f"GreenLight RMSE: {gl_metrics[0]:.4f}, RRMSE: {gl_metrics[1]:.4f}, ME: {gl_metrics[2]:.4f}\n"
                            f"Combined RMSE: {combined_metrics[0]:.4f}, RRMSE: {combined_metrics[1]:.4f}, ME: {combined_metrics[2]:.4f}")
            else:
                ax.set_title(f"{title}")

            ax.set_xlabel('Timesteps [5 minutes]')
            ax.set_ylabel(title)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
        # Save the plot to a file
        fig.savefig(filename, dpi=500)
    
    def plot_leaf_temperature(self, filename, time, leaf_temp_actual, leaf_temp_predicted_dnn, leaf_temp_predicted_gl, leaf_temp_combined=None,
                            metrics_dnn=None, metrics_gl=None, metrics_combined=None):
        '''
        Plot leaf_temperature parameter to make it easier to compare predicted vs actual values.

        Parameters:
        - time: List of time values
        - leaf_temp_actual: List of actual leaf temperature values
        - leaf_temp_predicted_dnn: List of predicted leaf temperature values from Neural Network
        - leaf_temp_predicted_gl: List of predicted leaf temperature values from GreenLight Model
        - leaf_temp_combined: List of combined leaf temperature predictions (optional)
        - metrics_dnn: Dictionary with RMSE, RRMSE, and ME for DNN predictions (optional)
        - metrics_gl: Dictionary with RMSE, RRMSE, and ME for GL predictions (optional)
        - metrics_combined: Dictionary with RMSE, RRMSE, and ME for Combined predictions (optional)
        '''

        # Create a single figure with size 9x6
        fig = plt.figure(figsize=(9, 6))

        # Plot actual leaf temperature
        if leaf_temp_actual is not None:
            plt.plot(time, leaf_temp_actual, label='Actual', color='blue')

        # Plot predicted leaf temperature from DNN
        plt.plot(time, leaf_temp_predicted_dnn, label='Predicted DNN', color='purple', linestyle='--')

        # Plot predicted leaf temperature from GreenLight model
        plt.plot(time, leaf_temp_predicted_gl, label='Predicted GL', color='green', linestyle=':')

        # Plot combined leaf temperature predictions if not None
        if leaf_temp_combined is not None:
            plt.plot(time, leaf_temp_combined, label='Predicted Combined', color='red', linestyle='-.')

        # Prepare the title, including metrics if available
        title = 'Leaf Temperature In [°C]'
        if metrics_dnn and metrics_gl and metrics_combined:
            title += (f"\nDNN RMSE: {metrics_dnn[0]:.4f}, RRMSE: {metrics_dnn[1]:.4f}%, ME: {metrics_dnn[2]:.4f}°C\n"
                    f"GL RMSE: {metrics_gl[0]:.4f}, RRMSE: {metrics_gl[1]:.4f}%, ME: {metrics_gl[2]:.4f}°C\n"
                    f"Combined RMSE: {metrics_combined[0]:.4f}, RRMSE: {metrics_combined[1]:.4f}%, ME: {metrics_combined[2]:.4f}°C")
                
        # Set title with or without metrics
        plt.title(title)

        # Add axis labels
        plt.xlabel('Timesteps [5 minutes]')
        plt.ylabel('Leaf Temperature In [°C]')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add legend to the plot
        plt.legend()

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

        # Save the plot to a file
        fig.savefig(filename, dpi=500)

    def export_to_excel(self, filename, time, ventilation_list, toplights_list, heater_list, reward_list,
                        co2_actual=None, temp_actual=None, rh_actual=None, par_actual=None, leaf_temp_actual=None,
                        co2_predicted_dnn=None, temp_predicted_dnn=None, rh_predicted_dnn=None, par_predicted_dnn=None, leaf_temp_predicted_dnn=None,
                        co2_predicted_gl=None, temp_predicted_gl=None, rh_predicted_gl=None, par_predicted_gl=None, leaf_temp_predicted_gl=None,
                        co2_predicted_combined=None, temp_predicted_combined=None, rh_predicted_combined=None, par_predicted_combined=None, leaf_temp_predicted_combined=None):
        '''
        Export all the appended data to an Excel file including both actual and predicted values.

        Parameters:
        - file_name: Name of the output Excel file
        
        - ventilation_list: List of action for fan/ventilation from DRL model or offline datasets
        - toplights_list: List of action for toplights from DRL model or offline datasets
        - heater_list: List of action for heater from DRL model or offline datasets
        - reward_list: List of reward for iterated step
        
        - co2_actual: List of actual CO2 values (optional)
        - temp_actual: List of actual temperature values (optional)
        - rh_actual: List of actual relative humidity values (optional)
        - par_actual: List of actual PAR values (optional)
        - leaf_temp_actual: List of actual leaf temperature (optional)
        
        - co2_predicted_dnn: List of predicted CO2 values from Neural Network
        - temp_predicted_dnn: List of predicted temperature values from Neural Network
        - rh_predicted_dnn: List of predicted relative humidity values from Neural Network
        - par_predicted_dnn: List of predicted PAR values from Neural Network
        - leaf_temp_predicted_dnn: List of predicted leaf temperature values from Neural Network
        
        - co2_predicted_gl: List of predicted CO2 values from the GreenLight Model
        - temp_predicted_gl: List of predicted temperature values from the GreenLight Model
        - rh_predicted_gl: List of predicted relative humidity values from the GreenLight Model
        - par_predicted_gl: List of predicted PAR values from the GreenLight Model
        - leaf_temp_predicted_gl: List of predicted leaf tempearture values from the GreenLight Model
        
        - co2_predicted_combined: List of combined predicted CO2 values
        - temp_predicted_combined: List of combined predicted temperature values
        - rh_predicted_combined: List of combined predicted relative humidity values
        - par_predicted_combined: List of combined predicted PAR values
        - leaf_temp_combined: List of combined predicted temperature values
        '''

        # Prepare the data dictionary with always-included columns
        data = {
            'Timesteps [5 minutes]': time,
            'Action Ventilation': ventilation_list,
            'Action Toplights': toplights_list,
            'Action Heater': heater_list,
            'Rewards': reward_list
        }
        
        if co2_predicted_dnn is not None:
            data['CO2 In (Predicted DNN)'] = co2_predicted_dnn
        if co2_predicted_gl is not None:
            data['CO2 In (Predicted GL)'] = co2_predicted_gl
        if co2_predicted_combined is not None:
            data['CO2 In (Predicted Combined)'] = co2_predicted_combined
        if co2_actual is not None:
            data['CO2 In (Actual)'] = co2_actual
            
        if temp_predicted_dnn is not None:
            data['Temperature In (Predicted DNN)'] = temp_predicted_dnn
        if temp_predicted_gl is not None:
            data['Temperature In (Predicted GL)'] = temp_predicted_gl
        if temp_predicted_combined is not None:
            data['Temperature In (Predicted Combined)'] = temp_predicted_combined
        if temp_actual is not None:
            data['Temperature In (Actual)'] = temp_actual
            
        if rh_predicted_dnn is not None:
            data['RH In (Predicted DNN)'] = rh_predicted_dnn
        if rh_predicted_gl is not None:
            data['RH In (Predicted GL)'] = rh_predicted_gl
        if rh_predicted_combined is not None:
            data['RH In (Predicted Combined)'] = rh_predicted_combined
        if rh_actual is not None:
            data['RH In (Actual)'] = rh_actual
            
        if par_predicted_dnn is not None:
            data['PAR In (Predicted DNN)'] = par_predicted_dnn
        if par_predicted_gl is not None:
            data['PAR In (Predicted GL)'] = par_predicted_gl
        if par_predicted_combined is not None:
            data['PAR In (Predicted Combined)'] = par_predicted_combined
        if par_actual is not None:
            data['PAR In (Actual)'] = par_actual
            
        if leaf_temp_predicted_dnn is not None:
            data['Leaf Temp (Predicted DNN)'] = leaf_temp_predicted_dnn
        if leaf_temp_predicted_gl is not None:
            data['Leaf Temp (Predicted GL)'] = leaf_temp_predicted_gl
        if leaf_temp_predicted_combined is not None:
            data['Leaf Temp (Predicted Combined)'] = leaf_temp_predicted_combined
        if leaf_temp_actual is not None:
            data['Leaf Temp (Actual)'] = leaf_temp_actual
            
        # Check if all lists have the same length
        lengths = [len(v) for v in data.values() if v is not None]  # Skip None values in length check
        if len(set(lengths)) != 1:
            raise ValueError("All arrays must be of the same length")

        # Create a DataFrame and export to Excel
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"Data successfully exported to {filename}")

    def plot_actions(self, filename, time, ventilation_list, toplights_list, heater_list):
        '''
        Plot the actions.

        Parameters:
        - filename: Name of the output file
        
        - time: List of time values
        
        - ventilation_list: List of actions for ventilation from DRL model or offline datasets
        - toplights_list: List of actions for toplights from DRL model or offline datasets
        - heater_list: List of actions for heater from DRL model or offline datasets
        '''

        # Create subplots
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 6))

        # Data to be plotted along with their titles
        data = [
            (ventilation_list, 'Ventilation Action [-]'),
            (toplights_list, 'Toplights Action [-]'),
            (heater_list, 'Heater Action [-]'),
        ]

        # Plot each dataset in a subplot
        for ax, (y_data, title) in zip(axes, data):
            ax.plot(time, y_data, label=title, color='green')  # Plot the data
            
            # ax.set_title(title)  # Set the title for each subplot
            ax.set_xlabel('Timesteps [5 minutes]')  # Set the x-axis label
            ax.set_ylabel(title)  # Set the y-axis label
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.legend()  # Add legend to each subplot

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()

        # Save the plot to a file
        fig.savefig(filename, dpi=500)
    
    def plot_rewards(self, filename, time, rewards_list):
        '''
        Plot the rewards and cumulative rewards.

        Parameters:
        - filename: Name of the output file
        
        - time: List of time values
        
        - rewards_list: List of rewards per timestep
        '''

        # Calculate cumulative rewards
        cumulative_rewards = [sum(rewards_list[:i+1]) for i in range(len(rewards_list))]

        # Create subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 8))

        # Plot rewards per timestep
        axes[0].plot(time, rewards_list, label='Rewards per Timestep', color='blue')
        axes[0].set_title('Rewards per Timestep')
        axes[0].set_xlabel('Timesteps [5 minutes]')
        axes[0].set_ylabel('Reward')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend()

        # Plot cumulative rewards
        axes[1].plot(time, cumulative_rewards, label='Cumulative Rewards', color='green')
        axes[1].set_title('Cumulative Rewards')
        axes[1].set_xlabel('Timesteps [5 minutes]')
        axes[1].set_ylabel('Cumulative Reward')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()

        # Save the plot to a file
        fig.savefig(filename, dpi=500)
    
    def export_rewards_to_excel(self, filename, time, rewards_list):
        '''
        Export the rewards per timestep and cumulative rewards to an Excel file.

        Parameters:
        - filename: Name of the output Excel file
        
        - time: List of time values
        
        - rewards_list: List of rewards per timestep
        '''

        # Calculate cumulative rewards
        cumulative_rewards = [sum(rewards_list[:i+1]) for i in range(len(rewards_list))]

        # Prepare the data dictionary
        data = {
            'Timesteps [5 minutes]': time,
            'Rewards per Timestep': rewards_list,
            'Cumulative Rewards': cumulative_rewards,
        }

        # Create a DataFrame and export to Excel
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"Rewards data successfully exported to {filename}")
    
    def export_evaluated_data_to_excel_table(self, filename, metrics_dnn, metrics_gl, metrics_combined):
        '''
        Export the evaluation metrics (RMSE, RRMSE, and ME) for DNN, GL, and Combined models to an Excel file.

        Parameters:
        - filename: Name of the output Excel file
        - metrics_dnn: Dictionary with RMSE, RRMSE, and ME for DNN predictions
        - metrics_gl: Dictionary with RMSE, RRMSE, and ME for GL predictions
        - metrics_combined: Dictionary with RMSE, RRMSE, and ME for Combined predictions
        '''

        # Prepare data for the table
        rows = []
        variables = ['PAR', 'Temperature', 'Humidity', 'CO2', 'Leaf Temperature']

        for variable in variables:
            # DNN metrics
            rmse_dnn, rrmse_dnn, me_dnn = metrics_dnn[variable]
            # GL metrics
            rmse_gl, rrmse_gl, me_gl = metrics_gl[variable]
            # Combined metrics
            rmse_combined, rrmse_combined, me_combined = metrics_combined[variable]

            # Set the appropriate units
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
                unit_rmse = "W/m²"  # Assuming PAR is in W/m²
                unit_me = "W/m²"

            unit_rrmse = "%"  # RRMSE is always in percentage

            # Append rows for each variable and model type with values and units in separate columns
            rows.append([f"{variable} (DNN)", rmse_dnn, unit_rmse, rrmse_dnn, unit_rrmse, me_dnn, unit_me])
            rows.append([f"{variable} (GL)", rmse_gl, unit_rmse, rrmse_gl, unit_rrmse, me_gl, unit_me])
            rows.append([f"{variable} (Combined)", rmse_combined, unit_rmse, rrmse_combined, unit_rrmse, me_combined, unit_me])

        # Create a DataFrame
        df_metrics = pd.DataFrame(rows, columns=["Model", "RMSE", "RMSE Unit", "RRMSE", "RRMSE Unit", "ME", "ME Unit"])

        # Export the DataFrame to an Excel file
        df_metrics.to_excel(filename, index=False)
        print(f"Metrics successfully exported to {filename}")
    
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
        
        return json_data
    
    def publish_mqtt_data(self, json_data, broker="192.168.1.56", port=1883, topic="greenhouse-iot-system/drl-controls"):
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

    def get_outdoor_indoor_measurements(self, broker="192.168.1.56", port=1883, topic="greenhouse-iot-system/outdoor-indoor-measurements"):
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
            # print(msg.topic + " " + str(msg.payload.decode())) # debugging when receiving the the JSON payload
            # Parse the JSON data
            data = json.loads(msg.payload.decode())
                        
            # Process the received data and return it
            self.return_indoor_outdoor_measurements = self.process_received_data(data)
        
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
        
        return self.return_indoor_outdoor_measurements # Return the processed
    
    def process_received_data(self, data):
        '''
        Process the outdoor and indoor measurements, handling NaN values.
        
        Outdoor and indoor measurements:
        - time: from main loop iteration in 1 s
        - lux: Need to be converted to W / m^2
        - temperature
        - humidity
        - co2
        - leaf temperature 
        '''
        
        # Extract variables
        time = data.get("time", [])
        par_out = data.get("par_out", [])
        temp_out = data.get("temp_out", [])
        hum_out = data.get("hum_out", [])
        co2_out = data.get("co2_out", [])
        par_in = data.get("par_in", [])
        temp_in = data.get("temp_in", [])
        hum_in = data.get("hum_in", [])
        co2_in = data.get("co2_in", [])
        leaf_temp = data.get("leaf_temp", [])

        # Define a helper function to replace NaN with the previous value
        def replace_nan_with_previous(values):
            for i in range(len(values)):
                if np.isnan(values[i]):
                    # If it's the first value and NaN, replace it with 0 (or another default)
                    if i == 0:
                        values[i] = 0
                    else:
                        # Replace NaN with the previous non-NaN value
                        values[i] = values[i-1]
            return values
        
        # Handle NaN values in all the data lists
        par_out = replace_nan_with_previous(par_out)
        temp_out = replace_nan_with_previous(temp_out)
        hum_out = replace_nan_with_previous(hum_out)
        co2_out = replace_nan_with_previous(co2_out)
        par_in = replace_nan_with_previous(par_in)
        temp_in = replace_nan_with_previous(temp_in)
        hum_in = replace_nan_with_previous(hum_in)
        co2_in = replace_nan_with_previous(co2_in)
        leaf_temp = replace_nan_with_previous(leaf_temp)

        # Print the extracted variables after handling NaN
        print("Received from ONLINE MEASUREMENTS")
        print("Time:", time)
        print("PAR Out:", par_out)
        print("Temperature Out:", temp_out)
        print("Humidity Out:", hum_out)
        print("CO2 Out:", co2_out)
        print("PAR In:", par_in)
        print("Temperature In:", temp_in)
        print("Humidity In:", hum_in)
        print("CO2 In:", co2_in)
        print("Leaf Temperature:", leaf_temp)

        # Create indoor and outdoor measurements dictionary
        outdoor_indoor_measurements = {
            'time': np.array(time).reshape(-1, 1),
            'par_out': np.array(par_out).reshape(-1, 1),
            'temp_out': np.array(temp_out).reshape(-1, 1),
            'hum_out': np.array(hum_out).reshape(-1, 1),
            'co2_out': np.array(co2_out).reshape(-1, 1),
            'par_in': np.array(par_in).reshape(-1, 1),
            'temp_in': np.array(temp_in).reshape(-1, 1),
            'hum_in': np.array(hum_in).reshape(-1, 1),
            'co2_in': np.array(co2_in).reshape(-1, 1),
            'leaf_temp': np.array(leaf_temp).reshape(-1, 1)
        }

        # Save measurements to a .mat file
        sio.savemat('outdoor-indoor.mat', outdoor_indoor_measurements)
        
        return outdoor_indoor_measurements