import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self, dataframe):
        self.data = dataframe
        
        # Assign the columns to class attributes
        self.time = dataframe['Timesteps [5 minutes]'].values
        self.co2_in_excel_mqtt = dataframe['CO2 In (Actual)'].values
        self.temp_in_excel_mqtt = dataframe['Temperature In (Actual)'].values
        self.rh_in_excel_mqtt = dataframe['RH In (Actual)'].values
        self.par_in_excel_mqtt = dataframe['PAR In (Actual)'].values

        self.co2_in_predicted_dnn = dataframe['CO2 In (Predicted DNN)'].values
        self.temp_in_predicted_dnn = dataframe['Temperature In (Predicted DNN)'].values
        self.rh_in_predicted_dnn = dataframe['RH In (Predicted DNN)'].values
        self.par_in_predicted_dnn = dataframe['PAR In (Predicted DNN)'].values
        
        self.co2_in_predicted_gl = dataframe['CO2 In (Predicted GL)'].values
        self.temp_in_predicted_gl = dataframe['Temperature In (Predicted GL)'].values
        self.rh_in_predicted_gl = dataframe['RH In (Predicted GL)'].values
        self.par_in_predicted_gl = dataframe['PAR In (Predicted GL)'].values
        
        self.co2_in_predicted_combined_models = dataframe['CO2 In (Predicted Combined)'].values
        self.temp_in_predicted_combined_models = dataframe['Temperature In (Predicted Combined)'].values
        self.rh_in_predicted_combined_models = dataframe['RH In (Predicted Combined)'].values
        self.par_in_predicted_combined_models = dataframe['PAR In (Predicted Combined)'].values
        
        # Leaf temperature data (assuming your dataframe has these columns)
        self.leaf_temp_actual = dataframe['Leaf Temp (Actual)'].values
        self.leaf_temp_predicted_dnn = dataframe['Leaf Temp (Predicted DNN)'].values
        self.leaf_temp_predicted_gl = dataframe['Leaf Temp (Predicted GL)'].values
        self.leaf_temp_predicted_combined = dataframe.get('Leaf Temp (Predicted Combined)', None)

    def evaluate_predictions(self):
        '''
        Evaluate the RMSE, RRMSE, and ME of the predicted vs actual values for `par_in`, `temp_in`, `rh_in`, `co2_in`, and `leaf_temp`.
        '''
        
        # Extract actual values
        y_true_par_in = self.par_in_excel_mqtt
        y_true_temp_in = self.temp_in_excel_mqtt
        y_true_rh_in = self.rh_in_excel_mqtt
        y_true_co2_in = self.co2_in_excel_mqtt
        y_true_leaf_temp = self.leaf_temp_actual

        # Extract predicted values from Neural Network (DNN)
        y_pred_par_in_dnn = self.par_in_predicted_dnn
        y_pred_temp_in_dnn = self.temp_in_predicted_dnn
        y_pred_rh_in_dnn = self.rh_in_predicted_dnn
        y_pred_co2_in_dnn = self.co2_in_predicted_dnn
        y_pred_leaf_temp_dnn = self.leaf_temp_predicted_dnn

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
        y_pred_leaf_temp_combined = self.leaf_temp_predicted_combined

        # Calculate RMSE, RRMSE, and ME for each variable
        def calculate_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rrmse = rmse / np.mean(y_true) * 100  # RRMSE in percentage
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

        # Print the results with appropriate units
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
                unit_rmse = "W/m²"  # Assuming PAR is in W/m²
                unit_me = "W/m²"

            unit_rrmse = "%"  # RRMSE is always in percentage
            
            print(f"{variable} (DNN): RMSE = {rmse_dnn:.4f} {unit_rmse}, RRMSE = {rrmse_dnn:.4f} {unit_rrmse}, ME = {me_dnn:.4f} {unit_me}")
            print(f"{variable} (GL): RMSE = {rmse_gl:.4f} {unit_rmse}, RRMSE = {rrmse_gl:.4f} {unit_rrmse}, ME = {me_gl:.4f} {unit_me}")
            print(f"{variable} (Combined): RMSE = {rmse_combined:.4f} {unit_rmse}, RRMSE = {rrmse_combined:.4f} {unit_rrmse}, ME = {me_combined:.4f} {unit_me}")

        return metrics_dnn, metrics_gl, metrics_combined

    def plot_all_data(self, filename, time, co2_actual, temp_actual, rh_actual, par_actual, 
                  co2_predicted_dnn, temp_predicted_dnn, rh_predicted_dnn, par_predicted_dnn,
                  co2_predicted_gl, temp_predicted_gl, rh_predicted_gl, par_predicted_gl,
                  co2_combined, temp_combined, rh_combined, par_combined,
                  metrics_dnn, metrics_gl, metrics_combined):
        '''
        Plot all the parameters to make it easier to compare predicted vs actual values.
        '''

        # Create subplots with 2 rows and 2 columns
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))

        # Data to be plotted along with their titles and predicted data
        data = [
            (co2_actual, co2_predicted_dnn, co2_predicted_gl, co2_combined, 'CO2 In [ppm]', metrics_dnn['CO2'], metrics_gl['CO2'], metrics_combined['CO2'], "ppm"),
            (temp_actual, temp_predicted_dnn, temp_predicted_gl, temp_combined, 'Temperature In [°C]', metrics_dnn['Temperature'], metrics_gl['Temperature'], metrics_combined['Temperature'], "°C"),
            (rh_actual, rh_predicted_dnn, rh_predicted_gl, rh_combined, 'RH In [%]', metrics_dnn['Humidity'], metrics_gl['Humidity'], metrics_combined['Humidity'], "%"),
            (par_actual, par_predicted_dnn, par_predicted_gl, par_combined, 'PAR In [W/m²]', metrics_dnn['PAR'], metrics_gl['PAR'], metrics_combined['PAR'], "W/m²")
        ]

        # Plot each dataset in a subplot
        for ax, (y_actual, y_pred_dnn, y_pred_gl, y_combined, title, metrics_dnn, metrics_gl, metrics_combined, unit) in zip(axes.flatten(), data):
            ax.plot(time, y_actual, label='Actual', color='blue')  # Plot actual data
            ax.plot(time, y_pred_dnn, label='Predicted DNN', color='purple', linestyle='--')  # Plot DNN predicted data
            ax.plot(time, y_pred_gl, label='Predicted GL', color='green', linestyle=':')  # Plot GL predicted data
            ax.plot(time, y_combined, label='Predicted Combined', color='red', linestyle='-.')  # Plot combined predicted data
            
            # Set the appropriate units for RMSE and ME
            unit_rmse = unit
            unit_me = unit
            unit_rrmse = "%"  # RRMSE is always in percentage
            
            # Add RMSE, RRMSE, and ME to the title
            # ax.set_title(f"{title}\nDNN RMSE: {metrics_dnn[0]:.4f} {unit_rmse}, RRMSE: {metrics_dnn[1]:.4f} {unit_rrmse}, ME: {metrics_dnn[2]:.4f} {unit_me}\n"
            #             f"GL RMSE: {metrics_gl[0]:.4f} {unit_rmse}, RRMSE: {metrics_gl[1]:.4f} {unit_rrmse}, ME: {metrics_gl[2]:.4f} {unit_me}\n"
            #             f"Combined RMSE: {metrics_combined[0]:.4f} {unit_rmse}, RRMSE: {metrics_combined[1]:.4f} {unit_rrmse}, ME: {metrics_combined[2]:.4f} {unit_me}")
            
            ax.set_xlabel('Timesteps [5 minutes / -]')  # Set the x-axis label
            ax.set_ylabel(title)  # Set the y-axis label
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.legend()  # Add legend to distinguish between actual and predicted data

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()
        
        # Save the plot to a file
        fig.savefig(filename, dpi=500)
    
    def plot_leaf_temperature(self, filename, time, leaf_temp_actual, leaf_temp_predicted_dnn, leaf_temp_predicted_gl, leaf_temp_combined=None,
                              metrics_dnn=None, metrics_gl=None, metrics_combined=None):
        '''
        Plot leaf_temperature parameter to make it easier to compare predicted vs actual values.
        '''

        # Create a single figure with size 9x6
        fig = plt.figure(figsize=(4.5, 3))

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
        # if metrics_dnn and metrics_gl and metrics_combined:
        #     title += (f"\nDNN RMSE: {metrics_dnn[0]:.4f}, RRMSE: {metrics_dnn[1]:.4f}%, ME: {metrics_dnn[2]:.4f}°C\n"
        #             f"GL RMSE: {metrics_gl[0]:.4f}, RRMSE: {metrics_gl[1]:.4f}%, ME: {metrics_gl[2]:.4f}°C\n"
        #             f"Combined RMSE: {metrics_combined[0]:.4f}, RRMSE: {metrics_combined[1]:.4f}%, ME: {metrics_combined[2]:.4f}°C")
                
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

def main():
    # Read the Excel file into a DataFrame
    filepath = 'output_simulated_data_online.xlsx'  # Replace with the path to your Excel file
    df = pd.read_excel(filepath)

    # Initialize the DataPlotter with the DataFrame
    plotter = DataPlotter(df)

    # Evaluate the predictions to obtain the metrics
    metrics_dnn, metrics_gl, metrics_combined = plotter.evaluate_predictions()

    # Plot the data and save the plot to a file
    plotter.plot_all_data(
        filename='predictions_comparison_plot_online.png', 
        time=plotter.time,
        co2_actual=plotter.co2_in_excel_mqtt,
        temp_actual=plotter.temp_in_excel_mqtt,
        rh_actual=plotter.rh_in_excel_mqtt,
        par_actual=plotter.par_in_excel_mqtt,
        co2_predicted_dnn=plotter.co2_in_predicted_dnn,
        temp_predicted_dnn=plotter.temp_in_predicted_dnn,
        rh_predicted_dnn=plotter.rh_in_predicted_dnn,
        par_predicted_dnn=plotter.par_in_predicted_dnn,
        co2_predicted_gl=plotter.co2_in_predicted_gl,
        temp_predicted_gl=plotter.temp_in_predicted_gl,
        rh_predicted_gl=plotter.rh_in_predicted_gl,
        par_predicted_gl=plotter.par_in_predicted_gl,
        co2_combined=plotter.co2_in_predicted_combined_models,
        temp_combined=plotter.temp_in_predicted_combined_models,
        rh_combined=plotter.rh_in_predicted_combined_models,
        par_combined=plotter.par_in_predicted_combined_models,
        metrics_dnn=metrics_dnn,
        metrics_gl=metrics_gl,
        metrics_combined=metrics_combined
    )
    
    # Plot the data and save the plot to a file
    plotter.plot_leaf_temperature(
        filename='leaf_temp_comparison_plot_online.png', 
        time=plotter.time,
        leaf_temp_actual=plotter.leaf_temp_actual,
        leaf_temp_predicted_dnn=plotter.leaf_temp_predicted_dnn,
        leaf_temp_predicted_gl=plotter.leaf_temp_predicted_gl,
        leaf_temp_combined=plotter.leaf_temp_predicted_combined,
        metrics_dnn=metrics_dnn['Leaf Temperature'],
        metrics_gl=metrics_gl['Leaf Temperature'],
        metrics_combined=metrics_combined['Leaf Temperature']
    )

if __name__ == "__main__":
    main()
