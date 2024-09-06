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

        self.co2_in_predicted_nn = dataframe['CO2 In (Predicted NN)'].values
        self.temp_in_predicted_nn = dataframe['Temperature In (Predicted NN)'].values
        self.rh_in_predicted_nn = dataframe['RH In (Predicted NN)'].values
        self.par_in_predicted_nn = dataframe['PAR In (Predicted NN)'].values
        
        self.co2_in_predicted_gl = dataframe['CO2 In (Predicted GL)'].values
        self.temp_in_predicted_gl = dataframe['Temperature In (Predicted GL)'].values
        self.rh_in_predicted_gl = dataframe['RH In (Predicted GL)'].values
        self.par_in_predicted_gl = dataframe['PAR In (Predicted GL)'].values
        
        self.co2_in_predicted_combined_models = dataframe['CO2 In (Predicted Combined)'].values
        self.temp_in_predicted_combined_models = dataframe['Temperature In (Predicted Combined)'].values
        self.rh_in_predicted_combined_models = dataframe['RH In (Predicted Combined)'].values
        self.par_in_predicted_combined_models = dataframe['PAR In (Predicted Combined)'].values
    
    def evaluate_predictions(self):
        '''
        Evaluate the RMSE, RRMSE, and ME of the predicted vs actual values for `par_in`, `temp_in`, `rh_in`, and `co2_in`.
        '''
        
        # Extract actual values
        y_true_par_in = self.par_in_excel_mqtt
        y_true_temp_in = self.temp_in_excel_mqtt
        y_true_rh_in = self.rh_in_excel_mqtt
        y_true_co2_in = self.co2_in_excel_mqtt

        # Extract predicted values
        y_pred_par_in_nn = self.par_in_predicted_nn
        y_pred_temp_in_nn = self.temp_in_predicted_nn
        y_pred_rh_in_nn = self.rh_in_predicted_nn
        y_pred_co2_in_nn = self.co2_in_predicted_nn
        
        y_pred_par_in_gl = self.par_in_predicted_gl
        y_pred_temp_in_gl = self.temp_in_predicted_gl
        y_pred_rh_in_gl = self.rh_in_predicted_gl
        y_pred_co2_in_gl = self.co2_in_predicted_gl

        # Extract combined model predictions
        y_pred_par_in_combined = self.par_in_predicted_combined_models
        y_pred_temp_in_combined = self.temp_in_predicted_combined_models
        y_pred_rh_in_combined = self.rh_in_predicted_combined_models
        y_pred_co2_in_combined = self.co2_in_predicted_combined_models

        # Calculate RMSE, RRMSE, and ME for each variable
        def calculate_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rrmse = rmse / np.mean(y_true) * 100
            me = np.mean(y_pred - y_true)
            return rmse, rrmse, me

        metrics_nn = {
            'PAR': calculate_metrics(y_true_par_in, y_pred_par_in_nn),
            'Temperature': calculate_metrics(y_true_temp_in, y_pred_temp_in_nn),
            'Humidity': calculate_metrics(y_true_rh_in, y_pred_rh_in_nn),
            'CO2': calculate_metrics(y_true_co2_in, y_pred_co2_in_nn)
        }

        metrics_gl = {
            'PAR': calculate_metrics(y_true_par_in, y_pred_par_in_gl),
            'Temperature': calculate_metrics(y_true_temp_in, y_pred_temp_in_gl),
            'Humidity': calculate_metrics(y_true_rh_in, y_pred_rh_in_gl),
            'CO2': calculate_metrics(y_true_co2_in, y_pred_co2_in_gl)
        }

        metrics_combined = {
            'PAR': calculate_metrics(y_true_par_in, y_pred_par_in_combined),
            'Temperature': calculate_metrics(y_true_temp_in, y_pred_temp_in_combined),
            'Humidity': calculate_metrics(y_true_rh_in, y_pred_rh_in_combined),
            'CO2': calculate_metrics(y_true_co2_in, y_pred_co2_in_combined)
        }

        # Print the results with appropriate units
        print("------------------------------------------------------------------------------------")
        print("EVALUATION RESULTS :")
        for variable in ['PAR', 'Temperature', 'Humidity', 'CO2']:
            rmse_nn, rrmse_nn, me_nn = metrics_nn[variable]
            rmse_gl, rrmse_gl, me_gl = metrics_gl[variable]
            rmse_combined, rrmse_combined, me_combined = metrics_combined[variable]
            
            if variable == 'Temperature':
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
            
            print(f"{variable} (NN): RMSE = {rmse_nn:.4f} {unit_rmse}, RRMSE = {rrmse_nn:.4f} {unit_rrmse}, ME = {me_nn:.4f} {unit_me}")
            print(f"{variable} (GL): RMSE = {rmse_gl:.4f} {unit_rmse}, RRMSE = {rrmse_gl:.4f} {unit_rrmse}, ME = {me_gl:.4f} {unit_me}")
            print(f"{variable} (Combined): RMSE = {rmse_combined:.4f} {unit_rmse}, RRMSE = {rrmse_combined:.4f} {unit_rrmse}, ME = {me_combined:.4f} {unit_me}")

        return metrics_nn, metrics_gl, metrics_combined

    def plot_all_data(self, filename, time, co2_actual, temp_actual, rh_actual, par_actual, 
                  co2_predicted_nn, temp_predicted_nn, rh_predicted_nn, par_predicted_nn,
                  co2_predicted_gl, temp_predicted_gl, rh_predicted_gl, par_predicted_gl,
                  co2_combined, temp_combined, rh_combined, par_combined,
                  metrics_nn, metrics_gl, metrics_combined):
        '''
        Plot all the parameters to make it easier to compare predicted vs actual values.
        '''

        # Create subplots with 2 rows and 2 columns
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

        # Data to be plotted along with their titles and predicted data
        data = [
            (co2_actual, co2_predicted_nn, co2_predicted_gl, co2_combined, 'CO2 In [ppm]', metrics_nn['CO2'], metrics_gl['CO2'], metrics_combined['CO2'], "ppm"),
            (temp_actual, temp_predicted_nn, temp_predicted_gl, temp_combined, 'Temperature In [°C]', metrics_nn['Temperature'], metrics_gl['Temperature'], metrics_combined['Temperature'], "°C"),
            (rh_actual, rh_predicted_nn, rh_predicted_gl, rh_combined, 'RH In [%]', metrics_nn['Humidity'], metrics_gl['Humidity'], metrics_combined['Humidity'], "%"),
            (par_actual, par_predicted_nn, par_predicted_gl, par_combined, 'PAR In [W/m²]', metrics_nn['PAR'], metrics_gl['PAR'], metrics_combined['PAR'], "W/m²")
        ]

        # Plot each dataset in a subplot
        for ax, (y_actual, y_pred_nn, y_pred_gl, y_combined, title, metrics_nn, metrics_gl, metrics_combined, unit) in zip(axes.flatten(), data):
            ax.plot(time, y_actual, label='Actual', color='blue')  # Plot actual data
            ax.plot(time, y_pred_nn, label='Predicted NN', color='purple', linestyle='--')  # Plot NN predicted data
            ax.plot(time, y_pred_gl, label='Predicted GL', color='green', linestyle=':')  # Plot GL predicted data
            ax.plot(time, y_combined, label='Predicted Combined', color='red', linestyle='-.')  # Plot combined predicted data
            
            # Set the appropriate units for RMSE and ME
            unit_rmse = unit
            unit_me = unit
            unit_rrmse = "%"  # RRMSE is always in percentage
            
            # Add RMSE, RRMSE, and ME to the title
            ax.set_title(f"{title}\nNN RMSE: {metrics_nn[0]:.4f} {unit_rmse}, RRMSE: {metrics_nn[1]:.4f} {unit_rrmse}, ME: {metrics_nn[2]:.4f} {unit_me}\n"
                        f"GL RMSE: {metrics_gl[0]:.4f} {unit_rmse}, RRMSE: {metrics_gl[1]:.4f} {unit_rrmse}, ME: {metrics_gl[2]:.4f} {unit_me}\n"
                        f"Combined RMSE: {metrics_combined[0]:.4f} {unit_rmse}, RRMSE: {metrics_combined[1]:.4f} {unit_rrmse}, ME: {metrics_combined[2]:.4f} {unit_me}")
            
            ax.set_xlabel('Timesteps [5 minutes / -]')  # Set the x-axis label
            ax.set_ylabel(title)  # Set the y-axis label
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
            ax.legend()  # Add legend to distinguish between actual and predicted data

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        # plt.show()
        
        # Save the plot to a file
        fig.savefig(filename, dpi=1000)

def main():
    # Read the Excel file into a DataFrame
    filepath = 'output_simulated_data_online.xlsx'  # Replace with the path to your Excel file
    df = pd.read_excel(filepath)

    # Initialize the DataPlotter with the DataFrame
    plotter = DataPlotter(df)

    # Evaluate the predictions to obtain the metrics
    metrics_nn, metrics_gl, metrics_combined = plotter.evaluate_predictions()

    # Plot the data and save the plot to a file
    plotter.plot_all_data(
        filename='predictions_comparison_plot.png', 
        time=plotter.time,
        co2_actual=plotter.co2_in_excel_mqtt,
        temp_actual=plotter.temp_in_excel_mqtt,
        rh_actual=plotter.rh_in_excel_mqtt,
        par_actual=plotter.par_in_excel_mqtt,
        co2_predicted_nn=plotter.co2_in_predicted_nn,
        temp_predicted_nn=plotter.temp_in_predicted_nn,
        rh_predicted_nn=plotter.rh_in_predicted_nn,
        par_predicted_nn=plotter.par_in_predicted_nn,
        co2_predicted_gl=plotter.co2_in_predicted_gl,
        temp_predicted_gl=plotter.temp_in_predicted_gl,
        rh_predicted_gl=plotter.rh_in_predicted_gl,
        par_predicted_gl=plotter.par_in_predicted_gl,
        co2_combined=plotter.co2_in_predicted_combined_models,
        temp_combined=plotter.temp_in_predicted_combined_models,
        rh_combined=plotter.rh_in_predicted_combined_models,
        par_combined=plotter.par_in_predicted_combined_models,
        metrics_nn=metrics_nn,
        metrics_gl=metrics_gl,
        metrics_combined=metrics_combined
    )

if __name__ == "__main__":
    main()
