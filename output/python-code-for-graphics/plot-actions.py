import matplotlib.pyplot as plt
import pandas as pd

class DataPlotter:
    def __init__(self, dataframe):
        self.data = dataframe
        
        # Assign the columns to class attributes
        self.time = dataframe['Timesteps [5 minutes]'].values
        self.ventilation = dataframe['Action Ventilation'].values
        self.toplights = dataframe['Action Toplights'].values
        self.heater = dataframe['Action Heater'].values
    
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
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 6))

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

def main():
    # Read the Excel file into a DataFrame
    filepath = 'output_simulated_data_online.xlsx'  # Replace with the path to your Excel file
    df = pd.read_excel(filepath)
    
    # Initialize the DataPlotter with the DataFrame
    plotter = DataPlotter(df)
    
    plotter.plot_actions(
        filename='actions_plot_online.png',
        time = plotter.time,
        ventilation_list = plotter.ventilation,
        toplights_list = plotter.toplights,
        heater_list = plotter.heater
    )

if __name__ == "__main__":
    main()