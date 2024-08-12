import pandas as pd

# Load the dataset
file_path = r"C:\Users\frm19\OneDrive - Wageningen University & Research\2. Thesis - Information Technology\3. Software Projects\mini-greenhouse-greenlight-model\Code\inputs\Mini Greenhouse\dataset7.xlsx"
mgh_data = pd.read_excel(file_path)

# Define the step size
step_size = 4

# Iterate through the dataframe in step_data
# for i in range(0, len(mgh_data), step_size):
for i in range(0, 16, step_size):
    # Slice the dataframe to get the rows for the current step
    step_data = mgh_data.iloc[i:i + step_size]
    
    # Extract the variables from the step_data
    time = step_data['Time'].values
    global_out = step_data['global out'].values
    global_in = step_data['global in'].values
    temp_in = step_data['temp in'].values
    temp_out = step_data['temp out'].values
    rh_in = step_data['rh in'].values
    rh_out = step_data['rh out'].values
    co2_in = step_data['co2 in'].values
    co2_out = step_data['co2 out'].values
    toplights = step_data['toplights'].values
    ventilation = step_data['ventilation'].values
    heater = step_data['heater'].values

    # Print the variables for this step
    print(f"Step {i // step_size + 1}:")
    print(f"time = {time}")
    print(f"global_out = {global_out}")
    print(f"global_in = {global_in}")
    print(f"temp_in = {temp_in}")
    print(f"temp_out = {temp_out}")
    print(f"rh_in = {rh_in}")
    print(f"rh_out = {rh_out}")
    print(f"co2_in = {co2_in}")
    print(f"co2_out = {co2_out}")
    print(f"toplights = {toplights}")
    print(f"ventilation = {ventilation}")
    print(f"heater = {heater}")
    print("\n")
