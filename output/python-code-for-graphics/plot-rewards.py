import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Load data from the Excel files
scheduled_df = pd.read_excel('rewards_list_scheduled.xlsx')
drl_df = pd.read_excel('rewards_list_drl.xlsx')

# Extract data from the dataframes  
timesteps_scheduled = scheduled_df['Timesteps [5 minutes]'].to_numpy()
rewards_scheduled = scheduled_df['Rewards per Timestep'].to_numpy()
cumulative_scheduled = scheduled_df['Cumulative Rewards'].to_numpy()

timesteps_drl = drl_df['Timesteps [5 minutes]'].to_numpy()
rewards_drl = drl_df['Rewards per Timestep'].to_numpy()
cumulative_drl = drl_df['Cumulative Rewards'].to_numpy()

# Polynomial curve fitting for rewards per timestep
p_drl = Polynomial.fit(timesteps_drl, rewards_drl, 3)
p_drl_cumulative = Polynomial.fit(timesteps_drl, cumulative_drl, 2)

# Plot Reward per Timestep
plt.figure(figsize=(6, 3.5))
plt.plot(timesteps_scheduled, rewards_scheduled, label='Scheduled Actions', linestyle='-.', color='blue')
plt.plot(timesteps_drl, rewards_drl, label='Self-adaptive PPO Actions (Moving Avg.)', color='green')
plt.plot(timesteps_drl, p_drl(timesteps_drl), linestyle='--', color='purple', label='Self-adaptive PPO Actions (Polynomial Curve Fitting)')

plt.xlabel('Timesteps [5 minutes]')
plt.ylabel('Reward Value')
plt.legend()
plt.tight_layout()  # Adjust the layout to fit all components within the figure
plt.savefig('reward_per_timestep_high_res.png', dpi=1000)  # Save with high resolution
plt.show()

# Plot Cumulative Rewards
plt.figure(figsize=(6, 3.5))
plt.plot(timesteps_scheduled, cumulative_scheduled, label='Scheduled Actions', linestyle='-.', color='blue')
plt.plot(timesteps_drl, cumulative_drl, label='Self-adaptive PPO Actions (Moving Avg.)', color='green')
plt.plot(timesteps_drl, p_drl_cumulative(timesteps_drl), linestyle='--', color='purple', label='Self-adaptive PPO Actions (Polynomial Curve Fitting)')

plt.xlabel('Timesteps [5 minutes]')
plt.ylabel('Cumulative Rewards')
plt.legend()
plt.tight_layout()  # Adjust the layout to fit all components within the figure
plt.savefig('cumulative_rewards_high_res.png', dpi=1000)  # Save with high resolution
plt.show()

