import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_vel = '/Users/mac/Tesis/Cut_Cubes/SO24_3_velocity.tsv'
path_freq = '/Users/mac/Tesis/Cut_Cubes/SO24_3_frequency.tsv'


channel = [90, 240]

vel = pd.read_csv(path_vel, skiprows=range(5), sep='\t')
vel.columns = ['Velocity', 'Value']

freq = pd.read_csv(path_freq, skiprows=range(5),sep='\t')
freq.columns = ['Frequency', 'Value']


fig, ax1 = plt.subplots(figsize=(17,5))

# Eje X inferior: Velocidad
ax1.step(vel.Velocity, vel.Value, color='white')
ax1.set_xlabel('Velocity [km/s]', fontsize=16)
ax1.set_ylabel('[mJy/Beam]', fontsize=16)
ax1.set_xlim(-321.524252,192.721283)

ax1.tick_params(axis='both', labelsize=14)

# Eje X superior: Frecuencia
ax2 = ax1.twiny()

ax2.step(freq.Frequency, freq.Value, color='black')
ax2.set_xlim(332.861608,332.291249)
ax2.set_xlabel('Frequency [GHz]', fontsize=16)
ax2.tick_params(axis='x', labelsize=14)


ax1.axvspan(vel.Velocity[230], vel.Velocity[90], facecolor='green', alpha=0.2,label='Range for fitting')

ax1.legend(fontsize=16)
plt.savefig('/Users/mac/Tesis/Cut_Cubes/Emission_Figures/SO2_freq_vel.png', dpi=300, bbox_inches='tight')
