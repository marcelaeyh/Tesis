#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cube_to_df as ctdf
from lmfit import Model
from lmfit.models import GaussianModel
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import cube_to_df as ctdf

line = '13CO'
path = '/home/marcela/Tesis Marcela/IRAS15445_recortados/I15445.mstransform_cube_contsub_'+line+'.fits'

box = [205, 205, 305, 305]

cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)

# Dimensiones de la imagen
nx = 100
ny = 100

# Tamaño de píxel en segundos de arco
delta = 3.611111111111e-06 * 3600


x = -np.concatenate([np.arange(-nx/2,0)*delta,np.arange(nx/2)*delta])
y = np.concatenate([np.arange(-ny/2,0)*delta,np.arange(ny/2)*delta])
ts = 10

x_tick_labels = np.round(x[::ts], 5)
y_tick_labels = np.round(y[::ts], 5)

for i in range(104,198):
    plt.figure(figsize=(10,10))
    im = plt.imshow(cube[i, :, :], origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],vmin=-0.005,vmax=0.04)
    
    plt.xticks(x_tick_labels, fontsize=12)
    plt.yticks(y_tick_labels, fontsize=12)
    
    plt.xlabel('J2000 RA offset [arcsec]',fontsize=14)
    plt.ylabel('J2000 DEC offset [arcsec]',fontsize=14)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Intensidad [Jy/Beam]', fontsize=15) 
    cbar.ax.tick_params(labelsize=14)
    
    plt.title(f'Canal {i}', fontsize=15)
    
    plt.savefig('/home/marcela/Tesis Marcela/IRAS15445_recortados/anim_13COline/'+str(i),dpi=300)
    

x_z_projection = np.mean(cube, axis=-2)  # Alternativamente: np.mean(cube, axis=0)
z_tick_labels = np.round(Molines_A_df.index[::40]).astype(int)

# Vista desde arriba 

plt.figure(figsize=(10,5))
im = plt.imshow(x_z_projection.T, origin='lower',aspect='auto')
plt.gca().invert_xaxis()
plt.yticks(ticks=np.arange(len(y_tick_labels))*10, labels=np.round(y_tick_labels, 2),fontsize=12)
plt.xticks(ticks=np.arange(len(z_tick_labels))*40, labels=np.round(z_tick_labels, 2),fontsize=12)

plt.xlabel('Radio Velocity [km/s]',fontsize=14)
plt.ylabel('J2000 RA offset [arcsec]',fontsize=14)

cbar = plt.colorbar(im)
cbar.set_label('Intensidad [Jy/Beam]', fontsize=15) 
cbar.ax.tick_params(labelsize=14)

plt.savefig('/home/marcela/Tesis Marcela/IRAS15445_recortados/Vistatoroidal.png',dpi=300)






