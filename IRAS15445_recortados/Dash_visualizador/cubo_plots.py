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
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyBboxPatch

line = '13CO'
#path = '/home/marcela/Tesis Marcela/IRAS15445_recortados/I15445.mstransform_cube_contsub_'+line+'.fits'
path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_'+line+'.fits'

beam = SpectralCube.read(path).beam

box = [205, 205, 305, 305]

cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)


# Ejes
ny = abs(box[2]-box[0])
nx = abs(box[3]-box[1])

# Tamaño de píxel en segundos de arco
delta = 3.61e-06 * 3600

x = -np.concatenate([np.arange(-nx/2,0)*delta, np.arange(nx/2)*delta])
y = np.concatenate([np.arange(-ny/2,0)*delta, np.arange(ny/2)*delta])


for i in range(104,198):
    plt.figure(figsize=(10,10))
    ax1 = plt.subplot()
    im = plt.imshow(cube[i, :, :], origin='lower',vmin=-0.005,vmax=0.04)
    
    ax1.set_xticks(np.linspace(0, nx-1, 7))
    ax1.set_xticklabels(np.round(np.linspace(x.max(), x.min()-0.01, 7), 2),fontsize=12)
    ax1.set_yticks(np.linspace(0, ny-1, 7))
    ax1.set_yticklabels(np.round(np.linspace(y.min()+0.01, y.max(), 7), 2),fontsize=12)

    
    plt.xlabel('J2000 RA offset [arcsec]',fontsize=14)
    plt.ylabel('J2000 DEC offset [arcsec]',fontsize=14)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Intensidad [Jy/Beam]', fontsize=15) 
    cbar.ax.tick_params(labelsize=14)
    
    plt.title(f'Canal {i}', fontsize=15)
    
    plt.savefig('/home/marcela/Tesis Marcela/IRAS15445_recortados/anim_13COline/'+str(i),dpi=300)
    

x_z_projection = np.mean(cube, axis=1) 

# Vista desde arriba 

plt.figure(figsize=(15,5))

ax1 = plt.subplot()
im = ax1.imshow(x_z_projection.T, origin='lower',aspect='auto')
plt.gca().invert_xaxis()

ax1.set_yticks(np.linspace(0, ny-1, 7))
ax1.set_yticklabels(np.round(np.linspace(y.min()+0.01, y.max(), 7), 2),fontsize=12)

ax1.set_xticks(np.linspace(0,len(Molines_A_df.index),8))
ax1.set_xticklabels(np.round(np.linspace(Molines_A_df.index[0],Molines_A_df.index[-1],8)).astype(int),fontsize=12)

ax1.set_xlabel('Radio Velocity [km/s]',fontsize=14)
ax1.set_ylabel('J2000 RA offset [arcsec]',fontsize=14)

cbar = plt.colorbar(im)
cbar.set_label('Intensity [Jy/Beam]', fontsize=15) 
cbar.ax.tick_params(labelsize=14)


# Tamaños UA

d = np.array([4.38,5.4,8.5])
xx = 800 #UA
ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi

#ax1.hlines(30,17,ang[0]/delta+17,color='k')

ax1.hlines(10,274,-ang[0]/delta+274,color='white')
ax1.text(274,3,'800 AU',fontsize=11,color='white')

# Beam
#beam_ellipse = Ellipse(
#    (10, 10), width=beam.minor.value/delta*3600, height=beam.major.value/delta*3600, angle=-beam.pa.value,
#    edgecolor='black', facecolor='lightgray', lw=1)

# Agregar la elipse al gráfico
#ax1.add_patch(beam_ellipse)

ax1.set_xlim(280,40)

ax1.set_aspect('equal')



plt.savefig('/Users/mac/Tesis/IRAS15445_recortados/Vistatoroidal.png',dpi=300)


