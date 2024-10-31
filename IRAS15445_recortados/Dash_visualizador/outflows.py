#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cube_to_df as ctdf
import Ajuste_Figuras as Ajuste
from lmfit import Model
from lmfit.models import GaussianModel
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

line = '13CO'
path = '/home/marcela/Tesis Marcela/IRAS15445_recortados/I15445.mstransform_cube_contsub_'+line+'.fits'

def moment2(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[90:225,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.011*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 # pasar a km/s
   
    return m2

m2 = moment2(path)
m2_escalar = m2.value*1.9

box = [220,225,300,285]
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)

# Pixeles para los cuales m2 es mayor o igual a 38.56
pix = np.argwhere(m2_escalar >= 38.56)

columns1 = [f'Pix_{x}_{y}' for y, x in pix if y < 40]
columns1 = [col for col in columns1 if col in Molines_A_df.columns]
Molines_filtrado1 = Molines_A_df[columns1]

columns2 = [f'Pix_{x}_{y}' for y, x in pix if y > 40]
columns2 = [col for col in columns2 if col in Molines_A_df.columns]

Molines_filtrado2 = Molines_A_df[columns2]

Molines_A_df['mean1'] = Molines_filtrado1.sum(axis=1)/595
Molines_A_df['mean2'] = Molines_filtrado2.sum(axis=1)/595

pars1,result1,fig1 = Ajuste.gauss_model_outflows(Molines_A_df, cube, 'mean1', [80,230], plot=True)
plt.title('Outflow Down',fontsize=14)

pars2,result2,fig2 = Ajuste.gauss_model_outflows(Molines_A_df, cube, 'mean2', [80,230], plot=True)
plt.title('Outflow Up',fontsize=14)

channel=[80,230]

x_datos1 = Molines_A_df['mean1'].index[channel[0]:channel[1]]
y_datos1 = Molines_A_df['mean1'].iloc[channel[0]:channel[1]]

x_datos2 = Molines_A_df['mean2'].index[channel[0]:channel[1]]
y_datos2 = Molines_A_df['mean2'].iloc[channel[0]:channel[1]]


plt.figure(figsize=(15,5))

plt.step(Molines_A_df['mean1'].index[channel[0]:channel[1]],Molines_A_df['mean1'].iloc[channel[0]:channel[1]],color='orangered')
plt.step(Molines_A_df['mean2'].index[channel[0]:channel[1]],Molines_A_df['mean2'].iloc[channel[0]:channel[1]],color='royalblue')

plt.title(line,fontsize=15)
plt.xlabel('Radio Velocity [km/s]',fontsize=13)
plt.ylabel('$[Jy/beam]$',fontsize=13)

# Formato de los ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(axis='both', direction='in', length=5, width=1.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)

# Ajuste 
plt.plot(x_datos1, result1.best_fit, '-', color='black')
plt.text(min(x_datos1)+min(x_datos1)/40,max(y_datos1)-max(y_datos1)/11,r'    Down Outflow',color='orangered',fontsize=13)
plt.text(min(x_datos1)+min(x_datos1)/40,max(y_datos1)-2*max(y_datos1)/11,r'$\mu =$ '+str(round(pars1['cen'].value,2))+' km/s',color='orangered',fontsize=13)
plt.text(min(x_datos1)+min(x_datos1)/40,max(y_datos1)-3*max(y_datos1)/11,r'$\sigma =$ '+str(round(pars1['wid'].value,2))+' km/s',color='orangered',fontsize=13)


# Ajuste 
plt.plot(x_datos2, result2.best_fit, '-', color='black')
plt.text(max(x_datos1)-max(x_datos1),max(y_datos1)-max(y_datos1)/11,r'       Up Outflow',color='royalblue',fontsize=13)
plt.text(max(x_datos1)-max(x_datos1),max(y_datos1)-2*max(y_datos1)/11,r'$\mu =$ '+str(round(pars2['cen'].value,2))+' km/s',color='royalblue',fontsize=13)
plt.text(max(x_datos1)-max(x_datos1),max(y_datos1)-3*max(y_datos1)/11,r'$\sigma =$ '+str(round(pars2['wid'].value,2))+' km/s',color='royalblue',fontsize=13)












