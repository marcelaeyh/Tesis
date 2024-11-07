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

def moment0(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[90:225,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.011*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m0 = cube_include.moment(order=0)/1000 
   
    return m0
def moment2(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[90:225,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.011*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 # pasar a km/s
   
    return m2

# Calcular momentos
m0 = moment0(path)
m2 = moment2(path)
m2_escalar = m2.value*1.9

box = [220,225,300,285]
channel=[80,230]
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)

# Pixeles para los cuales m2 es mayor o igual a 38.56
pix = np.argwhere(m2_escalar >= 38.56)
pix_center = np.argwhere(m2_escalar < 38.56)

columns1 = [f'Pix_{x}_{y}' for y, x in pix if y < 40]
columns1 = [col for col in columns1 if col in Molines_A_df.columns]

columns2 = [f'Pix_{x}_{y}' for y, x in pix if y > 40]
columns2 = [col for col in columns2 if col in Molines_A_df.columns]

columns3 = [f'Pix_{x}_{y}' for y, x in pix_center]
columns3 = [col for col in columns3 if col in Molines_A_df.columns]

Molines_filtrado1 = Molines_A_df[columns1]
Molines_filtrado2 = Molines_A_df[columns2]
Molines_filtrado3 = Molines_A_df[columns3]

Molines_A_df['mean1'] = Molines_filtrado1.sum(axis=1)/595
Molines_A_df['mean2'] = Molines_filtrado2.sum(axis=1)/595
Molines_A_df['mean3'] = Molines_filtrado3.sum(axis=1)/595

# Ajustes
# ------------------------------------------------------------------------------------------------------------------
pars1,result1,fig1 = Ajuste.gauss_model_outflows(Molines_A_df, cube, 'mean1', channel, plot=True)
plt.title('Outflow Down',fontsize=14)

pars2,result2,fig2 = Ajuste.gauss_model_outflows(Molines_A_df, cube, 'mean2', channel, plot=True)
plt.title('Outflow Up',fontsize=14)

pars3,comps3,result3,fig3 = Ajuste.gauss_model(Molines_A_df, cube, 'mean3', channel, 0.01,plot=True)
plt.title('Center',fontsize=14)
# -----------------------------------------------------------------------------------------------------------------

# Graficos
#------------------------------------------------------------------------------------------------------------------
# Regiones
m2_up = np.zeros_like(m2_escalar)
m2_down = np.zeros_like(m2_escalar)
m2_c = np.zeros_like(m2_escalar)

for x,y in pix:
    if x>40:
        m2_up[x, y] = m2_escalar[x, y]
    else:
        m2_down[x, y] = m2_escalar[x, y]

for x, y in pix_center:
    m2_c[x, y] = m2_escalar[x, y]

plt.figure(figsize=(15,7))

plt.subplot(1,3,1)
plt.title('Upper outflow',fontsize=14)
plt.imshow(m2_up, origin='lower', vmin=5, vmax=66, cmap='terrain_r')
plt.contour(m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
                    linewidths=2, colors='red', linestyles='--')
plt.contour(m2_up, levels=np.array([0.1,0.23, 0.28, 0.4, 0.65, 0.84, 0.99]) * np.nanmax(m2_up)*1.9, 
                        linewidths=0.7, colors='black')

plt.xlabel('J2000 RA offset [arcsec]',fontsize=12)
plt.ylabel('J2000 DEC offset [arcsec]',fontsize=12)

plt.tick_params(axis='both', direction='in', length=5, width=1.5,labelsize=11)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)


plt.subplot(1,3,2)
plt.title('Bottom outflow',fontsize=14)
plt.imshow(m2_down, origin='lower', vmin=5, vmax=66, cmap='terrain_r')
plt.contour(m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
                    linewidths=2, colors='red', linestyles='--')
plt.contour(m2_down, levels=np.array([0.1,0.23, 0.28, 0.4,0.8, 0.98, 0.99]) * np.nanmax(m2_down)*1.9, 
                        linewidths=0.7, colors='black')

plt.xlabel('J2000 RA offset [arcsec]',fontsize=12)
plt.ylabel('J2000 DEC offset [arcsec]',fontsize=12)

plt.tick_params(axis='both', direction='in', length=5, width=1.5,labelsize=11)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)

plt.subplot(1,3,3)
plt.title('Center',fontsize=14)
plt.imshow(m2_c, origin='lower', vmin=5, vmax=66, cmap='terrain_r')
plt.contour(m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
                    linewidths=2, colors='red', linestyles='--')

plt.contour(m2_c, levels=np.array([0.2,0.3,0.6,]) * np.nanmax(m2_c)*1.9, 
                        linewidths=0.7, colors='black')

plt.xlabel('J2000 RA offset [arcsec]',fontsize=12)
plt.ylabel('J2000 DEC offset [arcsec]',fontsize=12)

plt.tick_params(axis='both', direction='in', length=5, width=1.5,labelsize=11)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)

#------------------------------------------------------------------------------------------------------------------
# Espectros
x_datos1 = Molines_A_df['mean1'].index[channel[0]:channel[1]]
y_datos1 = Molines_A_df['mean1'].iloc[channel[0]:channel[1]]

x_datos2 = Molines_A_df['mean2'].index[channel[0]:channel[1]]
y_datos2 = Molines_A_df['mean2'].iloc[channel[0]:channel[1]]

x_datos3 = Molines_A_df['mean3'].index[channel[0]:channel[1]]
y_datos3 = Molines_A_df['mean3'].iloc[channel[0]:channel[1]]

plt.figure(figsize=(15,5))

plt.step(Molines_A_df['mean1'].index[channel[0]:channel[1]],Molines_A_df['mean1'].iloc[channel[0]:channel[1]],color='orangered')
plt.step(Molines_A_df['mean2'].index[channel[0]:channel[1]],Molines_A_df['mean2'].iloc[channel[0]:channel[1]],color='royalblue')
plt.step(Molines_A_df['mean3'].index[channel[0]:channel[1]],Molines_A_df['mean3'].iloc[channel[0]:channel[1]],color='purple')

plt.title(line,fontsize=15)
plt.xlabel('Radio Velocity [km/s]',fontsize=13)
plt.ylabel('$[Jy/beam]$',fontsize=13)

# Formato de los ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(axis='both', direction='in', length=5, width=1.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)

# Ajuste 1
plt.plot(x_datos1, result1.best_fit, '-', color='black')
plt.text(min(x_datos3)+min(x_datos3)/40,max(y_datos3)-max(y_datos3)/11,r'    Down Outflow',color='orangered',fontsize=13)
plt.text(min(x_datos3)+min(x_datos3)/40,max(y_datos3)-2*max(y_datos3)/11,r'$\mu =$ '+str(round(pars1['cen'].value,2))+' km/s',color='orangered',fontsize=13)
plt.text(min(x_datos3)+min(x_datos3)/40,max(y_datos3)-3*max(y_datos3)/11,r'$\sigma =$ '+str(round(pars1['wid'].value,2))+' km/s',color='orangered',fontsize=13)


# Ajuste 2
plt.plot(x_datos2, result2.best_fit, '-', color='black')
plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-max(y_datos3)/11,r'       Up Outflow',color='royalblue',fontsize=13)
plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-2*max(y_datos3)/11,r'$\mu =$ '+str(round(pars2['cen'].value,2))+' km/s',color='royalblue',fontsize=13)
plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-3*max(y_datos3)/11,r'$\sigma =$ '+str(round(pars2['wid'].value,2))+' km/s',color='royalblue',fontsize=13)


# Ajuste 3
plt.plot(x_datos3, result3.best_fit, '-', color='black')
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-max(y_datos3)/11,r'       Center',color='royalblue',fontsize=13)
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-2*max(y_datos3)/11,r'$\mu =$ '+str(round(pars3['cen'].value,2))+' km/s',color='royalblue',fontsize=13)
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-3*max(y_datos3)/11,r'$\sigma =$ '+str(round(pars3['wid'].value,2))+' km/s',color='royalblue',fontsize=13)












