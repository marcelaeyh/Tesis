#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cube_to_df as ctdf
import ajustenuevo as Ajuste
from lmfit import Model
from lmfit.models import GaussianModel
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

line = '13CO'
#path = '/home/marcela/Tesis Marcela/IRAS15445_recortados/I15445.mstransform_cube_contsub_'+line+'.fits'
path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_'+line+'.fits'

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

pix_des = np.argwhere(m0.value < 1.2)


# Separación de regiones
pix_updown = np.argwhere(m2_escalar >= 38.56)

pix_center = np.argwhere(m0.value > 2.415)

pix_brown = np.argwhere(m2_escalar <= 24.5)

set_center = set(map(tuple, pix_center))
pix_brown = np.array([pix for pix in pix_brown if tuple(pix) not in set_center])

pix_yellow = np.argwhere((m2_escalar < 38.56) & (m2_escalar > 24.5))


set_des = set(map(tuple, pix_des))

pix_updown = np.array([pix for pix in pix_updown if tuple(pix) not in set_des])
pix_center = np.array([pix for pix in pix_center if tuple(pix) not in set_des])
pix_yellow = np.array([pix for pix in pix_yellow if tuple(pix) not in set_des])
pix_brown = np.array([pix for pix in pix_brown if tuple(pix) not in set_des])

# Upper Outflow 
columns1 = [f'Pix_{x}_{y}' for y, x in pix_updown if y > 40]
columns1 = [col for col in columns1 if col in Molines_A_df.columns]

# Bottom Outflow 
columns2 = [f'Pix_{x}_{y}' for y, x in pix_updown if y < 40]
columns2 = [col for col in columns2 if col in Molines_A_df.columns]

# Green Outflows
columns3 = [f'Pix_{x}_{y}' for y, x in pix_updown]
columns3 = [col for col in columns3 if col in Molines_A_df.columns]

# Upper Yellow 
columns4 = [f'Pix_{x}_{y}' for y, x in pix_yellow if y >40]
columns4 = [col for col in columns4 if col in Molines_A_df.columns]

# Bottom Yellow 
columns5 = [f'Pix_{x}_{y}' for y, x in pix_yellow if y < 40]
columns5 = [col for col in columns5 if col in Molines_A_df.columns]

# Yellow 
columns6 = [f'Pix_{x}_{y}' for y, x in pix_yellow]
columns6 = [col for col in columns6 if col in Molines_A_df.columns]

# Left Brown 
columns7 = [f'Pix_{x}_{y}' for y, x in pix_brown if x <30]
columns7 = [col for col in columns7 if col in Molines_A_df.columns]

# Right Brown
columns8 = [f'Pix_{x}_{y}' for y, x in pix_brown if x >30]
columns8 = [col for col in columns8 if col in Molines_A_df.columns]

# Brown
columns9 = [f'Pix_{x}_{y}' for y, x in pix_brown]
columns9 = [col for col in columns9 if col in Molines_A_df.columns]

# Left Center
columns10 = [f'Pix_{x}_{y}' for y, x in pix_center if x<30]
columns10 = [col for col in columns10 if col in Molines_A_df.columns]

# Right Center
columns11 = [f'Pix_{x}_{y}' for y, x in pix_center if x> 30]
columns11 = [col for col in columns11 if col in Molines_A_df.columns]

# Center
columns12 = [f'Pix_{x}_{y}' for y, x in pix_center]
columns12 = [col for col in columns12 if col in Molines_A_df.columns]

Molines_filtrado1 = Molines_A_df[columns1]
Molines_filtrado2 = Molines_A_df[columns2]
Molines_filtrado3 = Molines_A_df[columns3]
Molines_filtrado4 = Molines_A_df[columns4]
Molines_filtrado5 = Molines_A_df[columns5]
Molines_filtrado6 = Molines_A_df[columns6]
Molines_filtrado7 = Molines_A_df[columns7]
Molines_filtrado8 = Molines_A_df[columns8]
Molines_filtrado9 = Molines_A_df[columns9]
Molines_filtrado10 = Molines_A_df[columns10]
Molines_filtrado11 = Molines_A_df[columns11]
Molines_filtrado12 = Molines_A_df[columns12]

Molines_A_df['mean1'] = Molines_filtrado1.sum(axis=1)/595
Molines_A_df['mean2'] = Molines_filtrado2.sum(axis=1)/595
Molines_A_df['mean3'] = Molines_filtrado3.sum(axis=1)/595
Molines_A_df['mean4'] = Molines_filtrado4.sum(axis=1)/595
Molines_A_df['mean5'] = Molines_filtrado5.sum(axis=1)/595
Molines_A_df['mean6'] = Molines_filtrado6.sum(axis=1)/595
Molines_A_df['mean7'] = Molines_filtrado7.sum(axis=1)/595
Molines_A_df['mean8'] = Molines_filtrado8.sum(axis=1)/595
Molines_A_df['mean9'] = Molines_filtrado9.sum(axis=1)/595
Molines_A_df['mean10'] = Molines_filtrado10.sum(axis=1)/595
Molines_A_df['mean11'] = Molines_filtrado11.sum(axis=1)/595
Molines_A_df['mean12'] = Molines_filtrado12.sum(axis=1)/595

# Ajustes
# ------------------------------------------------------------------------------------------------------------------
pars1, comps1, result1, fig1 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean1')
plt.title('Upper Outflow ',fontsize=14)

pars2, comps2, result2, fig2 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean2')
plt.title('Bottom Outflow ',fontsize=14)

pars3, comps3, result3, fig3 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean3')
plt.title('Green Outflows',fontsize=14)

pars4, comps4, result4, fig4 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean4')
plt.title('Upper Yellow ',fontsize=14)

pars5, comps5, result5, fig5 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean5')
plt.title('Bottom Yellow ',fontsize=14)

pars6, comps6, result6, fig6 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean6')
plt.title('Yellow',fontsize=14)

pars7, comps7, result7, fig7 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean7')
plt.title('Left Brown',fontsize=14)

pars8, comps8, result8, fig8 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean8')
plt.title('Right Brown ',fontsize=14)

pars9, comps9, result9, fig9 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean9')
plt.title('Brown',fontsize=14)

pars10, comps10, result10, fig10 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean10')
plt.title('Left Center',fontsize=14)

pars11, comps11, result11, fig11 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean11')
plt.title('Right Center',fontsize=14)

pars12, comps12, result12, fig12 = Ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean12')
plt.title('Center',fontsize=14)
# -----------------------------------------------------------------------------------------------------------------

# Graficos
#------------------------------------------------------------------------------------------------------------------
# Regiones
m2_1 = np.zeros_like(m2_escalar)
m2_2 = np.zeros_like(m2_escalar)
m2_3 = np.zeros_like(m2_escalar)
m2_4 = np.zeros_like(m2_escalar)
m2_5 = np.zeros_like(m2_escalar)
m2_6 = np.zeros_like(m2_escalar)
m2_7 = np.zeros_like(m2_escalar)
m2_8 = np.zeros_like(m2_escalar)
m2_9 = np.zeros_like(m2_escalar)
m2_10 = np.zeros_like(m2_escalar)
m2_11 = np.zeros_like(m2_escalar)
m2_12 = np.zeros_like(m2_escalar)

for x,y in pix_updown:
    m2_3[x, y] = m2_escalar[x, y]
    if x>40:
        m2_1[x, y] = m2_escalar[x, y]
    else:
        m2_2[x, y] = m2_escalar[x, y]

for x,y in pix_yellow:
    m2_6[x, y] = m2_escalar[x, y]
    if x>40:
        m2_4[x, y] = m2_escalar[x, y]
    else:
        m2_5[x, y] = m2_escalar[x, y]
        
for x,y in pix_brown:
    m2_9[x, y] = m2_escalar[x, y]
    if y<30:
        m2_7[x, y] = m2_escalar[x, y]
    else:
        m2_8[x, y] = m2_escalar[x, y]

for x,y in pix_center:
    m2_12[x, y] = m2_escalar[x, y]
    if y<30:
        m2_10[x, y] = m2_escalar[x, y]
    else:
        m2_11[x, y] = m2_escalar[x, y]
        
fig, axes = plt.subplots(4, 3, figsize=(15, 20), subplot_kw={'box_aspect': 1})

# Definir el colormap y los límites comunes
cmap = 'terrain_r'
vmin, vmax = 5, 66

# Lista de datos y títulos
data_list = [m2_1,m2_2,m2_3,m2_4,m2_5,m2_6,m2_7,m2_8,m2_9,m2_10,m2_11,m2_12]
titles = ['Upper Outflow', 'Bottom Outflow ','Green Outflows', 'Upper Yellow','Bottom Yellow','Yellow', 'Left Brown', 'Right Brown', 'Brown', 'Left Center','Right Center', 'Center']

# Aplanar la grilla de ejes y recorrer solo los necesarios
for i in range(len(data_list)):
    ax = axes.flat[i]  # Tomamos cada subplot en orden
    data = data_list[i]
    title = titles[i]

    im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.contour(m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
               linewidths=2, colors='red', linestyles='--')
    
    contour_levels = np.array([0.2, 0.3, 0.6]) * np.nanmax(data) * 1.9
    ax.contour(data, levels=contour_levels, linewidths=0.7, colors='black')
    
    ax.set_title(title, fontsize=14)
    #ax.set_xlabel('J2000 RA offset [arcsec]', fontsize=12)
    #ax.set_ylabel('J2000 DEC offset [arcsec]', fontsize=12)
    
    ax.set_xlabel('pix', fontsize=12)
    ax.set_ylabel('pix', fontsize=12)
    
    ax.tick_params(axis='both', direction='in', length=5, width=1.5, labelsize=11)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=2, width=1)

# Desactivar el último subplot vacío
#axes[2, 1].axis("off")

# Agregar la barra de colores común
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Intensity')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Ajustar para que la colorbar no se sobreponga


#------------------------------------------------------------------------------------------------------------------
# Espectros
x_datos1 = Molines_A_df['mean1'].index[channel[0]:channel[1]]
y_datos1 = Molines_A_df['mean1'].iloc[channel[0]:channel[1]]

x_datos2 = Molines_A_df['mean2'].index[channel[0]:channel[1]]
y_datos2 = Molines_A_df['mean2'].iloc[channel[0]:channel[1]]

x_datos3 = Molines_A_df['mean3'].index[channel[0]:channel[1]]
y_datos3 = Molines_A_df['mean3'].iloc[channel[0]:channel[1]]

x_datos4 = Molines_A_df['mean4'].index[channel[0]:channel[1]]
y_datos4 = Molines_A_df['mean4'].iloc[channel[0]:channel[1]]

x_datos5 = Molines_A_df['mean5'].index[channel[0]:channel[1]]
y_datos5 = Molines_A_df['mean5'].iloc[channel[0]:channel[1]]

x_datos6 = Molines_A_df['mean6'].index[channel[0]:channel[1]]
y_datos6 = Molines_A_df['mean6'].iloc[channel[0]:channel[1]]

x_datos7 = Molines_A_df['mean7'].index[channel[0]:channel[1]]
y_datos7 = Molines_A_df['mean7'].iloc[channel[0]:channel[1]]

x_datos8 = Molines_A_df['mean8'].index[channel[0]:channel[1]]
y_datos8 = Molines_A_df['mean8'].iloc[channel[0]:channel[1]]

x_datos9 = Molines_A_df['mean9'].index[channel[0]:channel[1]]
y_datos9 = Molines_A_df['mean9'].iloc[channel[0]:channel[1]]

x_datos10 = Molines_A_df['mean10'].index[channel[0]:channel[1]]
y_datos10 = Molines_A_df['mean10'].iloc[channel[0]:channel[1]]

x_datos11 = Molines_A_df['mean11'].index[channel[0]:channel[1]]
y_datos11 = Molines_A_df['mean11'].iloc[channel[0]:channel[1]]

x_datos12 = Molines_A_df['mean12'].index[channel[0]:channel[1]]
y_datos12 = Molines_A_df['mean12'].iloc[channel[0]:channel[1]]



plt.figure(figsize=(15,9))
plt.title(line,fontsize=15)
plt.xlabel('Radio Velocity [km/s]',fontsize=13)
plt.ylabel('$[Jy/beam]$',fontsize=13)

# Formato de los ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(axis='both', direction='in', length=5, width=1.5)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)


for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
    plt.step(Molines_A_df['mean'+str(i)].index[channel[0]:channel[1]],Molines_A_df['mean'+str(i)].iloc[channel[0]:channel[1]],label=titles[i-1])

plt.plot(x_datos1, result1.best_fit, '-', color='black')
plt.plot(x_datos2, result2.best_fit, '-', color='black')
plt.plot(x_datos3, result3.best_fit, '-', color='black')
plt.plot(x_datos4, result4.best_fit, '-', color='black')
plt.plot(x_datos5, result5.best_fit, '-', color='black')
plt.plot(x_datos6, result6.best_fit, '-', color='black')
plt.plot(x_datos7, result7.best_fit, '-', color='black')
plt.plot(x_datos8, result8.best_fit, '-', color='black')
plt.plot(x_datos9, result9.best_fit, '-', color='black')
plt.plot(x_datos10, result10.best_fit, '-', color='black')
plt.plot(x_datos11, result11.best_fit, '-', color='black')
plt.plot(x_datos12, result12.best_fit, '-', color='black')


plt.legend(fontsize=15)





# Ajuste 1
plt.plot(x_datos1, result1.best_fit, '-', color='black')
#plt.text(min(x_datos3)+min(x_datos3)/40,max(y_datos3)-max(y_datos3)/11,r'    Down Outflow',color='orangered',fontsize=13)
#plt.text(min(x_datos3)+min(x_datos3)/40,max(y_datos3)-2*max(y_datos3)/11,r'$\mu =$ '+str(round(pars1['cen'].value,2))+' km/s',color='orangered',fontsize=13)
#plt.text(min(x_datos3)+min(x_datos3)/40,max(y_datos3)-3*max(y_datos3)/11,r'$\sigma =$ '+str(round(pars1['wid'].value,2))+' km/s',color='orangered',fontsize=13)


# Ajuste 2
plt.plot(x_datos2, result2.best_fit, '-', color='black')
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-max(y_datos3)/11,r'       Up Outflow',color='royalblue',fontsize=13)
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-2*max(y_datos3)/11,r'$\mu =$ '+str(round(pars2['cen'].value,2))+' km/s',color='royalblue',fontsize=13)
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-3*max(y_datos3)/11,r'$\sigma =$ '+str(round(pars2['wid'].value,2))+' km/s',color='royalblue',fontsize=13)


# Ajuste 3
plt.plot(x_datos3, result3.best_fit, '-', color='black')
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-max(y_datos3)/11,r'       Center',color='royalblue',fontsize=13)
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-2*max(y_datos3)/11,r'$\mu =$ '+str(round(pars3['cen'].value,2))+' km/s',color='royalblue',fontsize=13)
#plt.text(max(x_datos3)-max(x_datos3),max(y_datos3)-3*max(y_datos3)/11,r'$\sigma =$ '+str(round(pars3['wid'].value,2))+' km/s',color='royalblue',fontsize=13)

# Ajuste 4
#plt.plot(x_datos4, result4.best_fit, '-', color='black')

# Ajuste 5
#plt.plot(x_datos5, result5.best_fit, '-', color='black')









