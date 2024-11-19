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

path = '/home/marcela/Tesis Marcela/IRAS15445_recortados/I15445.mstransform_cube_contsub_13CO.fits'
path_cont = '/home/marcela/Tesis Marcela/IRAS15445_recortados/All_spw_continuum_temp.fits'

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

m0 = moment0(path)
m2 = moment2(path)

continuum = SpectralCube.read(path_cont)
continuum  = continuum[:,220:300,225:285]

box = [220,225,300,285]
channel=[85,230]
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
Molines_A_df['mean'] = Molines_A_df.sum(axis=1)/4800

pars,comps, out,matplotlib_fig = Ajuste.gauss_model(Molines_A_df, cube, 'mean', channel, 0.1, plot=True)

#for i in range(len(cube_include)):
#    plt.figure()
#    plt.imshow(cube_include[i,:,:].value)

# Ejes
ny = abs(box[2]-box[0])
nx = abs(box[3]-box[1])

# Tamaño de píxel en segundos de arco
delta = 3.611111111111e-06 * 3600

x = -np.concatenate([np.arange(-nx/2,0)*delta, np.arange(nx/2)*delta])
y = np.concatenate([np.arange(-ny/2,0)*delta, np.arange(ny/2)*delta])

# ---------------------------------------------------------------------------------------
# Obtener el colormap 'rainbow_r' original
cmap = plt.get_cmap('rainbow')

# Crear una versión personalizada del colormap
n_colors = 256  # Número de colores en el colormap
colors = cmap(np.linspace(0, 1, n_colors))

# Desvanecer los últimos colores (la parte roja) hacia el blanco
fade_start = int(0.6 * n_colors)  # Ajusta el punto de inicio del desvanecimiento
for i in range(fade_start, n_colors):
    # Mezclar progresivamente con blanco
    blend_factor = (i - fade_start) / (n_colors - fade_start)
    colors[i] = blend_factor * np.array([1, 1, 1, 1]) + (1 - blend_factor) * colors[i]

# Crear un nuevo colormap con los colores modificados
new_cmap = LinearSegmentedColormap.from_list('rainbow_fade_red', colors)
# -----------------------------------------------------------------------------------------

plt.figure(figsize=(15, 12))

# Configuración de la cuadrícula
gs = plt.GridSpec(2, 2, height_ratios=[3, 1])  # 2 filas, 2 columnas; la primera fila tiene dos gráficos, la segunda solo uno.

# Primer gráfico (Moment 0)
ax1 = plt.subplot(gs[0, 0])  # Primer gráfico en la primera columna
im1 = ax1.imshow(m0.value, origin='lower', cmap=new_cmap.reversed())

# Cambiar los nombres de los ejes (usar x e y)
ax1.set_xticks(np.linspace(0, nx-1, 7))
ax1.set_xticklabels(np.round(np.linspace(x.max(), x.min(), 7), 1),fontsize=12)
ax1.set_yticks(np.linspace(0, ny-1, 7))
ax1.set_yticklabels(np.round(np.linspace(y.min(), y.max(), 7), 1),fontsize=12)

# Añadir la barra de colores
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('[Jy/Beam . km/s]', fontsize=12)
cbar1.ax.tick_params(labelsize=12)
# Contornos en el primer gráfico
contours1 = ax1.contour(m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
                        linewidths=0.7, colors='black')
ax1.clabel(contours1, inline=True, fontsize=12)
ax1.set_title('Moment 0', fontsize=14)
ax1.set_xlabel('J2000 RA offset [arcsec]',fontsize=12)
ax1.set_ylabel('J2000 DEC offset [arcsec]',fontsize=12)

# Contornos en continuo
cont1 = ax1.contour(continuum[0].value, levels=np.array([0.3, 0.5, 0.7, 0.9, 1]) * np.nanmax(continuum[0].value), 
                    linewidths=2, colors='red', linestyles='--')

dust_contour_legend = Line2D([], [], color='red', linestyle='--', linewidth=2, label='Dust Continuum Emission')
ax1.legend(handles=[dust_contour_legend], fontsize=14)

# Segundo gráfico (Moment 2)
ax2 = plt.subplot(gs[0, 1])  # Segundo gráfico en la segunda columna
m2_escalar = m2.value * 1.9
im2 = ax2.imshow(m2_escalar, origin='lower', vmin=5, vmax=66, cmap='terrain_r')

# Cambiar los nombres de los ejes (usar x e y)
ax2.set_xticks(np.linspace(0, nx-1, 7))
ax2.set_xticklabels(np.round(np.linspace(x.max(), x.min(), 7), 1),fontsize=12)
ax2.set_yticks(np.linspace(0, ny-1, 7))
ax2.set_yticklabels(np.round(np.linspace(y.min(), y.max(), 7), 1),fontsize=12)
ax2.set_xlabel('J2000 RA offset [arcsec]',fontsize=12)
ax2.set_ylabel('J2000 DEC offset [arcsec]',fontsize=12)

# Añadir la barra de colores
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('[km/s]', fontsize=12)
cbar2.ax.tick_params(labelsize=12)
# Contornos en el segundo gráfico
contours2 = ax2.contour(m2_escalar, levels=np.array([0.23, 0.28, 0.4, 0.65, 0.84, 0.99]) * np.nanmax(m2.value), 
                        linewidths=0.7, colors='black')
ax2.clabel(contours2, inline=True, fontsize=12)
ax2.set_title('Moment 2', fontsize=14)

# Contornos sobre el segundo gráfico
cont2 = ax2.contour(m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
                    linewidths=2, colors='red', linestyles='--')

dust_contour_legend = Line2D([], [], color='red', linestyle='--', linewidth=2, label='Moment 0 Contours')
ax2.legend(handles=[dust_contour_legend], fontsize=14)


plt.figure(figsize=(15,5))
# Tercer gráfico (gráfico adicional cubriendo toda la segunda fila)
#ax3 = plt.subplot(gs[1, :])  # Un solo gráfico que cubre ambas columnas en la segunda fila
ax3 = plt.subplot()
ax3.step(Molines_A_df.sum(axis=1).index[channel[0]:channel[1]],Molines_A_df.sum(axis=1).iloc[channel[0]:channel[1]]/4800,color='black')
ax3.set_title('13CO total line emission', fontsize=12)
ax3.set_xlabel('Radio Velocity [km/s]',fontsize=12)
ax3.set_ylabel('[Jy/Beam]',fontsize=12)

x = Molines_A_df['mean'].index[channel[0]:channel[1]]
y = Molines_A_df['mean'].iloc[channel[0]:channel[1]]

for i in range(2):
    if i == 0:
        c='blue'
    else:
        c = 'red'
    ax3.plot(x, comps['peak%d_' % (i+1)],'--',color=c)

ax3.tick_params(axis='both', direction='in', length=5, width=1.5,labelsize=12)
ax3.minorticks_on()
ax3.tick_params(axis='both', which='minor', direction='in', length=2, width=1)


vsys = round(x[list(out.best_fit).index(out.best_fit[out.best_fit==max(out.best_fit)])],2)
ax3.vlines(vsys, 0,  max(out.best_fit),color='green',label='Vsys = '+str(vsys)+'km/s')

# Ajuste 
ax3.plot(x, out.best_fit, '-', color='purple')
ax3.text(min(x)+min(x)/40,max(y)-max(y)/11,r'$\mu =$ '+str(round(pars['peak1_center'].value,2))+' km/s',color='blue',fontsize=13)
ax3.text(min(x)+min(x)/40,max(y)-2*max(y)/11,r'$\sigma =$ '+str(round(pars['peak1_sigma'].value,2))+' km/s',color='blue',fontsize=13)

ax3.text(min(x)+min(x)/40,max(y)-3*max(y)/11,r'$\mu =$ '+str(round(pars['peak2_center'].value,2))+' km/s',color='red',fontsize=13)
ax3.text(min(x)+min(x)/40,max(y)-4*max(y)/11,r'$\sigma =$ '+str(round(pars['peak2_sigma'].value,2))+' km/s',color='red',fontsize=13)

ax3.legend(fontsize=13)
# Ajustar el layout para evitar que los gráficos se superpongan
plt.tight_layout()

output_filename = '13COline.png'  # Cambia la extensión si prefieres otro formato como .pdf, .svg, etc.
plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución
