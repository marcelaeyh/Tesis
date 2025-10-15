#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
import ajustenuevo as ajuste
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import pandas as pd

path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_13CO.fits'
path_cont = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'
path_line = '/Users/mac/Tesis/Cut_Cubes/13CO_velocity_nc.txt' 

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

beam = SpectralCube.read(path).beam

m0 = moment0(path)
m2 = moment2(path)

continuum = SpectralCube.read(path_cont)
continuum  = continuum[:,220:300,225:285]

# Coordenadas del píxel de máximo continuo
max_index = np.unravel_index(np.argmax(continuum), continuum.shape)[1:]
max_value = continuum[0][max_index]

box = [220,225,300,285]
channel = [350,520]
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
Molines_A_df['mean'] = Molines_A_df.sum(axis=1)/4800


linea = pd.read_csv(path_line, skiprows=range(7), sep=' ')
linea.columns = ['Velocity', 'Value']
linea = linea.set_index('Velocity')

pars,comps, out,matplotlib_fig = ajuste.ajuste_chisqr(linea, channel, px='Value')


#for i in range(len(cube_include)):
#    plt.figure()
#    plt.imshow(cube_include[i,:,:].value)


# --- Ejes ---
ny = abs(box[2] - box[0])
nx = abs(box[3] - box[1])
delta = 3.61e-06 * 3600  # tamaño píxel en arcsec

# Offsets centrados en el máximo continuo
y0, x0 = max_index  # fila (DEC), columna (RA)
x_offsets = (np.arange(nx) - x0) * delta
y_offsets = (np.arange(ny) - y0) * delta

# extent para imshow
extent = [x_offsets[0], x_offsets[-1], y_offsets[0], y_offsets[-1]]

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

#plt.figure(figsize=(15, 12))

# Configuración de la cuadrícula
#gs = plt.GridSpec(2, 2, height_ratios=[3, 1])  # 2 filas, 2 columnas; la primera fila tiene dos gráficos, la segunda solo uno.

# Primer gráfico (Moment 0)
plt.figure(figsize=(15, 10))
#ax1 = plt.subplot(gs[0, 0]) # Primer gráfico en la primera columna
ax1 = plt.subplot()

# Imagen moment 0 con extent en arcsec
im1 = ax1.imshow(m0.value, origin='lower', cmap=new_cmap.reversed(), extent=extent)

idx_max = np.nanargmax(m0.value)

# coordenadas del valor máximo
DEC,RA = np.unravel_index(idx_max, m0.value.shape)
#ax1.plot(x_offsets[RA],y_offsets[DEC],'o')

# Ejes y etiquetas
ax1.set_xlabel('J2000 RA offset (′′)',fontsize=17)
ax1.set_ylabel('J2000 DEC offset (′′)',fontsize=17)

# Barra de color
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('(Jy/Beam . km/s)', fontsize=17)
cbar1.ax.tick_params(labelsize=16)

# Contornos moment 0
contours1 = ax1.contour(x_offsets, y_offsets, m0.value,
                        levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1),
                        linewidths=0.7, colors='black')
ax1.clabel(contours1, inline=True, fontsize=12)

# Contornos del continuo
cont1 = ax1.contour(x_offsets, y_offsets, continuum[0].value,
                    levels=np.array([0.3, 0.5, 0.7, 0.9, 1]) * np.nanmax(continuum[0].value),
                    linewidths=2, colors='red', linestyles='--')

# Leyenda
dust_contour_legend = Line2D([], [], color='red', linestyle='--', linewidth=2,
                             label='Dust Continuum Emission')
ax1.legend(handles=[dust_contour_legend], fontsize=17)
ax1.minorticks_on()
ax1.tick_params(axis='both', which='major', labelsize=17)  # números más grandes
ax1.tick_params(axis='both', which='minor', labelsize=12)  # si tienes minor ticks


# Beam
beam_major = beam.minor.value*3600
beam_minor = beam.major.value*3600

# padding desde los bordes en arcsec
pad_x = 0.05 * (extent[1] - extent[0])
pad_y = 0.05 * (extent[3] - extent[2])

beam_cx = extent[1] - pad_x - beam_major / 2
beam_cy = extent[2] + pad_y + beam_minor / 2

beam_ellipse = Ellipse((beam_cx, beam_cy),
                       width=beam_major, height=beam_minor,
                       angle=beam.pa.value,
                       edgecolor='black', facecolor='lightgray', lw=1, zorder=5)
ax1.add_patch(beam_ellipse)

# Escala en UA
d = np.array([4.38, 5.4, 8.5])  # pc
xx = 500  # tamaño en AU
ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi

scale_x_start = extent[0] +0.08
scale_x_end = scale_x_start + ang[0]
scale_y = extent[2] + pad_y +0.02

ax1.hlines(scale_y, scale_x_start, scale_x_end, colors='k', linewidth=2, zorder=6)
ax1.text(scale_x_start+0.05, scale_y - 0.02 * (extent[3] - extent[2]),
         f'{int(xx)} AU', ha='center', va='top', fontsize=16)



#ax1.hlines(12,16,ang[0]/delta+16,color='k')
#ax1.text(16,10,'500 UA',fontsize=9)
#ax1.text(5,11,'d=4.38 kpc',fontsize=11)

#ax1.hlines(9,16,ang[1]/delta+16,color='k')
#ax1.text(16,7,'500 UA',fontsize=9)
#ax1.text(5,8,'d=5.4 kpc',fontsize=11)

#ax1.hlines(6,16,ang[2]/delta+16,color='k')
#ax1.text(16,4,'500 UA',fontsize=9)
#ax1.text(5,5,'d=8.5 kpc',fontsize=11)


plt.savefig('/Users/mac/Tesis/Cut_Cubes/Emission_Figures/13CO_moment0_44kpc.png', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución


plt.figure(figsize=(15,10))
# Segundo gráfico (Moment 2)
#ax2 = plt.subplot(gs[0, 1])  # Segundo gráfico en la segunda columna
ax2 = plt.subplot()
m2_escalar = m2.value * 1.9

im2 = ax2.imshow(m2_escalar, origin='lower', cmap='terrain_r', extent=extent,
                 vmin=5, vmax=66)


ax2.set_xlabel('J2000 RA offset (′′)',fontsize=17)
ax2.set_ylabel('J2000 DEC offset (′′)',fontsize=17)

# Añadir la barra de colores
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('(km/s)', fontsize=17)
cbar2.ax.tick_params(labelsize=16)
# Contornos en el segundo gráfico
contours2 = ax2.contour(x_offsets,y_offsets,m2_escalar, levels=np.array([0.23, 0.28, 0.4, 0.65, 0.84]) * np.nanmax(m2.value), 
                        linewidths=0.7, colors='black')
ax2.clabel(contours2, inline=True, fontsize=12)

# Contornos sobre el segundo gráfico
cont2 = ax2.contour(x_offsets,y_offsets,m0.value, levels=np.array([0.4, 0.6, 0.8, 0.94]) * round(np.nanmax(m0.value), 1), 
                    linewidths=2, colors='red', linestyles='--')

#cont2 = ax2.contour(continuum[0].value, levels=np.array([0.3, 0.5, 0.7, 0.9, 1]) * np.nanmax(continuum[0].value), 
#                    linewidths=2, colors='red', linestyles='--')


ax2.legend(handles=[dust_contour_legend], fontsize=17)
ax2.minorticks_on()
ax2.tick_params(axis='both', which='major', labelsize=17)  # números más grandes
ax2.tick_params(axis='both', which='minor', labelsize=12)  # si tienes minor ticks


# Beam
beam_ellipse = Ellipse((beam_cx, beam_cy),
                       width=beam_major, height=beam_minor,
                       angle=beam.pa.value,
                       edgecolor='black', facecolor='lightgray', lw=1, zorder=5)
ax2.add_patch(beam_ellipse)

ax2.hlines(scale_y, scale_x_start, scale_x_end, colors='k', linewidth=2, zorder=6)
ax2.text(scale_x_start+0.05, scale_y - 0.02 * (extent[3] - extent[2]),
         f'{int(xx)} AU', ha='center', va='top', fontsize=16)


plt.savefig('/Users/mac/Tesis/Cut_Cubes/Emission_Figures/13CO_moment2_44kpc.png', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución


plt.figure(figsize=(15,5))

# Tercer gráfico (gráfico adicional cubriendo toda la segunda fila)
#ax3 = plt.subplot(gs[1, :])  # Un solo gráfico que cubre ambas columnas en la segunda fila
ax3 = plt.subplot()

x = linea['Value'].index[channel[0]:channel[1]]
y = linea['Value'].iloc[channel[0]:channel[1]]

#channel = [70,240]
#pars,comps, out,matplotlib_fig = ajuste.ajuste_chisqr(Molines_A_df, channel, px='mean')

#x = Molines_A_df['mean'].index[channel[0]:channel[1]]
ejey = Molines_A_df['mean'].iloc[70:240]

ax3.step(x,y,color='black')

ax3.set_xlabel('Velocity (km/s)',fontsize=17)
ax3.set_ylabel('(mJy/Beam)',fontsize=17)

for i in range(2):
    if i == 0:
        c='blue'
    else:
        c = 'red'
    ax3.plot(x, comps['peak%d_' % (i+1)],'--',color=c)


vsys = (round(pars['peak1_center'].value,2) + round(pars['peak2_center'].value,2)) / 2
ax3.axvline(vsys,label='Vsys = '+str(vsys)+' km/s',color='green')


ylabels = np.linspace(-1,8,10)
ax3.set_yticklabels([f'{int(val)}' for val in ylabels])

# Ajuste 
ax3.plot(x, out.best_fit, '-', color='purple')
ax3.text(min(x)+10,max(y)-max(y)/11,r'$\mu =$ '+str(round(pars['peak1_center'].value,2))+' km/s',color='blue',fontsize=15)
ax3.text(min(x)+10,max(y)-2*max(y)/11,r'$\sigma =$ '+str(round(pars['peak1_sigma'].value,2))+' km/s',color='blue',fontsize=15)

ax3.text(min(x)+10,max(y)-3*max(y)/11,r'$\mu =$ '+str(round(pars['peak2_center'].value,2))+' km/s',color='red',fontsize=15)
ax3.text(min(x)+10,max(y)-4*max(y)/11,r'$\sigma =$ '+str(round(pars['peak2_sigma'].value,2))+' km/s',color='red',fontsize=15)

ax3.minorticks_on()
ax3.tick_params(axis='both', which='major', labelsize=16)  # números más grandes
ax3.tick_params(axis='both', which='minor', labelsize=12)  # si tienes minor ticks

ax3.set_xlim(min(x),max(x))
ax3.legend(fontsize=17)
# Ajustar el layout para evitar que los gráficos se superpongan
plt.tight_layout()

output_filename = '/Users/mac/Tesis/Cut_Cubes/Emission_Figures/13CO_mean_emission.png'  
plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución


