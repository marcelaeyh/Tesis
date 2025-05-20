#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cube_to_df as ctdf
import ajustenuevo as ajuste
from lmfit import Model
from lmfit.models import GaussianModel
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyBboxPatch


path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_SO2_21_21.fits'
path_cont = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'

def moment0(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[70:160,230:290,235:278]
    cube_include = cube_cut.with_mask(cube_cut > 0.0095*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m0 = cube_include.moment(order=0)/1000 
   
    return m0

def moment2(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[70:160,230:290,235:278]
    cube_include = cube_cut.with_mask(cube_cut > 0.0095*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 # pasar a km/s
   
    return m2

m0 = moment0(path)
m2 = moment2(path)

continuum = SpectralCube.read(path_cont)
continuum  = continuum[:,230:290,235:278]

box = [230,235,290,278]
channel = [60,180]
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
Molines_A_df['mean'] = Molines_A_df.sum(axis=1)/4800

pars,comps, out,matplotlib_fig = ajuste.ajuste_chisqr(cube, Molines_A_df, channel, px='mean')

beam = SpectralCube.read(path).beam

#for i in range(len(cube_include)):
#    plt.figure()
#    plt.imshow(cube_include[i,:,:].value)

# Ejes
ny = abs(box[2]-box[0])
nx = abs(box[3]-box[1])

# Tamaño de píxel en segundos de arco
delta = 3.61e-06 * 3600

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

#plt.figure(figsize=(15,10))
# Configuración de la cuadrícula
gs = plt.GridSpec(2, 2, height_ratios=[3, 1])  # 2 filas, 2 columnas; la primera fila tiene dos gráficos, la segunda solo uno.

#ax1 = plt.subplot()
# Primer gráfico (Moment 0)
ax1 = plt.subplot(gs[0, 0])  # Primer gráfico en la primera columna

im1 = ax1.imshow(m0.value, origin='lower', cmap=new_cmap.reversed())

# Cambiar los nombres de los ejes (usar x e y)
ax1.set_xticks(np.linspace(0, nx-1, 7))
ax1.set_xticklabels([0.27,0.18,0.09,0,-0.09,-0.18,-0.27],fontsize=12)
ax1.set_yticks(np.linspace(0, ny-1, 7))
ax1.set_yticklabels([-0.36,-0.24,-0.12,0,0.12,0.24,0.36],fontsize=12)

# Añadir la barra de colores
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('[Jy/Beam . km/s]', fontsize=12)
cbar1.ax.tick_params(labelsize=12)
# Contornos en el primer gráfico
contours1 = ax1.contour(m0.value, levels=np.array([0.35,0.45, 0.6, 0.8, 0.9]) * round(np.nanmax(m0.value), 1), 
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
ax1.minorticks_on()

# Tamaños UA

d = np.array([4.38,5.4,8.5])
xx = 500 #UA
ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi

#ax1.hlines(30,17,ang[0]/delta+17,color='k')

ax1.hlines(5,7,ang[0]/delta+7,color='k')
ax1.text(7,2,'500 UA',fontsize=12)

# Beam
beam_ellipse = Ellipse(
    (38, 5), width=beam.minor.value/delta*3600, height=beam.major.value/delta*3600, angle=beam.pa.value,
    edgecolor='black', facecolor='lightgray', lw=1)

# Agregar la elipse al gráfico
ax1.add_patch(beam_ellipse)

#ax1.hlines(10,13,ang[0]/delta+13,color='k')
#ax1.text(13,8,'500 UA',fontsize=9)
#ax1.text(4,9,'d=4.38 kpc',fontsize=11)

#ax1.hlines(7,13,ang[1]/delta+13,color='k')
#ax1.text(13,5,'500 UA',fontsize=9)
#ax1.text(4,6,'d=5.4 kpc',fontsize=11)

#ax1.hlines(4,13,ang[2]/delta+13,color='k')
#ax1.text(13,2,'500 UA',fontsize=9)
#ax1.text(4,3,'d=8.5 kpc',fontsize=11)


#plt.savefig('SO2_21_21_moment0_44kpc.png', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución


# Segundo gráfico (Moment 2)
ax2 = plt.subplot(gs[0, 1])  # Segundo gráfico en la segunda columna
#plt.figure(figsize=(15,10))
#ax2 = plt.subplot()
m2_escalar = m2.value * 2.1
im2 = ax2.imshow(m2_escalar, origin='lower', vmin=7, vmax=46, cmap='terrain_r')

# Cambiar los nombres de los ejes (usar x e y)
ax2.set_xticks(np.linspace(0, nx-1, 7))
ax2.set_xticklabels([0.27,0.18,0.09,0,-0.09,-0.18,-0.27],fontsize=12)
ax2.set_yticks(np.linspace(0, ny-1, 7))
ax2.set_yticklabels([-0.36,-0.24,-0.12,0,0.12,0.24,0.36],fontsize=12)

ax2.set_xlabel('J2000 RA offset [arcsec]',fontsize=12)
ax2.set_ylabel('J2000 DEC offset [arcsec]',fontsize=12)

# Añadir la barra de colores
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('[km/s]', fontsize=12)
cbar2.ax.tick_params(labelsize=12)
# Contornos en el segundo gráfico
contours2 = ax2.contour(m2_escalar, levels=np.array([0.29, 0.38, 0.5,0.66, 0.85]) * np.nanmax(m2.value), 
                        linewidths=0.7, colors='black')
ax2.clabel(contours2, inline=True, fontsize=12)
ax2.set_title('Moment 2', fontsize=14)

# Contornos sobre el segundo gráfico
cont2 = ax2.contour(m0.value, levels=np.array([0.35,0.45, 0.6, 0.8, 0.9]) * round(np.nanmax(m0.value), 1), 
                    linewidths=2, colors='red', linestyles='--')

dust_contour_legend = Line2D([], [], color='red', linestyle='--', linewidth=2, label='Moment 0 Contours')
ax2.legend(handles=[dust_contour_legend], fontsize=14)
ax2.minorticks_on()

# Beam
beam_ellipse = Ellipse(
    (38, 5), width=beam.minor.value/delta*3600, height=beam.major.value/delta*3600, angle=beam.pa.value,
    edgecolor='black', facecolor='lightgray', lw=1,zorder=9)

# Agregar la elipse al gráfico
ax2.add_patch(beam_ellipse)

rec = FancyBboxPatch(
    (8, 1),       
    10, 5,       
    boxstyle="round,pad=0.02,rounding_size=0.5",  
    facecolor='white',      
    edgecolor='white',      
    linewidth=1.5,
    alpha=0.9,
    zorder=10
)

# Agregar la elipse al gráfico
ax2.add_patch(rec)

ax2.hlines(5,7,ang[0]/delta+7,color='k',zorder=15)
ax2.text(7,2,'500 UA',fontsize=12,zorder=14)



#plt.savefig('SO2_21_21_moment2_44kpc.png', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución


# Tercer gráfico (gráfico adicional cubriendo toda la segunda fila)
ax3 = plt.subplot(gs[1, :])  # Un solo gráfico que cubre ambas columnas en la segunda fila
#plt.figure(figsize=(15,5))
#ax3 = plt.subplot()
ax3.step(Molines_A_df.sum(axis=1).index[channel[0]:channel[1]],Molines_A_df.sum(axis=1).iloc[channel[0]:channel[1]]/4800,color='black')
ax3.set_title('SO2_21_21 mean line emission', fontsize=12)
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


vsys = (round(pars['peak1_center'].value,2) + round(pars['peak2_center'].value,2)) / 2
ax3.vlines(vsys, 0, max(out.best_fit),color='green',label='Vsys = '+str(vsys)+' km/s')

# Ajuste 
ax3.plot(x, out.best_fit, '-', color='purple')
ax3.text(min(x)+min(x)/40,max(y)-max(y)/11,r'$\mu =$ '+str(round(pars['peak1_center'].value,2))+' km/s',color='blue',fontsize=13)
ax3.text(min(x)+min(x)/40,max(y)-2*max(y)/11,r'$\sigma =$ '+str(round(pars['peak1_sigma'].value,2))+' km/s',color='blue',fontsize=13)

ax3.text(min(x)+min(x)/40,max(y)-3*max(y)/11,r'$\mu =$ '+str(round(pars['peak2_center'].value,2))+' km/s',color='red',fontsize=13)
ax3.text(min(x)+min(x)/40,max(y)-4*max(y)/11,r'$\sigma =$ '+str(round(pars['peak2_sigma'].value,2))+' km/s',color='red',fontsize=13)

ax3.legend(fontsize=13)
# Ajustar el layout para evitar que los gráficos se superpongan
plt.tight_layout()

output_filename = 'SO2_21_21_44kpc.png'  # Cambia la extensión si prefieres otro formato como .pdf, .svg, etc.
plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución
