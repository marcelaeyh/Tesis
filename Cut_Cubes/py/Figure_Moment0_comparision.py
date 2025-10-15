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
import pandas as pd

def moment0_13CO(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[90:225,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.011*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m0 = cube_include.moment(order=0)/1000 
   
    return m0
def moment2_13CO(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[90:225,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.011*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 # pasar a km/s
   
    return m2

def moment0_SO2_4_3(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[120:220,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.012*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m0 = cube_include.moment(order=0)/1000 
   
    return m0

def moment2_SO2_4_3(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[120:220,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.012*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 # pasar a km/s
   
    return m2

def moment0_SO2_21_21(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[70:160,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.0095*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m0 = cube_include.moment(order=0)/1000 
   
    return m0

def moment2_SO2_21_21(path):
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[70:160,220:300,225:285]
    cube_include = cube_cut.with_mask(cube_cut > 0.0095*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 # pasar a km/s
   
    return m2

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


names = ['13CO','SO2_4_3','SO2_21_21']
line_names = ['$^{13}$CO \nJ=3-2\n','SO$_2$ \nJ$_{K_a,K_c}$ = 4$_{3,1}$ - 3$_{2,2}$\n','SO$_2$ \nJ$_{K_a,K_c}$ = 21$_{2,20}$ - 21$_{1,21}$\n']

boxes = [[220,225,300,285],[228, 227, 288, 282],[230,235,290,278]]
channels = [[350,520],[90, 240],[60,180]]
contornos = [[0.4, 0.6, 0.8, 0.94],[0.45, 0.6, 0.8, 0.94],[0.45, 0.6, 0.8, 0.9]]

path_cont = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'
path_line = '/Users/mac/Tesis/Cut_Cubes/13CO_velocity_nc.txt' 


linea = pd.read_csv(path_line, skiprows=range(7), sep=' ')
linea.columns = ['Velocity', 'Value']
linea = linea.set_index('Velocity')


fig, axes = plt.subplots(1, 3, figsize=(20,11), sharey=True,gridspec_kw={'wspace': 0.05})

# Ajustar espacio a la derecha para colorbar
fig.subplots_adjust(right=0.85)

# Crear un eje para la colorbar a la derecha de los plots
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height] en fracción de la figura

# --- Obtener los límites de color de la imagen 13CO ---
path_13co = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_13CO.fits'
m0_13co = moment0_13CO(path_13co)
vmin = np.nanmin(m0_13co.value)
vmax = np.nanmax(m0_13co.value)

# Bucle de plots
for k in range(len(names)):
    path = f'/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_{names[k]}.fits'
    
    beam = SpectralCube.read(path).beam
    continuum = SpectralCube.read(path_cont)
    continuum  = continuum[:,220:300,225:285]
    
    if names[k] == '13CO':
        m0 = m0_13co  # ya calculado antes
        m2 = moment2_13CO(path)
    elif names[k] == 'SO2_4_3':
        m0 = moment0_SO2_4_3(path)
        m2 = moment2_SO2_4_3(path)
    elif names[k] == 'SO2_21_21':
        m0 = moment0_SO2_21_21(path)
        m2 = moment2_SO2_21_21(path)

    # Coordenadas del píxel de máximo continuo
    max_index = np.unravel_index(np.argmax(continuum), continuum.shape)[1:]
    max_value = continuum[0][max_index]
    
    box = boxes[k]
    channel = channels[k]
    
    cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
    Molines_A_df['mean'] = Molines_A_df.sum(axis=1)/4800

    
    if names[k] == '13CO':
        pars,comps, out,matplotlib_fig = ajuste.ajuste_chisqr(linea, channel, px='Value')
    else:
        pars,comps, out,matplotlib_fig = ajuste.ajuste_chisqr(Molines_A_df, channel, px='mean')

    # --- Ejes ---
    ny, nx = m0.value.shape
    
    delta = 3.61e-06 * 3600  # tamaño píxel en arcsec
    
    # Offsets centrados en el máximo continuo
    y0, x0 = max_index  # fila (DEC), columna (RA)
    x_offsets = (np.arange(nx) - x0) * delta
    y_offsets = (np.arange(ny) - y0) * delta
    
    # extent para imshow
    extent = [x_offsets[0], x_offsets[-1], y_offsets[0], y_offsets[-1]]
    
    ax1 =  axes[k]

    # Imagen moment 0 con extent en arcsec
    im1 = ax1.imshow(
        m0.value,
        origin='lower',
        cmap=new_cmap.reversed(),
        extent=extent,
        vmin=vmin,  # <<< mismo rango
        vmax=vmax   # <<< mismo rango
    )
    
    # Ejes y etiquetas
    ax1.set_xlabel('J2000 RA offset (′′)', fontsize=17)
    
    # --- ticks grandes ---
    ax1.tick_params(axis='both', which='major', labelsize=17, length=6, width=1.5, direction='in')
    ax1.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1, direction='in')

    # --- título ---
    ax1.set_title(line_names[k], fontsize=25, pad=10)

    # Contornos moment 0
    contours1 = ax1.contour(x_offsets, y_offsets, m0.value,
                            levels=np.array(contornos[k]) * round(np.nanmax(m0.value), 1),
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
    ax1.tick_params(axis='both', which='minor', labelsize=16)  # si tienes minor ticks


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

    scale_x_start = extent[0] +0.1
    scale_x_end = scale_x_start + ang[0]
    scale_y = extent[2] + pad_y +0.03

    ax1.hlines(scale_y, scale_x_start, scale_x_end, colors='k', linewidth=2, zorder=6)
    ax1.text(scale_x_start+0.05, scale_y - 0.02 * (extent[3] - extent[2]),
             f'{int(xx)} AU', ha='center', va='top', fontsize=17)
    
    
axes[0].set_ylabel('J2000 DEC offset (′′)', fontsize=17)
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.ax.tick_params(labelsize=16, length=6, width=1.5)
cbar.set_label('(Jy/Beam . km/s)', fontsize=18)
plt.tight_layout()
    
fig.savefig('Moment0_comparison.png', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
