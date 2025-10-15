from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

path = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'


continuum = SpectralCube.read(path)
continuum  = continuum[:,234:285,232:283]*1000

max_index = np.unravel_index(np.argmax(continuum), continuum.shape)[1:]
max_value = continuum[0][max_index]

beam = SpectralCube.read(path).beam
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
# ---------------------------------------------------------------------------------------
nx = continuum.shape[1]
ny = continuum.shape[2]

delta = 3.61e-06 * 3600

plt.figure(figsize=(8.5, 8))

ax1 = plt.subplot()
im1 = ax1.imshow(continuum[0].value,origin='lower',cmap= new_cmap.reversed())


ax1.set_xlabel('J2000 RA offset (′′)',fontsize=17)
ax1.set_ylabel('J2000 DEC offset (′′)',fontsize=17)

# Contornos en continuo
con = ax1.contour(continuum[0].value, levels=np.array([0.3, 0.5, 0.7, 0.9]) * np.nanmax(continuum[0].value), 
                    linewidths=1, colors='k', linestyles='-')

ax1.clabel(con, inline=True, fontsize=15)


# Añadir la barra de colores
cbar1 = plt.colorbar(im1, ax=ax1,pad=0.07)
cbar1.set_label('Intensity (mJy/Beam)', fontsize=15)
cbar1.ax.tick_params(labelsize=15)

ax1.set_xticks(np.arange(0,nx+1,8))
ax1.set_xticklabels(np.round((np.arange(0,nx+1,8)-max_index[1])*delta,1))

ax1.set_yticks(np.arange(0,ny+1,8))
ax1.set_yticklabels(np.round((np.arange(0,ny+1,8)-max_index[0])*delta,1))


ax1.tick_params(axis='both', which='major', labelsize=15)  # números más grandes
ax1.tick_params(axis='both', which='minor', labelsize=15)  # si tienes minor ticks

ax1.set_aspect('equal')
ax1.set_ylim(0,46)
ax1.set_xlim(0,50)
ax1.minorticks_on()

# Tamaños UA

d = np.array([4.38,5.4,8.5])
xx = 500 #UA
ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi

ax1.hlines(3,5,ang[0]/delta+5,color='k')
ax1.text(5,4,'500 AU',fontsize=14)

# Beam
beam_ellipse = Ellipse(
    (45, 6), width=beam.minor.value/delta*3600, height=beam.major.value/delta*3600, angle=beam.pa.value,
    edgecolor='black', facecolor='lightgray', lw=1)

# Agregar la elipse al gráfico
ax1.add_patch(beam_ellipse)

plt.savefig('/Users/mac/Tesis/Cut_Cubes/Emission_Figures/Dust_continuum', dpi=300, bbox_inches='tight')

