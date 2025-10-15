from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.patches as patches

path = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'

# Definir función para pasar de píxel a arcsec
def pix_to_arcsec(px_x, px_y):
    ra = (px_x - x0) * delta   # x_offsets
    dec = (px_y - y0) * delta  # y_offsets
    return ra, dec

continuum_all = SpectralCube.read(path)
continuum  = continuum_all[:,200:310,200:310]*1000

beam = SpectralCube.read(path).beam

# Coordenadas del píxel de máximo continuo
max_index = np.unravel_index(np.argmax(continuum), continuum.shape)[1:]
max_value = continuum[0][max_index]

tamaño = 10

cubo1 = [25,50,25+tamaño,50+tamaño]
cubo2 = [50,30,50+tamaño,30+tamaño]
cubo3 = [50,75,50+tamaño,75+tamaño]
cubo4 = [75,50,75+tamaño,50+tamaño]

cubo1_cont = continuum[:,cubo1[0]:cubo1[2],cubo1[1]:cubo1[3]]
cubo2_cont = continuum[:,cubo2[0]:cubo2[2],cubo2[1]:cubo2[3]]
cubo3_cont = continuum[:,cubo3[0]:cubo3[2],cubo3[1]:cubo3[3]]
cubo4_cont = continuum[:,cubo4[0]:cubo4[2],cubo4[1]:cubo4[3]]

RMS = (np.std(cubo1_cont[0].value) + np.std(cubo2_cont[0].value) + np.std(cubo3_cont[0].value) + np.std(cubo4_cont[0].value))/4
print(3*RMS)

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

# --- Ejes ---
nx = continuum.shape[1]
ny = continuum.shape[2]
delta = 3.61e-06 * 3600  # tamaño píxel en arcsec

# Offsets centrados en el máximo continuo
y0, x0 = max_index  # fila (DEC), columna (RA)
x_offsets = (np.arange(nx) - x0) * delta
y_offsets = (np.arange(ny) - y0) * delta

# extent para imshow
extent = [x_offsets[0], x_offsets[-1], y_offsets[0], y_offsets[-1]]


plt.figure(figsize=(8.5, 8))

ax1 = plt.subplot()
im1 = ax1.imshow(continuum[0].value,origin='lower',cmap= new_cmap.reversed(), extent=extent)

# Convertir esquina inferior izquierda
corner1 = pix_to_arcsec(cubo1[0], cubo1[1])
square1 = patches.Rectangle(corner1, tamaño*delta, tamaño*delta, 
                             edgecolor='blue', facecolor='none', linewidth=2)

corner2 = pix_to_arcsec(cubo2[0], cubo2[1])
square2 = patches.Rectangle(corner2, tamaño*delta, tamaño*delta, 
                             edgecolor='blue', facecolor='none', linewidth=2)

corner3 = pix_to_arcsec(cubo3[0], cubo3[1])
square3 = patches.Rectangle(corner3, tamaño*delta, tamaño*delta, 
                             edgecolor='blue', facecolor='none', linewidth=2)

corner4 = pix_to_arcsec(cubo4[0], cubo4[1])
square4 = patches.Rectangle(corner4, tamaño*delta, tamaño*delta, 
                             edgecolor='blue', facecolor='none', linewidth=2)

ax1.add_patch(square1)
ax1.add_patch(square2)
ax1.add_patch(square3)
ax1.add_patch(square4)


ax1.set_xlabel('J2000 RA offset (′′)',fontsize=17)
ax1.set_ylabel('J2000 DEC offset (′′)',fontsize=17)
ax1.minorticks_on()


# Contornos en continuo
ax1.contour(x_offsets, y_offsets,continuum[0].value, levels=np.array([0.3, 0.5, 0.7, 0.9]) * np.nanmax(continuum[0].value), 
                    linewidths=1, colors='k', linestyles='-')

#ax1.contour(continuum[0].value, levels=[3*np.std(continuum[0].value)], colors='red', linewidths=1.5, alpha=0.7)

#custom_lines = [Line2D([0], [0], color='red', lw=1.5, alpha=0.7, label=r'$3\sigma$')]
#ax1.legend(handles=custom_lines, fontsize=16)


# Añadir la barra de colores
cbar1 = plt.colorbar(im1, ax=ax1,pad=0.07)
cbar1.set_label('Intensity (mJy/Beam)', fontsize=15)
cbar1.ax.tick_params(labelsize=15)


ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=15)  # si tienes minor ticks

ax1.set_aspect('equal')
#ax1.set_ylim(5,41)
#ax1.set_xlim(7,43)

# Beam
beam_major = beam.minor.value*3600
beam_minor = beam.major.value*3600

# padding desde los bordes en arcsec
pad_x = 0.05 * (extent[1] - extent[0])
pad_y = 0.05 * (extent[3] - extent[2])

beam_cx = extent[1] - pad_x - beam_major / 2 -0.08
beam_cy = extent[2] + pad_y + beam_minor / 2 +0.06

beam_ellipse = Ellipse((beam_cx, beam_cy),
                       width=beam_major, height=beam_minor,
                       angle=beam.pa.value,
                       edgecolor='black', facecolor='lightgray', lw=1, zorder=5)
ax1.add_patch(beam_ellipse)

# Escala en UA
d = np.array([4.38, 5.4, 8.5])  # pc
xx = 500  # tamaño en AU
ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi

scale_x_start = extent[0] +0.15
scale_x_end = scale_x_start + ang[0]
scale_y = extent[2] + pad_y +0.1

ax1.hlines(scale_y, scale_x_start, scale_x_end, colors='k', linewidth=2, zorder=6)
ax1.text(scale_x_start+0.08, scale_y - 0.02 * (extent[3] - extent[2]),
         f'{int(xx)} AU', ha='center', va='top', fontsize=12)



plt.savefig('/Users/mac/Tesis/Cut_Cubes/Emission_Figures/Dust_continuum_regions.png', dpi=300, bbox_inches='tight')


