from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.patches as patches

path = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'


continuum_all = SpectralCube.read(path)
continuum  = continuum_all[:,210:310,200:310]*1000

beam = SpectralCube.read(path).beam
max_index = np.unravel_index(np.argmax(continuum), continuum.shape)[1:]

tamaño = 10

cubo1 = [25,45,25+tamaño,45+tamaño]
cubo2 = [50,25,50+tamaño,25+tamaño]
cubo3 = [50,60,50+tamaño,60+tamaño]
cubo4 = [75,45,75+tamaño,45+tamaño]

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
nx = continuum.shape[1]
ny = continuum.shape[2]

delta = 3.61e-06 * 3600

x = -np.concatenate([np.arange(-nx/2,0)*delta, np.arange(nx/2)*delta])-0.01
y = np.concatenate([np.arange(-ny/2,0)*delta, np.arange(ny/2)*delta])+0.01

plt.figure(figsize=(8.5, 8))

ax1 = plt.subplot()
im1 = ax1.imshow(continuum[0].value,origin='lower',cmap= new_cmap.reversed())

square1 = patches.Rectangle((cubo1[0],cubo1[1]), tamaño, tamaño, edgecolor='blue', facecolor='none', linewidth=2)
square2 = patches.Rectangle((cubo2[0],cubo2[1]), tamaño, tamaño, edgecolor='blue', facecolor='none', linewidth=2)
square3 = patches.Rectangle((cubo3[0],cubo3[1]), tamaño, tamaño, edgecolor='blue', facecolor='none', linewidth=2)
square4 = patches.Rectangle((cubo4[0],cubo4[1]), tamaño, tamaño, edgecolor='blue', facecolor='none', linewidth=2)


ax1.add_patch(square1)
ax1.add_patch(square2)
ax1.add_patch(square3)
ax1.add_patch(square4)

ax1.set_title('Dust Continuum', fontsize=17)
ax1.set_xlabel('J2000 RA offset [arcsec]',fontsize=17)
ax1.set_ylabel('J2000 DEC offset [arcsec]',fontsize=17)
ax1.minorticks_on()


# Contornos en continuo
ax1.contour(continuum[0].value, levels=np.array([0.3, 0.5, 0.7, 0.9]) * np.nanmax(continuum[0].value), 
                    linewidths=1, colors='k', linestyles='-')

#ax1.contour(continuum[0].value, levels=[3*np.std(continuum[0].value)], colors='red', linewidths=1.5, alpha=0.7)

custom_lines = [Line2D([0], [0], color='red', lw=1.5, alpha=0.7, label=r'$3\sigma$')]
#ax1.legend(handles=custom_lines, fontsize=16)


# Añadir la barra de colores
cbar1 = plt.colorbar(im1, ax=ax1,pad=0.07)
cbar1.set_label('Intensity [mJy/Beam]', fontsize=15)
cbar1.ax.tick_params(labelsize=15)

ax1.set_xticks(np.arange(0,nx+1,14))
ax1.set_xticklabels(np.round((np.arange(0,nx+1,14)-max_index[1])*delta,1))

ax1.set_yticks(np.arange(0,ny+1,15))
ax1.set_yticklabels(np.round((np.arange(0,ny+1,15)-max_index[0])*delta,1))


ax1.tick_params(axis='both', which='major', labelsize=15)

ax1.set_aspect('equal')
#ax1.set_ylim(5,41)
#ax1.set_xlim(7,43)

# Tamaños UA

d = np.array([4.38,5.4,8.5])
xx = 500 #UA
ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi

ax1.hlines(7,10,ang[0]/delta+10,color='k')
ax1.text(10,9,'500 AU',fontsize=14)

# Beam
beam_ellipse = Ellipse(
    (99, 9), width=beam.minor.value/delta*3600, height=beam.major.value/delta*3600, angle=beam.pa.value,
    edgecolor='black', facecolor='lightgray', lw=1)

# Agregar la elipse al gráfico
ax1.add_patch(beam_ellipse)

#plt.savefig('Dust_continuum', dpi=300, bbox_inches='tight')


