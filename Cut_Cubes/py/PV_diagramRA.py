# PV Diagrams 

from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
from matplotlib.lines import Line2D

box = [205, 205, 307, 307]
lines = ['13CO','SO2_4_3','SO2_21_21']
line_names = ['$^{13}$CO \nJ=3-2\n','SO$_2$ \nJ$_{K_a,K_c}$ = 4$_{3,1}$ - 3$_{2,2}$\n','SO$_2$ \nJ$_{K_a,K_c}$ = 21$_{2,20}$ - 21$_{1,21}$\n']

lims = [[204,104],[212, 112],[167,67]]
fig, axes = plt.subplots(1, 3, figsize=(15,9),sharey=True,gridspec_kw={'wspace': 0.05})

if len(lines) == 1:
    axes = [axes]  # Asegura que siempre sea iterable

for ax1, line,lim,line_name in zip(axes, lines,lims,line_names):
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + line + '.fits'
    path_cont = '/Users/mac/Tesis/IRAS15445_recortados/All_spw_continuum_temp.fits'

    beam = SpectralCube.read(path).beam
    cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
    
    continuum = SpectralCube.read(path_cont)
    continuum  = continuum[:,230:290,235:278]

    max_index = np.unravel_index(np.argmax(cube), cube.shape)[1:]
    max_value = cube[0][max_index]

    x_z_projection = np.mean(cube, axis=1) *1000

    ny = abs(box[2] - box[0])
    nx = abs(box[3] - box[1])
    delta = 3.61e-06 * 3600
    
    # Offsets centrados en el máximo continuo
    y0, x0 = max_index  # fila (DEC), columna (RA)
    x_offsets = (np.arange(nx) - x0) * delta
    y_offsets = Molines_A_df.index + 90

    # extent para imshow
    extent = [x_offsets[0], x_offsets[-1], y_offsets[0], y_offsets[-1]]
    

    im = ax1.imshow(x_z_projection, origin='lower', aspect='auto',extent=extent)
    
    #levels = np.linspace(0.001, np.max(x_z_projection), 5)[1:]
    #ax1.contour(x_z_projection.T, levels=levels, colors='black', linewidths=0.7, alpha=0.7)
    
    ax1.contour(x_offsets, y_offsets, x_z_projection, levels=[3*np.std(x_z_projection)], colors='red', linewidths=1, alpha=0.7)
    ax1.contour(x_offsets, y_offsets, x_z_projection, levels=np.array([2,4.2])*np.std(x_z_projection), colors='k', linewidths=0.7, alpha=0.7)
    
    ax1.invert_yaxis()

    ax1.set_xlabel('J2000 RA offset (′′)', fontsize=17)
    
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax1.minorticks_on()
    
    ax1.set_title(line_name,fontsize=22)
    
    # padding desde los bordes en arcsec
    pad_x = 0.05 * (extent[1] - extent[0])
    pad_y = 0.05 * (extent[3] - extent[2])
    
    # Escala en UA
    d = np.array([4.38, 5.4, 8.5])  # pc
    xx = 1000  # tamaño en AU
    ang = 2*np.arctan(xx/(2*d*2.063e+8))*3600*180/np.pi
    
    scale_x_start = -0.4
    scale_x_end = scale_x_start + ang[0]
    scale_y = -80
    
    ax1.hlines(scale_y, scale_x_start, scale_x_end, colors='white', linewidth=2, zorder=6)
    ax1.text(scale_x_start+0.11, scale_y - 0.02 * (extent[3] - extent[2]),
            f'{int(xx)} AU', ha='center', va='top', fontsize=14,color='white')

    #ax1.set_aspect('equal')
    ax1.set_ylim(-90,90)
    ax1.set_xlim(-0.5,0.5)

    custom_lines = [Line2D([0], [0], color='red', lw=1, alpha=0.7, label=r'$3\sigma$')]
    ax1.legend(handles=custom_lines, fontsize=17)
    

plt.tight_layout()  # Ajusta primero todo
axes[0].set_ylabel('Velocity offset (km/s)', fontsize=17)

# Barra de color única para los tres subplots
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',  pad=0.1, shrink=0.5)
cbar.set_label('Intensity (mJy/beam)', fontsize=17)
cbar.ax.tick_params(labelsize=17)

plt.savefig('/Users/mac/Tesis/Cut_Cubes/Emission_Figures/PV diagrams/PV diagram RA (offset)', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución

