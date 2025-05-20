# PV Diagrams 

from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
from matplotlib.lines import Line2D

box = [200, 200, 320, 320]
lines = ['13CO','SO2_4_3','SO2_21_21']
lims = [[212,92],[222, 102],[177,57]]
fig, axes = plt.subplots(1, 3, figsize=(15,7))

if len(lines) == 1:
    axes = [axes]  # Asegura que siempre sea iterable

for ax1, line,lim in zip(axes, lines,lims):
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + line + '.fits'
    
    beam = SpectralCube.read(path).beam
    cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)

    x_z_projection = np.mean(cube, axis=2) *1000

    ny = abs(box[2] - box[0])
    nx = abs(box[3] - box[1])
    delta = 3.61e-06 * 3600
    
    x = -np.concatenate([np.arange(-nx/2, 0) * delta, np.arange(nx/2) * delta])
    y = np.concatenate([np.arange(-ny/2, 0) * delta, np.arange(ny/2+1) * delta])

    im = ax1.imshow(x_z_projection, origin='lower', aspect='auto')
    
    # Añadir contornos blancos
    ax1.contour(x_z_projection, levels=[3*np.std(x_z_projection)], colors='red', linewidths=1, alpha=0.7)
    ax1.contour(x_z_projection, levels=np.array([2,4.2,5.5])*np.std(x_z_projection), colors='k', linewidths=0.7, alpha=0.7)
        
    
    ax1.invert_yaxis()

    ax1.set_xticks(np.arange(0,ny+1,15))
    ax1.set_xticklabels(np.round(y[np.arange(0,ny+1,15)],2), fontsize=12)

    ax1.set_yticks(np.linspace(lim[0],lim[1],7))
    ax1.set_yticklabels(np.array([-180,-150,-120,-90,-60,-30,0])+90, fontsize=12)

    ax1.set_ylabel('Radio Velocity [km/s]', fontsize=16)
    ax1.set_xlabel('J2000 DEC offset [arcsec]', fontsize=16)
    
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax1.minorticks_on()
    
    ax1.set_title(line,fontsize=13)
    # Tamaños UA
    d = np.array([4.38, 5.4, 8.5])
    xx = 1000  # UA
    ang = 2 * np.arctan(xx / (2 * d * 2.063e+8)) * 3600 * 180 / np.pi

    ax1.hlines(lim[0]-14, 16, ang[0]/delta + 16, color='white')
    ax1.text(16, lim[0]-7, str(xx)+' AU', fontsize=14, color='white')

    ax1.set_aspect('equal')
    ax1.set_ylim(lim[0],lim[1])
    ax1.set_xlim(10,110)
    
    custom_lines = [Line2D([0], [0], color='red', lw=1, alpha=0.7, label=r'$3\sigma$')]
    ax1.legend(handles=custom_lines, fontsize=14)
    

plt.tight_layout()  # Ajusta primero todo

# Barra de color única para los tres subplots
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',  pad=0.1, shrink=0.4)
cbar.set_label('Intensity [mJy/beam]', fontsize=16)
cbar.ax.tick_params(labelsize=12)


plt.savefig('PV diagram DEC (offset)', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución
