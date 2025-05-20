# PV Diagrams 

from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import cube_to_df as ctdf
from matplotlib.lines import Line2D


box = [205, 205, 307, 307]
lines = ['13CO','SO2_4_3','SO2_21_21']
lims = [[204,104],[212, 112],[167,67]]
fig, axes = plt.subplots(1, 3, figsize=(15,7))

if len(lines) == 1:
    axes = [axes]  # Asegura que siempre sea iterable

for ax1, line,lim in zip(axes, lines,lims):
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + line + '.fits'
    
    beam = SpectralCube.read(path).beam
    cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)

    x_z_projection = np.mean(cube, axis=1) *1000

    ny = abs(box[2] - box[0])
    nx = abs(box[3] - box[1])
    delta = 3.61e-06 * 3600
    
    y = -np.concatenate([np.arange(-nx/2, 0) * delta, np.arange(nx/2) * delta])
    x = np.concatenate([np.arange(-ny/2, 0) * delta, np.arange(ny/2+1) * delta])

    im = ax1.imshow(x_z_projection, origin='lower', aspect='auto')
    
    #levels = np.linspace(0.001, np.max(x_z_projection), 5)[1:]
    #ax1.contour(x_z_projection.T, levels=levels, colors='black', linewidths=0.7, alpha=0.7)
    
    ax1.contour(x_z_projection, levels=[3*np.std(x_z_projection)], colors='red', linewidths=1, alpha=0.7)
    ax1.contour(x_z_projection, levels=np.array([2,4.2])*np.std(x_z_projection), colors='k', linewidths=0.7, alpha=0.7)
    
    ax1.invert_xaxis()

    ax1.set_xticks(np.arange(0,nx+1,15))
    ax1.set_xticklabels([-0.6,-0.4,-0.2,0,0.2,0.4,0.6], fontsize=12)

    ax1.set_yticks(np.linspace(lim[0],lim[1],7))
    ax1.set_yticklabels(np.array([-180,-150,-120,-90,-60,-30,0])+90, fontsize=12)
    
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', labelsize=17)
    

    ax1.set_ylabel('Radio Velocity [km/s]', fontsize=16)
    ax1.set_xlabel('J2000 RA offset [arcsec]', fontsize=16)
    
    ax1.set_title(line,fontsize=13)
    # Tamaños UA
    d = np.array([4.38, 5.4, 8.5])
    xx = 1000  # AU
    ang = 2 * np.arctan(xx / (2 * d * 2.063e+8)) * 3600 * 180 / np.pi
    #ax1.vlines(115, 60, -ang[0]/delta + 60, color='white')

    ax1.hlines(lim[0]-11, 5, ang[0]/delta + 5, color='white')
    ax1.text(5, lim[0]-5, str(xx)+' AU', fontsize=14, color='white')

    ax1.set_aspect('equal')
    ax1.set_ylim(lim[0],lim[1])
    ax1.set_xlim(0,90)
    
    custom_lines = [Line2D([0], [0], color='red', lw=1, alpha=0.7, label=r'$3\sigma$')]
    ax1.legend(handles=custom_lines, fontsize=14)

plt.tight_layout()  # Ajusta primero todo

# Barra de color única para los tres subplots
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',  pad=0.1, shrink=0.4)
cbar.set_label('Intensity [mJy/beam]', fontsize=16)
cbar.ax.tick_params(labelsize=12)


plt.savefig('PV diagram RA (offset)', dpi=300, bbox_inches='tight')  # 'dpi=300' aumenta la resolución

