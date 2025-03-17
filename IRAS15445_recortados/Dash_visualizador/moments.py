#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spectral_cube import SpectralCube
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

def moments_params(linea):
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + linea + '.fits'

    # Información de las líneas
    # chans, box, includepix, params_contornos1 (start, end, size), params_contornos2 (start, end, size), zmax
    info = {'13CO': [[90,225],[220,300,225,285],0.011,[1.2,2.82,0.4049],[0.23*59.319,0.99*59.319,11],66],
            'SO2_4_3': [[120,220],[230,290,225,280],0.012,[0.3*2.4,0.94*2.4,0.4],[0.2*67.04,0.99*67.04,10],55],
            'SO2_21_21': [[70,160],[230,290,235,278],0.0095,[0.63,1.62,0.2475],[0.29*46.07,0.85*46.07,7],46]}
    return info[linea]

def moment0(linea):
    params = moments_params(linea)
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + linea + '.fits'
    
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[params[0][0]:params[0][1],params[1][0]:params[1][1],params[1][2]:params[1][3]]
    cube_include = cube_cut.with_mask(cube_cut > params[2]*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m0 = cube_include.moment(order=0)/1000 
   
    return m0

def moment2(linea):
    params = moments_params(linea)
    path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_' + linea + '.fits'
    
    cube_prueba = SpectralCube.read(path)
    
    cube_cut = cube_prueba[params[0][0]:params[0][1],params[1][0]:params[1][1],params[1][2]:params[1][3]]
    cube_include = cube_cut.with_mask(cube_cut > params[2]*u.Jy/u.beam)  
    #cube_include = cube_include.with_mask(cube_include < 0.03*u.Jy/u.beam)  
    
    m2 = cube_include.moment(order=2)/1e8 
   
    return m2

'''
for i in range(len(cube_include)):
    plt.figure(figsize=(7,7))
    plt.imshow(cube_include[i,:,:].value, origin='lower')
    plt.show()
'''