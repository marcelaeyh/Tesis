import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
import cube_to_df as ctdf


plt.figure(figsize=(15,10))

i = 25

path = '/media/marcela/MARCE_SATA/IRAS_15445-5449/member.uid___A001_X87d_Xb1f.IRAS_15445-5449_sci.spw'+str(i)+'.cube.I.pbcor.fits'


box = [1185,1225,1335,1375]
channel=[0,-1]
    
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
    

plt.subplot(3,1,2)
plt.xlabel('Frequency [GHz]',fontsize=12)
plt.ylabel('[mJy/Beam]',fontsize=12)

plt.step(Molines_A_df.sum(axis=1).index[channel[0]:channel[1]]*1e-3,Molines_A_df.sum(axis=1).iloc[channel[0]:channel[1]]/4800*1000,
         color='black',linewidth=0.9)

plt.vlines(321.43,-2.5,7.5,color='red')
plt.text(321.44,6.7,'SO$_2$ (18-17)',fontsize=12,color='red')

plt.ylim(-2.5,7.5)
plt.xlim(321.2,321.6)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  

plt.xlabel('Frequency [GHz]',fontsize=12)
plt.ylabel('[mJy/Beam]',fontsize=12)
      
'''
i = 27

path = '/media/marcela/MARCE_SATA/IRAS_15445-5449/member.uid___A001_X87d_Xb1f.IRAS_15445-5449_sci.spw'+str(i)+'.cube.I.pbcor.fits'


box = [1185,1225,1335,1375]
channel=[0,-1]
    
cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
    

plt.subplot(3,1,3)
plt.xlabel('Frequency [GHz]',fontsize=12)
plt.ylabel('[mJy/Beam]',fontsize=12)

plt.step(Molines_A_df.sum(axis=1).index[channel[0]:channel[1]]*1e-6,Molines_A_df.sum(axis=1).iloc[channel[0]:channel[1]]/4800*1000,
         color='black',linewidth=0.9)


plt.ylim(-2,2)
plt.xlim(318.25,320)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  
        
plt.subplot(3,1,1)  

plt.xlabel('Frequency [GHz]',fontsize=12)
plt.ylabel('[mJy/Beam]',fontsize=12)

for i in np.array([29,31]):

    path = '/media/marcela/MARCE_SATA/IRAS_15445-5449/member.uid___A001_X87d_Xb1f.IRAS_15445-5449_sci.spw'+str(i)+'.cube.I.pbcor.fits'
    
    cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
    
    plt.step(Molines_A_df.sum(axis=1).index[channel[0]:channel[1]]*1e-6,Molines_A_df.sum(axis=1).iloc[channel[0]:channel[1]]/4800*1000,
             color='black',linewidth=0.9)
    
    plt.vlines(330.69,-2.5,7.5,color='red')
    plt.text(330.705,6.6,'$^{13}$CO (3-2)',fontsize=12,color='red')
    
    plt.vlines(331.68,-2.5,7.5,color='red')
    plt.text(331.695,6.6,'SO$_2$ (11-12)',fontsize=12,color='red')
    
    plt.vlines(332.60,-2.5,7.5,color='red')
    plt.text(332.615,6.6,'SO$_2$ (4-3)',fontsize=12,color='red')
    
    plt.vlines(332.19,-2.5,7.5,color='red')
    plt.text(332.205,6.6,'SO$_2$ (21-21)',fontsize=12,color='red')
    

    plt.ylim(-1.5,7.5)
    plt.xlim(329.9,333.5)
    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)  

plt.tight_layout()

plt.savefig('/home/marcela/Desktop/spws25_27_29_31.png',dpi=300 )
'''