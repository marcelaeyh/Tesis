import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from matplotlib import rcParams

import time
import pandas as pd
#import seaborn as sns

from astropy.io.fits import getheader
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, proj_plane_pixel_scales

import astropy.units as u
from lmfit import Model
from lmfit.models import GaussianModel

#from sklearn.linear_model import LinearRegression


plt.rcParams.update({
    "text.usetex": False,
    "font.size": 8})


def timer(func):
    """A decorator that prints how long a function takes to run."""

    def wrapper(*args,**kwargs):
        t_start=time.time()
        
        result=func(*args,**kwargs)

        t_total=time.time()-t_start
        print('{} took {}s'.format(func.__name__, t_total))

        return result
    return wrapper


@timer
def PlGetCube(path):
    """
    input: path to the FITS file, containing the spectral cube (ALMA/SINFONI)
    output: Data cube : cube_data.
            cabe = Header of the imported file.
    """
    cube=fits.open(path)
    cube_data=cube[0].data[:]
    cabe=cube[0].header
    cube.close()
    
    return (cube_data,cabe)

@timer
def cropCube(Cubo,borders,nchan):

    """
    This funcion selects/returns a region within the image, given the blc and trc pixel values.
    """
    Clean_Cube=Cubo[0:nchan,borders[1]:borders[3],borders[0]:borders[2]]

    return Clean_Cube

@timer
def zaxisCube(cubeheader):
    """
    Cut edges and construct the array for frequencies/velocities/wavelengths
    params: 
    cubeheader: header of the fits file.
    """
    global NpixY
    NpixY=cubeheader['NAXIS2']
    global NpixX
    NpixX=cubeheader['NAXIS1']
    global number_chan
    number_chan=cubeheader['NAXIS3']
    global delta_lambda
    delta_freq=cubeheader['CDELT3']
    global ini_lambda
    ini_freq=cubeheader['CRVAL3']
    global refchan
    refchan=cubeheader['CRPIX3']
    
    freq_distribution=np.arange(-refchan,number_chan-refchan+1,1)*delta_freq+ini_freq
    Clean_freq_distribution=freq_distribution/1e3       ## GHz
    
    return (Clean_freq_distribution,freq_distribution)

@timer
def line2frame(Zchan_range,wavelength_vec,cubo):
    """ function line2frame creates dictionary and df from the cube passed in the params
    Zchan_range = number of bins along the Z-axis of the cube.
    wavelength_vec = array with wavelength values
    cuboo = Spectral Cube.
    
    returns Dataframe
    *Dataframe with wavelength axis.
    """
    
    cube_dictio={}
    global list_pixels_naam
    list_pixels_naam=[]
    
    for i in range(0,cubo.shape[2]):
        for j in range(0,cubo.shape[1]):
            pixel_naam=str(j)+'_'+str(i)
            list_pixels_naam.append(pixel_naam)


    for i in range(0,len(list_pixels_naam)):
        mementlist=list_pixels_naam[i].split('_',1)
    #    print(mementlist)
        keyname=str(mementlist[0])+'_'+str(mementlist[1]) 
        cube_dictio['Pix_{0}'.format(keyname)]=cubo[Zchan_range[0]:Zchan_range[1],int(mementlist[0]),int(mementlist[1])] #ji

    cube_dictio['Radio_velocity']=wavelength_vec[Zchan_range[0]:Zchan_range[1]]*1e-3 # pasar a km/s
    Fluxgrid_df=pd.DataFrame(data=cube_dictio).T
    df_emissionLine=Fluxgrid_df.transpose().set_index('Radio_velocity')
    
    return df_emissionLine


@timer
def pix_max(df):
    '''
    Returns the pix with higher emission and the value of intensity for a given data set.
    
    Params:
        df - Dataframe
    Output:
        maxpx - Pixel of higher emission
        peak_flux_maxpx - Value of higher emission
    '''
    info = df.describe().loc['max']
    maxpx = info[info==info.max()].index[0]
    peak_flux_maxpx = info.max()
    
    return maxpx,peak_flux_maxpx


def Cube_to_df(source_name, spw, coord=False):
    '''
    First, the fits cube is read and assigned to Cubepath.
    Then the PlGetCube function will take the Cubepath, and extract both Header and Data tables.
    The Clean_NIR_cube contains the data in the cube excluding cube borders. Borders are defined as
    the pixel rows and columns from the image in imview.
    The result of this is called Clean_NIR_Cube, which contains all the spaxels we want to use for our
    calculations.
    
    Parameters
    ----------
    source_name: str
        Name of the source in cubepath.
    spw: int
        Spectral Window.
    coord: bool
        If need information about source coordinates coord = True

    Returns
    -------
    Molines_A_df: dataframe
        Data cube converted to dataframe.
    '''
    if spw==25:
        spw_lbc_trc_pix = [1185,1225,1335,1375] # 150 x 150 px
    elif spw==27:
        spw_lbc_trc_pix = [1185,1225,1335,1375] # 150 x 150 px
    elif spw==29:
        spw_lbc_trc_pix = [205,210,305,310] # 100 x 100 px
    elif spw==31: 
        spw_lbc_trc_pix = [1185,1225,1335,1375] # 150 x 150 px 
    
    ####
    #ALMA cube name
    ###

    #Cubepath = '/media/marcela/MARCE_SATA/'+source_name+'/member.uid___A001_X87d_Xb1f.'+source_name+'_sci.spw'+str(spw)+'.cube.I.pbcor.fits'
    Cubepath = '/home/marcela/Tesis Marcela/IRAS15445/I15445.mstransform_cube_contsub_13CO.fits'
    data_in_cube,cabe=PlGetCube(Cubepath)
    
    w=wcs.WCS(cabe)  #Coordinates info
    if coord == True:
        print('-----------------------------------------------------------')
        print('COORDINATES INFO - '+source_name+' - spw ='+str(spw))
        print('-----------------------------------------------------------')
        print(w)
        print('-----------------------------------------------------------')

    ALMAb7_final=cropCube(data_in_cube[0],spw_lbc_trc_pix,cabe['NAXIS3'])

    seg_freq_array,freq_array=zaxisCube(cabe)

    Molines_A_df=line2frame([0,319],freq_array,ALMAb7_final)

    return ALMAb7_final,Molines_A_df


def plotpix(Molines_A_df,px,channel):

    fig = plt.figure(figsize=(10,5))
    plt.title(px,fontsize=14)
    plt.xlabel('Radio Velocity [km/s]',fontsize=12)
    plt.ylabel('$[Jy/beam]$',fontsize=12)
    plt.grid()

    plt.step(Molines_A_df[px].index[channel[0]:channel[1]],Molines_A_df[px].iloc[channel[0]:channel[1]])

    return fig

def clas_g(result):
    if result.summary()['params'][0][1] <-1e-3:
        a='No hay linea por 1'
    elif result.summary()['params'][0][7] == None or result.summary()['params'][1][7] == None or result.summary()['params'][2][7] == None:
        a = 'No hay linea por 3'
    elif abs(result.summary()['params'][0][7]*100/result.summary()['params'][0][1]) >=40:
        a = 'No hay linea por 3'
    elif abs(result.summary()['params'][1][7]*100/result.summary()['params'][1][1]) >=40:
        a = 'No hay linea por 3'
    elif abs(result.summary()['params'][2][7]*100/result.summary()['params'][2][1]) >=40:
        a = 'No hay linea por 3'
    else:
        a = 'Hay linea'
    return a
    
def clas(Molines_A_df,cube,px,channel,plot=False):

    x_datos = Molines_A_df[px].index[channel[0]:channel[1]]
    y_datos = Molines_A_df[px].iloc[channel[0]:channel[1]]
    
    # centro del canal
    med = (y_datos.index.max()-y_datos.index.min())/2 + y_datos.index.min()
    
    # separar datos por izquierda y por derecha
    y_datos_l = y_datos[y_datos.index <= med]
    x_datos_l = x_datos[x_datos <=med]
    
    y_datos_r = y_datos[y_datos.index >= med]
    x_datos_r = x_datos[x_datos >= med]
    
    # Secciones para wid
    secc1 = Molines_A_df[px].iloc[channel[0]+30:channel[1]-30]
    #secc2 = pd.concat([Molines_A_df[px].iloc[:channel[0]+30],Molines_A_df[px].iloc[channel[1]-30:]])
    #prop1 = secc1.describe()['mean']
    #prop2 = secc2.describe()['mean']

    # Modelo de gaussiana
    gmodel = Model(gaussian)
    result_l = gmodel.fit(y_datos_l, x=x_datos_l, amp=max(y_datos_l)/2, cen=y_datos_l[y_datos_l==max(y_datos_l)].index[0], wid=abs(secc1.index[0]-med)/2)
    result_r = gmodel.fit(y_datos_r, x=x_datos_r, amp=max(y_datos_r)/2, cen=y_datos_r[y_datos_r==max(y_datos_r)].index[0], wid=abs(secc1.index[0]-med)/2)
    

    s = np.sum(cube[channel[0]+30:channel[1]-30,:,:],axis=0)[int(px.split('_')[1]),int(px.split('_')[2])]

    if s<0.8:
        a = 'Ruido'
    elif clas_g(result_l) != 'Hay linea' or clas_g(result_r) != 'Hay linea':
        a = 'Ruido'
    elif result_r.summary()['params'][1][1]-result_l.summary()['params'][1][1] <0.043:
        a = 'Un pico'
    else:
        a = 'Dos picos'
    
    if plot == True:
        #plotpix(px)        
        plotpix(px,channel)
        
        plt.suptitle(a+' '+str(s))
        #plt.plot(x_datos_r, result_r.init_fit, '--', label='initial f')  
        #plt.plot(x_datos_l, result_l.init_fit, '--', label='initial f')  
        #plt.plot(x_datos,y_datos)
        plt.plot(x_datos_l, result_l.best_fit, '-', color='orangered')
        plt.plot(x_datos_r, result_r.best_fit, '-', color='orangered')
        plt.vlines(secc1.index[0],y_datos.min(),y_datos.max(),color='k',linestyle='--')
        plt.vlines(secc1.index[-1],y_datos.min(),y_datos.max(),color='k',linestyle='--')
        plt.vlines(med,y_datos.min(),y_datos.max(),color='lightcoral',linestyle='--')
    
    return a,result_l,result_r

def gauss_model(Molines_A_df,cube,px,channel,plot=False):
    
    pars=0
    a = clas(Molines_A_df,cube,px,[channel[0],channel[1]])
    
    if a[0] != 'Ruido':
        modell = a[1]
        modelr = a[2]
        
        x = Molines_A_df[px].index[channel[0]:channel[1]]
        y = Molines_A_df[px].iloc[channel[0]:channel[1]]
        
        #if a[0] == 'Dos picos':
        npeaks=2
        model=GaussianModel(prefix='peak1_')
        
        # Armar el modelo, en este caso tengo dos gaussianas
        for i in range(1,npeaks):
          model=model+GaussianModel(prefix='peak%d_' % (i+1))
        pars=model.make_params()
        
        # Llenar los parámetros iniciales
        for i,ff in zip(range(npeaks),[modell.summary()['params'],modelr.summary()['params']]):
          pars['peak%d_center' % (i+1)].set(value=ff[1][1],vary=False) # fix nu_ul
          pars['peak%d_sigma' % (i+1)].set(value=ff[2][1],min=10,max=50)
          pars['peak%d_amplitude' % (i+1)].set(value=ff[0][1],min=0,max=4)
            
        # Hacer el fit al modelo inicial hasta que se ajusten las gaussianas
        out=model.fit(y,pars,x=x) # run fitting algorithm
        comps = out.eval_components(x=x) # fit results for each line
        
        if plot == True:
            # Data
            fig = plotpix(Molines_A_df,px,[channel[0],channel[1]])
            # Componentes 
            for i in range(npeaks):
                plt.plot(x, comps['peak%d_' % (i+1)], label='peak'+str(i+1))
            
            # Ajuste 
            plt.plot(x, out.best_fit, '-', color='purple')
    else:
        fig = plotpix(Molines_A_df,px,[channel[0],channel[1]])
    return pars,fig
      
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def plotfont(cube,channel,px=0):
    plt.figure(figsize=(10,10))
    plt.imshow(cube[channel,:,:],vmin=-0.03, vmax=0.03,cmap='inferno',interpolation=None)
    plt.colorbar()
    plt.title(str(channel), fontsize=15)
    if px!=0:
        plt.plot(int(px[4:6]),int(px[7:]),'o',color='lime',marker='s',markersize=1)
        