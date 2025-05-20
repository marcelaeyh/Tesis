import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import pandas as pd
from astropy import wcs

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
def Cube_to_df(path, box, coord=False):
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
    ALMAb7_final: numpy array
        
    Molines_A_df: dataframe
        Data cube converted to dataframe.
    '''
                                 
    ####
    #ALMA cube name    
    ###

    data_in_cube,cabe=PlGetCube(path)
    
    w=wcs.WCS(cabe)  #Coordinates info
    
    
    # Dimensiones de la imagen
    nx = 60
    ny = 80
    
    # Tamaño de píxel en segundos de arco
    delta = 3.611111111111e-06 * 3600

    x = -np.concatenate([np.arange(-nx/2,0)*delta,np.arange(nx/2)*delta])
    y = np.concatenate([np.arange(-ny/2,0)*delta,np.arange(ny/2)*delta])
    
    if coord == True:
        print('-----------------------------------------------------------')
        print('COORDINATES INFO')
        print('-----------------------------------------------------------')
        print(w)
        print('-----------------------------------------------------------')

    ALMAb7_final=cropCube(data_in_cube[0],box,cabe['NAXIS3'])

    seg_freq_array,freq_array=zaxisCube(cabe)

    Molines_A_df=line2frame([0,len(ALMAb7_final)],freq_array,ALMAb7_final)

    return ALMAb7_final,Molines_A_df,[x,y]
