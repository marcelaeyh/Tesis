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
from tqdm import tqdm
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
from scipy.integrate import quad
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

    cube_dictio['Frequency']=wavelength_vec[Zchan_range[0]:Zchan_range[1]]*1e-9
    Fluxgrid_df=pd.DataFrame(data=cube_dictio).T
    df_emissionLine=Fluxgrid_df.transpose().set_index('Frequency')
    
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
        spw_lbc_trc_pix = [1185,1225,1335,1375] # 150 x 150 px
    elif spw==31: 
        spw_lbc_trc_pix = [1185,1225,1335,1375] # 150 x 150 px 
    
    ####
    #ALMA cube name
    ###

    Cubepath = '/media/marcela/MARCE_SATA/'+source_name+'/member.uid___A001_X87d_Xb1f.'+source_name+'_sci.spw'+str(spw)+'.cube.I.pbcor.fits'
    
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

    Molines_A_df=line2frame([0,958],freq_array,ALMAb7_final)

    return ALMAb7_final,Molines_A_df

def plotpix(px,channel):

    plt.figure(figsize=(10,5))
    plt.title(px,fontsize=14)
    plt.xlabel('Frequency [GHz]',fontsize=12)
    plt.ylabel('$[Jy/beam]$',fontsize=12)
    plt.grid()

    plt.step(Molines_A_df[px].index[channel[0]:channel[1]],Molines_A_df[px].iloc[channel[0]:channel[1]])



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
    
def clas(px,channel,plot=False):

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

    if s<0.09:
        a = 'Ruido sum'
    elif clas_g(result_l) != 'Hay linea' or clas_g(result_r) != 'Hay linea':
        a = 'Ruido gauss'
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

def gauss_model(px,channel,plot=False):
    
    pars=0
    a = clas(px,[channel[0],channel[1]])
    
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
          pars['peak%d_sigma' % (i+1)].set(value=ff[2][1],min=1e-3,max=0.1)
          pars['peak%d_amplitude' % (i+1)].set(value=ff[0][1],min=0,max=1e-2)
            
        # Hacer el fit al modelo inicial hasta que se ajusten las gaussianas
        out=model.fit(y,pars,x=x) # run fitting algorithm
        comps = out.eval_components(x=x) # fit results for each line
        
        
        
        if plot == True:
            # Data
            plotpix(px,[channel[0],channel[1]])
            # Componentes 
            for i in range(npeaks):
                plt.plot(x, comps['peak%d_' % (i+1)], label='peak'+str(i+1))
            
            # Ajuste 
            plt.plot(x, out.best_fit, '-', color='purple')
    return pars
      
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


def integral_gaussmodel(px,channel):
    
    pars = gauss_model(px, channel)
    
    amp1 = pars['peak1_amplitude'].value
    cen1 = pars['peak1_center'].value
    wid1 = pars['peak1_sigma'].value
    
    amp2 = pars['peak2_amplitude'].value
    cen2 = pars['peak2_center'].value
    wid2 = pars['peak2_sigma'].value
    
    lmin = min(Molines_A_df.iloc[channel[0]:channel[1]].index)
    lmax = max(Molines_A_df.iloc[channel[0]:channel[1]].index)
    
    resultado1, error1 = quad(lambda x: gaussian(x, amp1,cen1,wid1), lmin, lmax)
    resultado2, error2 = quad(lambda x: gaussian(x, amp2,cen2,wid2), lmin, lmax)
    
    return resultado1+resultado2



def plotfont(channel,px=0):
    plt.figure(figsize=(10,10))
    plt.imshow(cube[channel,:,:],vmin=-0.03, vmax=0.03,cmap='inferno',interpolation=None)
    plt.colorbar()
    plt.title(str(channel), fontsize=15)
    if px!=0:
        plt.plot(int(px[4:6]),int(px[7:]),'o',color='lime',marker='s',markersize=1)
        
        
'''
Here, the code should start.
'''
cube,Molines_A_df = Cube_to_df('IRAS_15445-5449',29)

Molines_A_df = Molines_A_df.sort_values('Frequency',ascending=False)

pm = pix_max(Molines_A_df)


for i in range(0,150):
    px = 'Pix_'+str(i)+'_78'
    #px='Pix_65_35'
    
    a = clas(px,[370,500],plot=True)
    
for i in range(0,150):
    px = 'Pix_65_'+str(i)
    #px='Pix_8_64'
    
    gauss_model(px, [120,680],plot=True)

integrals = []
pixels = []
for i in tqdm(range(0,150)):
    for j in range(0,150):
        
        px = 'Pix_'+str(i)+'_'+str(j)
        inte = integral_gaussmodel(px,[370,500])
        pixels.append(px)
        integrals.append(inte)

px = 'Pix_71_78'
gauss_model(px, [50,700],plot=True)

        

sumaCO=np.sum(cube[:,:,:],axis=0)
 
plt.imshow(sumaCO,origin='lower',cmap='Greys',vmin=0.0)

plt.colorbar()

plt.contour(sumaCO,levels=(0.4,1,1.2,1.4),colors='gray')

plt.show()



### MAPA PICOS

matrix = []
for i in range(0,150):
    fil = []
    for j in range(0,150):
        px = 'Pix_'+str(i)+'_'+str(j)
        a = clas(px,[120,700])
        
        fil.append(a[0])
    matrix.append(fil)

mapa = {"Ruido": 0, "Un pico": 1, "Dos picos": 2}
matrix_ = np.vectorize(mapa.get)(matrix)

cmap = ListedColormap(["lightcoral", "indianred", "brown"])


# Paso 4: Hacer el plot con imshow
plt.figure(figsize=(7,7))
plt.imshow(matrix_.T,cmap=cmap,interpolation=None)
plt.plot(1,1,label='Ruido',color='lightcoral')
plt.plot(1,1,label='Un pico',color='indianred')
plt.plot(1,1,label='Dos picos',color='brown')

plt.gca().invert_yaxis()

plt.legend(fontsize=9)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
#Many functions that could work, after some tweaks.

# @timer
# def TransitionStats(df):
#     """
#     Slices columns in DFs in three sectors

#     Params:
#     df(pandas.df): dataframe with the spectral cube information in the columns.
#     Returns: 
#     Tuple with lists of channel index and counts of peak flux per column(pixel): SectorA and SectorB are the outer thirds 
#     of the spectral range. While emisector is the third containing the rest_wavelenth.
#     """

# #    sectorA=int(df.shape[0]/3.0)
# #    sectorB=int(df.shape[0]*2.0/3.0)
#     sectorA = 60
#     sectorB = 220
#     channelPeak_emisector=np.zeros(160)
#     channelPeak_Asector=np.zeros(60)
#     channelPeak_Bsector=np.zeros(80)

#     for k in df.keys():
        
#         indexPeak_emisector=np.where(df[k].iloc[60:220]==df[k].iloc[60:220].max())[0][0]
#         channelPeak_emisector[indexPeak_emisector]=channelPeak_emisector[indexPeak_emisector]+1

#         indexPeak_Asector=np.where(df[k].iloc[0:sectorA]==df[k].iloc[0:sectorA].max())[0][0]
#         channelPeak_Asector[indexPeak_Asector]=channelPeak_Asector[indexPeak_Asector]+1

#         indexPeak_Bsector=np.where(df[k].iloc[sectorB:]==df[k].iloc[sectorB:].max())[0][0]
#         channelPeak_Bsector[indexPeak_Bsector]=channelPeak_Bsector[indexPeak_Bsector]+1

#     return (channelPeak_Asector,channelPeak_emisector,channelPeak_Bsector)

# @timer
# def velDistribution(df,PixList,flagLine):
#     """ Get the peakWavelength distribution of a df transition """
#     Pewa=[]
#     veloDict={}
#     for pix in df[PixList].columns[0:]:
#         maxvalue=df[pix].iloc[80:102].max()
#         maxIndex=df[pix].iloc[80:102].index[df[pix].iloc[80:102]==maxvalue][0]
#         Pewa.append(maxIndex)
# #        print(Dppler_vel_vector(Pewa,flagLine))
#         pixMaxLamda={pix : [Dppler_vel_vector(Pewa,flagLine),pix.split('_')[1],pix.split('_')[2]]}
#         veloDict.update(pixMaxLamda)
#         Pewa = []

#     return veloDict    
    
        
#     # if len(PixList) > 0: 
#     #     for k in PixList:
#     #         maxvalue=df[k].iloc[60:121].max()
#     #         Pewa.append(df[k].iloc[60:121].index[df[k].iloc[60:121]==maxvalue][0])
#     # else:
#     #     print(PixList)
#     #     maxvalue=df[PixList].iloc[60:121].max()
#     #     print(df[PixList].iloc[60:121].index[df[Pixlist].iloc[60:121]==maxvalue][0])
#     #     Pewa.append(df[PixList].iloc[60:121].index[df[Pixlist].iloc[60:121]==maxvalue][0])



# @timer
# def pixECDF(df,pix):
#     """ Sorts pix flux values in the emisector and plots the Empirical Cumulative Distribution Function """
# #    sns.set()
#     for k in range(0,len(pix)+1):
#         x=np.sort(df[pix].iloc[59:121])
#         y=np.arange(1,len(x)+1)/len(x)
#         _ = plt.plot(x,y,marker='.',linestyle='none',alpha=0.5)
#     plt.xlabel('Peak Flux (Jy)')
#     plt.ylabel('ECDF')
#     plt.margins(0.02)    

#     plt.savefig('ECDF_pix.pdf')
#     return print('ECDF in the figure ')


# @timer
# def pixECDFWavelength(listade):
#     """ Sorts pix flux values in the emisector and plots the Empirical Cumulative Distribution Function """
#     sns.set()
#     for k in range(0,len(listade)+1):
#         x=np.sort(listade)
#         y=np.arange(1,len(x)+1)/len(x)
#         _ = plt.plot(x,y,marker='.',linestyle='none',alpha=0.5)
#     plt.xlabel('Wavelenght (um)')
#     plt.ylabel('ECDF')
#     plt.margins(0.02)    

# #    plt.savefig('NewFigs_Oct2020/ECDF_pix.pdf')
#     return print('ECDF in the figure ')


# @timer
# def DescribeIT(frame):
    
#     frame_described = frame.describe()
    
#     return frame_described
    
# @timer
# def LineWaveRange(Clean_lambdas,restFreq):
#     """
#     Looking for the index of the restfrequency within our wavelength array
#     """
#     lamb_refence_bin=[np.abs(Clean_lambdas[i]-restFreq) for i in range(len(Clean_lambdas))]
#     bin_ref=np.where(lamb_refence_bin==np.min(lamb_refence_bin))[0][0] 
#     return bin_ref

# @timer
# def IntegS(df,pix,flag='H1'):
#     ''' 
#     Calculates the Area under the curve given by the spectral line 
#     df: DataFrame with the Spectral cube observations
#     pix: List with pixels names, formated as key values of the df
#     H2 1-0 S(1), wavelength (channel) range [82:99]
#     H2 2-1 S(1), wavelength (channel) range [80:94]
#     H2 1-0 Q(3), wavelength (channel) range [80:93]
#     Returns the integral as the Sum * deltaWavelength.
#     '''


#     if (flag == 'H1') :
#         LineSector = [82,99]
#     elif (flag == 'S0') :
#         LineSector = [80,98]
#     elif (flag == 'S2') :
#         LineSector = [81,93]
#     elif (flag == 'S3') :
#         LineSector = [27,40]
#     elif (flag == 'H2') :
#         LineSector = [80,94]
#     elif (flag == 'Q3') :
#         LineSector = [80,93]
#     elif (flag == 'Q1') :
#         LineSector = [82,92]
#     elif (flag == '13CO') :
#         LineSector = [50,220]
#     else:
#         print('better give the sector')
        
#     Sinteg=df[pix].iloc[LineSector[0]:LineSector[1]].sum()*delta_lambda*-1.0e-3

#     return Sinteg

# @timer
# def corta(naamlist):
#     North_trSigPix=[]
#     South_trSigPix=[]
#     for pixNaam in naamlist:
#         if(int(pixNaam.split('_')[1]) >= 150 and int(pixNaam.split('_')[1]) < 1250):
# #            if(int(pixNaam.split('_')[2]) >= 55 and int(pixNaam.split('_')[2]) < 111):
#             if(int(pixNaam.split('_')[2]) >= 50 and int(pixNaam.split('_')[2]) < 200):
#                 North_trSigPix.append(pixNaam)
#         else:
#             South_trSigPix.append(pixNaam)

#     return (North_trSigPix,South_trSigPix)


# def PlotMarbels(df,listMaxs):

#     d=f=0
#     a=b=0
#     e=0
#     vel0 = ''
    
# #    _ = plt.plot([0,400],[0,400],markersize=1e-5,label='km s$^{-1}$',linestyle='none',alpha=0.05)
    
#     for i in range(0,len(df[listMaxs].keys().to_list())):
        
#         deColor=df[listMaxs[i]]
#         if (deColor < 0.6):
#             c = 'white'
#         elif(deColor >= 0.6 and deColor < 2.8):
#              c= 'gray'
#         elif(deColor >= 2.8 and deColor < 3.5):
#              c= 'blue'
#         elif(deColor >= 3.5 and deColor <3.8):
#              c= 'green'
#         elif(deColor >= 3.80 and deColor < 4.1):
#              c= 'red'
#         else:
#              c = 'yellow'
           
#         plt.plot((int(df[listMaxs].keys()[i].split('_')[2])),(int(df[listMaxs].keys()[i].split('_')[1])),marker='o',linestyle='none',markersize=1.5**(df[df[listMaxs].keys()[i]].max()),color=c,alpha=0.5)
# #        plt.plot((int(df[listMaxs].columns[i].split('_')[2])),(int(df[listMaxs].columns[i].split('_')[1])),marker='o',linestyle='none',markersize=2.0*(df[df[listMaxs].columns[i]].max()),color='blue',alpha=0.5)

#     plt.title('13CO: Velocity components \n J = 3-2')
# #    plt.xticks([12,25,38,51],[1.3,0.65,0.0,-0.65])
# #    plt.yticks([10,23,36,49],[1.3,0.65,0.0,-0.65])

#     plt.xlabel('RA offset (arcsec)')
#     plt.ylabel('DEC offset (arcsec)')
#     plt.legend()
#     return print('Done!')



# def PlotMarbels2(df,listMaxs,c='gray'):

#     for i in range(0,len(listMaxs)):

#         indexPeak_emisector=np.where( df[listMaxs[i]] == df[listMaxs[i]].describe().loc['max'] )[0][0]
        
#         if indexPeak_emisector in range(170,220):
#             c='cyan'
#         elif indexPeak_emisector in range(155,170):
#             c='purple'
#         elif indexPeak_emisector in range(141,155):
#             c='blue'
#         elif indexPeak_emisector in range(130,141):
#             c='green'
#         elif indexPeak_emisector in range(120,130):  
#             c='orange'
#         elif indexPeak_emisector in range(110,120):  
#             c='red'
#         elif indexPeak_emisector in range(90,110):
#             c='darkred'
#         else:
#             c='gray'
            
#         plt.plot((int(df[listMaxs].columns[i].split('_')[2])),(int(df[listMaxs].columns[i].split('_')[1])),marker='o',linestyle='none',markersize=1.5e2*(df[df[listMaxs].columns[i]].max()),color=c,alpha=0.5)
        
# #    plt.xticks([12,25,38,51],[1.3,0.65,0.0,-0.65])
# #    plt.title('Molecular Hydrogen: Velocity components \n H$_{2}$ 1-0 S(1)')
# #    plt.yticks([10,23,36,49],[1.3,0.65,0.0,-0.65])
#     plt.xlabel('RA offset (arcsec)')
#     plt.ylabel('DEC offset (arcsec)')
#     plt.legend('')
#     return print('Done!')



# @timer
# def PixStats_max(df,sector='Line'):
#     """
#     Given a transition in a df, it finds the pixels with the max value and record the pix names
#     in a list
#     params:
#     df dataframe.
#     sector (str) Wavelength range of interest in the spectra.
#              A: 0 - 60
#              B: 120 - 180
#           Line: 60 - 120 (Index in the wavelength array with the spectral line of interest)
#     """

#     if dp == False:
#         if(sector == 'A'):
#             maxFluxes_obj=df.iloc[0:60].describe().loc['max']   ## Maximum value per pixel
#             peakFlux_obj=df.iloc[0:60].describe().loc['max'].max() ## Total Maximum value
#         elif(sector == 'B'):
#             maxFluxes_obj=df.iloc[220:].describe().loc['max']   ## Maximum value per pixel
#             peakFlux_obj=df.iloc[220:].describe().loc['max'].max() ## Total Maximum value
#         else:
#             maxFluxes_obj=df.iloc[70:220].describe().loc['max']   ## Maximum value per pixel
#             peakFlux_obj=df.iloc[70:220].describe().loc['max'].max() ## Total Maximum value

#     else:
#         if(sector == 'A'):
#             maxFluxes_obj=df.iloc[0:60].describe().loc['max']   ## Maximum value per pixel
#             peakFlux_obj=df.iloc[0:60].describe().loc['max'].max() ## Total Maximum value
#         elif(sector == 'B'):
#             maxFluxes_obj=df.iloc[220:].describe().loc['max']   ## Maximum value per pixel
#             peakFlux_obj=df.iloc[220:].describe().loc['max'].max() ## Total Maximum value
#         else:
#             maxFluxes_obj=df.iloc[70:220].describe().loc['max']   ## Maximum value per pixel
#             peakFlux_obj=df.iloc[70:220].describe().loc['max'].max() ## Total Maximum value



            
#     Peak_pix_naam=maxFluxes_obj[maxFluxes_obj==maxFluxes_obj.describe().loc['max']].index[0] ## Pixel name of the max
#     PlotIT(df,Peak_pix_naam)
#     return Peak_pix_naam

# @timer
# def PixStats_uPQu(df,sector='Line',ente=0.3):
#     """ Compares the max value in a pixel with the MAX value (peak flux) of the transition, within a given wavelength range 
#         (A=[0:60],B=[120:180],Line=[60:120]). If the peak flux in the pixel is above 69.99% of the MAX value, then the pixel
#         name (Pix_j_i) gets recorded.
#         params:
#         df dataframe with the transition info.
#     """
#     if(sector=='A'):
#         print('Wavelength range -> Sector A')
#         borderlist=[0,1,2,59,58,57]
#         UpperQua_Flux_naam=[]
#         peak_pix_val=df[PixStats_max(df)].iloc[0:60].describe().loc['max']
#         for pix in df.columns:
#             fast_cociente=df[pix].iloc[:60].describe().loc['max']/peak_pix_val
#             if (fast_cociente >= 0.98):
#                 jj=pix.split('_')[1]
#                 ii=pix.split('_')[2]
#                 if((int(ii) not in borderlist) & (int(jj) not in borderlist)):
#                      UpperQua_Flux_naam.append(str(pix))

#     elif(sector=='B'):
#         print('Wavelength range -> Sector B')
#         UpperQua_Flux_obj=df.iloc[220:].describe().loc['75%'] ##Upper Quartile value per pixel
#         borderlist=[0,1,2,59,58,57]
#         UpperQua_Flux_naam=[]
#         peak_pix_val=df[PixStats_max(df)].iloc[220:].describe().loc['max']
#         for pix in df.columns:
#             fast_cociente=df[pix].iloc[220:].describe().loc['max']/peak_pix_val
#             if (fast_cociente >= 0.98):
#                 jj=pix.split('_')[1]
#                 ii=pix.split('_')[2]
#                 if((int(ii) not in borderlist) & (int(jj) not in borderlist)):
#                     UpperQua_Flux_naam.append(str(pix))

#     else:
#         print('Wavelength range -> EmiSector')
#         #UpperQua_Flux_obj=df.iloc[int(df.shape[0]/3):int(df.shape[0]*2.0/3.0)].describe().loc['75%'] ##Upper Quartile value per pixel
#         borderlist=[0,1,2,59,58,57]
#         UpperQua_Flux_naam=[]
#         peak_pix_val=df[PixStats_max(df)].iloc[70:220].describe().loc['max']
#         for pix in df.columns:
#             fast_cociente=df[pix].iloc[70:220].describe().loc['max']/peak_pix_val
#             if (fast_cociente >= ente):
#                 jj=pix.split('_')[1]
#                 ii=pix.split('_')[2]
#                 if((int(ii) not in borderlist) & (int(jj) not in borderlist)):
#                     UpperQua_Flux_naam.append(str(pix))
                    
#     return UpperQua_Flux_naam


# @timer
# def PixStats_DoublePeak(df,sector='Line',ente=0.3):
#     """ Compares the max value in a pixel with the MAX value (peak flux) of the transition, within a given wavelength range 
#         (A=[0:60],B=[120:180],Line=[60:120]). If the peak flux in the pixel is above 69.99% of the MAX value, then the pixel
#         name (Pix_j_i) gets recorded.
#         params:
#         df dataframe with the transition info.
#     """
#     if(sector=='A'):
#         print('Wavelength range -> Sector A')
#         borderlist=[0,1,2,59,58,57]
#         UpperQua_Flux_naam=[]
#         peak_pix_val=df[PixStats_max(df)].iloc[108:120].describe().loc['max']
#         for pix in df.columns:
#             fast_cociente=df[pix].iloc[108:120].describe().loc['max']/peak_pix_val
#             if (fast_cociente >= 0.98):
#                 jj=pix.split('_')[1]
#                 ii=pix.split('_')[2]
#                 if((int(ii) not in borderlist) & (int(jj) not in borderlist)):
#                      UpperQua_Flux_naam.append(str(pix))

#     elif(sector=='B'):
#         print('Wavelength range -> Sector B')
#         UpperQua_Flux_obj=df.iloc[220:].describe().loc['75%'] ##Upper Quartile value per pixel
#         borderlist=[0,1,2,59,58,57]
#         UpperQua_Flux_naam=[]
#         peak_pix_val=df[PixStats_max(df)].iloc[138:150].describe().loc['max']
#         for pix in df.columns:
#             fast_cociente=df[pix].iloc[138:150].describe().loc['max']/peak_pix_val
#             if (fast_cociente >= 0.98):
#                 jj=pix.split('_')[1]
#                 ii=pix.split('_')[2]
#                 if((int(ii) not in borderlist) & (int(jj) not in borderlist)):
#                     UpperQua_Flux_naam.append(str(pix))

#     else:
#         print('Wavelength range -> EmiSector')
#         #UpperQua_Flux_obj=df.iloc[int(df.shape[0]/3):int(df.shape[0]*2.0/3.0)].describe().loc['75%'] ##Upper Quartile value per pixel
#         borderlist=[0,1,2,59,58,57]
#         UpperQua_Flux_naam=[]
#         peak_pix_val=df[PixStats_max(df)].iloc[122:138].describe().loc['max']
#         for pix in df.columns:
#             fast_cociente=df[pix].iloc[122:138].describe().loc['max']/peak_pix_val
#             if (fast_cociente >= ente):
#                 jj=pix.split('_')[1]
#                 ii=pix.split('_')[2]
#                 if((int(ii) not in borderlist) & (int(jj) not in borderlist)):
#                     UpperQua_Flux_naam.append(str(pix))
                    
#     return UpperQua_Flux_naam




# @timer
# def PixPicker(df,transi='13CO'):
#     '''
#     In a spectra with channels index from 60:121, finds the standart deviation of the channels in both sides of the spectral feauture.
#     df= dataFrame with the spectral datacube information
#     transi= label with name of transition 
#             H1 -->H2_10_S(1)
#             H2 -->H2_21_S(1)
#             Q1 -->H2_10_Q(1)
#             Q3 -->H2_10_Q(3)
#             Brg -->BRGamma
#     returns tuple(meanSTD,List3sgma,List5sigma,List10sigma)
#     '''
#     if (transi == 'H1') :
#         Asector=[60,82] 
#         LineSector = [82,99]
#         Bsector = [99,121]

#     elif (transi == 'H2') :
#         Asector = [30,60]
#         LineSector = [80,94]
#         Bsector = [94,108]
  
#     elif (transi == 'Q3') :
#         Asector = [50,70]
#         LineSector = [80,93]
#         Bsector = [95,106]

#     elif (transi == 'Q1') :
#         Asector = [53,66]
#         LineSector = [81,93]
#         Bsector = [66,79]

#     elif (transi == 'S3') :
#         Asector = [1,14]
#         LineSector = [27,40]
#         Bsector = [15,25]

#     elif (transi == 'S2') :
#         Asector = [65,80]
#         LineSector = [82,94]
#         Bsector = [95,105]

#     elif (transi == 'S0') :
#         Asector = [50,60]
#         LineSector = [78,100]
#         Bsector = [110,120]
        
#     elif( transi == 'Brg'):
#         Asector = [45,71]
#         LineSector = [71,105]
#         Bsector = [110,130]

#     elif( transi == '13CO'):
#         Asector = [0,60]
#         LineSector = [60,220]
#         Bsector = [220,300]

#     elif( transi == '13COPeaks'):
#         Asector = [90,107]
#         LineSector = [120,140]
#         Bsector = [140,180]
#         Xsector = [107,120]

#     print(transi,Asector)

#     Side_sectorsMean=np.mean([df.iloc[Asector[0]:Asector[1]].describe().loc['std'].mean(),df.iloc[Bsector[0]:Bsector[1]].describe().loc['std'].mean()])

#     # if (transi == 'H2'):
#     #     timeSig = 1.4
#     #     timeSigII = 3.0
#     #     timeSigIII = 6.0

#     if (transi == '13COPeaks'):                             

#         SigmaCore_pixel_list = []
#         SigmaRwing_pixel_list = []
#         SigmaBwing_pixel_list = []
#         SigmaXsector_pixel_list = []
#         SigmaRare_pixel_list = []

# #        Side_sectorAMean=df.iloc[Asector[0]:Asector[1]].describe().loc['mean'].mean()
# #        Side_sectorBMean=df.iloc[Bsector[0]:Bsector[1]].describe().loc['mean'].mean()
        
#         for k in df.columns.to_list():

#             Side_sectorAMean=df[k].iloc[Asector[0]:Asector[1]].describe().loc['mean'].mean()
#             Side_sectorBMean=df[k].iloc[Bsector[0]:Bsector[1]].describe().loc['mean'].mean()
#             Side_sectorXMean=df[k].iloc[Xsector[0]:Xsector[1]].describe().loc['mean'].mean()
#             Line_sectorMean=df[k].iloc[LineSector[0]:LineSector[1]].describe().loc['mean']


#             if (Side_sectorXMean > Line_sectorMean and Side_sectorXMean > Side_sectorBMean):
#                 SigmaXsector_pixel_list.append(k)
                
#             elif (Side_sectorBMean > Side_sectorXMean and Side_sectorBMean > Side_sectorAMean):
#                 SigmaBwing_pixel_list.append(k)

#             elif (Side_sectorAMean > Side_sectorXMean and Side_sectorAMean > Side_sectorBMean):
#                 SigmaRwing_pixel_list.append(k)

#             elif (Line_sectorMean > Side_sectorXMean and Line_sectorMean > Side_sectorBMean):
#                 SigmaCore_pixel_list.append(k)
                                
#             else:
#                 SigmaRare_pixel_list.append(k)
        
#         return (Side_sectorAMean,Side_sectorBMean,SigmaRare_pixel_list,SigmaBwing_pixel_list,SigmaRwing_pixel_list,SigmaCore_pixel_list,SigmaXsector_pixel_list)
    
#     else:
#         timeSig =1.5
#         timeSigII=2.5
#         timeSigIII=5.0


#         Sigma3_series = df.iloc[LineSector[0]:LineSector[1]].describe().loc['mean'] >= (Side_sectorsMean * timeSig)
#         Sigma3_pixel_list=Sigma3_series[Sigma3_series == True].index.tolist()
#         Sigma3_pixel_residual_list=Sigma3_series[Sigma3_series == False].index
        
#         Sigma5_series = df.iloc[LineSector[0]:LineSector[1]].describe().loc['mean'] >= (Side_sectorsMean * timeSigII) #emissionAbove5sigma
#         Sigma5_pixel_list=Sigma5_series[Sigma5_series == True].index.tolist()
#         Sigma5_pixel_residual_list=Sigma5_series[Sigma5_series == False].index
                                 
#         TenSigma_series = df.iloc[LineSector[0]:LineSector[1]].describe().loc['mean'] >= (Side_sectorsMean * timeSigIII) #emissionAbove10sigma
#         TenSigma_pixel_list= TenSigma_series[TenSigma_series == True].index.tolist()
#         TenSigma_pixel_residual_list=TenSigma_series[TenSigma_series == False].index

#         return (Side_sectorsMean,Sigma3_pixel_list,Sigma5_pixel_list,TenSigma_pixel_list)

# #    elif (transi == 'Q1'):
# #        Side_sectorsMean=0.004
# #        timeSig =3.0
# #        timeSigII=5.0
# #        timeSigIII=10.
#     # elif (transi == '13COPeaks'):
#     #    Side_sectorsMean=1.4
#     #    timeSig =0.0023
#     #    timeSigII=3.0
#     #    timeSigIII=6.0

# @timer
# def PlotIT(df,pix):
#     plt.step(df.index,df[pix])
#     plt.xlabel('vel (km~s^{-1})')
#     plt.ylabel('Flux (?Jy)')
#     plt.legend()
#     plt.show()
#     return print('DitIT :P')


# def Dppler_vel_vector(lista,flag):
#     """
#     Construct the array of velocities associated to the spectral line of interest.
#     """

#     speedC=2.99792458e5 ##(km/s)
#     ar_vertical=[2.1218,2.2477,2.4237,2.1660]
#     if flag == 'H1' :
#         lamb_refe=ar_vertical[0]
#     elif flag == 'H2':
#         lamb_refe=ar_vertical[1]
#     elif flag == 'Q3':
#         lamb_refe=ar_vertical[2]
#     elif flag == 'Brga':
#         lamb_refe=ar_vertical[3]
    

#     velo_array=[(speedC*(lambd-lamb_refe)/lamb_refe) for lambd in lista] 

#     return velo_array[0]

# def UpDown_pix(pixel_list):
#     up_pixel_naam=[]
#     down_pixel_naam=[]
#     for i in range(0,len(pixel_list)):
#         j=int(pixel_list[i].split('_')[1])
#         if j >= 30 :
#             up_pixel_naam.append(pixel_list[i])
#         else:
#             down_pixel_naam.append(pixel_list[i])
#     return (up_pixel_naam,down_pixel_naam)


# def SavetheList(lista,flag='H5'):
#     archi=open('Pixels_3Sma_'+flag+'.txt','w')
#     for line in lista:
#         archi.write(line)
#         archi.write('\n')
#     archi.close()    
    
#     return print('saved!')


#Cubepath='/Users/aperez/W-letter/IRAS15445/VLTstuff/Red2017/bg.f.IRAS15445-5449.new.fits'
# plt.step(ALMA_13coFrame.index[60:220],ALMA_13coFrame.index[60:220].T.describe().loc['mean'])
# plt.step(ALMA_13coFrame.index[60:90],ALMA_13coFrame.index[60:90].T.describe().loc['mean'],c='r')
# Plt.step(ALMA_13coFrame.index[190:220],ALMA_13coFrame.index[190:220].T.describe().loc['mean'],c='b')
# plt.show()

#Cubepath='I18286spw0_cube_X18b.fits'

#ALMAb7_Cube=cropCube(TrthCO_spec,[350,174,300,324])

##Masers_A_df=line2frame([0,1920],velo_array,Maser_spec_a)

#Masers_A_pxp_described = DescribeIT(Masers_A_df)

# TreceCO = PixPicker(ALMA_13coFrame,'13CO')

# Snu_13CO=IntegS(ALMA_13coFrame,TreceCO[1],'13CO')

# PlotMarbels(Snu_13CO,Snu_13CO.keys().tolist())

#TreceCO_peaks = PixPicker(ALMA_13coFrame,'13COPeaks')

#Testn,TestS=corta(TreceCO_peaks[1])

# Pixcrd_RA = np.array([[5,36,890],[5,10,890]],dtype=np.float64)
# pixcrd_DEC = np.array([[5,5,890],[51,5,890]],dtype=np.float64)

# inDegs_RA_A=w.wcs_pix2world(pixcrd_RA[[0]],0)
# inDegs_RA_B=w.wcs_pix2world(pixcrd_RA[[1]],0)

# inDegs_DEC_A=w.wcs_pix2world(pixcrd_DEC[[0]],0)
# inDegs_DEC_B=w.wcs_pix2world(pixcrd_DEC[[1]],0)


# c1_RA=SkyCoord(inDegs_RA_A[0][0]*u.deg,inDegs_RA_A[0][1]*u.deg)

# c2_RA=SkyCoord(inDegs_RA_B[0][0]*u.deg,inDegs_RA_B[0][1]*u.deg)

# c1_DEC=SkyCoord(inDegs_DEC_A[0][0]*u.deg,inDegs_DEC_A[0][1]*u.deg)

# c2_DEC=SkyCoord(inDegs_DEC_B[0][0]*u.deg,inDegs_DEC_B[0][1]*u.deg)

# angsep=c1_RA.separation(c2_RA)

# Clean_NIR_Cube=cropCube(NIR_Spectrum,[5,10,4,6])

# Clean_lambdas,lambdas=zaxisCube(cabe)

# range_H2_10s0=[LineWaveRange(Clean_lambdas,2.2235)-90,LineWaveRange(Clean_lambdas,2.2235)+90]
# range_H2_10s1=[LineWaveRange(Clean_lambdas,2.12134)-90,LineWaveRange(Clean_lambdas,2.12134)+90]
# range_H2_10s2=[LineWaveRange(Clean_lambdas,2.0338)-90,LineWaveRange(Clean_lambdas,2.0338)+90]
# range_H2_10s3=[LineWaveRange(Clean_lambdas,1.9576)-36,LineWaveRange(Clean_lambdas,1.9576)+36]

# range_H2_21s1=[LineWaveRange(Clean_lambdas,2.2477)-90,LineWaveRange(Clean_lambdas,2.2477)+90]

# range_H2_10q1=[LineWaveRange(Clean_lambdas,2.4066)-90,LineWaveRange(Clean_lambdas,2.4066)+90]
# range_H2_10q3=[LineWaveRange(Clean_lambdas,2.4237)-90,LineWaveRange(Clean_lambdas,2.4237)+90]

# range_H2_Brgamma=[LineWaveRange(Clean_lambdas,2.1660)-90,LineWaveRange(Clean_lambdas,2.1660)+90]
# range_ato_Fe=[LineWaveRange(Clean_lambdas,2.133)-90,LineWaveRange(Clean_lambdas,2.133)+90]
# range_ato_NaDuo=[LineWaveRange(Clean_lambdas,2.207)-90,LineWaveRange(Clean_lambdas,2.207)+90]

# ##MolecularLines
# LinedF_H2_10S3=line2frame(range_H2_10s3,Clean_lambdas,Clean_NIR_Cube)
# LinedF_H2_10S2=line2frame(range_H2_10s2,Clean_lambdas,Clean_NIR_Cube)
# LinedF_H2_10S1=line2frame(range_H2_10s1,Clean_lambdas,Clean_NIR_Cube)
# LinedF_H2_10S0=line2frame(range_H2_10s0,Clean_lambdas,Clean_NIR_Cube)

# LinedF_H2_21S1=line2frame(range_H2_21s1,Clean_lambdas,Clean_NIR_Cube)

# LinedF_H2_10Q1=line2frame(range_H2_10q1,Clean_lambdas,Clean_NIR_Cube)
# LinedF_H2_10Q3=line2frame(range_H2_10q3,Clean_lambdas,Clean_NIR_Cube)

# ##AtomicLines
# LinedF_H2_Brgamma=line2frame(range_H2_Brgamma,Clean_lambdas,Clean_NIR_Cube)
# LinedF_Ato_Fe=line2frame(range_ato_Fe,Clean_lambdas,Clean_NIR_Cube)
# LinedF_Ato_NaDuo=line2frame(range_ato_NaDuo,Clean_lambdas,Clean_NIR_Cube)

# ##Correcting bad pixels, which are columns
# ##LinedF_H2_10S0[H1S0[1]].iloc[50:90].describe().loc['max'][LinedF_H2_10S0[H1S0[1]].iloc[50:90].describe().loc['max'] == LinedF_H2_10S0[H1S0[1]].iloc[50:90].describe().loc['max'].max()]

# LinedF_H2_10Q3['Pix_59_30'].iloc[71:73]=LinedF_H2_10Q3['Pix_59_30'].iloc[73:83].mean()
# LinedF_H2_10Q3['Pix_59_31'].iloc[71:74]=LinedF_H2_10Q3['Pix_59_31'].iloc[74:83].mean()
# LinedF_H2_10S0['Pix_59_40'].iloc[70:76]=LinedF_H2_10S0['Pix_59_40'].iloc[60:70].mean()
# LinedF_H2_10S0['Pix_59_41'].iloc[68:75]=LinedF_H2_10S0['Pix_59_41'].iloc[58:70].mean()
# LinedF_H2_10S0['Pix_59_42'].iloc[68:75]=LinedF_H2_10S0['Pix_59_42'].iloc[58:70].mean()
# LinedF_H2_Brgamma['Pix_59_6'].iloc[138:140]=LinedF_H2_Brgamma['Pix_59_6'].iloc[130:138].mean()
# LinedF_H2_Brgamma['Pix_59_7'].iloc[138:140]=LinedF_H2_Brgamma['Pix_59_7'].iloc[130:138].mean()
# LinedF_H2_Brgamma['Pix_59_44'].iloc[171:176]=LinedF_H2_Brgamma['Pix_59_44'].iloc[165:169].mean()
# LinedF_H2_Brgamma['Pix_59_45'].iloc[171:175]=LinedF_H2_Brgamma['Pix_59_45'].iloc[165:169].mean()
# LinedF_H2_Brgamma['Pix_59_46'].iloc[171:175]=LinedF_H2_Brgamma['Pix_59_46'].iloc[165:169].mean()
# LinedF_H2_Brgamma['Pix_18_1'].iloc[98:101]=LinedF_H2_Brgamma['Pix_18_1'].iloc[90:97].mean()
# LinedF_H2_Brgamma['Pix_19_1'].iloc[98:101]=LinedF_H2_Brgamma['Pix_19_1'].iloc[90:97].mean()
# LinedF_H2_Brgamma['Pix_42_54'].iloc[20:23]=LinedF_H2_Brgamma['Pix_42_54'].iloc[23:30].mean()
# LinedF_H2_Brgamma['Pix_43_54'].iloc[20:23]=LinedF_H2_Brgamma['Pix_43_54'].iloc[23:30].mean()
# LinedF_H2_Brgamma['Pix_42_55'].iloc[20:23]=LinedF_H2_Brgamma['Pix_42_54'].iloc[23:30].mean()
# LinedF_H2_Brgamma['Pix_43_55'].iloc[20:23]=LinedF_H2_Brgamma['Pix_43_54'].iloc[23:30].mean()
# LinedF_H2_Brgamma['Pix_46_58'].iloc[69:73]=LinedF_H2_Brgamma['Pix_42_54'].iloc[73:80].mean()
# LinedF_H2_Brgamma['Pix_47_58'].iloc[69:73]=LinedF_H2_Brgamma['Pix_43_54'].iloc[73:80].mean()

# ###geting some numbers List of pixels to use#
# '''
# First, find the averaged value of the standard deviation at both sides of the spectral line, using spectra-feature-free channels. This is done by the
# function PixPicker.
 
# Example:For the H2 1-0 S1 line, the line free channels are (Asector [60:82]), (Line sector [82:99]), (Bsector [99:121]).

# Asector      LinedF_H2_10S1.iloc[60:82].describe().loc['std'].describe().mean() = 3.54952e-04
# Line Sector  LinedF_H2_10S1.iloc[82:99].describe().loc['mean'].describe().mean() = 2.724e-03
# Bsector      LinedF_H2_10S1.iloc[99:121].describe().loc['std'].describe().mean() = 3.554571e-04

# Get the mean between the mean values in sectors A and B
# np.mean([Asector,Bsector]) = 3.6e-4

# Then, select a list of Pixels whose MEAN flux value, in the line channel range, is X*np.mean(std) in line free channels.

# H1S1, H2S1, and H1Q3 contain lists in their 1,2,3 indexes with different Sigma levels. By default it should be 3, 5, and 10 times the mean of the standart deviation 
# in line-free channels next to the emission line. Because H2S1 is faint, the sigma*factor level for that transition hardcoded in the PixPicker function are 1.5, 3.0, and 6.0

# Now, calculate the integral within each one of those spaxels above the lowest sigma level set for each transition. That for the line sector.

# After finding the pixels with flux above the level that we need, the IntegS function will sum up and multiply by Delta_lambda the flux value for channels that belong to the line
# profile. This value should be used to estimate the Integrated flux Ratios between the transitions of interest. Because the fainter the transition, the less number of pixels, plus
# the fact that the pixels with emission in the H2 1-0 s1 line also have emission for other transitions, the ratios should be estimated using the pixels with emission on the fainter
# transition of those being compared.

# The 'vel_sys' is defined as follows: First, finding the wavelength of the emission peak flux density, and its channel index. Doppler shift is calculated using the wavelenght 
# of the peak with respect to the lambda_ref of each transition. Hence, each Pixel has an associated velocity.
# '''



# H1S1 = PixPicker(LinedF_H2_10S1,'H1')
# H2S1 = PixPicker(LinedF_H2_21S1,'H2')
# H1Q3 = PixPicker(LinedF_H2_10Q3,'Q3')
# BrGa=  PixPicker(LinedF_H2_Brgamma,'Brg')

# H1Q1 = PixPicker(LinedF_H2_10Q1,'Q1')
# H1S2 = PixPicker(LinedF_H2_10S2,'S2')
# H1S3 = PixPicker(LinedF_H2_10S3,'S3')
# H1S0 = PixPicker(LinedF_H2_10S0,'S0')

# LaListaDePeakWavelenghts_H1S1 = LinedF_H2_10S1[LinedF_H2_10S1[H1S1[1]] == LinedF_H2_10S1[H1S1[1]].describe().loc['max']].index.to_list()

# Snu_H1S1=IntegS(LinedF_H2_10S1,H1S1[1],'H1')
# Snu_H2S1=IntegS(LinedF_H2_21S1,H2S1[1],'H2')
# Snu_H1Q3=IntegS(LinedF_H2_10Q3,H1Q3[1],'Q3')
# Snu_H1Q1=IntegS(LinedF_H2_10Q1,H1Q1[1],'Q1')

# Snu_H1S3=IntegS(LinedF_H2_10S3,H1S3[1],'S3')
# Snu_H1S2=IntegS(LinedF_H2_10S2,H1S2[1],'S2')
# Snu_H1S0=IntegS(LinedF_H2_10S0,H1S0[1],'S0')

# Test_A=IntegS(LinedF_H2_10S1,H2S1[1],'H1')
# Test_B=Test_A/Snu_H2S1


# Test_Z=IntegS(LinedF_H2_10S1,H1Q3[1],'H1')
# Test_Y=Test_Z/Snu_H1Q3 


# ##Pixels differenciated by location in the spatial plane
# North_trSigPix_H1,South_trSigPix_H1 = corta(H1S1[1])
# North_trSigPix_H2,South_trSigPix_H2 = corta(H2S1[1])
# North_trSigPix_Q3,South_trSigPix_Q3 = corta(H1Q3[1])

# LaListaDePeakWavelenghts_North = LinedF_H2_10S1[LinedF_H2_10S1[North_trSigPix_H1] == LinedF_H2_10S1[North_trSigPix_H1].describe().loc['max']].index.to_list()
# LaListaDePeakWavelenghts_South = LinedF_H2_10S1[LinedF_H2_10S1[South_trSigPix_H1] == LinedF_H2_10S1[South_trSigPix_H1].describe().loc['max']].index.to_list()

# Waveindex_MaxCounts_N=np.where(TransitionStats(LinedF_H2_10S1[North_trSigPix_H1])[1] == np.max(TransitionStats(LinedF_H2_10S1[North_trSigPix_H1])[1]))[0][0]

# Waveindex_MaxCounts_S=np.where(TransitionStats(LinedF_H2_10S1[South_trSigPix_H1])[1] == np.max(TransitionStats(LinedF_H2_10S1[South_trSigPix_H1])[1]))[0][0]

# Waveindex_MaxCounts_Overall=np.where(TransitionStats(LinedF_H2_10S1[H1S1[1]])[1] == np.max(TransitionStats(LinedF_H2_10S1[H1S1[1]])[1]))[0][0]

# North_PeakFluxwavelength = LinedF_H2_10S1[North_trSigPix_H1].index[60:121][Waveindex_MaxCounts_N]
# South_PeakFluxwavelength = LinedF_H2_10S1[South_trSigPix_H1].index[60:121][Waveindex_MaxCounts_S]
# Overall_PeakFluxwavelength = LinedF_H2_10S1[H1S1[1]].index[60:121][Waveindex_MaxCounts_Overall]

# vel_frame_H1 = pd.DataFrame.from_dict(velDistribution(LinedF_H2_10S1,H1S1[1],flagLine='H1'),orient='index')
# vel_frame_H2 = pd.DataFrame.from_dict(velDistribution(LinedF_H2_21S1,H2S1[1],flagLine='H2'),orient='index')
# vel_frame_Q3 = pd.DataFrame.from_dict(velDistribution(LinedF_H2_10Q3,H1Q3[1],flagLine='Q3'),orient='index')
# vel_frame_BrGa = pd.DataFrame.from_dict(velDistribution(LinedF_H2_Brgamma,BrGa[1],flagLine='Brga'),orient='index')

# vel_frame_BrGa['vel_sys'] = vel_frame_BrGa[0]
# vel_frame_BrGa['xPix'] = vel_frame_BrGa[2]
# vel_frame_BrGa['yPix'] = vel_frame_BrGa[1]

# vel_frame_BrGa = vel_frame_BrGa.drop(axis=1,columns=0)
# vel_frame_BrGa = vel_frame_BrGa.drop(axis=1,columns=1)
# vel_frame_BrGa = vel_frame_BrGa.drop(axis=1,columns=2)

# vel_frame_H1['vel_sys'] = vel_frame_H1[0]
# vel_frame_H1['xPix'] = vel_frame_H1[2]
# vel_frame_H1['yPix'] = vel_frame_H1[1]

# vel_frame_H1 = vel_frame_H1.drop(axis=1,columns=0)
# vel_frame_H1 = vel_frame_H1.drop(axis=1,columns=1)
# vel_frame_H1 = vel_frame_H1.drop(axis=1,columns=2)

# vel_frame_H2['vel_sys'] = vel_frame_H2[0]
# vel_frame_H2['xPix'] = vel_frame_H2[2]
# vel_frame_H2['yPix'] = vel_frame_H2[1]

# vel_frame_H2 = vel_frame_H2.drop(axis=1,columns=0)
# vel_frame_H2 = vel_frame_H2.drop(axis=1,columns=1)
# vel_frame_H2 = vel_frame_H2.drop(axis=1,columns=2)

# vel_frame_Q3['vel_sys'] = vel_frame_Q3[0]
# vel_frame_Q3['xPx'] = vel_frame_Q3[2]
# vel_frame_Q3['yPix'] = vel_frame_Q3[1]

# vel_frame_Q3 = vel_frame_Q3.drop(axis=1,columns=0)
# vel_frame_Q3 = vel_frame_Q3.drop(axis=1,columns=1)
# vel_frame_Q3 = vel_frame_Q3.drop(axis=1,columns=2)

# North_vel_mean = vel_frame_H1.T[North_trSigPix_H1].mean(axis=1)
# South_vel_mean = vel_frame_H1.T[South_trSigPix_H1].mean(axis=1)

# red = vel_frame_H1['vel_sys'] == vel_frame_H1.groupby('vel_sys').describe().index[7]
# red_pix=red[red == True].index.to_list()

# yellow = vel_frame_H1['vel_sys'] == vel_frame_H1.groupby('vel_sys').describe().index[6]
# yellow_pix=yellow[yellow == True].index.to_list()

# green = vel_frame_H1['vel_sys'] == vel_frame_H1.groupby('vel_sys').describe().index[5]
# green_pix=green[green == True].index.to_list()

# blue = vel_frame_H1['vel_sys'] == vel_frame_H1.groupby('vel_sys').describe().index[4]
# blue_pix=blue[blue == True].index.to_list()

# purple = vel_frame_H1['vel_sys'] == vel_frame_H1.groupby('vel_sys').describe().index[3]
# purple_pix=purple[purple == True].index.to_list()

# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
#                   '#f781bf', '#a65628', '#984ea3',
#                   '#999999', '#e41a1c', '#dede00']

# sns.set_theme(style='whitegrid')
# plt.axes().set_aspect('equal','box')
# PlotMarbels(LinedF_H2_10S1,H1S1[3],c='white',vel0='')
# #PlotMarbels(LinedF_H2_10S1,LinedF_H2_10S1.columns.to_list(),c=CB_color_cycle[6],vel0='')
# PlotMarbels(LinedF_H2_10S1,red_pix,c=CB_color_cycle[7],vel0=vel_frame_H1.groupby('vel_sys').describe().index[7])
# PlotMarbels(LinedF_H2_10S1,yellow_pix,c=CB_color_cycle[8],vel0=vel_frame_H1.groupby('vel_sys').describe().index[6])
# PlotMarbels(LinedF_H2_10S1,green_pix,c=CB_color_cycle[2],vel0=vel_frame_H1.groupby('vel_sys').describe().index[5])
# PlotMarbels(LinedF_H2_10S1,blue_pix,c=CB_color_cycle[0],vel0=vel_frame_H1.groupby('vel_sys').describe().index[4])
# PlotMarbels(LinedF_H2_10S1,purple_pix,c=CB_color_cycle[3],vel0=vel_frame_H1.groupby('vel_sys').describe().index[3])
# #plt.savefig('NewFigs_Oct2020/AllVel_compII.pdf',format='pdf',bbox_inches='tight')
# #plt.savefig('NewFigs_Oct2020/AllVel_compII.eps',format='eps',bbox_inches='tight')
# #plt.savefig('NewFigs_Oct2020/AllVel_compII.ps',format='ps',bbox_inches='tight')
# plt.show()


# a=b=c=d=0
# plt.axes().set_aspect('equal','box')
# plt.plot([3,52],[5,60],linestyle='none')
# for i in range(0,len(Test_B)):
#     if ( (Test_B[Test_B.index[i]] >= 16.48) ):
#         if(a==10):
#             plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=0.01*Test_B[Test_B.index[i]],c=CB_color_cycle[0],alpha=0.5,label='[16.5 : 34.]')
#         elif(a!=10):
#             plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[0],alpha=0.5)

#         a=a+1
#     elif ( (Test_B[Test_B[i]] >= 10.36) and (Test_B[Test_B.index[i]] < 16.48) ):
#         if(c==8):
#             plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[5],alpha=0.5,label='[10.4 : 16.5)')
#         elif(c!=8):
#             plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[5],alpha=0.5)
#         c=c+1
#     elif (Test_B[Test_B.index[i]] < 10.36):
#         if(d==0):
#             plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[2],alpha=0.5,label='[5.5 : 10.4)')
#         elif(d!=0):
#             plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[2],alpha=0.5)
#         d=d+1
#     # elif ( (Test_B[Test_B.index[i]] >= 11.65) and (Test_B[Test_B.index[i]] < 16.48) ):
#     #     if(b==0):
#     #         plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[1],alpha=0.5,label='[ 11.7:16.5 )')
#     #     elif(b!=0):    
#     #         plt.plot(int(Test_B.index[i].split('_')[2]),int(Test_B.index[i].split('_')[1]),linestyle='none',marker='.',markersize=1.0*Test_B[Test_B.index[i]],c=CB_color_cycle[1],alpha=0.5)
#     #     b=b+1

# plt.xticks([12,25,38,51],[1.3,0.65,0.0,-0.65])
# plt.yticks([10,23,36,49],[1.3,0.65,0.0,-0.65])
# plt.title('Molecular Hydrogen: Intensity Ratio\n H$_{2}$ 1-0 S(1) / H$_{2}$ 2-1 S(1)')
# plt.ylabel('DEC offset (arcsec)')
# plt.xlabel('RA offset (arcsec)')

# plt.legend()

# #plt.savefig('NewFigs_Oct2020/Excitation.pdf',format='pdf',bbox_inches='tight')
# #plt.savefig('NewFigs_Oct2020/Excitation.eps',format='eps',bbox_inches='tight')
# #plt.savefig('NewFigs_Oct2020/Excitation.ps',format='ps',bbox_inches='tight')
# plt.show()


# plt.axes().set_aspect('equal','box')
# plt.plot([3,52],[5,60],linestyle='none')
# plt.xticks([12,25,38,51],[1.3,0.65,0.0,-0.65])
# plt.yticks([10,23,36,49],[1.3,0.65,0.0,-0.65])
# plt.title('Molecular Hydrogen: Intensity Ratio\n H$_{2}$ 1-0 S(1) / H$_{2}$ 1-0 Q(3)')
# plt.xlabel('RA offset (arcsec)')
# plt.ylabel('DEC offset (arcsec)')

# a=b=c=d=e=0
# for i in range(0,len(Test_Y)):
#     if (Test_Y[Test_Y.index[i]] < 1.3):
#         if(a==0):
#             plt.plot(int(Test_Y.index[i].split('_')[2]),int(Test_Y.index[i].split('_')[1]),linestyle='none',marker='o',markersize=3.5*Test_Y[Test_Y.index[i]],c=CB_color_cycle[0],alpha=0.5,label='[0.9 : 1.3)')
#         elif(a!=0):
#             plt.plot(int(Test_Y.index[i].split('_')[2]),int(Test_Y.index[i].split('_')[1]),linestyle='none',marker='o',markersize=3.5*Test_Y[Test_Y.index[i]],c=CB_color_cycle[0],alpha=0.5)
#         a=a+1
        
#     elif ( ( Test_Y[Test_Y.index[i]] >= 1.3 ) and ( Test_Y[Test_Y.index[i]] < 1.7) ):
#         if(b==0):
#             plt.plot(int(Test_Y.index[i].split('_')[2]),int(Test_Y.index[i].split('_')[1]),linestyle='none',marker='o',markersize=3.5*Test_Y[Test_Y.index[i]],c=CB_color_cycle[2],alpha=0.4,label='[1.3 : 1.7)')
#         elif(b!=0):
#             plt.plot(int(Test_Y.index[i].split('_')[2]),int(Test_Y.index[i].split('_')[1]),linestyle='none',marker='o',markersize=3.5*Test_Y[Test_Y.index[i]],c=CB_color_cycle[2],alpha=0.4)
#         b=b+1
#     else:
#         if(c==0):
#             plt.plot(int(Test_Y.index[i].split('_')[2]),int(Test_Y.index[i].split('_')[1]),linestyle='none',marker='o',markersize=3.5*Test_Y[Test_Y.index[i]],c=CB_color_cycle[4],alpha=0.5,label='[1.7 : 4.7]')
#         elif(c!=0):
#             plt.plot(int(Test_Y.index[i].split('_')[2]),int(Test_Y.index[i].split('_')[1]),linestyle='none',marker='o',markersize=3.5*Test_Y[Test_Y.index[i]],c=CB_color_cycle[4],alpha=0.5)
#         c=c+1
        
# plt.legend()
# #plt.savefig('NewFigs_Oct2020/Extinction.pdf',format='pdf',bbox_inches='tight')
# #plt.savefig('NewFigs_Oct2020/Extinction.eps',format='eps',bbox_inches='tight')
# #plt.savefig('NewFigs_Oct2020/Extinction.ps',format='ps',bbox_inches='tight')
# plt.show()


# ##Extinction: Equation taken from Davis et al.2011 
# Av=-114.0 * np.log10( 0.704 * Test_Y)
# plt.plot([10,55],[10,55],linestyle='none')
# plt.axes().set_aspect('equal','box')
# for i in range(0,len(Av.keys())):
#     plt.plot(int(Av.index[i].split('_')[2]),int(Av.index[i].split('_')[1]),marker='.',linestyle='none',color='gray',markersize=-Av[i]/3.,alpha=0.3)
#     if(Av[i] < -9.7):
#         plt.plot(int(Av.index[i].split('_')[2]),int(Av.index[i].split('_')[1]),marker='.',linestyle='none',color=CB_color_cycle[7])
#     elif(Av[i] >= -9.7 and Av[i] < -2.8):
#         plt.plot(int(Av.index[i].split('_')[2]),int(Av.index[i].split('_')[1]),marker='.',linestyle='none',color=CB_color_cycle[5])
#     elif(Av[i] >= -2.8 and Av[i] < -1.0):
#         plt.plot(int(Av.index[i].split('_')[2]),int(Av.index[i].split('_')[1]),marker='.',linestyle='none',color=CB_color_cycle[4])
#     elif(Av[i] >= -1.0 and Av[i] < 1.0):
#         plt.plot(int(Av.index[i].split('_')[2]),int(Av.index[i].split('_')[1]),marker='.',linestyle='none',color=CB_color_cycle[6])
#     else:
#         plt.plot(int(Av.index[i].split('_')[2]),int(Av.index[i].split('_')[1]),marker='.',linestyle='none',color=CB_color_cycle[8])

# plt.show()

# Eupper ={'Eup_10_Q1': 6149,
#          'Eup_10_Q3': 6956,
#          'Eup_10_S0': 6471,
#          'Eup_10_S1': 6956,
#          'Eup_10_S2': 7584,
#          'Eup_10_S3': 8365,
#          'Eup_21_S1': 12550}



# Eupper=pd.Series(Eupper)

# #IntegratedFrame_Nor=pd.DataFrame([Snu_H1Q3[LaListaDePeakWavelenghts_North],Snu_H1Q3[LaListaDePeakWavelenghts_North],Snu_H1S0[LaListaDePeakWavelenghts_North],Snu_H1S1[LaListaDePeakWavelenghts_North],Snu_H1S2[LaListaDePeakWavelenghts_North],Snu_H1S3[LaListaDePeakWavelenghts_North],Snu_H2S1[LaListaDePeakWavelenghts_North]],index=Eupper)

# IntegratedFrame=pd.DataFrame([Snu_H1Q1[H2S1[2]],Snu_H1Q3[H2S1[2]],Snu_H1S0[H2S1[2]],Snu_H1S1[H2S1[2]],Snu_H1S2[H2S1[2]],Snu_H1S3[H2S1[2]],Snu_H2S1[H2S1[2]]],index=Eupper)
# #plt.axes().set_aspect('equal','box')

# plt.xlim(5000,14000)
# for pix in IntegratedFrame.columns:
#     sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=pix)
# #    sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=pix
# #    sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=IntegratedFrame.columns[41])
# #    sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=IntegratedFrame.columns[42])
# plt.show()

# plt.xlim(5000,14000)
# sns.regplot(data=IntegratedFrame.T.describe(),x=IntegratedFrame.T.describe().columns,y= IntegratedFrame.T.describe().loc['mean'])
# plt.show()


# y = IntegratedFrame[IntegratedFrame.columns[1]].values

# x = IntegratedFrame.index.values.reshape((-1,1))

# fmodel=LinearRegression().fit(x,y)

# print(fmodel.intercept_,fmodel.coef_)


# #TtTtTtTtTtTtTtTtTtTtTtTtTt
# ##Integrals for population diagrams
# Diag_SnuQ1 = IntegS(LinedF_H2_10S1,H2S1[1],'Q1')
# Diag_SnuQ3 = IntegS(LinedF_H2_10Q3,H2S1[1],'Q3')
# Diag_SnuS0 = IntegS(LinedF_H2_10S0,H2S1[1],'S0')
# Diag_SnuS1 = IntegS(LinedF_H2_10S1,H2S1[1],'H1')
# Diag_SnuS2 = IntegS(LinedF_H2_10S2,H2S1[1],'S2')
# Diag_SnuS3 = IntegS(LinedF_H2_10S3,H2S1[1],'S3')

# Diag_SnuH2 = Snu_H2S1

# IntegratedFrame2=pd.DataFrame([Diag_SnuQ1,Diag_SnuQ3,Diag_SnuS0,Diag_SnuS1,Diag_SnuS2,Diag_SnuS3,Diag_SnuH2],index=Eupper)

# plt.xlim(5000,14000)
# for pix in IntegratedFrame2.columns:
#     sns.regplot(data=IntegratedFrame2,x=IntegratedFrame2.index,y=pix)
# #    sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=pix
# #    sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=IntegratedFrame.columns[41])
# #    sns.regplot(data=IntegratedFrame,x=IntegratedFrame.index,y=IntegratedFrame.columns[42])
# plt.show()

# plt.xlim(5000,14000)
# sns.regplot(data=IntegratedFrame2.T.describe(),x=IntegratedFrame2.T.describe().columns,y= IntegratedFrame2.T.describe().loc['mean'])
# plt.show()


# y = IntegratedFrame2[IntegratedFrame2.columns[1]].values

# x = IntegratedFrame2.index.values.reshape((-1,1))

# fmodel=LinearRegression().fit(x,y)


# # plt.axes().set_aspect('equal','box')




# # plt.plot([15,50],[5,60],linestyle='none')
# # for i in range(0,len(Snu_H1S1)): 
# #     plt.plot(int(Snu_H1S1.index[i].split('_')[2]),int(Snu_H1S1.index[i].split('_')[1]),linestyle='none',marker='.',markersize=9.0e4*Snu_H1S1[Snu_H1S1.index[i]],c='red',alpha=0.5)
# # plt.show()

# # plt.axes().set_aspect('equal','box')
# # plt.plot([15,50],[5,60],linestyle='none')
# # for i in range(0,len(Snu_H1S0)): 
# #     plt.plot(int(Snu_H1S0.index[i].split('_')[2]),int(Snu_H1S0.index[i].split('_')[1]),linestyle='none',marker='.',markersize=9.0e4*Snu_H1S0[Snu_H1S0.index[i]],c='blue',alpha=0.5)
# # plt.show()

    
# # plt.axes().set_aspect('equal','box')
# # plt.plot([15,50],[5,60],linestyle='none')
# # for i in range(0,len(Snu_H1S2)):
# #     plt.plot(int(Snu_H1S2.index[i].split('_')[2]),int(Snu_H1S2.index[i].split('_')[1]),linestyle='none',marker='.',markersize=9.0e4*Snu_H1S2[Snu_H1S2.index[i]],c='blue',alpha=0.5)
# # plt.show()


    
# # plt.axes().set_aspect('equal','box')
# # plt.plot([15,50],[5,60],linestyle='none')
# # for i in range(0,len(Snu_H1S3)):
# #     plt.plot(int(Snu_H1S3.index[i].split('_')[2]),int(Snu_H1S3.index[i].split('_')[1]),linestyle='none',marker='.',markersize=9.0e4*Snu_H1S3[Snu_H1S3.index[i]],c='blue',alpha=0.5)

# # plt.show()

# # plt.boxplot([Snu_H1S0[H1S0[1]],Snu_H1S1[H1S1[1]],Snu_H1Q3[H1Q3[1]],Snu_H1S2[H1S2[1]],Snu_H1S3[H1S3[1]]],notch=True,showmeans=True)

# # plt.boxplot([Snu_H1S0[H1S0[3]],Snu_H1S1[H1S1[3]],Snu_H1Q3[H1Q3[3]],Snu_H1S2[H1S2[3]],Snu_H1S3[H1S3[3]]])

# # plt.plot([Eupper['Eup_10_S0'],Eupper['Eup_10_S1'],Eupper['Eup_10_S2'],Eupper['Eup_10_S3'],Eupper['Eup_10_Q3'],Eupper['Eup_21_S1']],[Snu_H1S0[H1S0[1]].describe().loc['mean'].mean(),Snu_H1S1[H1S1[1]].describe().loc['mean'].mean(),Snu_H1S2[H1S2[1]].describe().loc['mean'].mean(),Snu_H1S3[H1S3[1]].describe().loc['mean'].mean(),Snu_H1Q3[H1Q3[1]].describe().loc['mean'].mean(),Snu_H2S1[H2S1[1]].describe().loc['mean'].mean()],marker='o',linestyle='none')

# # plt.plot([Eupper['Eup_10_S0'],Eupper['Eup_10_S1'],Eupper['Eup_10_S2'],Eupper['Eup_10_S3'],Eupper['Eup_10_Q3'],Eupper['Eup_21_S1']],[Snu_H1S0[Snu_H2S1.index[0]],Snu_H1S1[Snu_H2S1.index[0]],Snu_H1S2[Snu_H2S1.index[0]],Snu_H1S3[Snu_H2S1.index[0]],Snu_H1Q3[Snu_H2S1.index[0]],Snu_H2S1[Snu_H2S1.index[0]]],marker='o',linestyle='none')

# # plt.plot([Eupper['Eup_10_S0'],Eupper['Eup_10_S1'],Eupper['Eup_10_S2'],Eupper['Eup_10_S3'],Eupper['Eup_10_Q3'],Eupper['Eup_21_S1']],[Snu_H1S0[Snu_H2S1.index[1]],Snu_H1S1[Snu_H2S1.index[2]],Snu_H1S2[Snu_H2S1.index[3]],Snu_H1S3[Snu_H2S1.index[3]],Snu_H1Q3[Snu_H2S1.index[3]],Snu_H2S1[Snu_H2S1.index[3]]],marker='o',linestyle='none')

# # plt.plot([Eupper['Eup_10_S0'],Eupper['Eup_10_S1'],Eupper['Eup_10_S2'],Eupper['Eup_10_S3'],Eupper['Eup_10_Q3'],Eupper['Eup_21_S1']],[Snu_H1S0[Snu_H2S1.index[7]],Snu_H1S1[Snu_H2S1.index[7]],Snu_H1S2[Snu_H2S1.index[7]],Snu_H1S3[Snu_H2S1.index[7]],Snu_H1Q3[Snu_H2S1.index[7]],Snu_H2S1[Snu_H2S1.index[7]]],marker='o',linestyle='none')

# # plt.show()










# # plt.axes().set_aspect('equal','box')
# # plt.plot([15,50],[5,60],linestyle='none')
# # for i in range(0,len(Test_M)):
# #     if (Test_M[Test_M.index[i]] < 10.0):
# #         plt.plot(int(Test_M.index[i].split('_')[2]),int(Test_M.index[i].split('_')[1]),linestyle='none',marker='.',markersize=2.0*Test_M[Test_M.index[i]],c='gray',alpha=0.5)
# #     elif (Test_M[Test_M.index[i]] >= 17.0):
# #         plt.plot(int(Test_M.index[i].split('_')[2]),int(Test_M.index[i].split('_')[1]),linestyle='none',marker='.',markersize=2.0*Test_M[Test_M.index[i]],c='blue',alpha=0.4)
# #     else:
# #         plt.plot(int(Test_M.index[i].split('_')[2]),int(Test_M.index[i].split('_')[1]),linestyle='none',marker='.',markersize=2.0*Test_M[Test_M.index[i]],c='orange',alpha=0.5)
        
        
# #PlotMarbels(LinedF_H2_10S1,purple_pix,c=CB_color_cycle[3],vel0=vel_frame_H1.groupby('vel_sys').describe().index[3])
# #plt.savefig('NewFigs_Aug2020/Vel_compI.pdf',format='pdf')

# #PlotMarbels(LinedF_H2_10S1,blue_pix,c=CB_color_cycle[0],vel0=vel_frame_H1.groupby('vel_sys').describe().index[4])
# #plt.savefig('NewFigs_Aug2020/Vel_compII.pdf',format='pdf')

# #PlotMarbels(LinedF_H2_10S1,green_pix,c=CB_color_cycle[2],vel0=vel_frame_H1.groupby('vel_sys').describe().index[5])
# #PlotMarbels(LinedF_H2_10S1,yellow_pix,c=CB_color_cycle[8],vel0=vel_frame_H1.groupby('vel_sys').describe().index[6])
# # PlotMarbels(LinedF_H2_10S1,red_pix,c=CB_color_cycle[7],vel0=vel_frame_H1.groupby('vel_sys').describe().index[7])
# # plt.savefig('NewFigs_Aug2020/Vel_compV.pdf',format='pdf')

# # PlotMarbels(LinedF_H2_10S1,LinedF_H2_10S1.columns.to_list(),c=CB_color_cycle[6])
# # PlotMarbels(LinedF_H2_10S1,red_pix,c=CB_color_cycle[7])
# # PlotMarbels(LinedF_H2_10S1,blue_pix,c=CB_color_cycle[5])
# # PlotMarbels(LinedF_H2_10S1,yellow_pix,c=CB_color_cycle[1])
# # PlotMarbels(LinedF_H2_10S1,green_pix,c=CB_color_cycle[0])
# # PlotMarbels(LinedF_H2_10S1,purple_pix,c=CB_color_cycle[4])
# # plt.savefig('NewFigs_Aug2020/TestVelFiguresII.pdf',format='pdf')
# # plt.savefig('NewFigs_Aug2020/TestVelFiguresII.eps',format='eps')
# # plt.show()


# # PlotMarbels(LinedF_H2_10S1,LinedF_H2_10S1.columns.to_list(),c=CB_color_cycle[6])
# # PlotMarbels(LinedF_H2_10S1,red_pix,c='red')
# # PlotMarbels(LinedF_H2_10S1,blue_pix,c='blue')
# # PlotMarbels(LinedF_H2_10S1,yellow_pix,c='yellow')
# # PlotMarbels(LinedF_H2_10S1,green_pix,c='green')
# # PlotMarbels(LinedF_H2_10S1,purple_pix,c='purple')
# # plt.savefig('NewFigs_Aug2020/TestVelFiguresIII.pdf',format='pdf')
# # plt.savefig('NewFigs_Aug2020/TestVelFiguresIII.eps',format='eps')
# # plt.show()




# # trSigPix_H2 = H2S1[1]
# # North_trSigPix_H2 = []
# # South_trSigPix_H2 = []
# # for stringui in trSigPix_H2:
# #     if(int(stringui.split('_')[1]) > 30):
# #         North_trSigPix_H2.append(stringui)
# #     else:
# #         South_trSigPix_H2.append(stringui)


# # Side_sectorsMean=np.mean([LinedF_H2_10S1.iloc[60:82].describe().loc['std'].mean(),LinedF_H2_10S1.iloc[99:121].describe().loc['std'].mean()])

# # H1S1_series = LinedF_H2_10S1.iloc[82:99].describe().loc['mean'] > (Side_sectorsMean * 3.0)
# # H1S1_list=H1S1_series[H1S1_series == True].index.tolist()
# # h1s1_residual_list=H1S1_series[H1S1_series == False].index

# # H1S1_HF_series = LinedF_H2_10S1.iloc[82:99].describe().loc['mean'] > (Side_sectorsMean * 5.0) #emissionAbove5sigma
# # H1S1_HF_list=H1S1_HF_series[H1S1_HF_series == True].index.tolist()
# # h1s1_HF_residual_list=H1S1_series[H1S1_HF_series == False].index

# # H1S1_HHF_series = LinedF_H2_10S1.iloc[82:99].describe().loc['mean'] > (Side_sectorsMean * 10.0) #emissionAbove10sigma
# # H1S1_HHF_list= H1S1_HHF_series[H1S1_HHF_series == True].index.tolist()
# # h1s1_HHF_residual_list=H1S1_series[H1S1_HHF_series == False].index

# #MaxVals_naam_H2_10S1=PixStats_uPQu(LinedF_H2_10S1,'Line',0.5)
# #Snu_H2_10S1=LinedF_H2_10S1[MaxVals_naam_H2_10S1[:]].iloc[84:99].sum()*delta_lambda



# #stats_H2_10S1_Asector,stats_H2_10S1_LineSector,stats_H2_10S1_Bsector=TransitionStats(LinedF_H2_10S1)

# #stats_H2_21S1_Asector,stats_H2_21S1_LineSector,stats_H2_21S1_Bsector=TransitionStats(LinedF_H2_21S1)

# #stats_H2_10Q3_Asector,stats_H2_10Q3_LineSector,stats_H2_10Q3_Bsector=TransitionStats(LinedF_H2_10Q3)



# #MaxVals_naam_H2_10Q3=PixStats_uPQu(LinedF_H2_10Q3,'Line',0.05)

# #MaxVals_naam_H2_21S1=PixStats_uPQu(LinedF_H2_21S1,'Line',0.05)

# # plt.step(Clean_lambdas[range_H2_21s1_Ti[0]:range_H2_21s1_Ti[1]],stats_H2_21S1_Asector)
# # plt.step(Clean_lambdas[range_H2_21s1_TLine[0]:range_H2_21s1_TLine[1]],stats_H2_21S1_LineSector)
# # plt.step(Clean_lambdas[range_H2_21s1_Te[0]:range_H2_21s1_Te[1]],stats_H2_21S1_Bsector)
# # plt.xlabel('Wavelenght (um)')
# # plt.ylabel('Counts')
# # plt.title('Peak flux counts per wavelenght bin. H2 2-1 S(1)')
# # plt.legend()
# # plt.savefig('2to1_CountsPerWavelength.pdf')
# # plt.show()

# # plt.step(Clean_lambdas[range_H2_10s1_Ti[0]:range_H2_10s1_Ti[1]],stats_H2_10S1_Asector)
# # plt.step(Clean_lambdas[range_H2_10s1_TLine[0]:range_H2_10s1_TLine[1]],stats_H2_10S1_LineSector)
# # plt.step(Clean_lambdas[range_H2_10s1_Te[0]:range_H2_10s1_Te[1]],stats_H2_10S1_Bsector)
# # plt.title('Peak flux counts per wavelenght bin: H2 10 S(1)')
# # plt.xlabel('Wavelenght (um)')
# # plt.ylabel('Counts')
# # plt.legend()
# # plt.savefig('1to0S1_CountsPerWavelength.pdf')
# # plt.show()

# # plt.step(Clean_lambdas[range_H2_10q3_Ti[0]:range_H2_10q3_Ti[1]],stats_H2_10Q3_Asector)
# # plt.step(Clean_lambdas[range_H2_10q3_TLine[0]:range_H2_10q3_TLine[1]],stats_H2_10Q3_LineSector)
# # plt.step(Clean_lambdas[range_H2_10q3_Te[0]:range_H2_10q3_Te[1]],stats_H2_10Q3_Bsector)
# # plt.title('Peak flux counts per wavelenght bin: H2 10 Q(3)')
# # plt.xlabel('Wavelenght (um)')
# # plt.ylabel('Counts')
# # plt.legend()
# # plt.savefig('1to0Q3_CountsPerWavelength.pdf')
# # plt.show()


# # Snu_H2_21S1=LinedF_H2_21S1[MaxVals_naam_H2_21S1[:]].iloc[84:93].sum()*delta_lambda 

# # Snu_H2_10Q3=LinedF_H2_10Q3[MaxVals_naam_H2_10Q3[:]].iloc[83:92].sum()*delta_lambda

# # # _ = plt.plot([0,60],[0,60],linestyle='none')
# # # for i in range(0,len(LinedF_H2_21S1[MaxVals_naam_H2_21S1].columns)):
# # #     _ = plt.plot(int(LinedF_H2_21S1[MaxVals_naam_H2_21S1].columns[i].split('_')[2]),int(LinedF_H2_21S1[MaxVals_naam_H2_21S1].columns[i].split('_')[1]),marker='.',linestyle='none',color='yellow',alpha=0.2)
# # #     _ = plt.plot(int(LinedF_H2_10S1[MaxVals_naam_H2_10S1].columns[i].split('_')[2]),int(LinedF_H2_10S1[MaxVals_naam_H2_10S1].columns[i].split('_')[1]),marker='.',linestyle='none',color='blue',alpha=0.2) 
# # #     _ = plt.plot(int(LinedF_H2_10Q3[MaxVals_naam_H2_10Q3].columns[i].split('_')[2]),int(LinedF_H2_10Q3[MaxVals_naam_H2_10Q3].columns[i].split('_')[1]),marker='.',linestyle='none',color='blue',alpha=0.2)


# # # plt.show()

# # #maxPix=PixStats_max(LinedF_H2_10S1,sector='Line')
# # #pixECDF(LinedF_H2_10S1,maxPix)

# # # jj=[]
# # # ii=[]
# # # for i in Snu_H2_10Q3.index:
# # #     jj.append(int(Snu_H2_10Q3.index[1].split('_')[1]))
# # #     ii.append(int(Snu_H2_10Q3.index[1].split('_')[2]))

# # # fig,axs = plt.subplots(60,60)
# # # fig.suptitle('Grid')
# # # axs[j,i]=Snu_H2_10S1[]


# # # x=np.sort(LinedF_H2_21S1[MaxVals_naam_H2_21S1[0]])
# # # y=np.arange(1,len(x)+1)/len(x)
# # # _ = plt.plot(x,y,marker='.',linestyle='none')
# # # plt.margins(0.02)
# # # plt.show()

# # ###Lambda_peak distribution. WiP
# # # H10=velDistribution(LinedF_H2_10S1,MaxVals_naam_H2_10S1)
# # # H21=velDistribution(LinedF_H2_21S1,MaxVals_naam_H2_21S1)
# # # Q10=velDistribution(LinedF_H2_10Q3,MaxVals_naam_H2_10Q3)
# # # H2_10_aTrest=2.1218
# # # H2_21_aTrest=2.2477
# # # Q3_10_aTrest=2.4237



# # #equis_ticks=LinedF_H2_10Q3[MaxVals_naam_H2_10Q3[0:]].iloc[60:121].index.values.round(4)

# # #multiPixPlot(LinedF_H2_10S1,MaxVals_naam_H2_10S1)



# # #fig,ax = plt.subplots()
# # #ax.boxplot(LinedF_H2_10Q3[MaxVals_naam_H2_10Q3[0:]].iloc[60:121])
# # #ax.xaxis.set_major_formatter(plt.NullFormatter()) 
# # #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
# # #ax.set_xticklabels(equis_ticks[0:])
        
# # range_H2_10s1_Ti=[LineWaveRange(Clean_lambdas,2.12134)-90,LineWaveRange(Clean_lambdas,2.12134)-30]
# # range_H2_10s1_TLine=[LineWaveRange(Clean_lambdas,2.12134)-30,LineWaveRange(Clean_lambdas,2.12134)+30]
# # range_H2_10s1_Te=[LineWaveRange(Clean_lambdas,2.12134)+30,LineWaveRange(Clean_lambdas,2.12134)+90]

# # range_H2_21s1_Ti=[LineWaveRange(Clean_lambdas,2.2477)-90,LineWaveRange(Clean_lambdas,2.2477)-30]
# # range_H2_21s1_TLine=[LineWaveRange(Clean_lambdas,2.2477)-30,LineWaveRange(Clean_lambdas,2.2477)+30]
# # range_H2_21s1_Te=[LineWaveRange(Clean_lambdas,2.2477)+30,LineWaveRange(Clean_lambdas,2.2477)+90]

# # range_H2_10q3_Ti=[LineWaveRange(Clean_lambdas,2.4237)-90,LineWaveRange(Clean_lambdas,2.4237)-30]
# # range_H2_10q3_TLine=[LineWaveRange(Clean_lambdas,2.4237)-30,LineWaveRange(Clean_lambdas,2.4237)+30]
# # range_H2_10q3_Te=[LineWaveRange(Clean_lambdas,2.4237)+30,LineWaveRange(Clean_lambdas,2.4237)+90]



# ##FWHM Channels:
# ##H2 2-1 S1:[84:90]




# #def multiPixPlot(df,pixels):

# #     ar_vertical=[2.1218,2.2477,2.4237]
# #     equis_ticks=df[pixels[0:]].iloc[60:121].index.values.round(4)

# # ##1-0S1
# #     plt.boxplot(df[pixels[0:]].iloc[60:121],autorange=True)#,notch=True,meanline=True,showmeans=True)
# #     plt.show()
#     #   plt.vlines(x=33,ymin=-0.009,ymax=-0.005,linestyle='dashdot',color='blue')
#  #   plt.title('H_2 1-0 S(1)')
    
#  #   plt.xlabel('Wavelength (um)')
#  #   plt.ylabel('Flux (Jy)')
#  #   plt.xticks(np.arange(0,61,step=5),equis_ticks[0::5],rotation=0)
#  #   plt.tick_params(labelsize='x-small')

# #    plt.savefig('NewFigs_Aug2020/TestFigures.pdf',format='pdf')
# #    plt.savefig('NewFigs_Aug2020/TestFigures.eps',format='eps')

# ##1-0S1

# #    fig,ax=plt.subplots(2,2)
    
# ##1-0Q3
# #    plt.boxplot(df[pixels[0:]].iloc[60:121],notch=True,meanline=True,showmeans=True)
# #    plt.vlines(x=30,ymin=-0.025,ymax=-0.005,linestyle='dashdot',color='blue')
# #    plt.title('H_2 1-0 S(1)')

# ##1-0Q3

# ##2-1S1
# #    plt.boxplot(df[pixels[0:]].iloc[60:121],notch=True,meanline=True,showmeans=True)
# #    plt.vlines(x=30,ymin=-0.004,ymax=-0.002,linestyle='dashdot',color='blue')
# #    plt.title('H_2 2-1 S(1)')

# ##2-1S1
# #    return print('done?')
