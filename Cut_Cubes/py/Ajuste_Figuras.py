import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import GaussianModel
from scipy.integrate import quad
import cube_to_df as ctdf

def plotpix(Molines_A_df,px,channel):

    fig = plt.figure(figsize=(15,5))
    plt.title(px,fontsize=16)
    plt.xlabel('Radio Velocity [km/s]',fontsize=13)
    plt.ylabel('$[Jy/beam]$',fontsize=13)
    plt.step(Molines_A_df[px].index[channel[0]:channel[1]],Molines_A_df[px].iloc[channel[0]:channel[1]],color='k')

    return fig
    
def clas(Molines_A_df,cube,px,channel,ruido,plot=False):

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

    # Modelo de gaussiana
    gmodel = Model(gaussian)
    result_l = gmodel.fit(y_datos_l, x=x_datos_l, amp=max(y_datos_l), cen=y_datos_l[y_datos_l==max(y_datos_l)].index[0], wid=(secc1.index[0]-med)/2)
    result_r = gmodel.fit(y_datos_r, x=x_datos_r, amp=max(y_datos_r), cen=y_datos_r[y_datos_r==max(y_datos_r)].index[0], wid=(secc1.index[0]-med)/2)
    
    '''
    resultado1, error1 = quad(lambda x_datos_l: gaussian(x_datos_l, result_l.summary()['params'][0][1],result_l.summary()['params'][1][1],result_l.summary()['params'][2][1]), min(x_datos_l), max(x_datos_l))
    resultado2, error2 = quad(lambda x_datos_r: gaussian(x_datos_r, result_r.summary()['params'][0][1],result_r.summary()['params'][1][1],result_r.summary()['params'][2][1]), min(x_datos_r), max(x_datos_r))
    
    s = resultado1+resultado2
    '''
    #s = np.sum(cube[channel[0]:channel[-1],int(px.split('_')[1]),int(px.split('_')[2])],axis=0)
    s=1
    if s<ruido:
        a = 'Ruido'
    elif result_r.summary()['params'][1][1]-result_l.summary()['params'][1][1] <45:
        a = 'Un pico'
    else:
        a = 'Dos picos'
    
    if plot == True:
        #plotpix(px)        
        plotpix(Molines_A_df,px,channel)
        
        plt.suptitle(a)
        #plt.plot(x_datos_r, result_r.init_fit, '--', label='initial f')  
        #plt.plot(x_datos,y_datos)
        plt.plot(x_datos_l, result_l.best_fit, '-', color='orangered')
        plt.plot(x_datos_r, result_r.best_fit, '-', color='orangered')
        plt.vlines(secc1.index[0],y_datos.min(),y_datos.max(),color='k',linestyle='--')
        plt.vlines(secc1.index[-1],y_datos.min(),y_datos.max(),color='k',linestyle='--')
        plt.vlines(med,y_datos.min(),y_datos.max(),color='lightcoral',linestyle='--')
    
    return a,result_l,result_r

def gauss_model(Molines_A_df,cube,px,channel,ruido,plot=False):
    
    pars=0
    a = clas(Molines_A_df,cube,px,[channel[0],channel[1]],ruido)
    
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
        out=model.fit(y,pars,x=x) 
        comps = out.eval_components(x=x) 

        if plot == True:
            # Data
            fig = plotpix(Molines_A_df,px,[channel[0],channel[1]])
            # Componentes 
            for i in range(npeaks):
                if i == 0:
                    c='blue'
                else:
                    c = 'red'
                plt.plot(x, comps['peak%d_' % (i+1)],'--',color=c)
            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tick_params(axis='both', direction='in', length=5, width=1.5)
            
            plt.minorticks_on()
            plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)
            
            # Ajuste 
            plt.plot(x, out.best_fit, '-', color='purple')
            plt.text(min(x)+min(x)/40,max(y)-max(y)/11,r'$\mu =$ '+str(round(pars['peak1_center'].value,2))+' km/s',color='blue',fontsize=13)
            plt.text(min(x)+min(x)/40,max(y)-2*max(y)/11,r'$\sigma =$ '+str(round(pars['peak1_sigma'].value,2))+' km/s',color='blue',fontsize=13)
            
            plt.text(min(x)+min(x)/40,max(y)-3*max(y)/11,r'$\mu =$ '+str(round(pars['peak2_center'].value,2))+' km/s',color='red',fontsize=13)
            plt.text(min(x)+min(x)/40,max(y)-4*max(y)/11,r'$\sigma =$ '+str(round(pars['peak2_sigma'].value,2))+' km/s',color='red',fontsize=13)
    
    else:
        if plot == True:
            # Data
            fig = plotpix(Molines_A_df,px,[channel[0],channel[1]])
            
    return pars,comps,out,fig

def gauss_model_outflows(Molines_A_df,cube,px,channel,plot=False):
    
    x_datos = Molines_A_df[px].index[channel[0]:channel[1]]
    y_datos = Molines_A_df[px].iloc[channel[0]:channel[1]]
    
    # Secciones para wid
    secc1 = Molines_A_df[px].iloc[channel[0]+30:channel[1]-30]

    # Mitad de los canales
    med = (y_datos.index.max()-y_datos.index.min())/2 + y_datos.index.min()
    # Modelo de gaussiana
    gmodel = Model(gaussian)
    result = gmodel.fit(y_datos, x=x_datos, amp=max(y_datos), cen=y_datos[y_datos==max(y_datos)].index[0], wid=(secc1.index[0]-med)/2)
    pars= result.params
    
    if plot == True:
        # Data
        fig = plotpix(Molines_A_df,px,[channel[0],channel[1]])
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.tick_params(axis='both', direction='in', length=5, width=1.5)
        
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)
        
        # Ajuste 
        plt.plot(x_datos, result.best_fit, '-', color='purple')
        plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-max(y_datos)/11,r'$\mu =$ '+str(round(pars['cen'].value,2))+' km/s',color='green',fontsize=13)
        plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-2*max(y_datos)/11,r'$\sigma =$ '+str(round(pars['wid'].value,2))+' km/s',color='black',fontsize=13)
        
        vsys = round(pars['cen'].value,2)
        plt.vlines(vsys, -0.0005, max(result.best_fit),color='green',label='Vsys = '+str(vsys))
        
    return pars,result,fig
      
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

'''
path = '/Users/mac/Tesis/IRAS15445_recortados/I15445.mstransform_cube_contsub_13CO.fits'
box = [220, 225, 300, 285]

cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
channel = [90,210]

for i in range(0,80):
    px= 'Pix_30_'+str(i)

    gauss_model(Molines_A_df, cube, px, channel, 0.1,plot=True)
    #clas(Molines_A_df,cube,px,channel,0.1,plot=True)
'''



