import cube_to_df as ctdf
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import GaussianModel
from scipy.integrate import quad

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


path = '/home/marcela/Tesis Marcela/IRAS15445_recortados/I15445.mstransform_cube_contsub_13CO.fits'
box = [220,225,300,285]

cube, Molines_A_df, coord = ctdf.Cube_to_df(path, box)
channel = [90,225]

for k in range(0,80):
    px = 'Pix_30_27'#+str(k)
    
    x_datos = Molines_A_df[px].index[channel[0]:channel[1]]
    y_datos = Molines_A_df[px].iloc[channel[0]:channel[1]]
    
    # centro del canal
    cen = np.linspace(0.8,1.2,10)* -90
    cench= (y_datos.index.max()-y_datos.index.min())/2 + y_datos.index.min()
    
    cen = np.append(cen,cench)
    
    chisqrl = []
    chisqrr = []
    resultsl = []
    resultsr = []
    
    for j in cen:
        med=j
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
        
        chisqrl.append(result_l.summary()['chisqr'])
        chisqrr.append(result_r.summary()['chisqr'])
        resultsl.append(result_l)
        resultsr.append(result_r)
        
        '''
        fig = plt.figure(figsize=(15,5))
        plt.title(px,fontsize=16)
        plt.xlabel('Radio Velocity [km/s]',fontsize=13)
        plt.ylabel('$[Jy/beam]$',fontsize=13)
        plt.step(Molines_A_df[px].index[channel[0]:channel[1]],Molines_A_df[px].iloc[channel[0]:channel[1]],color='k')
    
        plt.plot(x_datos_l, result_l.best_fit, '-', color='orangered',label = result_l.summary()['chisqr'])
        plt.plot(x_datos_r, result_r.best_fit, '-', color='orangered',label = result_r.summary()['chisqr'])
        plt.vlines(secc1.index[0],y_datos.min(),y_datos.max(),color='k',linestyle='--')
        plt.vlines(secc1.index[-1],y_datos.min(),y_datos.max(),color='k',linestyle='--')
        plt.vlines(med,y_datos.min(),y_datos.max(),color='lightcoral',linestyle='--')
        
        plt.legend()    
        '''
        
    # minimo valor de chisqr a la derecha e izquierda
    modelr = resultsr[chisqrr.index(min(chisqrr))]
    modell = resultsl[chisqrl.index(min(chisqrl))]
    
    npeaks=2
    model=GaussianModel(prefix='peak1_')
    
    # Armar el modelo, en este caso tengo dos gaussianas
    for i in range(1,npeaks):
      model=model+GaussianModel(prefix='peak%d_' % (i+1))
    pars=model.make_params()
    
    # Llenar los parámetros iniciales
    for i,ff in zip(range(npeaks),[modell.summary()['params'],modelr.summary()['params'],]):
      pars['peak%d_center' % (i+1)].set(value=ff[1][1],vary=False) # fix nu_ul
      pars['peak%d_sigma' % (i+1)].set(value=ff[2][1],min=10,max=50)
      pars['peak%d_amplitude' % (i+1)].set(value=ff[0][1],min=0,max=4)
        
    # Hacer el fit al modelo inicial hasta que se ajusten las gaussianas
    out=model.fit(y_datos,pars,x=x_datos) # run fitting algorithm
    comps = out.eval_components(x=x_datos) # fit results for each line
    
    chisqrnew = out.summary()['chisqr']
    
    #--------------------------Version anterior de los ajustes-------------------------------------------------------------------------
    
    modelr_old = resultsr[-1]
    modell_old = resultsl[-1]
    
    npeaks=2
    model=GaussianModel(prefix='peak1_')
    
    # Armar el modelo, en este caso tengo dos gaussianas
    for i in range(1,npeaks):
      model=model+GaussianModel(prefix='peak%d_' % (i+1))
    pars_old=model.make_params()
    
    # Llenar los parámetros iniciales
    for i,ff in zip(range(npeaks),[modell_old.summary()['params'],modelr_old.summary()['params'],]):
      pars_old['peak%d_center' % (i+1)].set(value=ff[1][1],vary=False) # fix nu_ul
      pars_old['peak%d_sigma' % (i+1)].set(value=ff[2][1],min=10,max=50)
      pars_old['peak%d_amplitude' % (i+1)].set(value=ff[0][1],min=0,max=4)
        
    # Hacer el fit al modelo inicial hasta que se ajusten las gaussianas
    out_old=model.fit(y_datos,pars_old,x=x_datos) # run fitting algorithm
    comps_old = out_old.eval_components(x=x_datos) # fit results for each line
    
    chisqrold = out_old.summary()['chisqr']
    #---------------------------------------------------------------------------------------------------
    '''
    if chisqrold < chisqrnew:
        pars = pars_old
        out = out_old
        comps = comps_old
    '''
    # Grafica Resultados!!
    fig = plt.figure(figsize=(15,10))
    
    #old
    plt.subplot(2,1,1)
    plt.title(px+' - OLD',fontsize=16)
    plt.xlabel('Radio Velocity [km/s]',fontsize=13)
    plt.ylabel('$[Jy/beam]$',fontsize=13)
    plt.step(Molines_A_df[px].index[channel[0]:channel[1]],Molines_A_df[px].iloc[channel[0]:channel[1]],color='k')
    
    # Componentes 
    for i in range(npeaks):
        if i == 0:
            c='blue'
        else:
            c = 'red'
        plt.plot(x_datos, comps_old['peak%d_' % (i+1)],'--',color=c)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(axis='both', direction='in', length=5, width=1.5)
    
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)
    
    # Ajuste 
    plt.plot(x_datos, out_old.best_fit, '-', color='purple')
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-max(y_datos)/11,r'$\mu =$ '+str(round(pars_old['peak1_center'].value,2))+' km/s',color='blue',fontsize=13)
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-2*max(y_datos)/11,r'$\sigma =$ '+str(round(pars_old['peak1_sigma'].value,2))+' km/s',color='blue',fontsize=13)
    
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-3*max(y_datos)/11,r'$\mu =$ '+str(round(pars_old['peak2_center'].value,2))+' km/s',color='red',fontsize=13)
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-4*max(y_datos)/11,r'$\sigma =$ '+str(round(pars_old['peak2_sigma'].value,2))+' km/s',color='red',fontsize=13)

    #new
    plt.subplot(2,1,2)

    plt.title(px+' - NEW',fontsize=16)
    plt.xlabel('Radio Velocity [km/s]',fontsize=13)
    plt.ylabel('$[Jy/beam]$',fontsize=13)
    plt.step(Molines_A_df[px].index[channel[0]:channel[1]],Molines_A_df[px].iloc[channel[0]:channel[1]],color='k')
    
    # Componentes 
    for i in range(npeaks):
        if i == 0:
            c='blue'
        else:
            c = 'red'
        plt.plot(x_datos, comps['peak%d_' % (i+1)],'--',color=c)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(axis='both', direction='in', length=5, width=1.5)
    
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', direction='in', length=2, width=1)
    
    # Ajuste 
    plt.plot(x_datos, out.best_fit, '-', color='purple')
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-max(y_datos)/11,r'$\mu =$ '+str(round(pars['peak1_center'].value,2))+' km/s',color='blue',fontsize=13)
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-2*max(y_datos)/11,r'$\sigma =$ '+str(round(pars['peak1_sigma'].value,2))+' km/s',color='blue',fontsize=13)
    
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-3*max(y_datos)/11,r'$\mu =$ '+str(round(pars['peak2_center'].value,2))+' km/s',color='red',fontsize=13)
    plt.text(min(x_datos)+min(x_datos)/40,max(y_datos)-4*max(y_datos)/11,r'$\sigma =$ '+str(round(pars['peak2_sigma'].value,2))+' km/s',color='red',fontsize=13)

    plt.tight_layout()