#Se importan todos los paquetes que se van a necesitar.
from __future__ import division
import sys, platform, os

from matplotlib import pyplot as plt
import numpy as np

import camb
from camb import model, initialpower

import pylab as pl
#Se declara la forma de impresión.
font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='x-large'
plt.rc('font', **font)
plt.rcParams['legend.fontsize']='medium'

#import scipy
from scipy.interpolate import interp1d


from scipy import integrate
from scipy import linalg


#Parámetros cosmológicos necesarios.
c=3e5
pi=np.pi

hubble=0.678
omegab=0.022*pow(hubble,-2)
omegac=0.119*pow(hubble,-2)
om0=omegac+omegab
H00=100*hubble
Ass=2.14e-9
nss = 0.968

gamma=0.545


class parametros_CAMB():
    
    def parametrosCAMB():
        
        #Se preparan los parámetros de CAMB
        pars = camb.CAMBparams()
        
        #Set cosmology || Se preparan los datos cosmológicos a como se desean.
        pars.set_cosmology(H0 = H00, ombh2 = omegab*pow(hubble, 2), omch2=omegac*pow(hubble, 2), omk = 0, mnu = 0)
        pars.set_dark_energy() #LCDM (default)
        pars.InitPower.set_params(ns = nss, r = 0, As = Ass)
        pars.set_for_lmax(2500, lens_potential_accuracy = 0);
        
        #Se calculan resultados para esos parámetros
        results = camb.get_results(pars)
        
        #Get matter power spectrum at z=0: P(k,z=0)
        pars.set_matter_power(redshifts=[0.], kmax = 2.0)
        
        #Linear spectra
        pars.NonLinear = model.NonLinear_none
        results.calc_power_spectra(pars)
        kh, z, pk = results.get_matter_power_spectrum(minkh = 1e-4, maxkh = 2.0, npoints = 200)
        return kh, z, pk

    #Construct P(k,z=0) interpolating function, in units of Mpc (no h)
    def Pkz0():
        kh, z, pk = parametros_CAMB.parametrosCAMB()
        Pkz0 = interp1d(kh*hubble, pk[0]/pow(hubble, 3))
        return Pkz0
    
    """Se comienzan a definir todas las funciones necesarias para formar el Espectro de Potencias para diferentres redshift (z)"""
    #Defino E(z)
    def Ez(zc):
        return np.sqrt(1 - om0 + om0*pow(1 + zc, 3))

    #Define the comoving distance
    def drdz(zp):
        return (c/H00)/parametros_CAMB.Ez(zp)
    
    def rcom(zc):
        return integrate.romberg(parametros_CAMB.drdz, 0, zc)

    #Define the growth function in LCDM
    def fg(zz):
        omz = om0*pow(1 + zz, 3)/(om0*pow(1 + zz, 3) + 1 - om0)
        return pow(omz, gamma)

    #Get the growth factor 
    def Dg_dz(zz):
        return parametros_CAMB.fg(zz)/(1 + zz)
    
    def Dgz(zc):
        ans = integrate.romberg(parametros_CAMB.Dg_dz, 0.0, zc)
        return np.exp(-ans)

    #Fiducial HI abundance and bias fitting functions from SKA Cosmology Red Book 2018
    def OmHI(zc):
        return 0.00048 + 0.00039*zc - 0.000065*pow(zc, 2)

    def bHI(zc):
        return 0.67 + 0.18*zc + 0.05*pow(zc, 2)
    
    def Pkz(kk, zc):    
        return pow(parametros_CAMB.Dgz(zc), 2)*(parametros_CAMB.Pkz0)
    
    def Tb(zc): #in mK
        return 0.0559 + 0.2324*zc - 0.024*pow(zc, 2)

    #Construct  matter power spectrum P(k,z) - no RSDs
    #Se construye la ecuación del Espectro de Potencia para cualquier z P(k,z)
    #Construct P_HI(k,z) [mK^2]
    def PHI(kk, zc):
        return pow(parametros_CAMB.Tb(zc), 2)*pow(parametros_CAMB.bHI(zc), 2)*parametros_CAMB.Pkz(kk, zc)

print(parametros_CAMB.PHI(0.1, 0.5))













