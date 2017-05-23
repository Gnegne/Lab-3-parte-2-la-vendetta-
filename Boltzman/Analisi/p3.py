import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), 'Data Analysis'))
from ANALyzer import *
from uncertainties import *
from uncertainties import unumpy as un
import numpy as np

#########################################

#anmpli vari ed eventuali#

#i valori di queste variabili sono calcolati nello script di valerio

amp0 = un.uarray(amplis[0],errs[0])
amp1 = un.uarray(amplis[1],errs[1])
amp2 = un.uarray(amplis[2],errs[2])

amp0_ = (amp0[0]+amp0[1]+amp0[2])/3
amp1_ = (amp1[0]+amp1[1]+amp1[2])/3
amp2_ = (amp2[0]+amp2[1]+amp2[2])/3



#########################################

#passabanda#

dir= folder+"\\"
file="dati_passabanda"

def f(x, a, b, c):
    return a*x*np.sqrt(1/((x**2-b**2)**2+(x**2*b**2/c**2)))
    
p0=[1e3,6200,10]

def XYfun(a):
    return a[0]*1000, a[2]/a[1]

unit = [("freq", "osc"),("volt_ar_nc", "osc"),("volt_ar_nc", "osc")]

titolo = "Fit boltzmann"
Xlab = "R [kohm]"
Ylab = "V"

par0 = fit(dir, file, unit, f, p0, titolo, Xlab, Ylab, XYfun, Xscale="log", residuals=True)

amp3_ = par0[0]*par0[2]/par0[1]
#########################################

#RUMORE#

dir= folder+"\\"
file="dati_rummore"

def f(x, a, b, c):
    return a * np.sqrt(1 + x/b+(x**2/c**2))
    
p0=[0.05,3e3,30e3]

def XYfun(a):
    return a[0]*1000, a[1], a[2]

unit = [("ohm", "dig"),("volt", "dig"),("volt", "dig")]

titolo = "Fit boltzmann"
Xlab = "R [kohm]"
Ylab = "V"

par1 = fit(dir, file, unit, f, p0, titolo, Xlab, Ylab, XYfun, yerr=True, kx=1e-3, residuals=True)

A = amp0_*amp1_*amp2_*amp3_/np.sqrt(2)
T = 300
df = par0[1]/par0[2]*np.pi/2

K = par1[0]**2/(par1[1]*4*T*(A**2)*df)

print(K)







