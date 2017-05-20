import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import integrate
from lab import *
from iandons import *

##preliminare
Vs = 5.00
dVs = mme(Vs, 'volt')
Vm = -5.01
dVm = mme(Vm, 'volt')
print(dVs, dVm)

r1 = 984
r2 = 974
c1 = 112.5e-9
c2 = 105.7e-9

dr1 = mme(r1, 'ohm')
dr2 = mme(r2, 'ohm')
dc1 = mme(c1, 'farad')
dc2 = mme(c2, 'farad') 

print(r1, dr1,'//')
print(r2, dr2,'//')
print(c1, dc1,'//')
print(c2, dc2,'//')


##partitore
R1 = 987
R2 = 1.031e6
dR1 = mme(R1, 'ohm')
dR2 = mme(R2, 'ohm')
Av = R1/(R1 +R2)
dAv = (dR1/(R1 +R2))**2 + (R1*dR1/((R1 +R2)**2))**2 + (R1*dR2/((R1 +R2)**2))**2 
dAv = np.sqrt(dAv)
print('partitore',Av, dAv,'//')
print(R1, dR1,'//')
print(R2, dR2,'//')

##prea-amp
R1 = 971
R2 = 4.69e3
R3 = 71.9e3
Vref = 1.10
Vp = 21.0

dVp = mme(Vp,'volt','oscil')
dVref = mme(Vref,'volt','oscil')
dR1 = mme(R1, 'ohm')
dR2 = mme(R2, 'ohm')
dR3 = mme(R3, 'ohm')

print('/pre-amp/')
print(R1, dR1)
print(R2, dR2)
print(R3, dR3)
print(Vp,dVp,'tensione')
print(Vref,dVref,'tensione')

y = Vref/Vp
dy = (dVref/Vp )**2  +  (Vref*dVp/(Vp**2) )**2
dy = np.sqrt(dy)
xx = y/Av
dxx =  (dy/Av )**2  +  (dAv*y/(Av**2) )**2
dxx = np.sqrt(dxx)
print(xx, dxx)

##preamp-b

Vo = 5.68
Vx = 0.374
dVo = mme(Vo,'volt','oscil')
dVx = mme(Vx,'volt','oscil')
g = Vo/Vx
dg = (dVo/Vx )**2  +  (Vo*dVx/(Vx**2) )**2
dg = np.sqrt(dg)
print(Vx, dVx)
print(Vo, dVo)
print(g, dg)

##banda

R1 = 2.66e3
R2 = 117.9
R3 = 46.4e3
dR1 = mme(R1, 'ohm')
dR2 = mme(R2, 'ohm')
dR3 = mme(R3, 'ohm')
print('banda resist:')
print(R1, dR1)
print(R2, dR2)
print(R3, dR3)

##post
R1p = 972
R3p = 3.87e3
R2p = 33.1e3


dR1p = mme(R1p, 'ohm')
dR2p = mme(R2p, 'ohm')
dR3p = mme(R3p, 'ohm')
print('post resist:')
print(R1p, dR1p)
print(R2p, dR2p)
print(R3p, dR3p)
#(((((((((((