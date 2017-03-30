import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import integrate
from lab import *
from iandons import *


filename = "prov-t.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T
dt = mme(rawdata, 'time', 'oscil')
print(dt)

filename = "dati_4.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T
R1 = rawdata[0]
R2 = rawdata[1]
T = rawdata[2]
pls = rawdata[3]
dR1 = mme(rawdata[0], 'ohm')
dR2 = mme(rawdata[1], 'ohm')
dT = mme(rawdata[2], 'time', 'oscil')
dpls = mme(rawdata[3], 'time', 'oscil')
dty = T/pls
ddty = np.sqrt(dT**2 + (T*dpls/pls)**2) /pls
a = np.array((R1, dR1, R2, dR2, T, dT, pls, dpls, dty, ddty)) 
print(maketab(*a, errors='all'))
print(dR1)


T = 101e-6
up = 30.2e-6
v=5.01
dt = mme(T, 'time', 'oscil')
dp = mme(up, 'time', 'oscil')
dv = mme(v, 'volt')
print(dt, dp,dv)