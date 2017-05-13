import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import integrate
from lab import *
from iandons import *

Vs = 5.00
dVs = mme(Vs, 'volt')
Vm = -4.98
dVm = mme(Vm, 'volt')
Vpp = 2.02
f = 1.027505*1000
v2lastre = 0.840 

dVpp = mme(v2lastre, 'volt', 'oscil')
print(dVpp)
print(dVs)
print(dVm)

filename = "resistenze.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T
dR = mme(rawdata, 'ohm')
print(dR)