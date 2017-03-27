import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import integrate
from lab import *
from iandons import *


filename = "dati_1b_ii_ingressoalto_uscitabassa.txt"
rawdata = np.loadtxt(os.path.join(folder, 'dati', filename)).T

r2 = 100.8
vout = rawdata[0]
iout = rawdata[1] / r2
di = mme(rawdata[1], 'volt') / r2
dv = mme(vout, 'volt')

a = np.array((iout, di, vout, dv)) * np.array([[1e3], [2e3], [1], [1]])

filename = "dati_1b_ii_ingressobasso_uscitaalta.txt"
rawdata = np.loadtxt(os.path.join(folder, 'dati', filename)).T

r2 = 100.8
vout = rawdata[0]
iout = rawdata[1] / r2
di = mme(rawdata[1], 'volt') / r2
dv = mme(vout, 'volt')

b = np.array((iout, di, vout, dv)) * np.array([[-1e3], [2e3], [1], [1]])

print(maketab(*a, *b, errors='all'))
