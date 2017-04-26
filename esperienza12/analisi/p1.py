import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import integrate
from lab import *
from iandons import *

Ti =  76e-9
Td =  24.2e-9
dTi = mme(Ti, 'time', 'oscil')
dTd = mme(Td, 'time', 'oscil')
print(Ti, dTi)
print(Td, dTd)

# valerio

deltas = np.loadtxt(os.path.join(folder, 'data', '2_ritardo.txt'))[:,-1::-1] * 1e-9
errs = mme(deltas, 'time', 'oscil')
print(errs)
print()
print(*xe(deltas, errs)[0])
print(*xe(deltas, errs)[1])

# /valerio
