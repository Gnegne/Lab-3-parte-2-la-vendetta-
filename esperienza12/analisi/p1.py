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