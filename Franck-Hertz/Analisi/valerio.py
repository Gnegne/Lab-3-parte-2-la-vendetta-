import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

# inizio cose

oscilfile = os.path.join(folder, 'oscilloscopio', 'dati001.csv')

a, b = data_from_oscill(oscilfile)

a.draw()
b.draw()

plt.show()
