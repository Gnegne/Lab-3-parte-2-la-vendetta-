## Programma generico di fit numerico del minimo chi quadro.

import math
import pylab
import numpy
import sys, os
import scipy.special
from scipy.optimize import curve_fit
from numpy.linalg import solve
from matplotlib.legend_handler import HandlerLine2D
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('.\Desktop\LAB\lab3tras\0_Valerio_Utilities\Python'), '0_Valerio_Utilities', 'Python'))
#sys.path.append('.\Desktop\LAB\lab3tras\0_Valerio_Utilities\Python')
#import lab
from lab import *

## Import dei dati (x, dx, y, dy).
x, y= pylab.loadtxt('dati_1a.txt', unpack = True)
dx=dy=0.01
## Grafico y(x).
pylab.figure(1)
pylab.rc('font',size=14)
pylab.xlim(-0.3, 5.4)
pylab.ylim(0, 5)
pylab.title('$V_{out}$ $VS$ $V_{in}$ ai capi di NOT1')
pylab.grid(True, which="both", color = "gray")
pylab.ylabel('$V_{out}$ $[V]$')
pylab.xlabel('$V_{in}$ $[V]$')
#pylab.ylim(0.980,1)
pylab.errorbar(x, y, dy, dx, linestyle = '', color = 'blue', marker = '+')


func_grid = numpy.linspace(-1, 6, 10000000)

zeroline=func_grid*0 + 4.38
pylab.plot(func_grid,zeroline, '--', color = 'red')
zeroline=func_grid*0 + 2.82
pylab.plot(func_grid,zeroline, '--', color = 'red')

zeroline=func_grid*0 + 0.142
pylab.plot(func_grid,zeroline, '--', color = 'blue')

pylab.errorbar([0.954,0.954],[0,5],[0,0],linestyle = '--', color= 'purple', marker = '')
pylab.errorbar([1.092,1.092],[0,5],[0,0],linestyle = '--', color= 'green', marker = '')



pylab.show('Grafici')
