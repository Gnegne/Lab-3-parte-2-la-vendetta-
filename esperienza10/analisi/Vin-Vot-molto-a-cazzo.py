## Programma generico di fit numerico del minimo chi quadro.
# PLEASE USE RELATIVE PATHS
# PLEASE USE SPACES PROPERLY
import pylab 
import numpy
import sys, os
import scipy.special
from scipy.optimize import curve_fit
from numpy.linalg import solve
import getpass
folder = os.path.realpath('..')
sys.path.append(os.path.join(folder, '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
#from lab import *
#from iandons import *
from matplotlib.legend_handler import HandlerLine2D
folder = os.path.realpath('..')


## Import dei dati (x, dx, y, dy).
x, y = pylab.loadtxt('C:\\Users\\Roberto\\Documents\\GitHub\\Lab3.2\\esperienza10\\dati\\dati_1a.txt',unpack=True)
#sorry per il pahts ma non riesco a far funzionare i path relativi... e non c'ho cazzi di perderci tempo
volt = x
dx = mme(x,"volt","dig")
dy = mme(y,"volt","dig")
#for counter in range (0,len(x)):
#	if x[counter]  < 2:
#		dx[counter]=(0.001**2 +(x[counter]*0.5/100)**2)**0.5
#	else:
#		dx[counter]=(0.01**2 +(x[counter]*0.5/100)**2)**0.5
#for counter in range (0,len(y)):
#	if y[counter]  < 2:
#		dy[counter]=(0.0001**2 +(y[counter]*0.5/100)**2)**0.5
#	else:
#		dy[counter]=(0.01**2 +(y[counter]*0.5/100)**2)**0.5
#print( dy)
r = 114
dr = ((r*4/100)**2+9)**0.5
print( dr)
## Grafico y(x).
pylab.figure(1)
pylab.rc('font',size=14)
pylab.xlim(-0.3, 5.4)
pylab.ylim(0, 5)
pylab.title('Comportamento della porta NOT')
pylab.grid(True, which="both", color = "gray")
pylab.ylabel('$V_{OUT}$ $[V]$')
pylab.xlabel('$V_{IN}$ $[V]$')
#pylab.ylim(0.980,1)
pylab.errorbar(x, y, dy, dx, fmt=",",ecolor="black",capsize=0.5)


func_grid = numpy.linspace(-1, 6, 1000)

zeroline=func_grid*0 + 2.4
pylab.plot(func_grid,zeroline, '--', color = 'red')
zeroline=func_grid*0 + 0.4
pylab.plot(func_grid,zeroline, '--', color = 'blue')

#zeroline=func_grid*0 + 0.142
#pylab.plot(func_grid,zeroline, '--', color = 'blue')

pylab.errorbar([2,2],[0,5],[0,0],linestyle = '--', color= 'red', marker = '')
pylab.errorbar([0.8,0.8],[0,5],[0,0],linestyle = '--', color= 'blue', marker = '')



pylab.show('Grafici')
