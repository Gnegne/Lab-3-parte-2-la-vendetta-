import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

# inizio cose

datafile = "parteA.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Dati', datafile)).T

k = 1e-6 / rawdata[0]		# inverse of lambda in um^-1 units
ang = (360 - rawdata[1] - rawdata[2]/60) * np.pi / 180
dang = rawdata[3]/60 * np.pi / 180

fitline = createline()
# fitline.mask = rawdata[0] < 600e-9

uno = DataHolder(k, ang, dy=dang)
uno.fit_generic(fitline, method='ev')
uno.draw(fitline, resid=True)

na = (360 - 311 - 47/60) * np.pi / 180
dna = 2/60 * np.pi / 180

nal = lineheight(fitline.pars, fitline.cov, na, dna**2)

print("Sodium wavelenght: {} \pm {} nm".format(1e3/nal[0], np.sqrt(nal[1])/nal[0]**2*1e3))

plt.show()
