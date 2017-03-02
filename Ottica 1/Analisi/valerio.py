import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

# inizio cose

datafile = "parteA.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Dati', datafile)).T

abc = "{0:.1f} & {{{1:.0f}}}{{{2:.0f} ({3:.0f})}}"
fullround = np.array([0, 359, 60, 0])
mm = np.array([-1e9, 1, 1, -1])

for r in rawdata.T:
	print(abc.format(*(fullround - r*mm)))


k = 1e-6 / rawdata[0]		# inverse of lambda in um^-1 units
ang = (360 - rawdata[1] - rawdata[2]/60) * np.pi / 180
dang = rawdata[3]/60 * np.pi / 180 * 3/2

fitline = createline()
# fitline.mask = rawdata[0] < 600e-9

uno = DataHolder(k, ang, dy=dang)

uno.x.label = R"$1/\lambda$ [$\mu m^{-1}$]"
uno.y.label = R"$\alpha - \alpha_0$ [rad]"

uno.fit_generic(fitline, method='ev')
uno.draw(fitline, resid=True)

print(fit_norm_cov(fitline.cov))

na = (360 - 311 - 47/60) * np.pi / 180
dna = 2/60 * np.pi / 180
nal, nalvar = lineheight(fitline.pars, fitline.cov, na, dna**2)
print(R"Sodium wavelenght: \SI{{{0:.2f} ({1:.2f})}}{{\nm}}".format(1e3/nal, np.sqrt(nalvar)/nal**2*1e3))

plt.show()
