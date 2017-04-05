import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

# ******* mpl setup
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['lines.linewidth'] = 1.0

# ******* end setup

filename = "dati_2.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T

r = rawdata[0]
t = rawdata[1]
dt = mme(t*1e-6, 'time', 'oscil')*1e6/2
dr = 1 # * mme(r, 'ohm')

theory = createline()
theory.pars = [.1, 0]
theory.mask = r > 300
theory.linekw = dict(label='fit')

qq = DataHolder(r, t, dr, dt)
qq.fit_generic(theory)

print(fit_norm_cov(theory.cov)[0, 1])

qq.x.err = mme(r, 'ohm')
qq.y.err *= 2

qq.y.label = '$T_H$ [$\mu s$]'
qq.x.label = '$R_1$ [$\Omega$]'

qq.draw(theory, resid=True, legend=False)
qq.main_ax.scatter(253, 29.4, s=70, facecolors='none', edgecolors='r', label='outliers')
qq.main_ax.legend(loc='best')
plt.show()
