"""Tipo roba."""

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

# passabanda

filename = "dati_passabanda.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T

f = rawdata[0]
df = f / 1e3
g = rawdata[2] / rawdata[1]
g = 20*np.log10(g)
dg = np.sqrt(mme(rawdata[1], 'volt', 'oscil')**2 / rawdata[1]**2 + mme(rawdata[2], 'volt', 'oscil')**2 / rawdata[2]**2)*20*np.log10(np.e)

w = DataHolder(f, g, df, dg)
w.x.type = 'log'
w.draw()

# amplificazioni
freqs = [1e3] * 6
idx = ['', 1, 2, 'b', 'b1', 'b2']
amplis = [[], []]
errs = [[], []]

for freq, i in zip(freqs, idx):
	f = createwave()
	f.pars = [freq*2*np.pi, 1, 0, 0]
	h = createwave()
	h.pars = [freq*2*np.pi, 1, 0, 0]
	filo = 'preamp{}.csv'.format(i)
	a, b = data_from_oscill(os.path.join(folder, 'Data', filo))
	a.fit_generic(f)
	b.fit_generic(h)
	G = np.abs(f.pars[1]/h.pars[1])
	e = f.cov[1, 1]/f.pars[1]**2 + h.cov[1, 1]/h.pars[1]**2
	if G < 1:
		G = 1/G
	e = G * np.sqrt(e)
	amplis[0].append(G)
	errs[0].append(e)

print(amplis[0])
print(errs[0])
# rumore

filename = "dati_rummore.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T

r = rawdata[0] * 1e3
n = rawdata[1]
dn = rawdata[2]
dr = mme(r, 'ohm')

kb = DataHolder(r, n, dr, dn)
kb.draw()

plt.show()
