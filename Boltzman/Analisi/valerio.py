"""Tipo roba."""

import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

# ******* mpl setup
mpl.rcParams['errorbar.capsize'] = 1
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
#w.draw()

# *** amplificazioni ***
amplis = [[], [], []]
errs = [[], [], []]
# preamp primo stadio
freqs = [1e3] * 3
idx = ['', 1, 2]
parter = .987/1031

for freq, i in zip(freqs, idx):
	f = createwave()
	f.pars = [freq*2*np.pi, 1, 0, 0]
	h = createwave()
	h.pars = [freq*2*np.pi, 1, 0, 0]
	filo = 'preamp{}.csv'.format(i)
	a, b = data_from_oscill(os.path.join(folder, 'Data', filo))
	a.fit_generic(f, verbose=False)
	b.fit_generic(h, verbose=False)
	G = np.abs(h.pars[1]/f.pars[1]/parter)
	e = f.cov[1, 1]/f.pars[1]**2 + h.cov[1, 1]/h.pars[1]**2
	e = G * np.sqrt(e)
	amplis[0].append(G)
	errs[0].append(e)

# preamp secondo stadio
freqs = [1e3] * 3
idx = ['', 1, 2]

for freq, i in zip(freqs, idx):
	f = createwave()
	f.pars = [freq*2*np.pi, 1, 0, 0]
	h = createwave()
	h.pars = [freq*2*np.pi, 1, 0, 0]
	filo = 'preampb{}.csv'.format(i)
	a, b = data_from_oscill(os.path.join(folder, 'Data', filo))
	a.fit_generic(f, verbose=False)
	b.fit_generic(h, verbose=False)
	G = np.abs(f.pars[1]/h.pars[1])
	e = f.cov[1, 1]/f.pars[1]**2 + h.cov[1, 1]/h.pars[1]**2
	e = G * np.sqrt(e)
	amplis[1].append(G)
	errs[1].append(e)

# postamp
freqs = [1e3] * 3
idx = [1, 2, 3]

for freq, i in zip(freqs, idx):
	f = createwave()
	f.pars = [freq*2*np.pi, 1, 0, 0]
	h = createwave()
	h.pars = [freq*2*np.pi, 1, 0, 0]
	filo = 'postampo{}.csv'.format(i)
	a, b = data_from_oscill(os.path.join(folder, 'Data', filo))
	a.fit_generic(f, verbose=False)
	b.fit_generic(h, verbose=False)
	G = np.abs(h.pars[1]/f.pars[1])
	e = f.cov[1, 1]/f.pars[1]**2 + h.cov[1, 1]/h.pars[1]**2
	e = G * np.sqrt(e)
	amplis[2].append(G)
	errs[2].append(e)

amplis = np.array(amplis)
errs = np.array(errs)
print('Preamp stadio a:', *xe(amplis[0], errs[0]))
print('Preamp stadio b:', *xe(amplis[1], errs[1]))
print('Postamp:', *xe(amplis[2], errs[2]))

# rumore
filename = "dati_rummore.txt"
rawdata = np.loadtxt(os.path.join(folder, 'Data', filename)).T

r = rawdata[0] * 1e3
n = rawdata[1]
dn = rawdata[2]
dr = mme(r, 'ohm')


def noiso(R, Vn, Rt, Rn):
	return Vn * np.sqrt(1 + R/Rt + (R/Rn)**2)


noiso.deriv = lambda R, Vn, Rt, Rn: Vn/(2 * np.sqrt(1 + R/Rt + (R/Rn)**2)) * (1/Rt + 2*R/Rn**2)
noiso.pars = [0.05, 9e3, 17e3]
noiso.mask = (r > 4e3)

kb = DataHolder(r, n, dr, dn)
kb.fit_generic(noiso)
#kb.draw(noiso, resid=True)

print(fit_norm_cov(noiso.cov))
Vn, Rt, Rn = noiso.pars

Atot = 1e7
band = 1e3
boltzmann = Vn**2 / (4 * Rt * 300 * Atot * band)
print(boltzmann)

plt.show()
