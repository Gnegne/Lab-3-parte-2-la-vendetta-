import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import integrate
from lab import *
from iandons import *
from lab import _fit_generic_ev

# setup

camera_to_ruler = .568
bulb_radius = .05
bulb_thickness = .001
centre_to_ruler = .11

# contis (spero di aver azzeccato i segni)

def calcolacose():
	n = 1.5  # che vetro stiamo usando? in realtà non cambia na sega
	d = bulb_thickness
	r1 = -bulb_radius
	r2 = -bulb_radius - d
	p = (n-1) * (1/r1 - 1/r2 + (n-1)*d/(n * r1 * r2))  # lensmaker eq
	f = 1/p

	vo = 1/bulb_radius
	vi = p - vo  # optician thin lens eq
	di = 1/vi

	M = f/(f - bulb_radius)
	img_from_camera = camera_to_ruler - centre_to_ruler - bulb_radius - di
	k = img_from_camera / camera_to_ruler

	return M, k, img_from_camera, di

# omogeneità e cazzi


rawdata = np.loadtxt(os.path.join(folder, 'dati', 'mappatra.txt')).T
r = rawdata[0]
v = rawdata[1] / 11.09
b = v / 5e-3

field = DataHolder(r, b, .05, 1/11.09/10)


def integrand(x, k):
	return (1 - k*np.cos(x)) / (5/4 + k**2 - 2*k*np.cos(x))**(3/2)


def derint(x, k):
	return -((1 - k*np.cos(x))*3*(k - np.cos(x)) + np.cos(x)*(5/4 + k**2 - 2*k*np.cos(x))) / (5/4 + k**2 - 2*k*np.cos(x))**(5/2)


@np.vectorize
def b_z(r, a, ni):
	k = r/a
	val, err = integrate.quad(integrand, 0, 2*np.pi, args=(k,))
	return 2e-3 * ni/a * val


@np.vectorize
def db(r, zero, a, z):
	ni = 130 * 0.95 * z
	r = (r - zero) / 100
	k = r/a
	gg, _ = integrate.quad(derint, 0, 2*np.pi, args=(k,))
	return 2e-3 * ni/a**2 * gg


def bz_wrap(r, zero, a, z):
	return b_z((r - zero)/100, a, 130 * 0.95 * z)


bz_wrap.pars, cov, _ = _fit_generic_ev(bz_wrap, db, field.x.val, field.y.val, field.x.err, field.y.err, np.array([20.2, .158, 1.05]), 100)

field.x.label = "Posizione orizzontale [cm]"
field.y.label = "$B_z$ [Gs]"
field.title = "Mappatura del campo magnetico nella regione centrale"
bz_wrap.resd = (field.y.val - bz_wrap(field.x.val, *bz_wrap.pars)) / np.sqrt(field.y.err**2 + (field.x.err/100)**2 * db(field.x.val, *bz_wrap.pars) **2)

field.draw(bz_wrap, resid=True)
field.main_ax.axvline(x=bz_wrap.pars[0], color='black', ls='--')

# plt.figure()
# plt.plot(field.pts, db(field.pts, *bz_wrap.pars))


print(tell_chi2(bz_wrap.resd, len(field.y.val) - 3, style='latex'))
print(xe(bz_wrap.pars, np.sqrt(np.diag(cov))))
print(fit_norm_cov(cov))

print(bz_wrap(0, 0, *bz_wrap.pars[1:])/0.95)


print(maketab(*rawdata.T, errors='none'))
