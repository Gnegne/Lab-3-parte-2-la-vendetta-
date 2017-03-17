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


@np.vectorize
def b_z(r, a, ni):
	k = r/a
	val, err = integrate.quad(integrand, 0, 2*np.pi, args=(k,))
	return 2e-3 * ni/a * val


def bz_wrap(r, zero, a):
	return b_z((r- zero)/100, a, 130 * 0.95)


bz_wrap.pars, _, _ = _fit_generic_ev(bz_wrap, lambda x, *pars: 0, field.x.val, field.y.val, field.x.err, field.y.err, np.array([20, .15]), 10)
field.draw(bz_wrap)
plt.show()

print(bz_wrap.pars)
