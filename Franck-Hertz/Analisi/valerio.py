import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

plt.close('all')
# inizio cose

oscilfile = os.path.join(folder, 'oscilloscopio', 'dati001.csv')

a, b = data_from_oscill(oscilfile)
z = data_from_oscill(oscilfile, mode='xy')


def shift(arr, pos):
	hold.sort()
	n = np.zeros(np.shape(arr))
	for i in range(pos):
		n[i] = arr[0]
	n[pos:] = arr[:-pos]
	return n


def smoothing(hold, roll):
	hold.sort()
	newx = sum(hold.x.val[i:i-roll] for i in range(roll)) / roll
	errx = np.sqrt(sum(hold.x.err[i:i-roll]**2 for i in range(roll))) / roll
	newy = sum(hold.y.val[i:i-roll] for i in range(roll)) / roll
	erry = np.sqrt(sum(hold.y.err[i:i-roll]**2 for i in range(roll))) / roll

	return DataHolder(newx, newy, errx, erry)


def deltas(hold, dist):
	hold.sort()
	y = hold.y.val[dist-1:] - hold.y.val[:1-dist]
	x = (hold.x.val[dist-1:] + hold.x.val[:1-dist]) / 2
	errx = np.sqrt(hold.x.err[dist-1:]**2 + hold.x.err[:1-dist]**2) / 2
	erry = np.sqrt(hold.y.err[dist-1:]**2 + hold.y.err[:1-dist]**2) / 2
	return DataHolder(x, y, errx, erry)

def maximas(hold, splits):
	ret = []
	for s in splits:
		mask = hold.x.val < s
		x = hold.x.val[mask]
		y = hold.y.val[mask]
		i = np.argmax(y)
		ret.append((x[i], hold.x.err[mask][i]))
	return ret
