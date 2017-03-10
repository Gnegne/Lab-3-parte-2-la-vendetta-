import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), '0_Valerio_Utilities', 'Python'))
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from lab import *
from iandons import *

plt.close('all')
# inizio cose


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
	for t, s in ((splits[i], splits[i+1]) for i in range(len(splits) - 1)):
		mask = (hold.x.val < s) & (hold.x.val > t)
		x = hold.x.val[mask]
		y = hold.y.val[mask]
		i = np.argmax(y)
		ret.append((x[i], hold.x.err[mask][i]))
	return ret

def minimas(hold, splits):
	ret = []
	for t, s in ((splits[i], splits[i+1]) for i in range(len(splits) - 1)):
		mask = (hold.x.val < s) & (hold.x.val > t)
		x = hold.x.val[mask]
		y = hold.y.val[mask]
		i = np.argmin(y)
		ret.append((x[i], hold.x.err[mask][i]))
	return ret

datainfo = np.loadtxt(os.path.join(folder, 'oscilloscopio', 'dati_info.txt')).T

# for i in range(1, 12):
# 	oscilfile = os.path.join(folder, 'oscilloscopio', 'dati{0:03d}.csv'.format(i))
#
# 	# a, b = data_from_oscill(oscilfile)
# 	z = data_from_oscill(oscilfile, mode='xy')
#
# 	q = np.array(maximas(z, (0.5, 2.5, 4.5, 6.5, 200)))*10 - np.array((datainfo[2, i-1], 0))
# 	p = np.array(minimas(z, (1.9, 3.7, 5.7, 77)))*10 - np.array((datainfo[2, i-1], 0))
# 	z.x.label = '$U_A$ [V]'
# 	z.x.re = 10
# 	z.y.label = '$V_{out}$'
# 	z.title = '$U_E = {}$ V'.format(datainfo[1, i-1])
# 	# z.draw()
#
# 	print('file {0:03d}, U_E: {1}, deltas:'.format(i, datainfo[1][i-1]), *(xe(*q.T)))

# plt.show()

d = np.array([[18.2, .3], [20.5, .4], [20.1, .5]]).T
n = np.array([2, 3, 4])

def majick(m, r, E):
	return (1 + r * (2*m -1)) * E

pars, pcov = fit_generic(majick, n, d[0], None, d[1], [.1, 16])

print(pars)
print(np.sqrt(np.diag(pcov)))

print(tell_chi2((d[0] - majick(n, *pars))/d[1], 1))
