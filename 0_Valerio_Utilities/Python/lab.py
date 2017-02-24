# ********************** IMPORTS ***************************

import math
import inspect
import numpy as np
import time
from scipy import odr
from scipy.optimize import curve_fit, leastsq
import os
import sympy

# TODO
#
# fit_concatenate
# concatena modelli di fit per fittare simultaneamente condividendo parametri
#
# _fit_*_odr
# vedere se c'è qualcosa di meglio di leastsq (es. least_squares?)
#
# util_format
# opzione si=True per formattare come num2si
# opzione errdig=(intero) per scegliere le cifre dell'errore
# usare lo standard di arrotondamento del PDG
#
# fit_generic (nuova funzione)
# mangia anche le funzioni sympy calcolandone jacb e jacd, riconoscendo se può fare un fit analitico
# controlla gli argomenti in ingresso prima per dare errori sensati
# usa scipy.odr (anche fit implicito, covarianze, multidim, restart, fissaggio parametri)
# le cose multidim sono trasposte nel modo comodo per scrivere le funz. (quindi mi sa come scipy.odr), può trasporre lei con un'opzione
# riconosce se la f mangia un punto alla volta o tutti i dati insieme
# interfaccia tipo curve_fit (fare in modo che i casi base siano uguali a fit_linear e fit_generic_xyerr)
# ha un full_output e un print_info che fanno molte cose
# fit_generic(f, x, y=None, dx=None, dy=None, p0=None, pfix=None, dfdx=None, dfdp=None, dimorder='compfirst', absolute_sigma=True, full_output=False, print_info=False, restart=False)
# = par, cov,
# {'resx': residui x,
# 'resy': residui y,
# 'restart': oggetto ODR}
# se restart=True, ritorna l'oggetto ODR; se restart è un ODR, lo usa per ripartire da lì.
# pfix = [True, False ...] i True vengono bloccati
# pfix = [0, 3, 5...] i parametri a questi indici vengono bloccati

__all__ = [ # things imported when you do "from lab import *"
	'fit_norm_cov',
	'fit_generic',
	'fit_linear',
	'fit_const_yerr',
	'util_mm_er',
	'util_mm_list',
	'mme',
	'etastart',
	'etastr',
	'num2si',
	'num2sup',
	'num2sub',
	'unicode_pm',
	'xe',
	'xep',
	'util_format'
]

# __all__ += [ # things for backward compatibility
# 	'curve_fit_patched',
# 	'fit_generic_xyerr',
# 	'fit_generic_xyerr2'
# ]

__version__ = '2016.11'

# ************************** FIT ***************************

def _check_finite(array): # asarray_chkfinite is absent in old numpies
	for x in array.flat:
		if not np.isfinite(x):
			raise ValueError("array must not contain infs or NaNs")

def _patch_curve_fit(force_patch=False):
	args = inspect.getargspec(curve_fit).args
	
	if 'absolute_sigma' in args and 'check_finite' in args and not force_patch:
		return curve_fit
		
	elif 'absolute_sigma' in args and not force_patch:
		def curve_fit_patched(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
			if check_finite:
				_check_finite(xdata)
				_check_finite(ydata)
			return curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma, **kw)
		return curve_fit_patched
		
	else: # the case check_finite yes and absolute_sigma no does not exist
		def curve_fit_patched(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
			myp0 = p0
			if p0 is None: # we need p0 to implement absolute_sigma
				args = inspect.getargspec(f).args
				if len(args) < 2:
					raise ValueError("Unable to determine number of fit parameters.")
				myp0 = [1.0] * (len(args) - (2 if 'self' in args else 1))
			if np.isscalar(myp0):
				myp0 = np.array([myp0])
			if check_finite:
				_check_finite(xdata)
				_check_finite(ydata)
			rt = curve_fit(f, xdata, ydata, p0, sigma, **kw)
			if absolute_sigma and len(ydata) > len(myp0): # invert the normalization done by curve_fit
				popt = rt[0]
				s_sq = sum(((np.asarray(ydata) - f(xdata, *popt)) / (np.asarray(sigma) if sigma != None else 1.0)) ** 2) / (len(ydata) - len(myp0))
				pcov = rt[1] / s_sq
				rt = np.concatenate(([popt, pcov], rt[2:]))
			return rt
		return curve_fit_patched

curve_fit_patched = _patch_curve_fit()

def fit_norm_cov(cov):
	"""
	normalize a square matrix so that the diagonal is 1:
	ncov[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])

	Parameters
	----------
	cov : (N,N)-shaped array-like
		the matrix to normalize

	Returns
	-------
	ncov : (N,N)-shaped array-like
		the normalized matrix
	"""
	ncov = np.copy(cov)
	sigma = np.sqrt(np.diag(ncov))
	for i in range(len(ncov)):
		for j in range(len(ncov)):
			ncov[i,j] /= sigma[i]*sigma[j]
	return ncov

def _fit_generic_ev(f, dfdx, x, y, dx, dy, par, cov, absolute_sigma=True, conv_diff=1e-7, max_cycles=5, **kw):
	cycles = 1
	while True:
		if cycles >= max_cycles:
			cycles = -1
			break
		dyeff = np.sqrt(dy**2 + (dfdx(x, *par) * dx)**2)
		rt = curve_fit_patched(f, x, y, p0=par, sigma=dyeff, absolute_sigma=absolute_sigma, **kw)
		npar, ncov = rt[:2]
		error = abs(npar - par) / npar
		cerror = abs(ncov - cov) / ncov
		par = npar
		cov = ncov
		cycles += 1
		if (error < conv_diff).all() and (cerror < conv_diff).all():
			break
	return par, cov, cycles

def _fit_generic_odr(f, dfdx, dfdps, dfdpdxs, x, y, dx, dy, p0):
	dy2 = dy**2
	dx2 = dx**2
	def residual(p):
		return (y - f(x, *p)) / np.sqrt(dy2 + dfdx(x, *p)**2 * dx2)
	rt = np.empty((len(p0), len(x)))
	def jac(p):
		sdfdx = dfdx(x, *p)
		rad = dy2 + sdfdx**2 * dx2
		srad = np.sqrt(rad)
		res = (y - f(x, *p)) * dx2 * sdfdx / srad
		for i in range(len(p)):
			rt[i] = - (dfdps[i](x, *p) * srad + dfdpdxs[i](x, *p) * res) / rad
		return rt
	par, cov, _, _, _ = leastsq(residual, p0, Dfun=jac, col_deriv=True, full_output=True)
	return par, cov

class FitOutput:

	def chi2(self):
		"""compute chisquare. How to in case of non-linearized odr without inverse?
		in many cases an inverse could be given or built"""
		pass

class FitModel:
	
	def __init__(self, f, sym=True, dfdx=None, dfdp=None, dfdpdx=None, invf=None, implicit=False):
		"""if sym=True, use sympy to obtain derivatives from f
		or f is a scipy.odr.Model or a FitModel to copy"""
		if sym:
			args = inspect.getargspec(f).args
			xsym = sympy.symbols('x', real=True)
			psym = [sympy.symbols('p_%d' % i, real=True) for i in range(len(args) - 1)]
			syms = [xsym] + psym
			self._dfdx = sympy.lambdify(syms, f(*syms).diff(xsym), "numpy")
			self._dfdps = [sympy.lambdify(syms, f(*syms).diff(p), "numpy") for p in psym]
			self._dfdpdxs = [sympy.lambdify(syms, f(*syms).diff(xsym).diff(p), "numpy") for p in psym]
			self._f = sympy.lambdify(syms, f(*syms), "numpy")
			self._sym = True
		else:
			self._dfdx = dfdx
			self._dfdp = dfdp
			self._dfdpdx = dfdpdx
			self._f = f
			self._sym = False
	
	# def implicit(self):
	# 	"""return True if the model is implicit (no y)"""
	# 	pass
	#
	def f(self):
		"""return function"""
		return self._f
	
	def f_odrpack(self, length):
		rt = np.empty(length)
		def f_p(B, x):
			rt[:] = self._f(x, *B)
			return rt
		return f_p			
	
	def dfdx(self):
		"""return dfdx function"""
		return self._dfdx
	
	def dfdx_odrpack(self, length):
		"""return dfdx function with return format of scipy.odr's jacd"""
		rt = np.empty(length)
		def f_p(B, x):
			rt[:] = self._dfdx(x, *B)
			return rt
		return f_p
	
	def dfdps(self):
		"""return list of dfdp functions, one for each parameter"""
		return self._dfdps
	
	def dfdp_odrpack(self, length):
		"""return dfdp function with return format of scipy.odr's jacb"""
		rt = np.empty((len(self._dfdps), length))
		def f_p(B, x):
			for i in range(len(self._dfdps)):
				rt[i] = self._dfdps[i](x, *B)
			return rt
		return f_p
	
	def dfdp_curve_fit(self, length):
		rt = np.empty((len(self._dfdps), length))
		def f_p(*args):
			for i in range(len(self._dfdps)):
				rt[i] = self._dfdps[i](*args)
			return rt.T
		return f_p
	
	def dfdpdxs(self):
		return self._dfdpdxs

def fit_generic(f, x, y, dx=None, dy=None, p0=None, pfix=None, absolute_sigma=True, method='odrpack', full_output=False, print_info=False, **kw):
	"""f may be either callable or FitModel"""
	
	if isinstance(f, FitModel):
		model = f
	else:
		model = FitModel(f)
	
	if method == 'odrpack':
		fcn = model.f_odrpack(len(x))
		fjacb = model.dfdp_odrpack(len(x))
		fjacd = model.dfdx_odrpack(len(x))
		M = odr.Model(fcn, fjacb=fjacb, fjacd=fjacd)
		data = odr.RealData(x, y, sx=dx, sy=dy)
		ODR = odr.ODR(data, M, beta0=p0)
		output = ODR.run()
		par = output.beta
		cov = output.cov_beta
		if not absolute_sigma:
			chisq = ((output.eps / dy)**2 + (output.delta / dx)**2).sum() / (len(x) - len(par))
			cov *= chisq
		if print_info:
			output.pprint()
		
	elif method == 'linodr':
		f = model.f()
		dfdps = model.dfdps()
		dfdx = model.dfdx()
		dfdpdxs = model.dfdpdxs()
		par, cov = _fit_generic_odr(f, dfdx, dfdps, dfdpdxs, x, y, dx, dy, p0)
	
	elif method == 'ev':
		f = model.f()
		dfdx = model.dfdx()
		conv_diff = kw.pop('conv_diff', 1e-7)
		max_cycles = kw.pop('max_cycles', 5)
		par, cov = curve_fit_patched(f, x, y, p0=p0, absolute_sigma=absolute_sigma, **kw)
		par, cov, cycles = _fit_generic_ev(f, dfdx, x, y, dx, dy, par, cov, absolute_sigma=absolute_sigma, conv_diff=conv_diff, max_cycles=max_cycles, **kw)
		if cycles == -1:
			raise RuntimeError('Maximum number (%d) of fit cycles reached' % max_cycles)
	
	else:
		raise KeyError(method)
	
	return par, cov

def _fit_affine_odr(x, y, dx, dy):
	dy2 = dy**2
	dx2 = dx**2
	def residual(p):
		return (y - (p[0]*x + p[1])) / np.sqrt(dy2 + (p[0]*dx)**2)
	rt = np.empty((2, len(x)))
	def jac(p):
		rad = dy2 + p[0]**2 * dx2
		srad = np.sqrt(rad)
		res = (y - (p[0]*x + p[1])) * dx2 * p[0] / srad
		rt[0] = - (x * srad + res) / rad
		rt[1] = - 1 / srad
		return rt
	p0, _ = _fit_affine_unif_err(x, y)
	par, cov, _, _, _ = leastsq(residual, p0, Dfun=jac, col_deriv=True, full_output=True)
	return par, cov

def _fit_linear_odr(x, y, dx, dy):
	dy2 = dy**2
	dx2 = dx**2
	def residual(p):
		return (y - p[0]*x) / np.sqrt(dy2 + (p[0]*dx)**2)
	rt = np.empty((1, len(x)))
	def jac(p):
		rad = dy2 + p[0]**2 * dx2
		srad = np.sqrt(rad)
		res = (y - p[0]*x) * dx2 * p[0] / srad
		rt[0] = - (x * srad + res) / rad
		return rt
	p0, _ = _fit_affine_unif_err(x, y)
	par, cov, _, _, _ = leastsq(residual, (p0[0],), Dfun=jac, col_deriv=True, full_output=True)
	return np.array([par[0], 0]), np.array([[cov[0,0], 0], [0, 0]])

def _fit_affine_yerr(x, y, sigmay):
	dy2 = sigmay ** 2
	sy = (y / dy2).sum()
	sx2 = (x ** 2 / dy2).sum()
	sx = (x / dy2).sum()
	sxy = (x * y / dy2).sum()
	s1 = (1 / dy2).sum()
	denom = s1 * sx2 - sx ** 2
	a = (s1 * sxy - sy * sx) / denom
	b = (sy * sx2 - sx * sxy) / denom
	vaa = s1 / denom
	vbb = sx2 / denom
	vab = -sx / denom
	return np.array([a, b]), np.array([[vaa, vab], [vab, vbb]])

def _fit_linear_yerr(x, y, sigmay):
	dy2 = sigmay ** 2
	sx2 = (x ** 2 / dy2).sum()
	sxy = (x * y / dy2).sum()
	a = sxy / sx2
	vaa = 1 / sx2
	return np.array([a, 0]), np.array([[vaa, 0], [0, 0]])

def _fit_affine_unif_err(x, y):
	sy = y.sum()
	sx2 = (x ** 2).sum()
	sx = x.sum()
	sxy = (x * y).sum()
	s1 = len(x)
	denom = len(x) * sx2 - sx ** 2
	a = (s1 * sxy - sy * sx) / denom
	b = (sy * sx2 - sx * sxy) / denom
	vaa = s1 / denom
	vbb = sx2 / denom
	vab = -sx / denom
	return np.array([a, b]), np.array([[vaa, vab], [vab, vbb]])

def _fit_linear_unif_err(x, y):
	sx2 = (x ** 2).sum()
	sxy = (x * y).sum()
	a = sxy / sx2
	vaa = 1 / sx2
	return np.array([a, 0]), np.array([[vaa, 0], [0, 0]])

def _fit_affine_xerr(x, y, dx):
	par, cov = _fit_affine_yerr(y, x, dx)
	m, q = par
	dmm, dmq, _, dqq = cov.flat
	# par = np.array([1/m, -q/m])
	# J = np.array([[-1/m**2, 0], [q/m**2, -1/m]])
	# cov = J.dot(cov).dot(J.T)
	par[0] = 1 / m
	par[1] = -q / m
	m4 = m**4
	cov[0,0] = dmm / m4
	cov[1,1] = (-2 * dmq * m * q + dqq * m**2 + dmm * q**2) / m4
	cov[0,1] = (-dmm * q + dmq * m) / m4
	cov[1,0] = cov[0,1]
	return par, cov

def _fit_linear_xerr(x, y, dx):
	par, cov = _fit_linear_yerr(y, x, dx)
	m = par[0]
	dmm = cov[0,0]
	par[0] = 1 / m
	cov[0,0] = dmm / m**4
	return par, cov

def _fit_affine_ev(fun_fit, x, y, dx, dy, par, cov, absolute_sigma=True, conv_diff=1e-7, max_cycles=5):
	cycles = 1
	while True:
		if cycles >= max_cycles:
			cycles = -1
			break
		dyeff = np.sqrt(dy**2 + (par[0] * dx)**2)
		npar, ncov = fun_fit(x, y, dyeff)
		if not absolute_sigma:
			chisq_rid = (((y - npar[0]*x - npar[1]) / dyeff)**2).sum() / (len(x) - ddof)
			ncov *= chisq_rid
		error = abs(npar - par) / npar
		cerror = abs(ncov - cov) / ncov
		par = npar
		cov = ncov
		cycles += 1
		if (error < conv_diff).all() and (cerror < conv_diff).all():
			break
	return par, cov, cycles

_fit_lin_funcs = [
	[
		_fit_linear_odr,
		_fit_linear_yerr,
		_fit_linear_xerr,
		_fit_linear_unif_err
	],
	[
		_fit_affine_odr,
		_fit_affine_yerr,
		_fit_affine_xerr,
		_fit_affine_unif_err
	]
]

_fit_lin_ddofs = [1, 2]

def fit_linear(x, y, dx=None, dy=None, offset=True, absolute_sigma=True, method='odr', print_info=False, **kw):
	"""
	Fit y = m * x + q
	
	If offset=False, fit y = m * x

	Parameters
	----------
	x : M-length array
		x data
	y : M-length array
		y data
	dx : M-length array or None
		standard deviation of x
	dy : M-length array or None
		standard deviation of y
		If both dx and dy are None, the fit behaves as if absolute_sigma=False,
		dx=0 and dy was uniform. If only one of dx or dy is None, the fit
		behaves as if it is zero.
	offset : bool
		If True, fit y = m + x + q; else fit y = m * x. If False,
		the output format does not change.
	absolute_sigma : bool
		If True, compute standard error on parameters (maximum likelihood
		estimation assuming datapoints are normal). If False, rescale
		errors on parameters to values that would be obtained if the
		chisquare matched the degrees of freedom.
		Simply said: True for physicists, False for engineers
	method : string, one of 'odr', 'ev'
		fit method to use when there are errors on both x and y.
		'odr': use orthogonal distance regression
		'ev': use effective variance
	print_info : bool
		If True, print information about the fit.
	
	Keyword arguments
	-----------------
	When method='ev', the following parameters are meaningful:
	conv_diff : number
		relative difference for convergence
	max_cycles : integer
		The maximum number of fits done. If this maximum is reached, an exception
		is raised.
	
	Returns
	-------
	par:
		estimates (m, q)
	cov:
		covariance matrix m, q
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	offset = bool(offset)
	fitfun = _fit_lin_funcs[offset]
	ddof = _fit_lin_ddofs[offset]
	if not (dy is None):
		if not (dx is None): # dx, dy
			dx = np.asarray(dx)
			dy = np.asarray(dy)
			if method == 'odr':
				par, cov = fitfun[0](x, y, dx, dy)
				if not absolute_sigma:
					chisq_rid = ((y - par[0]*x - par[1])**2 / (dy**2 + (par[0]*dx)**2)).sum() / (len(x) - ddof)
					cov *= chisq_rid
			elif method == 'ev':
				ndy0 = np.asarray(dy != 0)
				if ndy0.sum() > ddof:
					par, cov = fitfun[1](x[ndy0], y[ndy0], dy[ndy0])
					if not absolute_sigma:
						chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - ddof)
						cov *= chisq_rid
				else:
					par, cov = fitfun[3](x, y)
					chisq_rid = (((y - par[0]*x - par[1]))**2).sum() / (len(x) - ddof)
					cov *= chisq_rid
				par, cov, cycles = _fit_affine_ev(fitfun[1], x, y, dx, dy, par, cov, absolute_sigma=absolute_sigma, **kw)
				if cycles == -1:
					raise RuntimeError('Max cycles %d reached' % max_cycles)
				if print_info:
					print("fit_linear: cycles: %d" % (cycles))
			else:
				raise KeyError(method)
		else: # dy
			dy = np.asarray(dy)
			par, cov = fitfun[1](x, y, dy)
			if not absolute_sigma:
				chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - ddof)
				cov *= chisq_rid
	else:
		if not (dx is None): # dx
			dx = np.asarray(dx)
			par, cov = fitfun[2](x, y, dx)
			if not absolute_sigma:
				chisq_rid = (((y - par[0]*x - par[1]) / (par[0]*dx))**2).sum() / (len(x) - ddof)
				cov *= chisq_rid
		else: # no error
			par, cov = fitfun[3](x, y)
			chisq_rid = ((y - par[0]*x - par[1])**2).sum() / (len(x) - ddof)
			cov *= chisq_rid
	return par, cov

def fit_const_yerr(y, sigmay):
	"""
		fit y = a
		
		Parameters
		----------
		y : M-length array
			dependent data
		sigmay : M-length array
			standard deviation of y
		
		Returns
		-------
		a : float
			optimal value for a
		vara : float
			variance of a
	"""
	y = np.asarray(y)
	sigmay = np.asarray(sigmay)
	dy2 = sigmay ** 2
	sy = (y / dy2).sum()
	s1 = (1 / dy2).sum()
	a = sy / s1
	vara = 1 / s1
	return a, vara

# *********************** MULTIMETERS *************************

def _find_scale(x, scales):
	# (!) scales sorted ascending
	for i in range(len(scales)):
		if x < scales[i]:
			return i
	return -1

def _find_scale_idx(scale, scales):
	# (!) scales sorted ascending
	for i in range(len(scales)):
		if scale == scales[i]:
			return i
		elif scale < scales[i]:
			return -1
	return -1

_util_mm_esr_data = dict(
	dm3900=dict(
		desc='multimeter Digimaster DM 3900 plus',
		type='digital',
		voltres=10e+6,
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		cdt=0.2,
		ampere=dict(
			scales=[2 * 10**z for z in range(-5, 2)],
			perc=[2, 0.5, 0.5, 0.5, 1.2, 1.2, 2],
			digit=[5, 1, 1, 1, 1, 1, 5]
		),
		ampere_ac=dict(
			scales=[2 * 10**z for z in range(-5, 2)],
			perc=[3, 1.8, 1, 1, 1.8, 1.8, 3],
			digit=[7, 3, 3, 3, 3, 3, 7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 8)],
			perc=[0.8] * 5 + [1],
			digit=[3, 1, 1, 1, 1, 2]
		)
	),
	lab3=dict(
		desc='multimeter from lab III course',
		type='digital',
		voltres=10e+6,
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		cdt=0.2,
		ampere=dict(
			scales=[2e-3, 20e-3, 0.2, 10],
			perc=[0.8, 0.8, 1.5, 2.0],
			digit=[1, 1, 1, 5]
		),
		ampere_ac=dict(
			scales=[2e-3, 20e-3, 0.2, 10],
			perc=[1, 1, 1.8, 3],
			digit=[3, 3, 3, 7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 9)],
			perc=[0.8] * 5 + [1, 5],
			digit=[3, 1, 1, 1, 1, 2, 10]
		),
		farad=dict(
			scales=[2e-9 * 10**z for z in range(1, 6)],
			perc=[4] * 5,
			digit=[3] * 5
		)
	),
	kdm700=dict(
		desc='multimeter GBC Mod. KDM-700NCV',
		type='digital',
		voltres=10e+6,
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		cdt=0.2,
		ampere=dict(
			scales=[2 * 10**z for z in range(-5, 0)] + [10],
			perc=[2, 0.8, 0.8, 0.8, 1.5, 2],
			digit=[5, 1, 1, 1, 1, 5]
		),
		ampere_ac=dict(
			scales=[2 * 10**z for z in range(-5, 0)] + [10],
			perc=[2, 1, 1, 1, 1.8, 3],
			digit=[5] * 5 + [7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 9)],
			perc=[0.8] * 5 + [1, 5],
			digit=[3, 1, 1, 1, 1, 2, 10]
		)
	),
	ice680=dict(
		desc='multimeter ICE SuperTester 680R VII serie',
		type='analog',
		volt=dict(
			scales=[0.1, 2, 10, 50, 200, 500, 1000],
			relres=[50] * 7, # scale / resolution
			valg=[1] * 7 # guaranteed error / scale * 100
		),
		volt_ac=dict(
			scales=[10, 50, 250, 750],
			relres=[50] * 3 + [37.5],
			valg=[2] * 3 + [100.0 / 37.5]
		),
		ampere=dict(
			scales=[50e-6, 500e-6, 5e-3, 50e-3, 500e-3, 5],
			relres=[50] * 6,
			valg=[1] * 6,
			cdt=[0.1, 0.294, 0.318] + [0.320] * 3
		),
		ampere_ac=dict(
			scales=[250e-6, 2.5e-3, 25e-3, 250e-3, 2.5],
			relres=[50] * 5,
			valg=[2] * 5,
			cdt=[2, 1.5, 1.6, 1.6, 1.9]
		)
	),
	oscil=dict(
		desc='oscilloscope from lab III course',
		type='oscil',
		volt=dict(
			scales=[ (8*d*10**s) for s in range(-3, 1) for d in [1, 2, 5] ],
			perc=[4] * 2 + [3] * 10,
			div=[ (d*10**s) for s in range(-3, 1) for d in [1, 2, 5] ]
		),
		time=dict(
			scales=[5e-09] + [ (10*d*10**s) for s in range(-9, 2) for d in [1, 2.5, 5] ]

		),
		freq=dict(
			scales=[1e9]
		),
		generic=dict(
		)
	)
)

def util_mm_list():
	l = []
	for meter in _util_mm_esr_data:
		l += [(meter, _util_mm_esr_data[meter]['type'], _util_mm_esr_data[meter]['desc'])]
	return l

def util_mm_er(x, scale, metertype='lab3', unit='volt', sqerr=False):
	"""
	Returns the uncertainty of x and the internal resistance of the multimeter.
	
	Parameters
	----------
	x : number
		the value measured, may be negative
	scale : number
		the fullscale used to measure x
	metertype : string
		one of the names returned by lab.util_mm_list()
		the multimeter used
	unit : string
		one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm', 'farad'
		the unit of measure of x
	sqerr : bool
		If True, sum errors squaring.
	
	Returns
	-------
	e : number
		the uncertainty
	r : number or None
		the internal resistance (if applicable)
	
	See also
	--------
	util_mm_esr, util_mm_esr2, mme
	"""
	
	x = abs(x)
	
	errsum = (lambda x, y: math.sqrt(x**2 + y**2)) if sqerr else (lambda x, y: x + y)
	
	meter = _util_mm_esr_data[metertype]
	info = meter[unit]
	typ = meter['type']
	
	s = scale
	idx = _find_scale_idx(s, info['scales'])
	if idx < 0:
		raise KeyError(s)
	r = None
	
	if typ == 'digital':
		e = errsum(x * info['perc'][idx] / 100.0, info['digit'][idx] * 10**(idx + math.log10(info['scales'][0] / 2.0) - 3))
		if unit == 'volt' or unit == 'volt_ac':
			r = info['voltres']
		elif unit == 'ampere' or unit == 'ampere_ac':
			r = info['cdt'] / s
	elif typ == 'analog':
		e = s * errsum(0.5 / info['relres'][idx], info['valg'][idx] / 100.0)
		if unit == 'volt' or unit == 'volt_ac':
			r = 20000 * s
		elif unit == 'ampere' or unit == 'ampere_ac':
			r = info['cdt'][idx] / s
	elif typ == 'oscil':
		e = info['div'][idx] / 25
		r = 10e6
	else:
		raise KeyError(typ)
		
	return e, r

def util_mm_esr(x, metertype='lab3', unit='volt', sqerr=False):
	"""
	determines the fullscale used to measure x with a multimeter,
	supposing the lowest possible fullscale was used, and returns the
	uncertainty, the fullscale and the internal resistance.
	
	Parameters
	----------
	x : number
		the value measured, may be negative
	metertype : string
		one of the names returned by util_mm_list()
		the multimeter used
	unit : string
		one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm', 'farad'
		the unit of measure of x
	sqerr : bool
		If True, sum errors squaring.
	
	Returns
	-------
	e : number
		the uncertainty
	s : number
		the full-scale
	r : number or None
		the internal resistance (if applicable)
	
	See also
	--------
	util_mm_er, util_mm_esr2, mme
	"""
	
	x = abs(x)
	info = _util_mm_esr_data[metertype][unit]
	idx = _find_scale(x, info['scales'])
	if idx < 0:
		raise ValueError("value '%.4g %s' too big for all scales" % (x, unit))
	s = info['scales'][idx]
	e, r = util_mm_er(x, s, metertype=metertype, unit=unit, sqerr=sqerr)
	return e, s, r

_util_mm_esr_vect_error = np.vectorize(lambda x, y, z, t: util_mm_esr(x, metertype=y, unit=z, sqerr=t)[0], otypes=[np.number])
_util_mm_esr_vect_scale = np.vectorize(lambda x, y, z, t: util_mm_esr(x, metertype=y, unit=z, sqerr=t)[1], otypes=[np.number])
_util_mm_esr_vect_res = np.vectorize(lambda x, y, z, t: util_mm_esr(x, metertype=y, unit=z, sqerr=t)[2], otypes=[np.number])
_util_mm_esr2_what = dict(
	error=_util_mm_esr_vect_error,
	scale=_util_mm_esr_vect_scale,
	res=_util_mm_esr_vect_res
)

def util_mm_esr2(x, metertype='lab3', unit='volt', what='error', sqerr=False):
	"""
	Vectorized version of lab.util_mm_esr
	
	Parameters
	----------
	what : string
		one of 'error', 'scale', 'res'
		what to return
	
	Returns
	-------
	z : number
		either the uncertainty, the fullscale or the internal resistance.
	
	See also
	--------
	util_mm_er, util_mm_esr, mme
	"""
	if unit == 'ohm' and what == 'res':
		raise ValueError('asking internal resistance of ohmmeter')
	return _util_mm_esr2_what[what](x, metertype, unit, sqerr)

def mme(x, unit, metertype='lab3', sqerr=False):
	"""
	determines the fullscale used to measure x with a multimeter,
	supposing the lowest possible fullscale was used, and returns the
	uncertainty of the measurement.
	
	Parameters
	----------
	x : (X-shaped array of) number 
		the value measured, may be negative
	unit : (X-shaped array of) string
		one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm', 'farad'
		the unit of measure of x
	metertype : (X-shaped array of) string
		one of the names returned by util_mm_list()
		the multimeter used
	sqerr : bool
		If True, sum errors squaring.
	
	Returns
	-------
	e : (X-shaped array of) number
		the uncertainty
	
	See also
	--------
	util_mm_er, util_mm_esr, util_mm_esr2
	"""
	return util_mm_esr2(x, metertype=metertype, unit=unit, what='error', sqerr=sqerr)

# *********************** FORMATTING *************************

d = lambda x, n: int(("%.*e" % (n - 1, abs(x)))[0])
ap = lambda x, n: float("%.*e" % (n - 1, x))
nd = lambda x: math.floor(math.log10(abs(x))) + 1
def _format_epositive(x, e, errsep=True, minexp=3):
	# DECIDE NUMBER OF DIGITS
	if d(e, 2) < 3:
		n = 2
		e = ap(e, 2)
	elif d(e, 1) < 3:
		n = 2
		e = ap(e, 1)
	else:
		n = 1
	# FORMAT MANTISSAS
	dn = int(nd(x) - nd(e)) if x != 0 else -n
	nx = n + dn
	if nx > 0:
		ex = nd(x) - 1
		if nx > ex and abs(ex) <= minexp:
			xd = nx - ex - 1
			ex = 0
		else:
			xd = nx - 1
		sx = "%.*f" % (xd, x / 10**ex)
		se = "%.*f" % (xd, e / 10**ex)
	else:
		le = nd(e)
		ex = le - n
		sx = '0'
		se = "%#.*g" % (n, e)
	# RETURN
	if errsep:
		return sx, se, ex
	return sx + '(' + ("%#.*g" % (n, e * 10 ** (n - nd(e))))[:n] + ')', '', ex

def util_format(x, e, pm=None, percent=False, comexp=True, nicexp=False):
	"""
	format a value with its uncertainty
	
	Parameters
	----------
	x : number (or something understood by float(), ex. string representing number)
		the value
	e : number (or as above)
		the uncertainty
	pm : string, optional
		The "plusminus" symbol. If None, use compact notation.
	percent : bool
		if True, also format the relative error as percentage
	comexp : bool
		if True, write the exponent once.
	nicexp : bool
		if True, format exponent like ×10¹²³
	
	Returns
	-------
	s : string
		the formatted value with uncertainty
	
	Examples
	--------
	util_format(123, 4) --> '123(4)'
	util_format(10, .99) --> '10.0(10)'
	util_format(1e8, 2.5e6) --> '1.000(25)e+8'
	util_format(1e8, 2.5e6, pm='+-') --> '(1.000 +- 0.025)e+8'
	util_format(1e8, 2.5e6, pm='+-', comexp=False) --> '1.000e+8 +- 0.025e+8'
	util_format(1e8, 2.5e6, percent=True) --> '1.000(25)e+8 (2.5 %)'
	util_format(nan, nan) = 'nan +- nan'
	
	See also
	--------
	xe, xep
	"""
	x = float(x)
	e = abs(float(e))
	if not math.isfinite(x) or not math.isfinite(e) or e == 0:
		return "%.3g %s %.3g" % (x, '+-', e)
	sx, se, ex = _format_epositive(x, e, not (pm is None))
	if ex == 0:
		es = ''
	elif nicexp:
		es = "×10" + num2sup(ex, format='%d')
	else:
		es = "e%+d" % ex
	if pm is None:
		s = sx + es
	elif comexp and es != '':
		s = '(' + sx + ' ' + pm + ' ' + se + ')' + es
	else:
		s = sx + es + ' ' + pm + ' ' + se + es
	if (not percent) or sx == '0':
		return s
	pe = e / x * 100.0
	return s + " (%.*g %%)" % (2 if pe < 100.0 else 3, pe)

_util_format_vect = np.vectorize(util_format, otypes=[str])

def xe(x, e, pm=None, comexp=True, nicexp=False):
	"""
	Vectorized version of util_format with percent=False,
	see lab.util_format and numpy.vectorize.
	
	Example
	-------
	xe(['1e7', 2e7], 33e4) --> ['1.00(3)e+7', '2.00(3)e+7']
	xe(10, 0.8, pm=unicode_pm) --> '10.0 ± 0.8'
	
	See also
	--------
	xep, num2si, util_format
	"""
	return _util_format_vect(x, e, pm, False, comexp, nicexp)

def xep(x, e, pm=None, comexp=True, nicexp=False):
	"""
	Vectorized version of util_format with percent=True,
	see lab.util_format and numpy.vectorize.
	
	Example
	-------
	xep(['1e7', 2e7], 33e4) --> ['1.00(3)e+7 (3.3 %)', '2.00(3)e+7 (1.7 %)']
	xep(10, 0.8, pm=unicode_pm) --> '10.0 ± 0.8 (8 %)'
	
	See also
	--------
	xe, num2si, util_format
	"""
	return _util_format_vect(x, e, pm, True, comexp, nicexp)

unicode_pm = u'±'

# this function taken from stackoverflow and modified
# http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
def num2si(x, format='%.15g', si=True, space=' '):
	"""
	Returns x formatted using an exponent that is a multiple of 3.
	
	Parameters
	----------
	x : number
		the number to format
	format : string
		printf-style string used to format the mantissa
	si : boolean
		if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
		e-9 etc. If the exponent would be greater than 24, numerical exponent is
		used anyway.
	space : string
		string interposed between the mantissa and the exponent
	
	Returns
	-------
	fx : string
		the formatted value
	
	Example
	-------
	     x     | num2si(x)
	-----------|----------
	   1.23e-8 |  12.3 n
	       123 |  123
	    1230.0 |  1.23 k
	-1230000.0 |  -1.23 M
	         0 |  0
	
	See also
	--------
	util_format, xe, xep
	"""
	x = float(x)
	if x == 0:
		return format % x + space
	exp = int(math.floor(math.log10(abs(x))))
	exp3 = exp - (exp % 3)
	x3 = x / (10 ** exp3)
	
	if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
		exp3_text = space + 'yzafpnμm kMGTPEZY'[(exp3 - (-24)) // 3]
	elif exp3 == 0:
		exp3_text = space
	else:
		exp3_text = 'e%s' % exp3 + space
	
	return (format + '%s') % (x3, exp3_text)

_subscr  = '₀₁₂₃₄₅₆₇₈₉₊₋ₑ․'
_subscrc = '0123456789+-e.'
_supscr  = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵉ·'

def num2sub(x, format=None):
	"""
	Format a number as subscript.
	
	Parameters
	----------
	x : string or number
		The number to format.
	format : None or string
		If None, x is interpreted as string and formatted subscript as-is.
		If string, it is a %-format used to format x before converting to subscript.
	
	Returns
	-------
	s : string
		x written in subscript.
	"""
	if format is None:
		x = str(x)
	else:
		x = format % float(x)
	for i in range(len(_subscrc)):
		x = x.replace(_subscrc[i], _subscr[i])
	return x

def num2sup(x, format=None):
	"""
	Format a number as superscript.
	
	Parameters
	----------
	x : string or number
		The number to format.
	format : None or string
		If None, x is interpreted as string and formatted superscript as-is.
		If string, it is a %-format used to format x before converting to superscript.
	
	Returns
	-------
	s : string
		x written in superscript.
	"""
	if format is None:
		x = str(x)
	else:
		x = format % float(x)
	for i in range(len(_subscrc)):
		x = x.replace(_subscrc[i], _supscr[i])
	return x

# ************************** TIME *********************************

def util_timecomp(secs):
	"""
		convert a time interval in seconds to hours, minutes, seconds
		
		Parameters
		----------
		secs : number
			the time interval expressed in seconds
		
		Returns
		-------
		hours : int
			hours, NOT bounded to 24
		minutes : int
			minutes, 0--59
		seconds : int
			seconds, 0--59
	
		See also
		--------
		util_timestr
	"""
	hours = int(secs / 3600)
	minutes = int((secs - hours * 3600) / 60)
	seconds = secs - hours * 3600 - minutes * 60
	return hours, minutes, seconds

def util_timestr(secs):
	"""
		convert a time interval in seconds to a string with hours, minutes, seconds
		
		Parameters
		----------
		secs : number
			the time interval expressed in seconds
		
		Returns
		-------
		str : str
			string representing the interval
	
		See also
		--------
		util_timecomp
	"""
	return "%02d:%02d:%02d" % util_timecomp(secs)

_eta_start = 0

def etastart():
	"""
	Call at the startpoint of something you want to compute the eta (estimated
	time of arrival) of.
	
	Returns
	-------
	An object containing the starting time, to be given as argument to etastr().
	
	Example
	-------
	>>> eta = etastart()
	>>> for i in range(N):
	>>>     print('elapsed time: %s, remaining time: %s' % etastr(eta, i / N))
	>>>     # do something
	
	See also
	--------
	etastr
	"""
	now = time.time()
	return [now, now]

def etastr(eta, progress, mininterval=np.inf):
	"""
	Compute the eta given a startpoint returned from etastart() and the progress.
	
	Parameters
	----------
	eta :
		object returned by etastart()
	progress : number in [0,1]
		the progress on a time-linear scale where 0 means still nothing done and
		1 means finished.
	
	Returns
	-------
	timestr : string
		elapsed time
	etastr : string
		estimated time remaining
	
	Example
	-------
	>>> eta = etastart()
	>>> for i in range(N):
	>>>     print('elapsed time: %s, remaining time: %s' % etastr(eta, i / N))
	>>>     # do something
	
	See also
	--------
	etastart
	"""
	now = time.time()
	interval = now - eta[0]
	if 0 < progress <= 1:
		etastr = util_timestr((1 - progress) * interval / progress)
	elif progress == 0:
		etastr = "--:--:--"
	else:
		raise RuntimeError("progress %.2f out of bounds [0,1]" % progress)
	timestr = util_timestr(interval)
	if now - eta[1] >= mininterval:
		print('elapsed time: %s, remaining time: %s' % (timestr, etastr))
		eta[1] = now
	return timestr, etastr

# *************************** FILES *******************************

def sanitizefilename(name, windows=True):
	"""
	Removes characters not allowed by the filesystem, replacing
	them with similar unicode characters.
	
	Parameters
	----------
	name : string
		The file name to sanitize. It can not be a path, since slashes are
		replaced.
	windows : bool
		If True, also replace characters not allowed in Windows.
	
	Return
	------
	filename : string
		The sanitized file name.
	"""
	name = name.replace('/', '∕').replace('\0', '')
	if windows:
		name = name.replace('\\', '⧵').replace(':', '﹕')
	return name

def nextfilename(base, ext, idxfmt='%02d', prepath=None, start=1, sanitize=True):
	"""
	Consider the following format:
		<base><index><ext>
	This functions search for the pattern with the lowest index that is not
	the path of an existing file.
	
	Parameters
	----------
	base : string
		Tipically the name of the file, without extension.
	ext : string
		Tipically the file type extension (with dot).
	idxfmt : string
		The %-format used to format the index.
	prepath : None or string
		A path that is prepended to <base> with a slash:
			<prepath>/<base>...
	start : number
		The index to start with.
	sanitize : bool
		If True, process <base> and <ext> with sanitizefilename. In this case,
		<base> should not contain a path since slashes are replaced. Use
		<prepath> instead.
	
	Returns
	-------
	filename : string
		File name of non-existing file.
	"""
	if sanitize:
		base = sanitizefilename(base)
		ext = sanitizefilename(ext)
	i = start
	while True:
		filename = ('%s%s-' + idxfmt + '%s') % ((prepath + '/') if prepath != None else '', base, i, ext)
		if not os.path.exists(filename):
			break
		i += 1
	return filename

# ************************ COMPATIBILITY ****************************

def fit_generic_xyerr(f, dfdx, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True, conv_diff=0.001, max_cycles=5, **kw):
	"""
	THIS FUNCTION IS DEPRECATED
	"""
	model = FitModel(f, dfdx=dfdx, sym=False)
	return fit_generic(model, x, y, dx=sigmax, dy=sigmay, p0=p0, absolute_sigma=absolute_sigma, print_info=print_info, method='ev', conv_diff=conv_diff, max_cycles=max_cycles, **kw)

def fit_generic_xyerr2(f, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True):
	"""
	THIS FUNCTION IS DEPRECATED
	"""
	model = FitModel(f, sym=False)
	return fit_generic(model, x, y, dx=sigmax, dy=sigmay, p0=p0, absolute_sigma=absolute_sigma, print_info=print_info, method='odrpack')