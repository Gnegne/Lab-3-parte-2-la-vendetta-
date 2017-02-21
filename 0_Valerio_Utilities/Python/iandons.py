
from itertools import count as itercount
import numpy as np
import scipy.stats.distributions as dists

# *********************** UTILITIES *************************


def parallel(*Zn):
	"""Calculate impedance parallel."""
	invZ = 0
	for Zi in Zn:
		if np.abs(Zi) < np.inf:
			invZ += 1/Zi
	return 1/invZ


def lineheight(linepars, linecov, height, hvar=0):
	"""Find intersection (with variance) of straight line and constant height."""
	m, q = linepars
	h = height
	x = (h - q) / m
	derivs = [(q - h)/m**2, -1/m, 1/m]
	covmat = np.zeros((3, 3))
	covmat[:2, :2] = linecov
	covmat[2, 2] = hvar
	varx = derivs @ covmat @ derivs
	return x, varx


def tell_chi2(resd, dofs, style='normal'):
	"""
	Chi^2 prettyprinter.

	Calculate chi squared from normalized redisuals and return
	a string telling it along with p-value and degrees of freedom.

	Parameters
	----------
	resd: (iterable of) number
		the normalised residuals of a fitted dataset
	dofs: unsigned integer
		the number of degrees of freedom of the fit
	style: stirng, optional (default 'normal')
		if 'latex', the string will be formatted with LaTeX-like symbols macro

	Returns
	-------
	chi2msg: string
		the message printed
	"""
	chi2 = np.sum(resd**2)
	pval = dists.chi2.sf(chi2, dofs)
	if style == 'normal':
		chiname = r"ChiSquare"
		dofname = r"DoFs,"
	elif style == 'latex':
		chiname = r"\chi^2"
		dofname = r"\dof , \ "
	chi2msg = "{0} = {1:.2f} ({2} {3} p = {4:.4f})"
	return chi2msg.format(chiname, chi2, dofs, dofname, pval)


def maketab(*columns, errors='all', precision=3):
	"""
	Print data in tabular form.

	Creates a LaTeX (table environment with S columns from siunitx) formatted
	table using inputs as columns; the input is assumed to be numerical.
	Errors can be given as columns following the relevant numbers column
	and their positions specified in a tuple given as the kwarg 'errors';
	they will be nicely formatted for siunitx use.
	The precision (number of signifitant digits) of the numbers representation
	will be inferred from errors, or when absent from the 'precision' kwarg.

	Parameters
	----------
	*columns: N iterables of lenght L with numerical items
		the lists of numbers that will make up the columns of the table,
		must be of uniform lenght.
	errors: tuple(-like) of ints, 'all' or 'none', optional (default 'all')
		the columns with positions corresponging to errors items will
		be considered errors corresponding to the previous column and formatted
		accordingly; if 'all' every other column will be considered an error,
		if 'none' no column will be considered an error.
	precision: (lenght-L tuple of) int, optional (default 3)
		number of significant digits to be used when error is not given;
		if a tuple(-like) is given, each column will use the correspondingly
		indexed precision.

	Returns
	-------
	tab: string
		the formatted text constituting the LaTeX table.
	"""
	vals = np.asarray(columns).T
	cols = np.alen(vals.T)
	precision = np.array(precision) * np.ones(cols)
	if errors == 'all':
		errors = range(1, cols, 2)
	if errors == 'none':
		errors = []
	beginning = (
		r"\begin{table}" "\n\t"
		r"\begin{tabular}{*{"
		+ str(cols - len(errors)) +
		r"}{S}}"
		"\n\t\t"
		r"\midrule" "\n"
	)
	inner = ""
	for i, row in enumerate(vals):
		rows = enumerate(row, start=1)
		inner += "\t"
		for pos, v in rows:
			num = v if np.isfinite(v) else "{-}"
			space = "&" if pos < cols else "\\\\ \n"
			err = ""
			prec = -precision[pos-1]
			if pos in errors:
				prec = np.floor(np.log10(row[pos]))
				if row[pos]/10**prec < 2.5:
					prec -= 1
				err = "({0:.0f})".format(round(row[pos]/10**prec))
				if next(rows, (-1, None))[0] >= cols:
					space = "\\\\ \n"
				num = round(num, int(-prec))
			inner += "\t{0:.{digits:.0f}f} {1}\t{2}".format(num, err, space, digits=max(0, -prec))
	ending = (
		"\t" r"\end{tabular}" "\n"
		"\t" r"\caption{some caption}" "\n"
		r"\label{t:somelabel}" "\n"
		r"\end{table}"
	)
	return beginning + inner + ending

# *********************** FACTORIES *************************


_dline = np.vectorize(lambda x, m, q: m)
_dlogline = np.vectorize(lambda x, m, q: m*np.log10(np.e)/x)
_nullfunc = np.vectorize(lambda *args: 0)
_const = np.vectorize(lambda x, q: q)


def createline(type='linear', name=""):
	"""
	Factory of linear(-like) functions.

	Parameters
	----------
	type: str, optional (default 'linear')
		if 'log', returns a function linear in log10(x).
	name: str, optional
		if provided, the function returned will have this name,
		otherwise it will be named "line" (or "logline" if type is 'log').

	Returns
	-------
	func: callable
		func(x, m, q) = mx + q (or m*log10(x) + q if type is 'log').
	"""
	if type == 'log':
		def logline(x, m, q):
			"""f: (x, m, q) --> m*log10(x) + q ."""
			return m*np.log10(x) + q
		logline.deriv = _dlogline
		func = logline
	elif type == 'linear':
		def line(x, m, q):
			"""f: (x, m, q) --> m*x + q ."""
			return m*x + q
		line.deriv = _dline
		func = line
	elif type == 'const':
		def flatline(x, q):
			"""f: (x, q) --> q ."""
			return q*np.ones(len(x))
		flatline.deriv = _nullfunc
		func = flatline
	if name:
		func.__name__ = name
	return func

# *********************** OBJECTS *************************


class NakedObject(object):
	"""Simple container of stuff."""
	pass


class DataHolder(object):

	def __init__(self, name=None):
		pass
