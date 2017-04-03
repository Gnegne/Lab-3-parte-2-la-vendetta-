# ********************** IMPORTS ***************************

# generic

import numpy as np
from numpy import array, asarray, isfinite, sqrt, diag, vectorize, number, isscalar
import math
from math import log10, fsum, floor
import inspect
import time
from matplotlib import gridspec, pyplot

# scipy

from scipy import odr
from scipy.optimize import curve_fit
from scipy.stats import chisqprob
#import scipy.stats
#from scipy.stats import distributions

# lab flavour
from pylab import *
#from pylab import loadtxt, transpose, matrix, zeros, figure, title, xlabel, ylabel, xscale, yscale, grid, errorbar, savefig, plot, clf, logspace, linspace, legend, rc

from uncertainties import unumpy, ufloat
import uncertainties
#from lab import mme, fit_generic_xyerr2, xep, xe

__all__ = [ # things imported when you do "from lab import *"
    'xep',
    'xe',
    'mme',
    'load_data',
    'errors',
    'plot_fit',
    'chi2_calc',
    'pretty_print_chi2',
    'latex_table',
    'fit',
    'fast_plot',
    'umme',
    'mme'

]

__version__ = 'Bob.0'

# ************************** FIT ***************************

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
    ncov /= np.outer(sigma, sigma)
    return ncov

def fit_generic_xyerr2(f, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True):
    """
        fit y = f(x, *params)

        Parameters
        ----------
        f : callable
            the function to fit
        x : M-length array
            independent data
        y : M-length array
            dependent data
        sigmax : M-length array
            standard deviation of x
        sigmay : M-length array
            standard deviation of y
        p0 : N-length sequence
            initial guess for parameters
        print_info : bool, optional
            If True, print information about the fit
        absolute_sigma : bool, optional
            If False, compute asymptotic errors, else standard errors for parameters

        Returns
        -------
        par : N-length array
            optimal values for parameters
        cov : (N,N)-shaped array
            covariance matrix of par

        Notes
        -----
        This is a wrapper of scipy.odr
    """
    f_wrap = lambda params, x: f(x, *params)
    model = odr.Model(f_wrap)
    data = odr.RealData(x, y, sx=sigmax, sy=sigmay)
    Odr = odr.ODR(data, model, beta0=p0)
    output = Odr.run()
    par = output.beta
    cov = output.cov_beta
    if print_info:
        output.pprint()
    if (not absolute_sigma) and len(y) > len(p0):
        s_sq = sum(((np.asarray(y) - f(x, *par)) / (np.asarray(sigmay))) ** 2) / (len(y) - len(p0))
        cov *= s_sq
    return par, cov



# *********************** MULTIMETERS *****************************

def _find_scale(x, scales):
    # (!) scales sorted ascending
    """
        Explore the scales list for automatic find scale of the value x.
    """
    for i in range(len(scales)):
        if x < scales[i]:
            return i
    return -1

def _find_scale_idx(scale, scales):
    # (!) scales sorted ascending
    """
        Find the index of the scale given in the scales list.
    """
    for i in range(len(scales)):
        if scale == scales[i]:
            return i
        elif scale < scales[i]:
            return -1
    return -1

_util_mm_esr_data = dict(
    dm3900 = dict(
        desc = 'multimeter Digimaster DM 3900 plus',
        type = 'digital',
        volt = dict(
            scales = [0.2, 2, 20, 200, 1000],
            perc = [0.5] * 4 + [0.8],
            digit = [1, 1, 1, 1, 2]
        ),
        volt_ac = dict(
            scales = [0.2, 2, 20, 200, 700],
            perc = [1.2, 0.8, 0.8, 0.8, 1.2],
            digit = [3] * 5
        ),
        ampere = dict(
            scales = [2 * 10**z for z in range(-5, 2)],
            perc = [2, 0.5, 0.5, 0.5, 1.2, 1.2, 2],
            digit = [5, 1, 1, 1, 1, 1, 5]
        ),
        ampere_ac = dict(
            scales = [2 * 10**z for z in range(-5, 2)],
            perc = [3, 1.8, 1, 1, 1.8, 1.8, 3],
            digit = [7, 3, 3, 3, 3, 3, 7]
        ),
        ohm = dict(
            scales = [2 * 10**z for z in range(2, 8)],
            perc = [0.8] * 5 + [1],
            digit = [3, 1, 1, 1, 1, 2]
        )
    ),
    dig = dict(
        desc = 'multimeter from lab III course',
        type = 'digital',
        volt = dict(
            scales = [0.2, 2, 20, 200, 1000],
            perc = [0.5] * 4 + [0.8],
            digit = [1, 1, 1, 1, 2]
        ),
        volt_ac = dict(
            scales = [0.2, 2, 20, 200, 700],
            perc = [1.2, 0.8, 0.8, 0.8, 1.2],
            digit = [3] * 5
        ),
        ampere = dict(
            scales = [2e-3, 20e-3, 0.2, 10],
            perc = [0.8, 0.8, 1.5, 2.0],
            digit = [1, 1, 1, 5]
        ),
        ampere_ac = dict(
            scales = [2e-3, 20e-3, 0.2, 10],
            perc = [1, 1, 1.8, 3],
            digit = [3, 3, 3, 7]
        ),
        ohm = dict(
            scales = [2 * 10**z for z in range(2, 9)],
            perc = [0.8] * 5 + [1, 5],
            digit = [3, 1, 1, 1, 1, 2, 10]
        ),
        farad = dict(
            scales = [2e-9 * 10**z for z in range(1, 6)],
            perc = [4] * 5,
            digit = [3] * 5
        )
    ),
    kdm700 = dict(
        desc = 'multimeter GBC Mod. KDM-700NCV',
        type = 'digital',
        volt = dict(
            scales = [0.2, 2, 20, 200, 1000],
            perc = [0.5] * 4 + [0.8],
            digit = [1, 1, 1, 1, 2]
        ),
        volt_ac = dict(
            scales = [0.2, 2, 20, 200, 700],
            perc = [1.2, 0.8, 0.8, 0.8, 1.2],
            digit = [3] * 5
        ),
        ampere = dict(
            scales = [2 * 10**z for z in range(-5, 0)] + [10],
            perc = [2, 0.8, 0.8, 0.8, 1.5, 2],
            digit = [5, 1, 1, 1, 1, 5]
        ),
        ampere_ac = dict(
            scales = [2 * 10**z for z in range(-5, 0)] + [10],
            perc = [2, 1, 1, 1, 1.8, 3],
            digit = [5] * 5 + [7]
        ),
        ohm = dict(
            scales = [2 * 10**z for z in range(2, 9)],
            perc = [0.8] * 5 + [1, 5],
            digit = [3, 1, 1, 1, 1, 2, 10]
        )
    ),
    ice680 = dict(
        desc = 'multimeter ICE SuperTester 680R VII serie',
        type = 'analog',
        volt = dict(
            scales = [0.1, 2, 10, 50, 200, 500, 1000],
            relres = [50] * 7,
            valg = [1] * 7
        ),
        volt_ac = dict(
            scales = [10, 50, 250, 750],
            relres = [50] * 3 + [37.5],
            valg = [2] * 3 + [100.0 / 37.5]
        ),
        ampere = dict(
            scales = [50e-6, 500e-6, 5e-3, 50e-3, 500e-3, 5],
            relres = [50] * 6,
            valg = [1] * 6,
            cdt = [0.1, 0.294, 0.318] + [0.320] * 3
        ),
        ampere_ac = dict(
            scales = [250e-6, 2.5e-3, 25e-3, 250e-3, 2.5],
            relres = [50] * 5,
            valg = [2] * 5,
            cdt = [2, 1.5, 1.6, 1.6, 1.9]
        )
    ),
    osc = dict(
        desc = 'oscilloscope from lab III course',
        type = 'osc',
        volt = dict(
            scales  =  [8*2e-3] + [8*5e-3] + [(8*d*10**s) for s in range(-2, 1) for d in [1, 2, 5]],
            perc  =  [4]*2 + [3]*9,
            div  =  [1e-3]*2 + [(d*10**s) for s in range(-2, 1) for d in [1, 2, 5]]
        ),
        volt_ar = dict(
            scales = [4*2e-3] + [4*5e-3] + [(4*d*10**s) for s in range(-2, 1) for d in [1, 2, 5]],
            perc = [4]*2 + [3]*9,
            div = [1e-3]*2 + [(d*10**s) for s in range(-2, 1) for d in [1, 2, 5]]
        ),
        volt_nc = dict(
            scales = [8*2e-3] + [8*5e-3] + [(8*d*10**s) for s in range(-2, 1) for d in [1, 2, 5]],
            perc = [0]*11,
            div = [1e-3]*2 + [(1*d*10**s) for s in range(-2, 1) for d in [1, 2, 5]]
        ),
        volt_ar_nc = dict(
            scales = [4*2e-3] + [4*5e-3] + [(4*d*10**s) for s in range(-2, 1) for d in [1, 2, 5]],
            perc = [0]*11,
            div = [1e-3]*2 + [(1*d*10**s) for s in range(-2, 1) for d in [1, 2, 5]]
        ),
        time = dict(
            scales=[5*10e-09] + [ (10*d*10**s) for s in range(-8, 2) for d in [1, 2.5, 5] ],
            perc=[100e-6]*34,
            div=[1e-09] + [ (1*d*10**s) for s in range(-8, 2) for d in [1, 2.5, 5] ]  
        ),
        freq = dict(
            scales = [1e9], 
            perc = [1]*11,
            div = [0]*11
        )
    )
)

def util_mm_list():
    l = []
    for meter in _util_mm_esr_data:
        l += [(meter, _util_mm_esr_data[meter]['type'], _util_mm_esr_data[meter]['desc'])]
    return l

def util_mm_er(x, scale, metertype='dig', unit='volt', sqerr=True):
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
            r = 10e+6
        elif unit == 'ampere' or unit == 'ampere_ac':
            r = 0.2 / s
    elif typ == 'analog':
        e = x * errsum(0.5 / info['relres'][idx], info['valg'][idx] / 100.0 * s)
        if unit == 'volt' or unit == 'volt_ac':
            r = 20000 * s
        elif unit == 'ampere' or unit == 'ampere_ac':
            r = info['cdt'][idx] / s
    elif typ == 'osc':
        e = errsum(x*info['perc'][idx]/100.0, info['div'][idx] / 25)
        r = 10e6
        if unit=='time':
            e=errsum(0.5e-9,e)
    else:
        raise KeyError(typ)

    return e, r

def util_mm_esr(x, metertype='dig', unit='volt', sqerr=True):
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

def util_mm_esr2(x, metertype='dig', unit='volt', what='error', sqerr=True):
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

def util_format(x, e, pm=None, percent=False, comexp=True):
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
    es = "e%+d" % ex if ex != 0 else ''
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


# this function taken from stackoverflow and modified
# http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
def num2si(x, format='%.16g', si=True, space=' '):
    """
    Returns x formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

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
    util_format, util_format_comp, xe, xep
    """
    x = float(x)
    if x == 0:
        return format % x
    exp = int(math.floor(math.log10(abs(x))))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[(exp3 - (-24)) // 3]
    elif exp3 == 0:
        exp3_text = ''
        space = ''
    else:
        exp3_text = 'e%s' % exp3

    return (format + '%s%s') % (x3, space, exp3_text)

# ************************ SHORTCUTS ******************************

def mme(x, unit, metertype='dig', sqerr=True):
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

_util_format_vect = np.vectorize(util_format, otypes=[str])

unicode_pm = u'±'

def xe(x, e, pm=None, comexp=True):
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
    return _util_format_vect(x, e, pm, False, comexp)

def xep(x, e, pm=None, comexp=True):
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
    return _util_format_vect(x, e, pm, True, comexp)

# ********************** LAB3_DEVELOPMENTS ************************

def _XYfunction(a): # default for the x-y columns from the file entries
    return a[0], a[1]

def load_data(directory,file_):
    # load the data matrix from the data file 
    
    data = loadtxt(directory + "data/" + file_ + ".txt", unpack = True)    
    if type(data[0]) is np.float64:    # check if the first column is a column 
        data=array(transpose(matrix(data)))

    return data

def errors(data, units, XYfun):
    # performs the error calculation, using mme

    # calculate data error with mme
    data_err = zeros((len(data),len(data[0])))
    for i in range(len(data)):
        data_err[i]=mme(data[i],*units[i])
    
    # extract from data x,y values with errors
    entries = unumpy.uarray(data,data_err)
    
    X_err = XYfun(entries)[0]
    Y_err = XYfun(entries)[1]
    
    X=unumpy.nominal_values(X_err)
    Y=unumpy.nominal_values(Y_err)
    dX=unumpy.std_devs(X_err)
    dY=unumpy.std_devs(Y_err)

    return X, Y, dX, dY, data_err

def _preplot(directory, file_, X, Y, dX, dY, title_="", fig="^^", 
             Xscale="linear", Yscale="linear", Xlab="", Ylab=""):
    # print a raw plot of the data, see fast_plot for the public function

    figure(fig+"_2")
    if (fig == file_):
        clf()
    title(title_)
    xlabel(Xlab)
    ylabel(Ylab)
    if Xscale=="log":
        xscale("log")
    if Yscale=="log":
        yscale("log")
    grid(b=True)
    errorbar(X,Y,dY,dX, fmt=",",ecolor="black",capsize=0.1, color="black")
    savefig(directory+"Figs-Tabs/fast_plot_"+fig+".pdf")
    savefig(directory+"Figs-Tabs/fast_plot_"+fig+".png")
    show()
    
def _outlier_(directory, file_, units, X, XYfun):
    # mark the outlier on the data plot, read them from a specific file

    data_ol = load_data(directory,file_+"_ol")
    X_ol, Y_ol, dX_ol, dY_ol, data_ol_err = errors(data_ol, units, XYfun)

    smin=min(min(X_ol),min(X))
    smax=max(max(X_ol),max(X))

    return X_ol, Y_ol, dX_ol, dY_ol, smin, smax

def _residuals(fig, gne, gs, ax1, f, par, out, X, dX, Xlab, Xscale, Y, dY, kx, X_ol=[], Y_ol=[], dY_ol=[], dX_ol=[]):
    # performs the calculation of residuals and plot it in a fashion way

    figure(fig+"_1")

    #subplot(212)
    ax2 = gne.add_subplot(gs[3,:], sharex=ax1)
    rc('ytick', labelsize=12)
    #title("Scarti normalizzati")
    xlabel(Xlab) #
    ylabel("Scarti")
    if Xscale=="log":
        xscale("log")
    grid(b=True)
    df = (f(X+dX/1e6,*par)-f(X,*par)) /(dX/1e6)
    plot(X*kx, (Y-f(X,*par))/sqrt(dY**2+(df*dX)**2), ".", color="black")
    
    #if out ==True:
    #   df_ol = (f(X_ol+dX_ol/1e6,*par)-f(X_ol,*par)) /(dX_ol/1e6)
    #   plot(kx*X_ol, (Y_ol-f(X_ol,*par))/sqrt(dY_ol**2+(df_ol*dX_ol)**2), "x", color="green")
    
    show()
    
def plot_fit(directory, file_, title_, units, f, par, X, Y, dX, dY, kx, ky,
             out=False, fig="^^", residuals=False,
             xlimp=[100,100], XYfun=_XYfunction,
             Xscale="linear", Yscale="linear", Xlab="", Ylab=""):
    """
        Parameters
        ----------    

        Returns
        -------

    """    
    if (fig == "^^"):
        fig == file_

    gs = gridspec.GridSpec(4, 1)
    gne = figure(fig+"_1")
    if (fig == file_):
        clf()
    if residuals==True:
        ax1 = gne.add_subplot(gs[:-1,:])
        setp(ax1.get_xticklabels(), visible=False)
        
        #subplot(211)
    title(title_)
    if Xscale=="log":
        xscale("log")
    if Yscale=="log":
        yscale("log")

    if residuals==False :
        xlabel(Xlab)
    ylabel(Ylab)
    xlima = array(xlimp)/100

    if out ==True:
        X_ol, Y_ol, dX_ol, dY_ol, smin, smax = _outlier_(directory, file_, units, X, XYfun)
        
    else:
        smin = min(X)
        smax = max(X)
        
    if Xscale=="log":
        l=logspace(log10(smin)*xlima[0],log10(smax*xlima[1]),1000)
    else:
        l=linspace(smin*xlima[0],smax*xlima[1],1000)
    grid(b=True)
    graph_fit = plot(l*kx,f(l,*par)*ky,"red",label="fit")
    
    graph_data = errorbar(X*kx,Y*ky,dY*ky,dX*kx, fmt=",",ecolor="black",capsize=0.1, color="black",label="data")
    
    if out==True:
        outlier = errorbar(X_ol*kx,Y_ol*ky,dY_ol*ky,dX_ol*kx, fmt="gx",ecolor="black",capsize=0.1, color="black",label="outlier")
        legend([graph_data, outlier], ["data","outlier"],loc="best")
    else:
        legend([graph_data],["data"], loc="best")
    if residuals==True:
        if out==True:
            _residuals(fig, gne, gs, ax1, f, par, out, X, dX, Xlab, Xscale, Y, dY, kx, X_ol, Y_ol, dY_ol, dX_ol)
        _residuals(fig, gne, gs, ax1, f, par, out, X, dX, Xlab, Xscale, Y, dY, kx)
    
    savefig(directory+"Figs-Tabs/fit_"+fig+".pdf")
    savefig(directory+"Figs-Tabs/fit_"+fig+".png")
    show()

def chi2_calc(f, par, X, Y, dY, dX, cov):
    """
        Parameters
        ----------    

        Returns
        -------

    """  
    df = (f(X+dX/1e6,*par)-f(X,*par)) /(dX/1e6)
    chi = sum((Y-f(X,*par))**2/(dY**2+(df*dX)**2))
    
    p = chisqprob(chi, len(X)-len(par))
    sigma = sqrt(diag(cov))
    
    normcov = zeros((len(par),len(par)))
    
    for i in range(len(par)):
        for j in range(len(par)):
            normcov[i,j]=cov[i, j]/(sigma[i]*sigma[j])

    return chi, sigma, normcov, p

def pretty_print_chi2(file_, par, sigma, chi, X, normcov, p):
    """
        Parameters
        ----------    

        Returns
        -------

    """    
    print("________________________________")
    print("\nFIT RESULT %s\n" % file_)
    for i in range(len(par)):
        print("p%s = %s" % (i,xep(par[i],sigma[i],",")))
        #print("p%s = %s" % (i,par))
    print("\nchi / ndof = %.1f / %s" %(chi,len(X)-len(par)))
    print("p_value = %.2f %%" %(p*100))
    if len(par)>1 :
        print("covarianza normalizzata=\n", normcov)

def latex_table(directory, file_, data, data_err, tab, out, data_ol=[], data_err_ol=[]):
    """
        Parameters
        ----------    

        Returns
        -------
    
    """


    with open(directory+"Figs-Tabs/tab_"+file_+".txt", "w") as text_file:
        text_file.write("\\begin{tabular}{c")
        for z in range (1,len(data)):
            text_file.write("|c")
        text_file.write("} \n")
        text_file.write("%s" % tab[0])
        for z in range (1,len(data)):
            text_file.write(" & %s" % tab[z])
        text_file.write("\\\\\n\hline\n")
        for i in range (len(data[0])):
            text_file.write("%s" % xe(data[0][i], data_err[0][i], "$\pm$"))
            for j in range (1,len(data)):
                text_file.write(" & %s" % xe(data[j][i], data_err[j][i], "$\pm$"))
            text_file.write("\\\\\n")
        if out==True:
            for i in range (len(data_ol[0])):
                text_file.write("%s" % xe(data_ol[0][i], data_err_ol[0][i], "$\pm$"))
                for j in range (1,len(data_ol)):
                    text_file.write(" & %s" % xe(data_ol[j][i], data_err_ol[j][i], "$\pm$"))
                text_file.write("\\\\\n")
        text_file.write("\\end{tabular}")
        text_file.close()
        

def fit(directory, file_, units, f, p0, 
        title_="", Xlab="", Ylab="", XYfun=_XYfunction, 
        preplot=False, Xscale="linear", Yscale="linear", 
        xlimp = array([97.,103.]), residuals=False, 
        table=False, tab=[""], fig="^^", out=False, kx =1, ky = 1):
    
    """
        Interface for the fit functions of lab library.
        It performs the following tasks:
            - make a fast plot of the datas, with errors of course
            - make the fit of the data and print the plot
            - print the residuals plot
            - recognize the outlier and mark them on the fit plot
            - print a file with the latex tables of data, 
                ready to import in the .tex file

        Parameters
        ----------
        directory:
        file_:
        units: array of tuples, 
               each tuple must contains two elements (unit, metertype)
        f:
        p0:
        

        Returns
        -------
        1, if all is gone well.

        Notes
        -----
        
    """
    data = load_data(directory,file_)
    X, Y, dX, dY, data_err = errors(data, units, XYfun)

    # define a default for the figure name
    if fig=="^^":
        fig=file_
    
    # print a fast plot of the data    
    if preplot==True :
        _preplot(directory, file_, title_, fig, X, Y, dX, dY, Xscale, Yscale, Xlab, Ylab)
    
    #Fit
    par, cov = fit_generic_xyerr2(f,X,Y,dX,dY,p0)
    
    #Plotto il grafico con il fit e gli scarti
    plot_fit(directory, file_, title_, units, f, par,
             X, Y, dX, dY, kx, ky,
             out, fig, residuals, xlimp, XYfun,
             Xscale, Yscale, Xlab, Ylab)

    #Calcolo chi, errori e normalizzo la matrice di cov
    chi, sigma, normcov, p = chi2_calc(f, par, X, Y, dY, dX, cov)

    #Stampo i risultati, il chi e la matrice di cov
    pretty_print_chi2(file_, par, sigma, chi, X, normcov, p)

    if out ==True:
        data_ol = load_data(directory,file_+"_ol")
        X_ol, Y_ol, dX_ol, dY_ol, data_err_ol = errors(data_ol, units, XYfun)
    else:
        data_ol=[]
        data_err_ol=[]
    #Salvo la tabella formattata latex
    if table==True:
        latex_table(directory, file_, data, data_err, tab, out, data_ol, data_err_ol)

    par_err = uncertainties.correlated_values(par,cov)
    
    return par_err

def fast_plot(directory, file_, units, XYfun=_XYfunction, title_="",
              fig="^^", Xscale="linear", Yscale="linear", Xlab="", Ylab=""):
    """
        Parameters
        ----------    
        directory: string
            the pwd
        file_: string
            the txt file with the data to crunch
        X: (N-shaped array of) numbers
        Y: (N-shaped array of) numbers
        dX: (N-shaped array of) numbers
        dY: (N-shaped array of) numbers
        title_: string, optional
            plot title
        fig: string, optional
        Xscale: string, optional
        Yscale: string, optional
        Xlab: string, optional
        Ylab: string, optional

        Returns
        -------
        1, if all goes well

    """    
    data = load_data(directory,file_)
    X, Y, dX, dY, data_err = errors(data, units, XYfun)

    # define a default for the figure name
    if fig=="^^":
        fig=file_
    
    # print a fast plot of the data    
    if preplot==True :
        _preplot(directory, file_, title_, fig, X, Y, dX, dY, Xscale, Yscale, Xlab, Ylab)

    return 1
    
def umme(value, unit="volt_ar", instrument="osc"):
    #Shortcut to generate an ufloat type with the error given by the mme function.
    return ufloat(value,mme(value,unit,instrument))