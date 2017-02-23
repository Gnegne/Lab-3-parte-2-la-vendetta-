import getpass
import sys
from uncertainties import unumpy
from uncertainties import *
import numpy as np

if getpass.getuser() == "Roberto":
    path = "C:\\Users\\Roberto\\Documents\\GitHub\\Lab3.2\\"
    path1 = "C:\\Users\\Roberto\\Documents\\GitHub\\Lab3.2\\Data Analysis\\Bob\\"
elif getpass.getuser() == "Studenti":
    path = "C:\\Users\\Studenti\\Desktop\\Lab3\\"
else:
    raise Error("unknown user, please specify it and the path in the file Esercitazione*.py")
sys.path = sys.path + [path1]

from ANALizer import *
from pylab import log10, arctan, pi, sin

dir= path + "Ottica 2/"

###########################################################################

#LOOP GAIN MODULI#

file="He-ne"

D = ufloat(2.09,0)
d = ufloat(0.001,0)
rif = ufloat(0.138,0.0005)
dritto = ufloat(0,0.0005)
zero = (rif+dritto)/2

def f(x, a, b):
    return -a*x+sin(b)

p0=[630e-9,90]

def XYfun(a):
    return a[0], unumpy.sin(pi/2-unumpy.arctan((a[1]-zero)/D))

unit = [("noerror","osc"),("generic", "osc"),("generic","osc")]

titolo = "Fit lunghezza d'onda laser He-Ne"
Xlab = "m"
Ylab = "sin"
xlimp = [50, 105]

tab = ["frequency [Hz]", "$V_A$ [V]", "$\\varphi$ [s]", "$V_A$ [V]"]

fit(dir, file, unit, f, p0, titolo, Xlab, Ylab, XYfun, xlimp=xlimp)