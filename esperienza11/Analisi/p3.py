import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), 'Data Analysis'))
from ANALyzer import *
from uncertainties import *
from uncertainties import unumpy as un

#########################################

#PARTE 3 - MULTIVIBRATORE ASTABILE#

dir= folder+"\\"
file="dati_3"
#fig="dati_3_mix"

def f(x, a, b):
    return a*x+b

p0=[0,0]

def XYfun(a):
    return a[0], a[1]*1e-6

unit = [("ohm", "dig"),("time", "osc")]

titolo = "Linearità multivibratore astabile"
Xlab = "Resistenza [$\Omega$]"
Ylab = "Periodo [$\mu s$]"

tab = ["Resistenza [$\Omega$]", "Periodo [$\mu s$]"]

fit(dir, file, unit, f, p0, titolo, Xlab, Ylab, XYfun, residuals=True, ky=1e6, tab=tab, table=True)

#########################################

#PARTE 2 - MULTIVIBRATORE MONOSTABILE#

dir= folder+"\\"
file="dati_2"

def f(x, a, b):
    return a*x+b

p0=[0,0]

def XYfun(a):
    return a[0], a[1]*1e-6

unit = [("ohm", "dig"),("time", "osc")]

titolo = "Linearità multivibratore monostabile"
Xlab = "Resistenza [$\Omega$]"
Ylab = "Periodo [$\mu s$]"

#tab = ["frequency [Hz]", "$V_A$ [V]", "$\\varphi$ [s]", "$V_A$ [V]"]

#fit(dir, file, unit, f, p0, titolo, Xlab, Ylab, XYfun, residuals=True, ky=1e6, out=True)

#######################################

R = umme(982,"ohm","dig")
C = umme(109.1e-9,"farad","dig")
V_H = umme(4.48,"volt_nc","osc")
V_c = umme(1.44, "volt_nc","osc")

T = R*C*un.log(V_H/V_c)

print ("T=",T*1e6,"us")