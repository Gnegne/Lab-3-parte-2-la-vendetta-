import sys, os
folder = os.path.realpath('..')
sys.path.append(os.path.join(os.path.realpath('..\..'), 'Data Analysis'))
from ANALyzer import *
from uncertainties import *
from uncertainties import unumpy as un
import numpy as np

#########################################

#PARTE 3 - MULTIVIBRATORE ASTABILE#

dir= folder+"\\"
file="data"

def f(x, a, b, c):
    return a*np.exp(-x/b)+c
    
p0=[2.1,4.5,0]

def XYfun(a):
    return a[0], a[1]

unit = [("fix", "osc"),("volt", "dig")]

titolo = "Assorbimento Mylar"
Xlab = "Numero di lastre"
Ylab = "Tensione output lock-in [V]"

tab = ["# lastre", "Tensione [V]"]

par = fit(dir, file, unit, f, p0, titolo, Xlab, Ylab, XYfun, residuals=True, tab=tab, table=True, out=True)

#########################################

R_18 = umme(3.19e6,"ohm","dig")
R_20 = umme(1.219e6,"ohm","dig")
R_21 = umme(1.451e6,"ohm","dig")
R_22 = umme(38.2e3,"ohm","dig")
R_23 = umme(32.5e3,"ohm","dig")
R_24 = umme(38.3e3,"ohm","dig")
R_26 = umme(32.0e3,"ohm","dig")

#print(0.5*(R_23+R_22)/R_23*R_24/(R_24+R_26) - 0.5*R_22/R_23)
#print(0.5*(R_23+R_22)/R_23*R_24/(R_24+R_26) + 0.5*R_22/R_23)