from numpy import *
#from math import *
from pylab import *
from uncertainties import unumpy
from uncertainties import *

d=45.8
r=6.4
n=1.6
s=0.1

##################### CORREZIONE BULBO

def t(mis):
    return arctan(mis/d)
def b(mis):
    return arcsin(d*sin(t(mis))/(r*n))
def c(mis):
    return pi-arcsin(d*sin(t(mis))/r)
def g(mis):
    return pi-t(mis)-c(mis)
def xi(mis):
    return b(mis)-g(mis)
def x(mis):
    return s/cos(b(mis))
def dist(mis):
    return  sqrt(r**2+d**2-2*r*d*cos(g(mis)))
def y(mis):
    return tan(b(mis))*s
def y1(mis):
    return (-sqrt(r**2*tan(b(mis))**(-2)-2*r*s-s**2)+r*tan(b(mis))**(-1)+s*tan(b(mis))**(-1))/(tan(b(mis))**(-2)+1)
def s1(mis):
    return tan(b(mis))**(-1)*(-sqrt(r**2*tan(b(mis))**(-2)-2*r*s-s**2)+r*tan(b(mis))**(-1)+s*tan(b(mis))**(-1))/(tan(b(mis))**(-2)+1)
def x1(mis):
    return sqrt(y1(mis)**2+s1(mis)**2)
def dg(mis):
    return arctan(y1(mis)/r)
def tcorr(mis):
    return arcsin(n*sin(dg(mis)+b(mis)))-g(mis)-dg(mis)
def remain(mis):
    return d-dist(mis)*cos(t(mis))-x1(mis)*cos(xi(mis))
def corr(mis):
    return dist(mis)*sin(t(mis))+x(mis)*sin(xi(mis))+remain(mis)*sin(tcorr(mis))
    
##################### DATI NON NOSTRI

l=linspace(0,6.4,10000)

#xlabel("raggio misurato [cm]")
#ylabel("fattore di correzione")
#title("Correzione distorsione ottica")
#plot(l,l/corr(l),"blue")
#plot(l,corr(l),"blue")
#plot(l,l,"red")

N1, I1, V1, r1 = loadtxt('C:\\Users\\Roberto\\Desktop\\megacazzi2.txt', unpack = True)

r1 = r1/203
R = sqrt(2*V1/((7.8e-4*I1)**2*1.7587e11))*100
em1 = 2*V1/(7.78e-4*I1*corr(r1))**2
#figure(1)
#plt.plot(N1, R/corr(r1), 'go')
#plt.plot(N1, R/r1, 'ro')


##################### DATI NOSTRI

N2, V2, I2, r2, dr2 = loadtxt('C:\\Users\\Roberto\\Desktop\\megacazzi.txt', unpack = True)

em = 1.7587

#CORREZIONE GEOMETRICA
cal = ufloat(181.9,0.3)
#56.8		45.8		11
d1 = ufloat(56.5,0.1)
d2 = ufloat(11,0.1)
d3 = d1-d2
prosp =d1/d3
cal_corr = cal*prosp

#CALCOLO B
V = ufloat(0.409,0.001)
G = ufloat(11.1,0.1)
m = ufloat(5e-3,0.1e-3)
I_m = ufloat(0.95,0.01)
B = V/G/m/I_m*1e-4

r2_e = unumpy.uarray(r2,dr2)
V2_e = unumpy.uarray(V2,1)
I2_e = unumpy.uarray(I2,0.01)

r2_e = r2_e/(cal_corr*100)
r2=r2/(nominal_value(cal_corr)*100)

R = sqrt(2*130/((7.78e-4*I2)**2*1.7587e11))*100

em2_e1 = 2*V2_e/(B*I2_e*r2_e)**2/1e11
em2 = 2*V2/(unumpy.nominal_values(B)*I2*corr(r2))**2/1e11
em2_e = unumpy.uarray(em2,unumpy.std_devs(em2_e1))
dm = em2_e-em

#figure(2)
#plt.errorbar(N2, unumpy.nominal_values(dm),unumpy.std_devs(dm), fmt=",")
#plt.plot(N2, R, 'ro')
#plt.plot(N2, corr(r2*100), 'bo')


dm, ddm, Vm = loadtxt('C:\\Users\\Roberto\\Desktop\\dm.txt', unpack = True)

figure(3)
ylabel("Experimental-Theoretical x $10^{11}$ [C/Kg]")
xlabel("Tensione sul cannone elettronico [V]")
title("Differenze esperimento-teoria vs V")
errorbar(Vm, dm, ddm, fmt=".",ecolor="black",color="black")
