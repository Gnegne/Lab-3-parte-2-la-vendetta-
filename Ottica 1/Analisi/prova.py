import pylab
import numpy
from scipy.optimize import curve_fit
from scipy import stats

#importo le misure
X,dX,d,t_i  = pylab.loadtxt('sodio.txt',unpack=True)
a,da,od,t_i_o  = pylab.loadtxt('sodio.txt',unpack=True)
b,db,od,t_i_1  = pylab.loadtxt('sodio.txt',unpack=True)
c,dc,od,t_i_2  = pylab.loadtxt('sodio.txt',unpack=True)

x_0 = 0
dx_0 = 1/0
x_0_i=a
dx_0_i = da 
dt_i = b
dt_d = c

for counter in range (0,len(a)):
	x_0_i[counter]=X[counter]-x_0
	dx_0_i[counter] = dX[counter] + dx_0


dt_i = x_0_i*numpy.pi/180
t_d = (numpy.pi - t_i - x_0_i*numpy.pi/180)

dt_d = dt_i + dx_0_i*numpy.pi/180


l=d*(numpy.sin(t_i)-numpy.sin(t_d))


dl = d*(numpy.cos(t_i)*dt_i + numpy.cos(t_d)*dt_d)


print l
