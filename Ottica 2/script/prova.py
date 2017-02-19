import pylab
import numpy
from scipy.optimize import curve_fit
from scipy import stats

#importo le misure
raw = pylab.loadtxt('He-ne.txt',unpack=True)

zero = raw[1][0]/2
m = raw[0][1:]
h = raw[1][1:] - zero
dh = raw[2,1:]
print(h, dh)
#distanza dal tavolo
D = 209
dD = 3

#sen(thetad)

senx = D/(numpy.sqrt(D**2 + h**2))
dx = numpy.sqrt(2*(dh/h)**2 + 2*(dD/D)**2)
x = h**2/(D**2)
Dx = 2*x*(dh/h + dD/D)
dsenx = 0.5*(Dx/(1 + x)**(1.5))

#for counter in range (0,len(senx)):
#   print '$(%f \pm %f )10^3$' % (senx[counter]*1000,dsenx[counter]*1000)

# define the fit function (NOTE THE INDENT!)
def fit_function(t, b, a):
    return  (a*t + b)

# set the array of initial value(s)
initial_values = [1,-0.0006]



# call the minimization procedure (NOTE THE ARGUMENTS)
pars, covm = curve_fit(fit_function, m, senx, initial_values,dsenx)
'''
gradient, intercept, r_value, p_value, std_err = stats.linregress(m,senx)

print gradient
print intercept
print r_value
print p_value
print pars
'''

#print covm
print ('coefficiente angolare = %f \pm %f ' % (pars[1], numpy.sqrt(covm[1,1])))


# bellurie 
pylab.rc('font',size=16)
pylab.xlabel('$m$ ')
pylab.ylabel('$sen(\\theta _d)$')
pylab.xlim(-1, 20)
#pylab.ylim(,60)
pylab.minorticks_on()
pylab.title('$sen(\\theta _d)$ vs m')

# data plot (NOTE THE ORDER OF ARGUMENTS)
pylab.errorbar(m,senx,Dx,linestyle = '', color = 'black', marker = '.')

func_grid = numpy.linspace(-1,20,100)

pylab.plot(func_grid, fit_function(func_grid,pars[0],pars[1]), color = 'blue')





# save the plot as a pdf file somewhere (in my own directories!)
pylab.savefig('fit.png')

# show the plot on screen
pylab.show()
