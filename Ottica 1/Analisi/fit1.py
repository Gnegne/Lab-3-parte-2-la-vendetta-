import pylab
import numpy
from scipy.optimize import curve_fit
from scipy import stats

#importo le misure
l,g,p,e = pylab.loadtxt('stocazzo.txt')[1:].T
l= 1/l/1e6

x = 360  - g - p/60
dx = e/60

# define the fit function (NOTE THE INDENT!)
def fit_function(x, b, a):
    return  (a*x + b)

# set the array of initial value(s)
initial_values = [0,0]



# call the minimization procedure (NOTE THE ARGUMENTS)
pars, covm = curve_fit(fit_function, l, x, initial_values,dx)

#print

print('intercetta = %f \pm %f' %(pars[0], numpy.sqrt(covm[0,0])))
print('coefficiente = %f \pm %f' %(pars[1], numpy.sqrt(covm[1,1])))
# bellurie 
pylab.rc('font',size=16)
pylab.xlabel('$ \\frac{1}{\lambda}$ ')
pylab.ylabel('$  \\alpha - \\alpha _0$')
pylab.minorticks_on()
pylab.title('Angoli di rifrazione lampada Cd')

# data plot (NOTE THE ORDER OF ARGUMENTS)
pylab.errorbar(l,x,dx,linestyle = '', color = 'black', marker = '.')

func_grid = numpy.linspace(0,2.5,100)

pylab.plot(func_grid, fit_function(func_grid,pars[0],pars[1]), color = 'blue')





# save the plot as a pdf file somewhere (in my own directories!)
pylab.savefig('fit1.png')

# show the plot on screen
pylab.show()
