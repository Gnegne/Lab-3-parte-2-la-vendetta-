from PIL import Image
from uncertainties import unumpy
from uncertainties import *
import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
from scipy.optimize import curve_fit

soglia= 254
#for img in (7,8,9,10,11,19,20,21,22,30,31,32,33,42,43,44,53,54,55,65,66):
for img in range(9,11):
        im = Image.open('C:\\Users\\Roberto\\Desktop\\stograncazzo\\gne\\JPEG\\('+str(img)+').jpg', 'r')
        pix = im.load()
        
        D11=0
        D12=0
        D13=0
        D21=0
        D22=0
        D23=0
        D31=0
        D32=0
        D33=0
        E1=0
        E2=0
        E3=0
        for x in range(0,im.size[0]):
            for y in range(0,im.size[1]):
                if pix[x,y][0] > soglia:
                    D11 += 2*x
                    D12 += 2*y
                    D13 += 1
                    D21 += 2*x**2
                    D22 += 2*x*y
                    D23 += x
                    D32 += 2*y**2
                    D33 += y
                    E1  += (x**2+y**2)
                    E2  += (x**3+y**2*x)
                    E3  += (y**3+x**2*y)
        
        D31 = D22
        
        D = np.matrix([[D11,D12,D13],[D21,D22,D23],[D31,D32,D33]])
        E = np.matrix([[E1],[E2],[E3]])
        Q = D**(-1)*E
        R = np.sqrt(Q[2]+Q[0]**2+Q[1]**2)
    
        
        N=20
        data=np.zeros((N)*2)
        for x in range(0,im.size[0]):
            for y in range(0,im.size[1]):
                if pix[x,y][0] > soglia:
                    a = round(float(np.sqrt((x-Q[0])**2+(y-Q[1])**2)-R))
                    if a <= N-1 and a >=-N:
                        data[a+N]+=1
                    
        for i in range (0,N*2):
            data[i]=data[i]/(R+(i-N))
        pos = np.arange(-N+1,N+1)
        width = 1.0     # gives histogram aspect to the bar diagram
        

        ax = plt.axes()
        ax.set_xticks(pos-0.5)
        ax.set_xticklabels( pos, rotation=90 )
        #ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.title("Distribuzione dei raggi normalizzati")
        plt.xlabel("differenze rispetto al raggio fittato $[pixel]$")
        plt.ylabel("occorrenze normalizzate $[pixel^{-1}]$")
        plt.bar(pos, data, width, color='r')
        plt.show()
        
        def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
            #+a1*np.exp(-(x-x01)**2/(2*sigma1**2))
            
        popt, pcov = curve_fit(gauss_function, pos, data, p0 = [1, 0, 20,],maxfev=10000)
        plt.plot(pos, gauss_function(pos, *popt), label='fit', color="black")
        plt.savefig("C:\\Users\\Roberto\\Desktop\\stograncazzo\\gne\\JPEG\\fit2\\fit_"+str(img)+"_"+str(soglia)+"_1g.pdf")
        plt.clf()
        #print(img, "    ", soglia," ",round(float(Q[0]),1), "   ",round(float(Q[1]),1)," ", round(float(R[0]),1),"    ",round(popt[2],1)," ",round(popt[5],1)," ",round(popt[1],1)," ",round(popt[4],1))
        print(img, "    ",round(float(R[0]),1),"   ",round(popt[2],1))