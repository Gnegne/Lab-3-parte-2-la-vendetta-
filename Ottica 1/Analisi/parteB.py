from numpy import *
from uncertainties import *
from uncertainties import unumpy


err = 3/60

#zero = ufloat(171+22/60, err)
#refl = ufloat(229+45/60,err)
zero = ufloat(171+21/60, err)
refl = ufloat(229+46/60,err)
primo = ufloat(277+16/60,err)
secondo = ufloat(315+53/60,err)

Hrefl = ufloat(229+38/60,err)
H1viola = ufloat(269+15/60,err)
H1azzurro = ufloat(273+3/60,err)
H1rosso=ufloat(284+57/60,err)
H2viola=ufloat(299+33/60,err)
H2azzurro=ufloat(306+58.5/60,err)

Na0 = ufloat(229+18/60,err)
Na1 = ufloat(280+17/60,err)
Na2 = ufloat(280+15/60,err)

lamb = 546.074e-9

ti = 0.5*(180-(refl-zero))*pi/180
td1 = pi-ti-(primo-zero)*pi/180
td2 = pi-ti-(secondo-zero)*pi/180

d1 = lamb/(unumpy.sin(ti)-unumpy.sin(td1))
p1=(d1**-1)/1000

d2 = 2*lamb/(unumpy.sin(ti)-unumpy.sin(td2))
p2=(d2**-1)/1000
#print(p1)
#print(p2)
print((p1+p2)/2)

d = (d1/unumpy.std_devs(d1)+d2/unumpy.std_devs(d2))/(1/unumpy.std_devs(d1)+1/unumpy.std_devs(d2))

hti = 0.5*(180-(Hrefl-zero))*pi/180
htv1 = pi - hti-(H1viola-zero)*pi/180
htv2 = pi - hti-(H2viola-zero)*pi/180
Hlv1 = d*(unumpy.sin(hti)-unumpy.sin(htv1))
Hlv2 = d*(unumpy.sin(hti)-unumpy.sin(htv2))/2

Hlv = (Hlv1/unumpy.std_devs(Hlv1)+Hlv2/unumpy.std_devs(Hlv2))/(1/unumpy.std_devs(Hlv1)+1/unumpy.std_devs(Hlv2))
print(Hlv*1e9)

hta1 = pi - hti-(H1azzurro-zero)*pi/180
hta2 = pi - hti-(H2azzurro-zero)*pi/180
Hla1 = d*(unumpy.sin(hti)-unumpy.sin(hta1))
Hla2 = d*(unumpy.sin(hti)-unumpy.sin(hta2))/2
Hla = (Hla1/unumpy.std_devs(Hla1)+Hla2/unumpy.std_devs(Hla2))/(1/unumpy.std_devs(Hla1)+1/unumpy.std_devs(Hla2))
print(Hla*1e9)

htr = pi - hti-(H1rosso-zero)*pi/180
Hlr = d*(unumpy.sin(hti)-unumpy.sin(htr))
print(Hlr*1e9)

nati = 0.5*(180-(Na0-zero))*pi/180
nat1 = pi - nati-(Na1-zero)*pi/180
nat2 = pi - nati-(Na2-zero)*pi/180
Nal1 = d*(unumpy.sin(nati)-unumpy.sin(nat1))
Nal2 = d*(unumpy.sin(nati)-unumpy.sin(nat2))
Hla = (Hla1/unumpy.std_devs(Hla1)+Hla2/unumpy.std_devs(Hla2))/(1/unumpy.std_devs(Hla1)+1/unumpy.std_devs(Hla2))
print(Nal1*1e9)
print(Nal2*1e9)
print((Nal1-Nal2)*1e9)
