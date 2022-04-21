"""
Registration : 012-1111-0461-20
Roll         : 203012-21-0008
Description  : Harmonic Oscillator Shooting Method
Author       : Chitrak Roychowdhury
"""


import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.integrate import simps
from scipy.optimize import brentq
from scipy.integrate import odeint

L=10.0
h=0.01
x=np.arange(-L,+L,h)
e=np.arange(0.95,10.1,0.0001)
all_zeros=[]

#Function for determining eigenenergy of harmonic oscillator in shooting method
def psiAtL(e):
     p0,p1=0.0,1.0
     x=-L+h
     while x<=+L:
         p0,p1=p1,(2-h**2*(e-x**2))*p1-p0
         x+=h
     return p1

#plotting given function
plt.plot(e,psiAtL(e))
plt.grid(True)
plt.show()

#first five eigenenergies evaluated
#we need to find zeros for the symmetrical case
s=np.sign(psiAtL(e))
for i in range(len(s)-1):
    if s[i]+s[i+1]==0:
        zero=brentq(psiAtL,e[i],e[i+1])
        all_zeros.append(zero) 

print ("Eigenenergies: ", all_zeros)
j=int(input("input a number(n) to select the wavefunction: "))
E=all_zeros[j]
xi=-4.;xf=+4.;h1=0.001
xs=np.arange(xi,xf,h1)
psi_s=[]

def Schrodinger(w,x):
    psi=w[0]
    psi1=w[1]
    dpsidx=psi1
    dpsi1dx=-(E-x**2.)*psi
    return np.array([dpsidx,dpsi1dx])

w0=[0.,-1.]
sol=odeint(Schrodinger,w0,xs)
psi=sol[:,0]
psi1=sol[:,1]

#normalization constant using simpsons rule from scipy
sq_arr_psi=psi**2

#normalization constant
N=1./simps(sq_arr_psi,dx=(xs[1]-xs[0]))

#Normalization wavefunction
psi=N*psi
plt.grid(True)
plt.plot(xs,psi,'o',c='r')
plt.title('Plotting wavefunctions of harmonic oscillator')
plt.xlabel("$x\longrightarrow$",fontsize=16)
plt.ylabel('$\psi(x)\longrightarrow$',fontsize=16)
plt.axhline(linestyle='--',c='0.03')
plt.show()
