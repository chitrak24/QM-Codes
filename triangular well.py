"""
Registration : 012-1111-0461-20
Roll         : 203012-21-0008
Description  : Triangular Well Shooting Method
Author       : Chitrak Roychowdhury
"""


# Triangular_Well_shooting
# Energy levels reproduced in scale of ((e*F*h_bar)**2/(2*m))**
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.integrate import simps
from scipy.optimize import brentq

L=10.0;
h=0.002;
all_zeros=[];
lam=np.arange(0, 10.,0.0001)

#Function for determining eigen energy in shooting method
def psiAlt(lam):
    p0, p1=0.0, 1.0
    x=+h
    while x<=+L:
        p0, p1=p1, (2-h**2*(lam-x))*p1-p0
        x+=h
    return p1

#Ploting Given Function
plt.plot(lam,psiAlt(lam))
plt.grid(True)
plt.show()

#find the zeros for the symmetrical case
s=np.sign(psiAlt(lam))
for i in range(len(s)-1):
    if s[i]+s[i+1]==0:
        zero=brentq(psiAlt, lam[i], lam[i+1])
        all_zeros.append(zero)

#Energy eigenvalues in scale of ((e*F*h_bar**2/(2*m)**(1/3)
print ("Numerical values: ", all_zeros)

#Apprximated Analytic results for energy eigenvalues in scales of ((e*F*h_bar**2/(2*m)**(1/3)
print ("Theoritical values: ", (1.5*np.pi*np.arange(0.75,6.0,1.0))**(2./3))
psi=0; psid=1.; #initial psi and first dervative of psi
I=np.array([psi,psid])
xi=-0.; xf=+8.; h1=0.002; x1=xi
n=int(input("select the wavefunction: "))
e=all_zeros[n]
xs=[]; psi_s=[]

def RK4(x1,I):
    p=dI(x1,I)
    q=dI(x1+h1/2, I+h1/2*p)
    r=dI(x1+h1/2, I+h1/2*q)
    s=dI(x1+h1, I+h1*r)
    I+=(p+2*q+2*r+s)/6*h1
    x1+=h1
    return x1,I

def dI(x1,I):
    I=psi,psid
    dphi_dx=psid
    dpsid_dx=-(e-x1)*psi
    return np.array([dphi_dx,dpsid_dx])
while x1<xf:
    x1,I=RK4(x1,I)
    psi,psid=I
    xs.append(x1)
    psi_s.append(psi)
    arr_xs=np.asarray(xs)
    arr_psi=np.asarray(psi_s)
    sq_arr_psi=arr_psi**2

#Normalization constant
N=1./simps(sq_arr_psi,dx=h1)
print ("Normalization constant is: ", N)
arr_psi=N*arr_psi
plt.plot(arr_xs,arr_psi,lw='3',c='g')
plt.grid(True)
plt.xlabel("$x\longrightarrow$",fontsize=16)
plt.ylabel("$\psi(x)\longrightarrow$",fontsize=16)
plt.title("Plotting wavefunctions of triangular well")
plt.show()
 
