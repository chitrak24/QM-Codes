'''
CU REG.- 012-1111-0890-20
CU ROLL- 203012-210135
DESCRIPTION : Symmetric Potential well- Finding Eigenvalues by solving Transcendental Equations and plotting Eigenfunctions
DATE : 24/02/22
'''

#import required libraries
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps


#define variable in Hartree atomic units
a=4/0.529177# two times width of potential well in atomic units
N=1000# Number of points to take
V_0=14/27.211# parameters determining depth of well(in atomic units)
en=np.linspace(.00001,V_0,1000)#vector of energies whose to look for the bound states
m=1;hbar=1;


#definition of dimensionless variables
u_0=np.sqrt(m*a**2*V_0/(2*hbar**2))
v=np.sqrt(2*m*en/hbar**2)*(a/2.0)

#functions to solve for transcendental Equation
f_sym=lambda v:v*np.tan(v)-np.sqrt((u_0)**2-(v)**2)
f_asym=lambda v:(-1/np.tan(v))*v-np.sqrt((u_0)**2-(v)**2)

#plotting the transcedental functions
def f_sym1(v):
    return v*np.tan(v)

def f_circle(v):
    return np.sqrt((u_0)**2-(v)**2)

def f_asym1(v):
    return (-1/np.tan(v))*v

plt.grid(True)
plt.plot(v,f_sym1(v))
plt.plot(v,f_asym1(v))
plt.ylim(0,10)
plt.legend(["Symmetric State Solution","Asymmetrrical State Solution"])
plt.xlabel(r'$v\longrightarrow$')
plt.ylabel(r'$f(v)\longrightarrow$')
plt.title("Solving of Transcendental Equations for v ")
plt.show()
v_zeros_s=[]#shows zeros of 'v' symmetric
v_zeros_a=[]#shows zeros of 'v' anti-symmetric
v_zeros=[]#shows all zeros of 'v'
Eigenenergies=[]

#Find the zeros for symmetrical case
s=np.sign(f_sym(v)) 
for i in range(len(s)-1):
    if s[i]+s[i+1]==0:
        Zero=brentq(f_sym,v[i],v[i+1])
        v_zeros_s.append(Zero)
l1=v_zeros_s[::2]#selecting non-divergent point of tan(v)

#find zeros for assymetrical case
s=np.sign(f_asym(v))
for i in range(len(s)-1):
    if s[i]+s[i+1]==0:
        Zero=brentq(f_asym,v[i],v[i+1])
        v_zeros_a.append(Zero)
l2=v_zeros_a[::2]
v_zeros=l1+l2#filter zeros
print('Value of v at which functions are satisfied:',v_zeros)

#Calculate eigenenergies corresponding to v
for i in v_zeros:
    Eigenenergies.append(2*i**2*hbar**2/(m*a**2))
print( 'Eigenenergy values',Eigenenergies, "Atomic Units")
Eigenenergies=np.asarray(Eigenenergies)
print( 'Eigenenergy values',27.211*Eigenenergies, 'Electron volts')
l=len(Eigenenergies)

#Potential function in the finite square well of width 'a' and depth is global variable 'V_0'
def V(x):
    a= 4/0.529177 #in atomic units
    if abs(x)>a/2:
        return V_0
    else:
        return 0

#Function to solve Schrodinger's eq using odeint
def Schrodinger(W,x,E):
    psi= W[0]
    psi1=W[1]
    dpsidx = psi1
    dpsi1dx = -2*m*((E-V(x))/(hbar**2))*psi
    return np.array([dpsidx,dpsi1dx])

#plotting
pl=np.array([]);pl_t=np.array([])
for i in range(l):
    if(i<l/2.):
        W_0 = [1.0, 0] #initial condition for even/symmetric states
    else:
        W_0 = [0, 1.] #initial condition for odd/antisymmetric states
    x_p = np.linspace(0,a,N) #positive x-axis
    E=Eigenenergies[i]#Selecting eigenenergy for ground state
    sol= odeint(Schrodinger,W_0,x_p,args=(E,))
    psi_plus = sol[:,0]
    psi1_plus = sol[:,1]
    x_n = np.linspace(0,-a,N) #negative x-axis
    sol1= odeint(Schrodinger,W_0,x_n,args=(E,))
    psi_minus = sol1[:,0]
    psi1_minus = sol1[:,1]
    plt.grid(True)
    #construction at psix and psi
    psix = np.concatenate((x_n[::-1],x_p[1:]))
    psi = np.concatenate((psi_minus[::-1],psi_plus[1:]))
    #Normalization constant using Simpson's rule from Scipy
    sq_arr_psi = psi**2
    #print ("Scipy based Simpson's Integration result as Normalization constant",simps(sq_arr_psi, dx=(x_p[1]-x_p[0])))
    Norm=1/simps(sq_arr_psi, dx=(x_p[1]-x_p[0])) #Normalization Constant
    psi = Norm * psi
    plt.plot(psix,psi,'-',c='b',lw=3)
    plt.title(r'$\Psi$'+' for Eigen-Energy value '+r'$E_n=$'+'{:.4f}ev'.format(27.211*E))
    plt.axhline(y=0, color='.1',lw=1)
    plt.axvline(x=0, color='.1',lw=1)
    plt.xlabel('x'+r'$\longrightarrow$')
    plt.ylabel(r'$\Psi(x)\longrightarrow$')
    plt.show()
