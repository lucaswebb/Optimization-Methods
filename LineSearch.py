import math
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

##Subroutines
def vlist(x):##Converts vectors to lists
    return x.flatten().tolist()[0]

##Main Function
def linetrace(F,J,alpha,P,x0,tol):##Modulator Form Solves x_n+1=x_n+alpha*p(x_n)
    x=x0
    p=P(F,J)##Produces a function p(x)
    p0=p(x)
    log=[[],[]]
    ##sol=np.matrix([0.7937005259840997, 0.7937005259840997]).T #F1 solution
    sol=np.matrix([1.,1.,1.,1.]) #F2 Solution
    for n in range(20000):
        v=np.matrix(x).T
        a=alpha(F,J,p0,x,v)##Finds the alpha
        v+=a*p0
        x=vlist(v.T)
        px=p(x)
        er=np.linalg.norm(px-p0)
        log[0].append(n)
        log[1].append(np.linalg.norm(sol-v))##Real Error
        #log[1].append(er)##Assumed error
        if er<10.**(-tol):
            return x, n, log, 0
        p0=px
    return x, 'MAX',log, 1

##P(x) Functions

def gradg(F,J):##Produces -grad b for p(x)
    def grad(x):
        return -1.*J(x).T*F(x)
    return grad

def Newton(F,J):
    def newton(x):
        return -1*J(x)**-1*F(x)
    return newton


##alpha functions
def hquad(F,J,p,x,v):##Creates a 1D quadratic intelopalation for g(x_n+alpha*p(x_n))in terms of alpha.
    def g(x):##Calculates sum fi^2 might need extension in future
        f=F(x)
        g=f.T*f
        return vlist(g)[0]
    h0=g(x)
    b=1.
    for i in range(1075):##Number goes to machine minimum
        h4=g((v+b*p).T.flatten().tolist()[0])
        if h0>h4:##Checks if the interval is too large in an attempt to find a basin of convergence.
            b1=b*(.5*np.cos(math.pi/6.)+.5) ##The intelopalation runs off of Chebyshev points for accuracy.
            b2=b*(.5*np.cos(3*math.pi/6.)+.5)
            b3=b*(.5*np.cos(5*math.pi/6.)+.5)
            h1=g(vlist((v+b1*p).T))
            h2=g(vlist((v+b2*p).T))
            h3=g(vlist((v+b3*p).T))
            a1=(h2-h1)/(b2-b1)
            a2=(h2-a1*(b2-b1))/(b3-b1)/(b3-b2)
            bm=(a2*(b1+b2)-a1)/(2.*a2)
            hm=g(vlist((v+bm*p).T))
            if hm<h4:##The 0 of the derivative should usually be a local minima but just in
                return bm
            else:
                return b
        b*=.5
    return b ##If no basin is found it is likely due to machine error, and the approximation cannot continue further.

##Identical to hquad but doesn't use Chebyshev nodes
def hquad2(F,J,p,x,v):##Creates a 1D quadratic intelopalation for g(x_n+alpha*p(x_n))in terms of alpha.
    def g(x):##Calculates sum fi^2 might need extension in future
        f=F(x)
        g=f.T*f
        return vlist(g)[0]
    h0=g(x)
    b=1.
    for i in range(1075):##Number goes to machine minimum
        h4=g((v+b*p).T.flatten().tolist()[0])
        if h0>h4:##Checks if the interval is too large in an attempt to find a basin of convergence.
            b1=0##Uses standard nodes
            b2=.5*b
            b3=b
            h1=h0
            h2=g(vlist((v+b2*p).T))
            h3=h4
            a1=(h2-h1)/(b2-b1)
            a2=(h2-a1*(b2-b1))/(b3-b1)/(b3-b2)
            bm=(a2*(b1+b2)-a1)/(2.*a2)
            hm=g(vlist((v+bm*p).T))
            if hm<h4:##The 0 of the derivative should usually be a local minima but just in
                return bm
            else:
                return b
        b*=.5
    return b ##If no basin is found it is likely due to machine error, and the approximation cannot continue further.

def cons(F,J,p,x,v):##constant alpha
    return 1

def Wolfe(F,J,p,x,v):
    def g(x):##Calculates sum fi^2 might need extension in future
        f=F(x)
        g=f.T*f
        return vlist(g)[0]
    h0=g(x)
    Dg=gradg(F,J)
    dg=Dg(x)
    c1=10**-4
    c2=.1
    a=1.
    for i in range(1075):##Number goes to machine minimum
        h=g((v+a*p).T.flatten().tolist()[0])
        dgx=Dg((v+a*p).T.flatten().tolist()[0])
        Arm=vlist(dg.T*p)[0]
        Cur=vlist(dgx.T*p)[0]
        if (h<=h0+c1*a*Arm and Cur>=c2*Arm):#Wolfe conditions
            return a
        a*=.5
    return a

##Test Functions
def F1(x): ##Test function
    f=[[1./(x[0]**2+x[1]**2)-x[0]],[1./(x[0]**2+x[1]**2)-x[1]]]
    return np.matrix(f)
def J1(x):
    j=[[-2.*x[0]/(x[0]**2+x[1]**2)**2-1,-2.*x[1]/(x[0]**2+x[1]**2)**2],[-2.*x[0]/(x[0]**2+x[1]**2)**2,-2.*x[1]/(x[0]**2+x[1]**2)**2-1]]
    return np.matrix(j)
def F2(x):##g(x,y)=Rosenbrock
    f=[[1-x[0]],[10*(x[1]-x[0]**2)]]
    return np.matrix(f)
def J2(x):
    j=[[-1,0],[-20*x[0],10]]
    return np.matrix(j)
def F3(x):##g(x1,x2,x3,x4)=Rosenbrock4D
    f=[[1-x[0]],[10*(x[1]-x[0]**2)],[1-x[2]],[10*(x[3]-x[2]**2)]]
    return np.matrix(f)
def J3(x):
    j=[[-1,0,0,0],[-20*x[0],10,0,0],[0,0,-1,0],[0,0,-20*x[2],10]]
    return np.matrix(j)
##Executions
##F1
[x,n,log,er]=linetrace(F1,J1,hquad,gradg,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'g')
plt.yscale("log")
[x,n,log,er]=linetrace(F1,J1,hquad2,gradg,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'b')
plt.yscale("log")
[x,n,log,er]=linetrace(F1,J1,cons,Newton,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'r')
plt.yscale("log")
[x,n,log,er]=linetrace(F1,J1,Wolfe,gradg,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'m')
plt.yscale("log")
plt.show()
#F2
[x,n,log,er]=linetrace(F2,J2,hquad,gradg,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'g')
plt.yscale("log")
[x,n,log,er]=linetrace(F2,J2,hquad2,gradg,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'b')
plt.yscale("log")
[x,n,log,er]=linetrace(F2,J2,cons,Newton,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'r')
plt.yscale("log")
[x,n,log,er]=linetrace(F2,J2,Wolfe,gradg,[1.5,1.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'m')
plt.yscale("log")
plt.show()
##F3
[x,n,log,er]=linetrace(F3,J3,hquad,gradg,[1.5,1.,2.,1.2],15.)
print(x,n,er)
plt.plot(log[0],log[1],'g')
plt.yscale("log")
[x,n,log,er]=linetrace(F3,J3,hquad2,gradg,[1.5,1.,2.,1.2],15.)
print(x,n,er)
plt.plot(log[0],log[1],'b')
plt.yscale("log")
[x,n,log,er]=linetrace(F3,J3,cons,Newton,[1.5,1.,2.,1.2],15.)
print(x,n,er)
plt.plot(log[0],log[1],'r')
plt.yscale("log")
[x,n,log,er]=linetrace(F3,J3,Wolfe,gradg,[1.5,1.,2.,1.2],15.)
print(x,n,er)
plt.plot(log[0],log[1],'m')
plt.yscale("log")
plt.show()
