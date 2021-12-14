import numpy as np

##Subroutines
def vlist(x):##Converts vectors to lists
    return x.flatten().tolist()[0]

##Main Function
fn=0
jn=0
def linetrace(F,J,alpha,P,x0,tol,sol):##Modulator Form Solves x_n+1=x_n+alpha*p(x_n)
    x=x0
    global fn,jn
    fn=0
    jn=0
    p=P(F,J)##Produces a function p(x)
    p0=p(x)
    log=[[],[],[],[]]
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
        log[2].append(fn)
        log[3].append(jn)
        if er<10.**(-tol):
            return x, n, log, 0
        p0=px
    return x, 'MAX',log, 1

##P(x) Functions

def gradg(F,J):##Produces -grad b for p(x)
    def grad(x):
        global fn,jn
        fn+=1
        jn+=1
        return -1.*J(x).T*F(x)
    return grad


##alpha functions

def quadratic(F,J,p,x,v):##Creates a 1D quadratic intelopalation for g(x_n+alpha*p(x_n))in terms of alpha.
    def g(x):##Calculates sum fi^2
        f=F(x)
        global fn
        fn+=1
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
            a2=(h3-h2+h1)/(b3-b1)/(b3-b2)
            bm=(b1+b2-a1/a2)/2.
            hm=g(vlist((v+bm*p).T))
            if hm<=h4:##The 0 of the derivative should usually be a local minima but in case it is a local max, the upper bound is used.
                return bm
            return b
        b*=.5
    return b ##If no basin is found it is likely due to machine error, and the approximation cannot continue further.

def Wolfe(F,J,p,x,v):
    def g(x):##Calculates sum fi^2
        f=F(x)
        global fn
        fn+=1
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
