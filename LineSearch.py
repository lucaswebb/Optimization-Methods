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
            b1=b*np.sin(math.pi/12.)**2 ##The intelopalation runs off of Chebyshev points for accuracy.
            b2=b*.5
            b3=b*np.cos(math.pi/12.)**2
            h1=g(vlist((v+b1*p).T))
            h2=g(vlist((v+b2*p).T))
            h3=g(vlist((v+b3*p).T))
            a1=(h2-h1)/(b2-b1)
            a2=(h3-h2+h1)/(b3-b1)/(b3-b2)
            bm=(b1+b2-a1/a2)/2.
            hm=g(vlist((v+bm*p).T))
            if hm<=h4:##The 0 of the derivative should usually be a local minima but just in
                return bm
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
            a2=(h3-h2+h1)/(b3-b1)/(b3-b2)
            bm=(b1+b2-a1/a2)/2.
            hm=g(vlist((v+bm*p).T))
            if hm<=h4:##The 0 of the derivative should usually be a local minima but just in
                return bm
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



import numpy as np
from sympy import ordered, Matrix, hessian, lambdify
from sympy.abc import x, y



# use sympy to return a lambda function for the gradient and hessian of an equation
def compute_gradient_hessian(eq, *args):
    f = eq(*args)
    vars = list(ordered(f.free_symbols))

    gradient = lambda func, vars : Matrix([func]).jacobian(vars)

    grad_lambda = lambda x : lambdify(vars, gradient(f, vars))(*x).flatten()
    hess_lambda = lambda x : lambdify(vars, hessian(f, vars))(*x)

    return grad_lambda, hess_lambda

def quadratic_model(f_k, grad_k, hess_k):
    return lambda p : f_k + p.dot(grad_k) + 0.5 * p.dot(hess_k).dot(p)

# return the positive root of Tau using the quadratic formula
def find_tau(z_j, d_j, radius):
    a = z_j.dot(z_j)
    b = z_j.dot(d_j)
    c = d_j.dot(d_j)
    return (np.sqrt(c*(radius**2 - a) + b**2) - b) / c

# page 171 in Nocedal
# find Tau directly by using the quadratic formula
def steihaug(radius, grad_k, hess_k, eps=1e-8):
    z_j = np.zeros(grad_k.size)
    r_j = np.copy(grad_k)
    d_j = -np.copy(grad_k)

    if np.linalg.norm(grad_k) < eps:
        return z_j

    while True:
        dBd = d_j.dot(hess_k).dot(d_j)

        # Negative curvature condition
        if dBd <= 0:
            # return the intersection of the current direction with the trust-region boundary
            return z_j + find_tau(z_j, d_j, radius) * d_j

        alpha_j = (r_j.dot(r_j)) / dBd
        z_old = np.copy(z_j)
        z_j += alpha_j * d_j

        # Trust region condition
        if np.linalg.norm(z_j) >= radius:
            # return the intersection of the current direction with the trust-region boundary
            return z_j + find_tau(z_j, d_j, radius) * d_j

        r_old = np.copy(r_j)
        r_j += alpha_j * hess_k.dot(d_j)

        # Stopping condition
        if np.linalg.norm(r_j) < eps:
            return z_j

        beta_j = (r_j.dot(r_j)) / (r_old.dot(r_old))
        d_j = beta_j * d_j - r_j



# n must be between 0 and 0.25
# from Nocedal page 69
def trust_region(f, f_grad, f_hess, x_0, tol=1e-15, model=quadratic_model, radius_0=1.0, radius_max=300.0, n=1e-3):
    k = 0
    x_k = x_0
    radius = radius_0

    # evaluate function, gradient, hessian, and model function at x_k
    f_k = f(x_k)
    grad_k = f_grad(x_k)
    hess_k = f_hess(x_k)
    model_k = model(f_k, grad_k, hess_k)

    error = []
    error.append(np.linalg.norm(np.subtract(x_0, [1,1])))

    while np.linalg.norm(f_grad(x_k)) > tol:
        # solve subproblem using Steihaug's Method
        p_k = steihaug(radius, grad_k, hess_k)

        # evaluate agreement between model function and actual function
        rho_k = (f(x_k) - f(x_k + p_k)) / (f(x_k) - model_k(p_k))

        if rho_k < 0.25:
            # poor approximation so shrink the trust-radius
            radius = 0.25 * radius
        elif rho_k > 0.75 and np.abs(np.linalg.norm(p_k) - radius) < tol:
            # good approximation and a full-step (within tolerance) was taken, expand trust_radius
            radius = min(2 * radius, radius_max)
        # otherwise leave the radius unchanged

        # if approximation was good, update x to the new step. Otherwise leave x unchanged
        if rho_k > n:
            x_k = x_k + p_k

            # update function evaluation, gradient, hessian, and model function at x_k+1
            f_k = f(x_k)
            grad_k = f_grad(x_k)
            hess_k = f_hess(x_k)
            model_k = model(f_k, grad_k, hess_k)

        k += 1
        error.append(np.linalg.norm(np.subtract(x_k, [1, 1])))

        if np.linalg.norm(p_k) < tol:
            break

    return [x_k, k, error]


# Rosenbrock function
a = 1
b = 100
rosenbrock = lambda x : (a-x[0])**2 + b*(x[1]-x[0]**2)**2

rosen_grad, rosen_hess = compute_gradient_hessian(rosenbrock, [x, y])

trust_rosen_results = trust_region(rosenbrock, rosen_grad, rosen_hess, [5.0, 5.0])





##Executions
##F1
# sol=np.matrix([0.7937005259840997, 0.7937005259840997]).T
# [x,n,log,er]=linetrace(F1,J1,hquad,gradg,[1.5,1.],15.)
# print(x,n,er)
# plt.plot(log[0],log[1],'g', label='Quadratic, Chebychev')
# plt.yscale("log")
# [x,n,log,er]=linetrace(F1,J1,hquad2,gradg,[1.5,1.],15.)
# print(x,n,er)
# plt.plot(log[0],log[1],'b', label='Quadratic')
# plt.yscale("log")
# [x,n,log,er]=linetrace(F1,J1,cons,Newton,[1.5,1.],15.)
# print(x,n,er)
# plt.plot(log[0],log[1],'r', label='Newton')
# plt.yscale("log")
# [x,n,log,er]=linetrace(F1,J1,Wolfe,gradg,[1.5,1.],15.)
# print(x,n,er)
# plt.plot(log[0],log[1],'m', label='Wolfe')
# plt.yscale("log")
# plt.legend()
# plt.xlabel('# of iterations')
# plt.ylabel('Absolute Error')
# plt.show()
##F2
sol=np.matrix([1.,1.]).T
[x,n,log,er]=linetrace(F2,J2,hquad,gradg,[20.0,20.0],15.)
print(x,n,er)
plt.plot(log[0],log[1],'g', label='Quadratic, Chebychev')
plt.yscale("log")
[x,n,log,er]=linetrace(F2,J2,hquad2,gradg,[20.,20.],15.)
print(x,n,er)
plt.plot(log[0],log[1],'b', label='Quadratic')
plt.yscale("log")
# [x,n,log,er]=linetrace(F2,J2,cons,Newton,[5.0,5.0],15.)
# print(x,n,er)
# plt.plot(log[0],log[1],'r', label='Newton')
plt.yscale("log")
[x,n,log,er]=linetrace(F2,J2,Wolfe,gradg,[20.0,20.0],15.)
print(x,n,er)
plt.plot(log[0],log[1],'m', label='Wolfe')

plt.plot(trust_rosen_results[2], label='Trust-Region')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Rosenbrock starting at (5,5)")
# plt.xlim((0, 7500))
plt.xlabel('# of Iterations')
plt.ylabel('Log of Absolute Error')
plt.show()
