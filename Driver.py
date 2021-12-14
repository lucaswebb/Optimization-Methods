from LineSearch import *
from TrustRegion import *

import matplotlib.pyplot as plt



# Rosenbrock function
a = 1
b = 100
rosenbrock = lambda x : (a-x[0])**2 + b*(x[1]-x[0]**2)**2

rosen_grad, rosen_hess = compute_gradient_hessian(rosenbrock, [x, y])

trust_rosen_results = trust_region(rosenbrock, rosen_grad, rosen_hess, [5.0, 5.0])


##Test Functions

def rosenbrock2d(x):##g(x,y)=Rosenbrock
    f=[[1-x[0]],[10*(x[1]-x[0]**2)]]
    return np.matrix(f)
def rosen_jacobian(x):
    j=[[-1,0],[-20*x[0],10]]
    return np.matrix(j)

##Executions

sol=np.matrix([1.,1.]).T
# Quadratic
[x,n,log,er]=linetrace(rosenbrock2d,rosen_jacobian,quadratic,gradg,[5.0,5.0],15., sol)
plt.plot(log[0],log[1],'b', label='Quadratic')

# Wolfe
[x,n,log,er]=linetrace(rosenbrock2d,rosen_jacobian,Wolfe,gradg,[5.0,5.0],15., sol)
plt.plot(log[0],log[1],'m', label='Wolfe')

#Trust Region
plt.plot(trust_rosen_results[2], label='Trust-Region')


# plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Rosenbrock starting at (5,5)")
# plt.xlim((0, 7500))
plt.xlabel('# of Iterations')
plt.ylabel('Log of Absolute Error')
plt.show()
